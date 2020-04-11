"""
TODO - explain compressed format
Compressed file format:
    Needs to have info about
    - z vector
    - info about generator (which generator to use for uncompress())

    img.ganz/
        z.pt.gz
        info.json

    info.json = { "gen_ckpt_name" : "...",
                  "checksum" : "...",
                  "PSNR" : ... }
"""
import math
import shutil
import hashlib
import gzip
import json
import numpy as np
import os
import torch
from tqdm import trange
import torch.nn.functional as F
from torchvision.utils import save_image
from torch_model import SizedGenerator
from model import SimpleGenerator

from utils import load_target_image, load_trained_generator, psnr
import unittest
import params as P

LATENT_VECTOR_FILENAME = 'z.npz'
INFO_JSON_FILENAME = 'info.json'
DEFAULT_GEN_CLASS = SimpleGenerator
DEFAULT_GEN_CKPT = './checkpoints/celeba_ELU_latent_dim_64/gen_ckpt.49.pt'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
GEN_LATENT_DIM = 64  # TODO - in order to load, need to match whatever latent dim the gen has. Should figure this out from gen_ckpt filename or something, maybe store in info_json
# Compression


def compress(img,
             compression_ratio=None,
             skip_linear_layer=True,
             no_linear_layer=False,
             output_filename=None,
             compressive_sensing=False,
             n_measurements=4000,
             n_steps=5000,
             n_measurements=5000,
             gen_ckpt=DEFAULT_GEN_CKPT,
             latent_dim=GEN_LATENT_DIM):
    """
    Args:
        img - filename of image to be compressed. Must be 128x128x3 pixels
        compression_ratio - (dimension of input) / (dimension of output)
        output_filename - optional, will save the z vector and the generator checkpoint checksum + name

    - figure out whether to skip_linear_layer
    - make a starting z vector
    - optimize
    - return final z vector and the final reconstructed img
    """
    if isinstance(img, str):
        img = load_target_image(img)
    else:
        if not isinstance(img, torch.Tensor):
            raise ValueError(
                "Must provide filename or preprocessed torch.Tensor!")

    x = img.to(DEVICE)
    g = load_trained_generator(DEFAULT_GEN_CLASS,
                               gen_ckpt,
                               latent_dim=latent_dim).to(DEVICE)
    g.eval()

    if skip_linear_layer and no_linear_layer:
        latent_dim = get_latent_dim(compression_ratio)
        assert latent_dim == 8192, "input shape mismatch for CNN"
        linear_layer = lambda z: z
    elif skip_linear_layer:
        latent_dim = get_latent_dim(compression_ratio)
        linear_layer = get_linear_layer(latent_dim)
    else:
        latent_dim = g.latent_dim

    z = torch.randn(latent_dim, device=DEVICE)
    z = torch.nn.Parameter(torch.clamp(z, -1, 1))

    # lr = 0.3/compression_ratio
    lr = 0.05
    #  target_lr_fraction = 0.01  # end at X% of starting LR
    # gamma = 2**(math.log2(target_lr_fraction) / n_steps)
    optimizer = torch.optim.Adam([z], lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    if compressive_sensing:
        A = torch.randn(n_measurements, np.prod(x.shape),
                        device=DEVICE) / math.sqrt(n_measurements)

    for j in trange(n_steps, leave=False):
        optimizer.zero_grad()
        if skip_linear_layer:
            model_input = linear_layer(z)
            x_hat = g(model_input, skip_linear_layer=True).squeeze(0)
        else:
            x_hat = g(z, skip_linear_layer=False).squeeze(0)

        if compressive_sensing:
            mse = F.mse_loss(A @ x_hat.view(-1), A @ x.view(-1))
        else:
            mse = F.mse_loss(x_hat, x)

        mse.backward()
        optimizer.step()
        scheduler.step()

    p = psnr(x, x_hat)
    if output_filename is not None:
        if not output_filename.endswith('.ganz'):
            output_filename += '.ganz'
        file_size = write_ganz(output_filename, z, p, gen_ckpt)
        return x_hat, z, p, file_size
    else:
        return x_hat, z, p


def get_latent_dim(compression_ratio):
    # Round down to nearest integer for latent_dim
    return (128 * 128 * 3) // compression_ratio


def get_linear_layer(latent_dim, output_dim=8192):
    torch.manual_seed(0)
    np.random.seed(0)
    return torch.nn.Linear(latent_dim, output_dim).to(DEVICE)


def write_ganz(output_filename, z, psnr, gen_ckpt):
    info_json = {
        'gen_ckpt_name': gen_ckpt,
        'sha1sum': sha1sum(gen_ckpt),
        'PSNR': psnr
    }
    _write_ganz(output_filename, z, info_json)
    return get_size(output_filename)


def _write_ganz(output_filename, z, info_json):
    os.makedirs(output_filename)
    # Save latent vector

    np.savez_compressed(os.path.join(output_filename, LATENT_VECTOR_FILENAME),
                        z.detach().cpu().numpy())

    # Save info JSON
    with open(os.path.join(output_filename, INFO_JSON_FILENAME),
              'w') as json_handle:
        json.dump(info_json, json_handle)
    return


# Uncompression
def uncompress(inp, output_filename=None, gen_ckpt=DEFAULT_GEN_CKPT):
    if isinstance(inp, str):
        z, info_json = read_ganz(inp)
        if sha1sum(gen_ckpt) != info_json['sha1sum']:
            raise ValueError(
                'Generator Checkpoints do not match - cannot uncompress!')
    else:
        z = inp

    z = z.to(DEVICE)
    latent_dim = z.shape[0]
    print(f'during uncompress, latent_dim {latent_dim}')
    linear_layer = get_linear_layer(latent_dim)

    g = load_trained_generator(DEFAULT_GEN_CLASS,
                               gen_ckpt,
                               latent_dim=GEN_LATENT_DIM).to(DEVICE)
    g.eval()

    model_input = linear_layer(z)
    x_hat = g(model_input, skip_linear_layer=True)
    if output_filename is not None:
        save_image(x_hat, output_filename)

    return x_hat


def read_ganz(input_filename):
    # Read latent vector
    z = np.load(os.path.join(input_filename, LATENT_VECTOR_FILENAME))
    z = torch.from_numpy(z['arr_0'])

    # Read info JSON
    with open(os.path.join(input_filename, INFO_JSON_FILENAME), 'r') as f:
        info_json = json.load(f)

    return z, info_json


def sha1sum(gen_ckpt):
    BUF_SIZE = 65536

    sha1 = hashlib.sha1()
    with open(gen_ckpt, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()


def get_size(start_path):
    if os.path.isfile(start_path):
        return os.path.getsize(start_path) / 1024.
    else:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for d in dirnames:
                total_size += get_size(os.path.join(dirpath, d))
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):  # skip links
                    total_size += os.path.getsize(fp)
        return total_size / 1024.


class TestGANZ(unittest.TestCase):

    def test_all(self):
        img = './images/bananas.jpg'

        output_filename = './images/bananas.ganz'
        if os.path.exists(output_filename):
            shutil.rmtree(output_filename)
        x_hat, z, psnr, fs = compress(img,
                                      compression_ratio=10,
                                      output_filename=output_filename,
                                      n_steps=5000)
        self.assertTrue(psnr > 25)

        result_filename = './images/degraded_bananas.jpg'
        if os.path.exists(result_filename):
            os.remove(result_filename)

        uncompress(output_filename, result_filename)


if __name__ == '__main__':
    unittest.main()
