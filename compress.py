"""
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

LATENT_VECTOR_FILENAME = 'z.pt.gz'
INFO_JSON_FILENAME = 'info.json'
DEFAULT_GEN_CLASS = SizedGenerator # TODO - switch to using SimpleGenerator
DEFAULT_GEN_CKPT = './checkpoints/celeba_cropped/gen_ckpt.49.pt'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
GEN_LATENT_DIM = 64 # TODO - in order to load, need to match whatever latent dim the gen has. Should figure this out from gen_ckpt filename or something, maybe store in info_json
# Compression

def compress(img, compression_ratio, output_filename=None, n_steps=5000, gen_ckpt=DEFAULT_GEN_CKPT):
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
    x = load_target_image(img).to(DEVICE)
    g = load_trained_generator(DEFAULT_GEN_CLASS, gen_ckpt, latent_dim=GEN_LATENT_DIM, num_filters=P.num_filters, image_size=P.size, num_ups=P.num_ups).to(DEVICE)
    g.eval()


    # Decide how to handle linear layer, based on compression_ratio
    torch.manual_seed(0)
    np.random.seed(0)

    # Round down to nearest integer for latent_dim
    latent_dim = (128*128*3) // compression_ratio

    linear_layer = torch.nn.Linear(latent_dim, 8192).to(DEVICE)

    z = torch.randn(latent_dim, device=DEVICE)
    z = torch.nn.Parameter(torch.clamp(z, -1, 1))

    optimizer = torch.optim.Adam([z], lr=0.05, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    for j in trange(n_steps, leave=False):
        optimizer.zero_grad()
        model_input = linear_layer(z)
        x_hat = g(model_input, skip_linear_layer=True).squeeze(0)
        mse = F.mse_loss(x_hat, x)
        mse.backward()
        optimizer.step()
        scheduler.step()

    p = psnr(x, x_hat)
    if output_filename is not None:
        if not output_filename.endswith('.ganz'):
            output_filename += '.ganz'
        write_ganz(output_filename, z, p, gen_ckpt)

    return x_hat, z, p


def write_ganz(output_filename, z, psnr, gen_ckpt):
    info_json = { 'gen_ckpt_name': gen_ckpt, 
                  'sha1sum': sha1sum(gen_ckpt),
                  'PSNR': psnr }
    _write_ganz(output_filename, z, info_json)
    return


def _write_ganz(output_filename, z, info_json):
    os.makedirs(output_filename)
    # Save latent vector
    with gzip.open(os.path.join(output_filename, LATENT_VECTOR_FILENAME), 'wb') as z_handle:
        z_handle.write(z)

    # Save info JSON
    with open(os.path.join(output_filename, INFO_JSON_FILENAME), 'w') as json_handle:
        json.dump(info_json, json_handle)
    return


# Uncompression
def uncompress(input_filename, output_filename=None, gen_ckpt=DEFAULT_GEN_CKPT):
    z, info_json = read_ganz(input_filename)
    if sha1sum(gen_ckpt) != info_json['sha1sum']:
        raise ValueError('Generator Checkpoints do not match - cannot uncompress!')

    g = load_trained_generator(DEFAULT_GEN_CLASS, gen_ckpt, latent_dim=GEN_LATENT_DIM, num_filters=P.num_filters, image_size=P.size, num_ups=P.num_ups).to(DEVICE)
    g.eval()

    x_hat = g(z)
    if output_filename is not None:
        save_image(x_hat, output_filename)
    
    return x_hat


def read_ganz(input_filename):
    # Read latent vector
    with gzip.open(os.path.join(input_filename, LATENT_VECTOR_FILENAME), 'rb') as f:
        z = f.read()

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


class TestGANZ(unittest.TestCase):
    def test_compress_toFile(self):
        img = './images/bananas.jpg'

        output_filename = './images/bananas.ganz'
        if os.path.exists(output_filename):
            shutil.rmtree(output_filename)
        compress(img, 10, output_filename, n_steps=1000)

    def test_compress_noFile(self):
        img = './images/bananas.jpg'
        x_hat, z, psnr = compress(img, 10, n_steps=1000)
        print(psnr)
        self.assertTrue(psnr > 50)

    def test_uncompress(self):
        raise NotImplementedError()

if __name__ == '__main__':
    unittest.main()
