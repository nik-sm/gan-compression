import argparse
from datetime import datetime
import math
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from PIL import Image
from torch_model import SizedGenerator
import os
from tqdm import trange

import params as P
from utils import save_img_tensorboard, load_trained_generator


def load_target_image(filename):
    if filename.endswith('.pt'):
        x = torch.load(filename)
    else:
        image = Image.open(filename)
        t = transforms.Compose([
            #transforms.CenterCrop(P.size), # NOTE - we should carefully crop target images
            transforms.Resize(P.size),
            transforms.ToTensor()])
        x = t(image)
    return x

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        raise ValueError("how do we handle a perfect reconstruction?")
    pixel_max = torch.tensor(1.0)
    return 20 * torch.log10(pixel_max) - 10 * torch.log10(mse)

def output_to_imshow(v):
    return v.squeeze(0).detach().to('cpu').numpy().transpose(1,2,0)

def main(args):

    logdir = f'tensorboard_logs/search/{args.run_name}'
    os.makedirs(logdir, exist_ok=True) # TODO - decide whether to clobber or what?

    writer = SummaryWriter(logdir)

    device='cuda:0' 

    x = load_target_image(args.image).to(device)
    save_img_tensorboard(x.squeeze(0).detach().cpu(), writer, f'original')

    g = load_trained_generator(SizedGenerator, args.generator_checkpoint, device, latent_dim=64, num_filters=P.num_filters, image_size=P.size, num_ups=P.num_ups)
    g.eval()

    if args.latent_dim != g.latent_dim:
        args.skip_linear_layer = True
    else:
        args.skip_linear_layer = False

    if args.skip_linear_layer and args.latent_dim < 8192:
        # Then we need a new linear layer to map to dimension 8192
        linear_layer = torch.nn.Linear(args.latent_dim, 8192).to(device)
    else:
        linear_layer = lambda x: x

    save_every_n = 50

    for i in trange(args.n_restarts):
        seed = i
        torch.manual_seed(seed)
        np.random.seed(seed)

        # NOTE - based on quick experiments:
        # - std=1.0 better than std=0.1 or std=0.01
        # - uniform and normal performed nearly identical
        if args.initialization == 'uniform':
            z = (2 * args.std) * torch.rand(args.latent_dim, device=device) - args.std
        elif args.initialization == 'normal':
            z = torch.randn(args.latent_dim, device=device) * args.std
        elif args.initialization == 'ones':
            mask = torch.rand(args.latent_dim) < 0.5
            z = torch.ones(args.latent_dim, device=device)
            z[mask] = -1
        else:
            raise NotImplementedError(args.initialization)

        # network only saw [-1, 1] during training
        z = torch.nn.Parameter(torch.clamp(z, -1, 1))

        z_initial = z.data.clone()
        optimizer = torch.optim.Adam([z], lr=0.05, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_steps)

        with torch.no_grad():
            model_input = linear_layer(z_initial)
            save_img_tensorboard(g(model_input, skip_linear_layer=args.skip_linear_layer).squeeze(0).detach().cpu(), writer, f'restart_{i}/beginning')

        for j in trange(args.n_steps, leave=False):
            optimizer.zero_grad()
            model_input = linear_layer(z)
            x_hat = g(model_input, skip_linear_layer=args.skip_linear_layer).squeeze(0)
            mse = F.mse_loss(x_hat, x)
            mse.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar(f'MSE/{i}', mse, j)
            writer.add_scalar(f'PSNR/{i}', psnr(x, x_hat), j)

            if j % save_every_n == 0:
                save_img_tensorboard(x_hat.squeeze(0).detach().cpu(), writer, f'restart_{i}/reconstruction', j)

        save_img_tensorboard(x_hat.squeeze(0).detach().cpu(), writer, f'restart_{i}/final')

def get_latent_dims(x):
    x = int(x)
    if x > 8192:
        raise ValueError('give a latent_dim between [1, 8192]')
    return x

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--generator_checkpoint', default='./checkpoints/celeba_cropped/gen_ckpt.49.pt', help="Path to generator checkpoint")
    p.add_argument('--image', required=True)
    p.add_argument('--run_name', default=datetime.now().isoformat())
    p.add_argument('--n_restarts', type=int, default=3)
    p.add_argument('--n_steps', type=int, default=3000)
    p.add_argument('--initialization', choices=['uniform', 'normal', 'ones'], default='normal')
    p.add_argument('--std', type=float, default=1.0, help='for normal dist, the std. for uniform, the min and max val')
    p.add_argument('--latent_dim', type=get_latent_dims, default=4096, help='int between [1, 8192]')
    args = p.parse_args()

    # TODO - if model used latent_dim=64 and you also wanan reconstruct from 64, 
    # does it hurt to just skip linear layer?
    main(args)
