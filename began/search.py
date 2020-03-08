import argparse
from datetime import datetime
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
import sys
from PIL import Image
from torch_model import SizedGenerator, SizedDiscriminator
import os
from tqdm import trange

import params as P
from utils import save_img_tensorboard

def load_trained_generator(gen_checkpoint):
    gen = SizedGenerator(P.latent_dim, P.num_filters, P.size, P.num_ups)
    try:
        gen.load_state_dict(torch.load(gen_checkpoint)['model_state_dict'])
    except:
        gen.load_state_dict(torch.load(gen_checkpoint))

    gen.eval()
    return gen.to('cuda:0')

def load_target_image(filename):
    image = Image.open(filename)
    t = transforms.Compose([
        #transforms.CenterCrop(P.size),
        transforms.Resize(P.size),
        transforms.ToTensor()])
    x = t(image)
#    x.unsqueeze_(0)
    return x.to('cuda:0')

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
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    x = load_target_image(args.image)
    save_img_tensorboard(x.squeeze(0).detach().cpu(), writer, f'original')

    g = load_trained_generator(args.gen_checkpoint)

    save_every_n = 50

    for i in trange(args.n_restarts):
        seed = i
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Good options for starting z:
        # - start at origin (all 0)
        # - start from a normal centered at 0, try various values of std (starting smallest, maybe try 3 values)
        mean = 0
        std = 1
        z = torch.nn.Parameter(torch.randn(64, device='cuda:0') * std + mean)

        z_initial = z.data.clone()
        optimizer = torch.optim.AdamW([z], lr=0.5e-2)
        gamma = 2 ** (math.log(0.25, 2) / args.n_steps) # Want to hit 0.25*initial_lr after n_steps steps (arbitrary design)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

        with torch.no_grad():
            save_img_tensorboard(g(z_initial).squeeze(0).detach().cpu(), writer, f'restart_{i}/beginning')

        for j in trange(args.n_steps, leave=False):
            optimizer.zero_grad()
            #x_hat = (g(z).squeeze(0) + 1) / 2
            x_hat = g(z).squeeze(0)
            mse = F.mse_loss(x_hat, x)
            mse.backward()
            optimizer.step()
            scheduler.step()

            writer.add_scalar(f'MSE/{i}', mse, j)
            writer.add_scalar(f'PSNR/{i}', psnr(x, x_hat), j)

            if j % save_every_n == 0 or j == n_steps - 1:
                save_img_tensorboard(x_hat.squeeze(0).detach().cpu(), writer, f'restart_{i}/reconstruction', j)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--generator_checkpoint', required=True, help="Ex.: --generator_checkpoint ./checkpoints/celeba_50k/gen_ckpt.20.pt")
    p.add_argument('--image', required=True)
    p.add_argument('--run_name', default=datetime.now().isoformat())
    p.add_argument('--n_restarts', type=int, default=3)
    p.add_argument('--n_steps', type=int, default=3000)
    args = p.parse_args()
    main(args)
