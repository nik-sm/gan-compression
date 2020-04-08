import os
from utils import load_target_image, psnr, save_gif
import torch
from compress import compress
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

from wavelet import wavelet_threshold
from compress import get_latent_dim


def make_gifs():
    folder = 'checkpoints'
    dataset = 'celeba'

    for X in [64, 128, 512]:
        gen_folder = os.path.join(folder, f'{dataset}_ELU_latent_dim_{X}')

        checkpoints = [
            os.path.join(gen_folder, f'gen_ckpt.{i}.pt') for i in range(50)
        ]

        save_gif(X, checkpoints, f'./figures/latent_dim_{X}.gif')


def make_compression_series(img_fp, ratios=[20, 40, 60, 100, 1000]):
    orig_img = load_target_image(img_fp).numpy().transpose((1, 2, 0))

    # Original image after transform (now size 128 x 128 x 3)
    transformed_img = Image.fromarray((orig_img * 255).astype(np.uint8))
    p, ext = os.path.splitext(img_fp)
    bn = os.path.basename(p)
    transformed_img_fp = f'{p}_transformed{ext}'
    transformed_img.save(transformed_img_fp)

    fig, axes = plt.subplots(len(ratios),
                             2,
                             figsize=(12, 6),
                             sharex=True,
                             sharey=True,
                             gridspec_kw={
                                 'hspace': 0,
                                 'wspace': 0
                             })

    for i, c, in enumerate(ratios):
        # GAN compression
        x_hat, _, p = compress(transformed_img_fp, c, n_steps=1)
        gan_img = x_hat.detach().cpu().numpy().transpose((1, 2, 0))
        axes[i, 0].imshow(gan_img)
        axes[i, 0].set_title(f'PSNR={p:.2f}dB')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # Wavelet compression
        wavelet_img = wavelet_threshold(np.array(transformed_img), c)
        wavelet_img = wavelet_img / 255.
        print(wavelet_img.min(), wavelet_img.max())
        axes[i, 1].imshow(wavelet_img)
        p = psnr(torch.from_numpy(np.array(transformed_img)),
                 torch.from_numpy(wavelet_img))
        axes[i, 1].set_title(f'PSNR={p:.2f}dB')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        axes[i, 0].set_ylabel(f'Ratio={c}')

    plt.tight_layout()
    fig.savefig(f"./figures/{bn}.compression_series.png")


if __name__ == "__main__":
    os.makedirs("./figures", exist_ok=True)
    make_gifs()
    # make_compression_series("./images/monarch.png")
