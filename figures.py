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


def make_compression_series(img_fp, ratios=[10, 20, 50, 100, 768]):
    # TODO - we can add one more target ratio,
    # e.g. for GAN64 it would be 768, for GAN128 it would be 384, etc
    # This means we would be making these series for various gans,
    # and compress() needs to take an argument specifying which one
    # (so that we can tell out here what the gen.latent_dim is)
    np_img = load_target_image(img_fp).numpy().transpose((1, 2, 0))

    transformed_img = Image.fromarray((np_img * 255).astype(np.uint8))
    p, ext = os.path.splitext(img_fp)
    bn = os.path.basename(p)
    transformed_img_fp = f'{p}_transformed.png'
    transformed_img.save(transformed_img_fp)

    fig, axes = plt.subplots(len(ratios)+1,
                             2,
                             figsize=(5, 17),
                             gridspec_kw={
                                 'wspace': 0,
                                 'hspace': 0.13
                             },
                             constrained_layout=True)

    # Show original image and keep_linear version
    axes[0, 0].set_title('Original', fontsize=18)
    axes[0, 1].set_title('Keep_Linear', fontsize=18)

    axes[0, 0].imshow(np_img)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    x_hat, _, psnr_gan = compress(transformed_img_fp, skip_linear_layer=False, n_steps=5000)
    gan_img = x_hat.detach().cpu().numpy().transpose((1, 2, 0))
    axes[0, 1].imshow(gan_img)
    axes[0, 1].set_xlabel(f'PSNR={psnr_gan:.2f}dB', fontsize=14)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # Show varying compression ratios
    axes[1, 0].set_title('GAN', fontsize=18)
    axes[1, 1].set_title('Wavelet', fontsize=18)
    for i, c, in enumerate(ratios):
        i += 1 # offset by 1
        # GAN compression
        x_hat, _, psnr_gan = compress(transformed_img_fp, c, n_steps=5000)
        gan_img = x_hat.detach().cpu().numpy().transpose((1, 2, 0))
        axes[i, 0].imshow(gan_img)
        axes[i, 0].set_xlabel(f'PSNR={psnr_gan:.2f}dB', fontsize=14)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 0].set_ylabel(f'Ratio={c}', fontsize=16)

        # Wavelet compression
        wavelet_img = wavelet_threshold(np_img, c)
        axes[i, 1].imshow(wavelet_img)
        psnr_wav = psnr(torch.from_numpy(np_img), torch.from_numpy(wavelet_img))
        axes[i, 1].set_xlabel(f'PSNR={psnr_wav:.2f}dB', fontsize=14)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

    fig.savefig(f"./figures/{bn}.compression_series.png")


def make_psnr_scatterplot():
    """
    For a list of images, compute the PSNR from GANZ and from Wavelets,
    and use these as the X,Y coords in a scatterplot

    images = []
    p_ganz = []
    p_wave = []
    for image in images:
        p_ganz.append(...)
        p_wave.append(...)

    plt.scatter(p_ganz, p_wave)
    plt.set_title('PSNR
    """
    pass





if __name__ == "__main__":
    os.makedirs("./figures", exist_ok=True)
    # Make gifs for different latent dims
    # make_gifs()

    # Celeba Train
    #make_compression_series("./dataset/celeba_preprocessed/train/034782.pt")

    # Celeba Test
    # make_compression_series("./dataset/celeba_preprocessed/test/193124.pt")
    # make_compression_series("./dataset/celeba_preprocessed/test/196479.pt")
    # make_compression_series("./dataset/celeba_preprocessed/test/197511.pt")

    # # FFHQ Test
    # make_compression_series("./dataset/ffhq_preprocessed/test/17205.pt")
    # make_compression_series("./dataset/ffhq_preprocessed/test/41272.pt")
    # make_compression_series("./dataset/ffhq_preprocessed/test/61998.pt")

    # # Random
    # make_compression_series("./images/astronaut.png")
    # make_compression_series("./images/bananas.jpg")
    make_compression_series("./images/jack.jpg")
    # make_compression_series("./images/lena.png")
    # make_compression_series("./images/monarch.png")
    # make_compression_series("./images/night.jpg")
    # make_compression_series("./images/ocean.jpg")

