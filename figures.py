import os
from utils import load_target_image, psnr, save_gif
import torch
from compress import compress
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
from tqdm import tqdm

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
    bn, _ = os.path.splitext(os.path.basename(img_fp))

    CS = True # compressive sensing?
    torch_img = load_target_image(img_fp)
    np_img = torch_img.numpy().transpose((1, 2, 0))

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

    x_hat, _, psnr_gan = compress(torch_img, skip_linear_layer=False, 
            compressive_sensing=CS, n_steps=5)
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
        x_hat, _, psnr_gan = compress(torch_img, c, compressive_sensing=CS, n_steps=5)
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
    return


def make_psnr_scatterplot(imgs_train, imgs_test, imgs_extra):
    """
    For a list of images, compute the PSNR from GANZ and from Wavelets,
    and use these as the X,Y coords in a scatterplot
    """
    fig, ax = plt.subplots(1, 1, figsize=(16,9), constrained_layout=True)
    ax.set_title(f'PSNR')
    ax.set_ylabel('GANZ')
    ax.set_xlabel('Wavelet')

    _scatter(ax, imgs_train, 'CelebA_Train', 'r', '+')
    _scatter(ax, imgs_test, 'CelebA_Test', 'g', 'd')
    _scatter(ax, imgs_extra, 'ImageNet', 'b', 'o')

    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)

    fig.legend()
    plt.savefig(f'./figures/psnr_scatterplot.png')
    return


def _scatter(ax, imgs, label, color, marker, compressive_sensing=False):
    compression_ratio=20
    psnr_ganz = []
    psnr_wave = []
    for img in tqdm(imgs):
        # Find PSNR from GANZ compression
        torch_img = load_target_image(img)
        _, _, p_gan = compress(torch_img,
                compression_ratio=compression_ratio,
                skip_linear_layer=True, 
                compressive_sensing=compressive_sensing, 
                n_steps=5)
        psnr_ganz.append(p_gan)

        # Find PSNR from Wavelet compression
        np_img = torch_img.numpy().transpose((1, 2, 0))
        wavelet_img = wavelet_threshold(np_img, compression_ratio)
        p_wav = psnr(torch.from_numpy(np_img), torch.from_numpy(wavelet_img))
        psnr_wave.append(p_wav)

    ax.scatter(psnr_wave, psnr_ganz, c=color, marker=marker, alpha=0.5, label=label)
    return


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
    # make_compression_series("./images/jack.jpg")
    # make_compression_series("./images/lena.png")
    # make_compression_series("./images/monarch.png")
    # make_compression_series("./images/night.jpg")
    # make_compression_series("./images/ocean.jpg")


    train = ['astronaut.png', 'bananas.jpg']
    test = ['flowers.jpg', 'night.jpg']
    extra = ['ocean.jpg', 'jack.jpg']

    imgs_train = [f'./images/{x}' for x in train]
    imgs_test = [f'./images/{x}' for x in test]
    imgs_extra = [f'./images/{x}' for x in extra]

    make_psnr_scatterplot(imgs_train, imgs_test, imgs_extra)

