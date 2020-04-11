import os
from utils import load_target_image, psnr, save_gif
import torch
from compress import compress, get_size, read_ganz
from PIL import Image
import numpy as np
import math
from tqdm import tqdm
import pathlib
import tempfile

from wavelet import wavelet_threshold
from compress import get_latent_dim

# Use non-interactive backend
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_gifs():
    folder = 'checkpoints'
    dataset = 'celeba'

    for X in [64, 128, 512]:
        gen_folder = os.path.join(folder, f'{dataset}_ELU_latent_dim_{X}')

        checkpoints = [
            os.path.join(gen_folder, f'gen_ckpt.{i}.pt') for i in range(50)
        ]

        save_gif(X, checkpoints, f'./figures/latent_dim_{X}.gif')


def single_image(img_fp,
                 cratio=6,
                 no_linear_layer=False,
                 n_steps=7500,
                 n_measure=5000,
                 CS=False):
    if no_linear_layer and cratio != 6:
        raise ValueError('bad combination')

    bn, _ = os.path.splitext(os.path.basename(img_fp))
    torch_img = load_target_image(img_fp)
    np_img = torch_img.numpy().transpose((1, 2, 0))

    fig, axes = plt.subplots(1,
                             2,
                             figsize=(16, 9),
                             gridspec_kw={
                                 'wspace': 0,
                                 'hspace': 0.13
                             },
                             constrained_layout=True)

    axes[0].set_title('Original', fontsize=52)
    axes[0].imshow(np_img)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].set_title(f'GANZ, CR={cratio}, CS={CS}', fontsize=52)
    x_hat, _, psnr_gan = compress(torch_img,
                                  skip_linear_layer=True,
                                  no_linear_layer=no_linear_layer,
                                  compression_ratio=cratio,
                                  compressive_sensing=CS,
                                  n_measurements=n_measure,
                                  n_steps=n_steps)
    gan_img_2 = x_hat.detach().cpu().numpy().transpose((1, 2, 0))
    axes[1].imshow(gan_img_2)
    axes[1].set_xlabel(f'PSNR={psnr_gan:.2f}dB', fontsize=48)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    if CS:
        name = "./figures/{}.comp.CR={}.CS={}.n_steps={}.n_measure={}.png".format(
            bn, cratio, CS, n_steps, n_measure)
    else:
        name = "./figures/{}.comp.CR={}.CS={}.n_steps={}.png".format(
            bn, cratio, CS, n_steps)

    fig.savefig(name)
    return psnr_gan


# Unfinished code
def compressive_sensing_psnr(split='test', measure_fraction=0.1):
    pass
    imgs_test = sorted(
        pathlib.Path('./data/celeba_preprocessed/test').rglob('*'))[:n_imgs]
    imgs_test = [str(x) for x in imgs_test]

    n_measurements = int(measure_fraction * (128 * 128 * 3))

    comp_sens_psnr = []
    for image in imgs_test:
        torch_img = load_target_image(img_fp)
        _, _, psnr_gan = compress(torch_img,
                                  skip_linear_layer=True,
                                  no_linear_layer=False,
                                  compression_ratio=6,
                                  compressive_sensing=True,
                                  n_measurements=n_measurements,
                                  n_steps=n_steps)


def side_by_side_8192(img_fp):
    bn, _ = os.path.splitext(os.path.basename(img_fp))
    n_steps = 7500

    torch_img = load_target_image(img_fp)
    np_img = torch_img.numpy().transpose((1, 2, 0))

    fig, axes = plt.subplots(1,
                             2,
                             figsize=(16, 9),
                             gridspec_kw={
                                 'wspace': 0,
                                 'hspace': 0.13
                             },
                             constrained_layout=True)

    axes[0].set_title('Square_Linear', fontsize=52)
    x_hat_1, _, psnr_gan_1 = compress(torch_img,
                                      skip_linear_layer=True,
                                      no_linear_layer=False,
                                      compression_ratio=6,
                                      compressive_sensing=False,
                                      n_steps=n_steps)
    gan_img_1 = x_hat_1.detach().cpu().numpy().transpose((1, 2, 0))
    axes[0].imshow(gan_img_1)
    axes[0].set_xlabel(f'PSNR={psnr_gan_1:.2f}dB', fontsize=48)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].set_title('No_Linear', fontsize=52)
    x_hat_2, _, psnr_gan_2 = compress(torch_img,
                                      skip_linear_layer=True,
                                      no_linear_layer=True,
                                      compression_ratio=6,
                                      compressive_sensing=False,
                                      n_steps=n_steps)
    gan_img_2 = x_hat_2.detach().cpu().numpy().transpose((1, 2, 0))
    axes[1].imshow(gan_img_2)
    axes[1].set_xlabel(f'PSNR={psnr_gan_2:.2f}dB', fontsize=48)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.savefig(f"./figures/{bn}.square_linear_layer_comparison.png")
    return


def make_compression_series(img_fp, ratios=[10, 20, 50, 100, 768]):
    # TODO - we can add one more target ratio,
    # e.g. for GAN64 it would be 768, for GAN128 it would be 384, etc
    # This means we would be making these series for various gans,
    # and compress() needs to take an argument specifying which one
    # (so that we can tell out here what the gen.latent_dim is)

    # TODO : compressed sensing with keep linear_layer, latent_dim=8192 try 3 different images for each?

    # Change to png first
    bn, _ = os.path.splitext(os.path.basename(img_fp))

    CS = False  # compressive sensing?
    torch_img = load_target_image(img_fp)
    np_img = torch_img.numpy().transpose((1, 2, 0))

    fig, axes = plt.subplots(len(ratios) + 1,
                             2,
                             figsize=(5, 18),
                             gridspec_kw={
                                 'wspace': 0,
                                 'hspace': 0.13
                             },
                             constrained_layout=True)

    # Show original image and keep_linear version
    axes[0, 0].set_title('Original', fontsize=24)
    axes[0, 1].set_title('Keep_Linear', fontsize=24)

    # TODO change to file size of transformed image
    with tempfile.TemporaryDirectory() as tmpdir:
        axes[0, 0].imshow(np_img)
        transformed_img_fp = os.path.join(tmpdir, 'transformed_img.npz')
        np.savez_compressed(transformed_img_fp, np_img)
        axes[0, 0].set_xlabel(f'Size={get_size(transformed_img_fp):.2f}kB',
                              fontsize=18)
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])

        x_hat, _, psnr_gan, fs = compress(torch_img,
                                          skip_linear_layer=False,
                                          compressive_sensing=CS,
                                          n_steps=7500,
                                          output_filename=tmpdir + "/")
    gan_img = x_hat.detach().cpu().numpy().transpose((1, 2, 0))
    axes[0, 1].imshow(gan_img)
    axes[0, 1].set_xlabel(f'PSNR={psnr_gan:.2f}dB\nSize={fs:.2f}kB',
                          fontsize=18)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # Show varying compression ratios
    axes[1, 0].set_title('GAN', fontsize=24)
    axes[1, 1].set_title('Wavelet', fontsize=24)

    for i, c, in enumerate(ratios):
        with tempfile.TemporaryDirectory() as tmpdir:
            i += 1  # offset by 1
            # GAN compression
            x_hat, _, psnr_gan, fs_gan = compress(torch_img,
                                                  c,
                                                  compressive_sensing=CS,
                                                  n_steps=7500,
                                                  output_filename=tmpdir + "/")
            gan_img = x_hat.detach().cpu().numpy().transpose((1, 2, 0))
            axes[i, 0].imshow(gan_img)
            axes[i, 0].set_xlabel(f'PSNR={psnr_gan:.2f}dB\nSize={fs_gan:.2f}kB',
                                  fontsize=18)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            axes[i, 0].set_ylabel(f'Ratio={c}', fontsize=24)

            # Wavelet compression
            wavelet_img, _ = wavelet_threshold(np_img, c)
            axes[i, 1].imshow(wavelet_img)
            psnr_wav = psnr(torch.from_numpy(np_img),
                            torch.from_numpy(wavelet_img))
            axes[i, 1].set_xlabel(f'PSNR={psnr_wav:.2f}dB', fontsize=18)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])

    fig.savefig(f"./figures/{bn}.compression_series.png")
    return


def make_psnr_scatterplot():
    """
    For a list of images, compute the PSNR from GANZ and from Wavelets,
    and use these as the X,Y coords in a scatterplot

    Settings:
    - 7500 steps for each compression
    - 20 images per group (train/test/extra)
    - 4 generator checkpoints
    - 2 target compression ratios
    """
    n_imgs = 20
    imgs_train = sorted(
        pathlib.Path('./dataset/celeba_preprocessed/train').rglob('*'))[:n_imgs]
    imgs_test = sorted(
        pathlib.Path('./dataset/celeba_preprocessed/test').rglob('*'))[:n_imgs]
    imgs_extra = sorted(
        pathlib.Path('./images/out_of_domain/').rglob('*'))[:n_imgs]

    imgs_train = [str(x) for x in imgs_train]
    imgs_test = [str(x) for x in imgs_test]
    imgs_extra = [str(x) for x in imgs_extra]

    gen_ckpts = {
        './checkpoints/celeba_ELU_latent_dim_64/gen_ckpt.49.pt': ('r', 64),
        './checkpoints/celeba_ELU_latent_dim_128/gen_ckpt.49.pt': ('y', 128),
        './checkpoints/celeba_ELU_latent_dim_256/gen_ckpt.49.pt': ('g', 256),
        './checkpoints/celeba_ELU_latent_dim_512/gen_ckpt.49.pt': ('b', 512)
    }

    cratios = {10: 'x', 30: 'o'}

    # Training_dim
    _scatter(gen_ckpts, {10: 'x'}, imgs_train, 'Train (training_dim)', CR=False)
    _scatter(gen_ckpts, {10: 'x'}, imgs_test, 'Test (training_dim)', CR=False)
    _scatter(gen_ckpts, {10: 'x'},
             imgs_extra,
             'Out-of-Domain (training_dim)',
             CR=False)

    # Compression Ratios
    _scatter(
        {'./checkpoints/celeba_ELU_latent_dim_64/gen_ckpt.49.pt': ('r', 64)},
        cratios,
        imgs_train,
        'Train (compression ratio)',
        CR=True)
    _scatter(
        {'./checkpoints/celeba_ELU_latent_dim_64/gen_ckpt.49.pt': ('r', 64)},
        cratios,
        imgs_test,
        'Test (compression ratio)',
        CR=True)
    _scatter(
        {'./checkpoints/celeba_ELU_latent_dim_64/gen_ckpt.49.pt': ('r', 64)},
        cratios,
        imgs_extra,
        'Out-of-Domain (compression ratio)',
        CR=True)

    return


def _scatter(gen_ckpts, cratios, imgs, title, CR=False):
    compressive_sensing = False

    fig, ax = plt.subplots(1, 1, figsize=(9, 9), constrained_layout=True)
    ax.set_title(f'PSNR on {title}', fontsize=28)
    ax.set_ylabel('GANZ', fontsize=28)
    ax.set_xlabel('Wavelet', fontsize=28)

    lims = [10, 50
           ]  # TODO - Check that this is a safe lower bound, or set empirically
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.plot(lims, lims, 'k--', alpha=0.3)
    # ax.plot(ax.get_xlim(), ax.get_ylim(), 'k--', alpha=0.3)

    for g, (color, latent_dim) in gen_ckpts.items():
        for cratio, marker in cratios.items():
            psnr_ganz = []
            psnr_wave = []
            for img in tqdm(imgs):
                # Find PSNR from GANZ compression
                torch_img = load_target_image(img)
                _, _, p_gan = compress(torch_img,
                                       compression_ratio=cratio,
                                       skip_linear_layer=True,
                                       compressive_sensing=compressive_sensing,
                                       n_steps=7500,
                                       gen_ckpt=g,
                                       latent_dim=latent_dim)
                psnr_ganz.append(p_gan)

                # Find PSNR from Wavelet compression
                np_img = torch_img.numpy().transpose((1, 2, 0))
                wavelet_img = wavelet_threshold(np_img, cratio)
                p_wav = psnr(torch.from_numpy(np_img),
                             torch.from_numpy(wavelet_img))
                psnr_wave.append(p_wav)
            if CR:
                label = f'CR={cratio}'
            else:
                label = f'Training_dim={latent_dim}'
            ax.scatter(psnr_wave,
                       psnr_ganz,
                       c=color,
                       marker=marker,
                       s=(len(imgs) * len(gen_ckpts) * len(cratios)) * 1.5,
                       alpha=0.5,
                       label=label)

    ax.legend(loc='upper_left', fontsize=20)
    fig.savefig(f'./figures/psnr_scatterplot.{title}.png')
    return


if __name__ == "__main__":
    os.makedirs("./figures", exist_ok=True)
    # Make gifs for different latent dims
    # make_gifs()

    # Celeba Train
    # make_compression_series("./dataset/celeba_preprocessed/train/034782.pt")

    # Celeba Test
    # make_compression_series("./dataset/celeba_preprocessed/test/196479.pt")

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
    # make_compression_series("./images/horsehead_nebula.jpg")

    # make_psnr_scatterplot()

    side_by_side_8192("./dataset/celeba_preprocessed/train/034782.pt")
    side_by_side_8192("./dataset/celeba_preprocessed/test/196479.pt")
    side_by_side_8192("./images/jack.jpg")
    side_by_side_8192("./images/ferns.jpg")
    side_by_side_8192("./images/horsehead_nebula.jpg")
    side_by_side_8192("./images/subway.jpg")
    side_by_side_8192("./images/campfire.jpg")
    side_by_side_8192("./images/ocean.jpg")
    side_by_side_8192("./images/snow_tree.jpg")

    # single_image("./dataset/celeba_preprocessed/train/034782.pt",
    #              cratio=6,
    #              n_steps=7500,
    #              CS=True,
    #              n_measure=4000)
    # single_image("./dataset/celeba_preprocessed/test/196479.pt",
    #              cratio=6,
    #              n_steps=7500,
    #              CS=True,
    #              n_measure=4000)
    # single_image("./images/jack.jpg",
    #              cratio=6,
    #              n_steps=7500,
    #              CS=True,
    #              n_measure=5000)
