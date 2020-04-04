from os.path import join, splitext
from utils import jpeg_compress, load_target_image, psnr, save_gif
import torch
from compress import compress
import matplotlib.pyplot as plt
from PIL import Image, ImageMath
import numpy as np
"""
1) make GIFs of training
2) for each generator, make side-by-side columns 
    at the top, the original image
    below, on left: GANz. on right: JPEG
    from top to bottom: decreasing quality/increasing compression ratio

"""


def make_gifs():
    folder = 'checkpoints'
    dataset = 'celeba'

    for X in [64, 128, 512]:
        gen_folder = join(folder, f'{dataset}_ELU_latent_dim_{X}')

        checkpoints = [join(gen_folder, f'gen_ckpt.{i}.pt') for i in range(50)]

        save_gif(X, checkpoints, f'latent_dim_{X}.gif')


def make_compression_series(img_fp, ratios=[2, 4, 6, 8]):
    orig_img = load_target_image(img_fp).numpy().transpose((1, 2, 0))

    # Original image after transform (now size 128 x 128 x 3)
    transformed_img = Image.fromarray((orig_img * 255).astype(np.uint8))
    p, ext = splitext(img_fp)
    transformed_img_fp = f'{p}_transformed{ext}'
    transformed_img.save(transformed_img_fp)

    fig, axs = plt.subplots(len(ratios) + 1, 2, sharex=True, sharey=True)
    axs[0, 0].imshow(orig_img)
    axs[0, 0].set_title('Original')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    fig.delaxes(ax=axs[0, 1])

    for c, i in zip(ratios, range(1, len(ratios) + 1)):
        # GAN compression
        x_hat, _, p = compress(transformed_img_fp, c)
        gan_img = x_hat.detach().cpu().numpy().transpose((1, 2, 0))
        axs[i, 0].imshow(gan_img)
        axs[i, 0].set_title(f'GAN: PSNR={p:.2f}dB')
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])

        # JPEG2000 Compression
        jpeg_img = np.asarray(jpeg_compress(transformed_img_fp, [c],
                                            'rates')) / 255.
        axs[i, 1].imshow(jpeg_img)
        p = psnr(torch.from_numpy(orig_img), torch.from_numpy(jpeg_img))
        axs[i, 1].set_title(f'JPEG2000: PSNR={p:.2f}dB')
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])

        axs[i, 0].set_ylabel(f'Ratio={c}')

    fig.savefig("./images/out.png")


if __name__ == "__main__":
    make_compression_series("./images/bananas.jpg")
