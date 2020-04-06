import os
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
        gen_folder = os.path.join(folder, f'{dataset}_ELU_latent_dim_{X}')

        checkpoints = [
            os.path.join(gen_folder, f'gen_ckpt.{i}.pt') for i in range(50)
        ]

        save_gif(X, checkpoints, f'./figures/latent_dim_{X}.gif')


def make_compression_series(img_fp, ratios=[5, 10, 15, 20]):
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

    # axes[0, 0].imshow(orig_img)
    # axes[0, 0].set_title('Original')
    # axes[0, 0].set_xticks([])
    # axes[0, 0].set_yticks([])

    ax[0].annotate()

    for c, i in zip(ratios, range(len(ratios))):
        # GAN compression
        x_hat, _, p = compress(transformed_img_fp, c, n_steps=1000)
        gan_img = x_hat.detach().cpu().numpy().transpose((1, 2, 0))
        axes[i, 0].imshow(gan_img)
        axes[i, 0].set_title(f'PSNR={p:.2f}dB')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # JPEG2000 Compression
        jpeg_img = np.asarray(jpeg_compress(transformed_img_fp, [c],
                                            'dB')) / 255.
        axes[i, 1].imshow(jpeg_img)
        p = psnr(torch.from_numpy(orig_img), torch.from_numpy(jpeg_img))
        axes[i, 1].set_title(f'PSNR={p:.2f}dB')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        axes[i, 0].set_ylabel(f'Ratio={c}')

    fig.savefig(f"./figures/{bn}.png")


if __name__ == "__main__":
    os.makedirs("./figures", exist_ok=True)
    make_gifs()
    make_compression_series("./images/bananas.jpg")
