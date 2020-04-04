from os.path import join
from utils import save_gif
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

    for X in [64, 128, 256, 512]:
        gen_folder = join(folder, f'{dataset}_ELU_latent_dim_{X}')

        checkpoints = [join(gen_folder, f'gen_ckpt.{i}.pt') for i in range(50)]

        save_gif(X, checkpoints, f'latent_dim_{X}.gif')


def make_compression_series():
    pass


if __name__ == "__main__":
    make_gifs()
