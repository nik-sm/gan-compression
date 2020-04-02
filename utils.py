import io
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from model import SimpleGenerator
import numpy as np
from os.path import join
import params as P
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def _gen_img(img):
    plt.figure(figsize=(16, 9))
    plt.imshow(img)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf


def save_img_tensorboard(img, writer, tag, epoch=None):
    # Rescale to [0, 1]
    img -= img.min()
    img /= img.max()

    img_buf = _gen_img(img.numpy().transpose(1, 2, 0))
    img = np.array(Image.open(img_buf))
    writer.add_image(tag, img, global_step=epoch, dataformats='HWC')
    return


def save_grid_tensorboard(img_list, writer, tag, epoch=None):
    grid = make_grid(img_list, scale_each=True).numpy().transpose(1, 2, 0)
    img_buf = _gen_img(grid)
    img = np.array(Image.open(img_buf))
    writer.add_image(tag, img, global_step=epoch, dataformats='HWC')
    return


def load_trained_generator(generator_class, generator_checkpoint, *gen_args, **gen_kwargs):
    gen = generator_class(*gen_args, **gen_kwargs)
    try:
        ckpt = torch.load(generator_checkpoint)['model_state_dict']
        fixed_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith('module.'):
                fixed_ckpt[k[7:]] = v
            else:
                fixed_ckpt[k] = v
        gen.load_state_dict(fixed_ckpt)
    except Exception as e:
        print(e)
        gen.load_state_dict(torch.load(generator_checkpoint))

    return gen


def jpeg_compress(img, quality_layers, quality_mode='rates'):
    """
        quality_mode: 'rates' - compression ratio. 'dB' - SNR value in decibels

    Example usage:
        i = 'images/bananas.jpg'
        results = jpeg_compress(i, [30], 'dB')
        results.save('test.jpg')

    """
    img = Image.open(img)
    outputIoStream = io.BytesIO()
    img.save(outputIoStream, "JPEG2000", quality_mode=quality_mode, quality_layers=quality_layers)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)


def save_gif(gen_checkpoints, n_row=2, n_col=3, gen_class=SimpleGenerator):
    """
    TODO - work-in-progress
    Input:
        gen_checkpoints: list of filenames of pretrained generator checkpoints
        n_row: the final GIF will have 'n_row' rows
        n_col: the final GIF will have 'n_col' cols
        gen_class: the python class of the generator

    Output:
        saves 1 GIF, containing 1 frame per checkpoint. In each frame, we see the 
        same (n_row x n_col) latent points, and how they look at the current timestep.
    """
    # TODO - settings
    latent_dim = 64 
    gen_args = []
    gen_kwargs = {}
    device = 'cuda:0'
    output_dir = 'movies'
    output_name = 'TEST.gif'

    # We need to select n_row * n_col points to track through time
    z = torch.randn(n_row * n_col, latent_dim, device=device)
    movie = []

    # Generate the frames
    for c in gen_checkpoints:
        g = load_trained_generator(gen_class, c, *gen_args, **gen_kwargs)
        g.eval()

        # Produce a batch of images, shape ((n_row * n_col) x H x W x Channel)
        batch = g(z).detach().to('cpu')
        img_height = batch.shape[1]
        img_width = batch.shape[2]
        n_channels = batch.shape[3]

        # Normalize each face in the frame separately
        batch -= batch.view(n_row * n_col, -1).min(1).values[:,None,None,None]
        batch /= batch.view(n_row * n_col, -1).max(1).values[:,None,None,None]

        # Reshape the batch, stacking the individual faces around into a grid of shape
        # n_row by n_col
        frame = batch.numpy().reshape(n_row * img_height, n_col * img_width, n_channels)

        movie.append(frame)

    # Write the GIF
    fps = 2
    filename = join(output_dir, output_name)
    ImageSequenceClip(movie).write_gif(filename, fps=fps)
    return


def load_target_image(filename):
    if filename.endswith('.pt'):
        x = torch.load(filename)
    else:
        image = Image.open(filename)
        t = transforms.Compose([
            # TODO - ideal is:
            # - if img rectangular, cut into square
            # - then resize to (P.size, P.size)
            transforms.Resize((P.size, P.size)),
            transforms.ToTensor()])
        x = t(image)
    return x


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        raise ValueError("how do we handle a perfect reconstruction?")
    pixel_max = torch.tensor(1.0)
    p = 20 * torch.log10(pixel_max) - 10 * torch.log10(mse)
    if isinstance(p. torch.Tensor):
        p = p.item()
    return p
