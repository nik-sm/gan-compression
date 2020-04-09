import io
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from model import SimpleGenerator
import numpy as np
from os.path import join
import params as P
from PIL import Image, ImageDraw
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


def load_trained_generator(generator_class, generator_checkpoint, *gen_args,
                           **gen_kwargs):
    gen = generator_class(*gen_args, **gen_kwargs)
    try:
        ckpt = torch.load(generator_checkpoint,
                          map_location='cpu')['model_state_dict']
        fixed_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith('module.'):
                fixed_ckpt[k[7:]] = v
            else:
                fixed_ckpt[k] = v
        gen.load_state_dict(fixed_ckpt)
    except Exception as e:
        print(e)
        gen.load_state_dict(torch.load(generator_checkpoint,
                                       map_location='cpu'))

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
    img.save(outputIoStream,
             "JPEG2000",
             quality_mode=quality_mode,
             quality_layers=quality_layers)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)


def save_gif(latent_dim,
             gen_checkpoints,
             output_filename,
             n_row=2,
             n_col=3,
             gen_class=SimpleGenerator):
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
    gen_args = [latent_dim]
    gen_kwargs = {}
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # We need to select n_row * n_col points to track through time
    torch.manual_seed(0)
    np.random.seed(0)
    z = torch.randn(n_row * n_col, latent_dim, device=device)
    movie = []

    # Generate the frames
    for i, c in enumerate(gen_checkpoints):
        g = load_trained_generator(gen_class, c, *gen_args,
                                   **gen_kwargs).to(device)
        g.eval()

        # Produce a batch of images, shape ((n_row * n_col) x H x W x Channel)
        batch = g(z).detach().to('cpu')

        frame = (make_grid(batch, nrow=n_col, normalize=True,
                           scale_each=True).numpy().transpose(
                               (1, 2, 0)) * 255).astype(np.uint8)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text((5, n_row * 128 - 10),
                  f'{0.19247 * (i+1):.1f}M images shown',
                  fill=(77, 5, 232, 255))
        movie.append(np.asarray(img))

    # Write the GIF
    fps = 2
    clip = ImageSequenceClip(movie, fps=fps).resize(2)
    clip.write_gif(output_filename, fps=fps)
    return


def load_target_image(img):
    if img.endswith('.pt'):
        x = torch.load(img)
    else:
        image = Image.open(img)
        height, width = image.size

        if height > width:
            crop = transforms.CenterCrop((width, width))
        else:
            crop = transforms.CenterCrop((height, height))

        t = transforms.Compose([
            # TODO - ideal is:
            # - if img rectangular, cut into square
            # - then resize to (P.size, P.size)
            crop,
            transforms.Resize((P.size, P.size)),
            transforms.ToTensor()
        ])
        x = t(image)
    return x


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        raise ValueError("how do we handle a perfect reconstruction?")
    pixel_max = torch.tensor(1.0)
    p = 20 * torch.log10(pixel_max) - 10 * torch.log10(mse)
    if isinstance(p, torch.Tensor):
        p = p.item()
    return p
