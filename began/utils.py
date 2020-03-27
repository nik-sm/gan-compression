import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import torch


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


def load_trained_generator(generator_class, generator_checkpoint, device, *gen_args, **gen_kwargs):
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

    return gen.to(device)

def load_trained_disc(d_class, d_checkpoint, device, *d_args, **d_kwargs):
    d = d_class(*d_args, **d_kwargs)
    try:
        ckpt = torch.load(d_checkpoint)['model_state_dict']
        fixed_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith('module.'):
                fixed_ckpt[k[7:]] = v
            else:
                fixed_ckpt[k] = v
        d.load_state_dict(fixed_ckpt)
    except Exception as e:
        print(e)
        d.load_state_dict(torch.load(d_checkpoint))

    return d.to(device)
