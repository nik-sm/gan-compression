import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.utils import make_grid

def _gen_img(img):
    plt.figure(figsize=(16,9))
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
    
    img_buf = _gen_img(img.numpy().transpose(1,2,0))
    img = np.array(Image.open(img_buf))
    writer.add_image(tag, img, global_step=epoch, dataformats='HWC')
    return

def save_grid_tensorboard(img_list, writer, tag, epoch=None):
    grid = make_grid(img_list, scale_each=True).numpy().transpose(1,2,0)
    img_buf = _gen_img(grid)
    img = np.array(Image.open(img_buf))
    writer.add_image(tag, img, global_step=epoch, dataformats='HWC')
    return
