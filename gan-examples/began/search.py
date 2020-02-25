import torch
import torch.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

def load_trained_generator():
    pass

def load_target_image():
    pass

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        raise ValueError("how do we handle a perfect reconstruction?")
    pixel_max = 1.0
    return 20 * torch.log10(pixel_max) - 10 * torch.log10(mse)

logdir = 'tensorboard_logs'

writer = SummaryWriter(logdir)

x = load_target_image()

g = load_trained_generator()

z = torch.Variable(torch.randn(64), requires_grad=True)

optimizer = torch.optim.AdamW(z)

scheduler = torch.otim.lr_scheduler.ExponentialLR(optimizer, 0.98)

fig, ax = plt.subplots(1,2, figsize=(16,9))
ax[0].imshow(x)
for i in range(100):
    print(f"Iteration: {i}")

    x_hat = g(z)
    mse = F.mse_loss(x_hat, x)
    mse.backward()
    optimizer.step()
    scheduler.step()

    writer.add_scalar('MSE', mse, i)
    writer.add_scalar('PSNR', psnr(x, x_hat), i)

ax[1].imshow(x_hat)
