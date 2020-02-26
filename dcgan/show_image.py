import torch
import matplotlib.pyplot as plt
from dcgan import Generator
import numpy as np

g = Generator(ngpu=0)

# images are [batch idx, channels, height, width] NCHW
# conceptually, the 100-dim latent code provides 100 "features" which are iid
# random gaussian
latent = np.random.normal(size=(1, 100, 1, 1)) 

latent = torch.tensor(latent, dtype=torch.float)

g.load_state_dict(torch.load("celeba_64x64/dcgan_G.pt",
                             map_location=torch.device("cpu")))

im = g.forward(latent)

im = (im + 1)/2 
# this operation undoes a normalization special to GANs
# a conventional dcgan architecture, by design, has trouble dealing with values
# near the origin

# (if I remember correctly, passing an image of all zeros into a discriminator
# will kill all gradients for that point. the generater learns this as a
# suprious minima, but you can get around it by renormalizing images from [0,
# 1] to [-1, 1]

im = im.detach().numpy()
im = im[0].transpose((1, 2, 0))
plt.imshow(im)
plt.show()
