import torch.nn as nn
import torch
import math

# 100 x 128 x 128pix x 128pix
# 100 x 256 x 64pix x 64pix
#...
# 100 x 1024 x 16pix x 16pix
def downsampling_layer(k, l):
    c1 = nn.Sequential(
        nn.Conv2d(l*k, l*k, kernel_size=(3, 3), padding=1, bias=False),
        nn.ELU())
    c2 = nn.Sequential(
        nn.Conv2d(l*k, (l+1)*k, kernel_size=(3, 3), padding=1, stride=2, bias=False),
        nn.ELU()
    )
    return nn.Sequential(c1, c2)

def upsampling_layer(k, with_skip=True):
    if(with_skip):
        c1 = nn.Sequential(
            nn.Conv2d(2*k, k, kernel_size=(3, 3), padding=1, bias=False), # this line is different
            nn.ELU())
        c2 = nn.Sequential(
            nn.Conv2d(k, k, kernel_size=(3, 3), padding=1, bias=False),
            nn.ELU())
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), c1, c2)
    else:
        c1 = nn.Sequential(
            nn.Conv2d(k, k, kernel_size=(3, 3), padding=1, bias=False),
            nn.ELU()
        )
        c2 = nn.Sequential(
            nn.Conv2d(k, k, kernel_size=(3, 3), padding=1, bias=False),
            nn.ELU()
        )
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            c1,
            c2)

class SizedGenerator(nn.Module):
    def __init__(self, latent_dim, num_filters, image_size, num_ups, output_act='elu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.output_size = image_size
        self.initial_size = int(image_size / (2**num_ups))

        self.initial_fmaps = nn.Linear(latent_dim,
                                       (self.initial_size ** 2) * num_filters,
                                       bias=False)

        self.carry = torch.tensor(1, dtype=torch.float, requires_grad=False)

        conv = [upsampling_layer(self.num_filters, with_skip=False)]
        skip = []

        # output and second to last layer are not receiving a skip
        for i in range(num_ups):
            conv.append(upsampling_layer(self.num_filters, with_skip=True))

        for i in range(num_ups - 1):
            skip.append(nn.Upsample(scale_factor=2**(i+1), mode="nearest"))

        self.conv = nn.ModuleList(conv)
        self.skip = nn.ModuleList(skip)

        self.output_fmap = nn.Conv2d(num_filters, 3, kernel_size=(3, 3), padding=1, bias=False)
        if output_act == 'elu':
            self.output_act = nn.ELU()
        elif output_act == 'sigmoid':
            self.output_act = nn.Sigmoid()
        else:
            raise NotImplementedError()

    def forward(self, inp, skip_linear_layer=False):

        k = self.num_filters
        s = self.initial_size
        if skip_linear_layer:
            fmaps = inp.view(-1, k, s, s) # 32, 128, 8, 8
        else:
            fmaps = self.initial_fmaps(inp).view(-1, k, s, s) # 32, 128, 8, 8
        chain = self.conv[0](fmaps) # 32, 128, 16, 16
        for conv, skip in zip(self.conv[1:], self.skip):
            chain = conv( torch.cat((chain, skip(fmaps)), dim=1 ))

        return self.output_act(self.output_fmap(chain) )

class SizedDiscriminator(nn.Module):
    def __init__(self, latent_dim, num_filters, image_size, num_dns, output_act='elu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.output_size = image_size
        self.initial_size = int(image_size / (2**num_dns))
        #print(self.initial_size)
        self.num_dns = num_dns

        self.initial_fmaps = nn.Conv2d(3, self.num_filters, kernel_size=(3, 3), padding=1)
        self.initial_activ = nn.ELU()

        conv = [downsampling_layer(self.num_filters, i+1) for i in range(num_dns)]
        self.conv = nn.ModuleList(conv)

        self.output = nn.Linear((num_dns+1) * num_filters * (self.initial_size**2), latent_dim, bias=False)
        self.decoder = SizedGenerator(latent_dim, num_filters, image_size, num_dns, output_act)

    def forward(self, inp):

        fmaps = self.initial_activ(self.initial_fmaps(inp))
        chain = fmaps
        for conv in self.conv:
            chain = conv(chain)
        latent = self.output(chain.view((-1, (self.num_dns + 1) * self.num_filters * (self.initial_size**2))))
            # reshape (n_batch x -1)
        return self.decoder.forward(latent)

"""
gen: SizedGenerator(
  (initial_fmaps): Linear(in_features=32, out_features=8192, bias=False)
  (conv): ModuleList(
    (0): Sequential(
      (0): Upsample(scale_factor=2.0, mode=nearest)
      (1): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
      (2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
    )
    (1): Sequential(
      (0): Upsample(scale_factor=2.0, mode=nearest)
      (1): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
      (2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
    )
    (2): Sequential(
      (0): Upsample(scale_factor=2.0, mode=nearest)
      (1): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
      (2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
    )
    (3): Sequential(
      (0): Upsample(scale_factor=2.0, mode=nearest)
      (1): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
      (2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
    )
    (4): Sequential(
      (0): Upsample(scale_factor=2.0, mode=nearest)
      (1): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
      (2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
    )
  )
  (skip): ModuleList(
    (0): Upsample(scale_factor=2.0, mode=nearest)
    (1): Upsample(scale_factor=4.0, mode=nearest)
    (2): Upsample(scale_factor=8.0, mode=nearest)
  )
  (output_fmap): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (output_act): ELU(alpha=1.0)
)
disc: SizedDiscriminator(
  (initial_fmaps): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (initial_activ): ELU(alpha=1.0)
  (conv): ModuleList(
    (0): Sequential(
      (0): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
      (1): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
    )
    (1): Sequential(
      (0): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
      (1): Sequential(
        (0): Conv2d(256, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
    )
    (2): Sequential(
      (0): Sequential(
        (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
      (1): Sequential(
        (0): Conv2d(384, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
    )
    (3): Sequential(
      (0): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
      (1): Sequential(
        (0): Conv2d(512, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): ELU(alpha=1.0)
      )
    )
  )
  (output): Linear(in_features=40960, out_features=32, bias=False)
  (decoder): SizedGenerator(
    (initial_fmaps): Linear(in_features=32, out_features=8192, bias=False)
    (conv): ModuleList(
      (0): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): ELU(alpha=1.0)
        )
        (2): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): ELU(alpha=1.0)
        )
      )
      (1): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): Sequential(
          (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): ELU(alpha=1.0)
        )
        (2): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): ELU(alpha=1.0)
        )
      )
      (2): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): Sequential(
          (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): ELU(alpha=1.0)
        )
        (2): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): ELU(alpha=1.0)
        )
      )
      (3): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): Sequential(
          (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): ELU(alpha=1.0)
        )
        (2): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): ELU(alpha=1.0)
        )
      )
      (4): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): Sequential(
          (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): ELU(alpha=1.0)
        )
        (2): Sequential(
          (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): ELU(alpha=1.0)
        )
      )
    )
    (skip): ModuleList(
      (0): Upsample(scale_factor=2.0, mode=nearest)
      (1): Upsample(scale_factor=4.0, mode=nearest)
      (2): Upsample(scale_factor=8.0, mode=nearest)
    )
    (output_fmap): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (output_act): ELU(alpha=1.0)
  )
)

"""




"""
Our result:




gen: SimpleGenerator(
  (convnet): Sequential(
    (0): UpBlock(
      (conv): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): ConvBlock(
          (net): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): ELU(alpha=1.0)
          )
        )
        (2): ConvBlock(
          (net): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): ELU(alpha=1.0)
          )
        )
      )
      (skip): Upsample(scale_factor=2.0, mode=nearest)
    )
    (1): UpBlock(
      (conv): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): ConvBlock(
          (net): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): ELU(alpha=1.0)
          )
        )
        (2): ConvBlock(
          (net): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): ELU(alpha=1.0)
          )
        )
      )
      (skip): Upsample(scale_factor=4.0, mode=nearest)
    )
    (2): UpBlock(
      (conv): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): ConvBlock(
          (net): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): ELU(alpha=1.0)
          )
        )
        (2): ConvBlock(
          (net): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): ELU(alpha=1.0)
          )
        )
      )
      (skip): Upsample(scale_factor=8.0, mode=nearest)
    )
    (3): UpBlock(
      (conv): Sequential(
        (0): Upsample(scale_factor=2.0, mode=nearest)
        (1): ConvBlock(
          (net): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): ELU(alpha=1.0)
          )
        )
        (2): ConvBlock(
          (net): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): ELU(alpha=1.0)
          )
        )
      )
    )
  )
)
"""
