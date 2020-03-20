import torch.nn as nn
import torch
import math


#class Residual(nn.Module):
#    def __init__(self, weight, layer):
#        super().__init__()
#        self.weight = weight
#        self.l = layer
#    def forward(self, inp):
#        return self.weight * inp + (1 - self.weight) * self.l(inp)

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

    def forward(self, inp):

        k = self.num_filters
        s = self.initial_size
        fmaps = self.initial_fmaps(inp).view(-1, k, s, s)
        chain = self.conv[0](fmaps)
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

        self.carry = torch.tensor(1, dtype=torch.float, requires_grad=False)

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
        return self.decoder.forward(latent)

class AdjustedSizedGenerator(nn.Module):
    def __init__(self, latent_dim, num_filters, image_size, num_ups, n_layers_skipped=0, output_act='elu'):
        super().__init__()
        latent_dim = latent_dim * 2**n_layers_skipped
        num_ups -= n_layers_skipped
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

    def forward(self, inp):

        k = self.num_filters
        s = self.initial_size
        fmaps = self.initial_fmaps(inp).view(-1, k, s, s)
        chain = self.conv[0](fmaps)
        for conv, skip in zip(self.conv[1:], self.skip):
            chain = conv( torch.cat((chain, skip(fmaps)), dim=1 ))

        return self.output_act(self.output_fmap(chain) )

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim, act='elu'):
        """
        Works for images of size 128x128
        """
        super().__init__()
        self.latent_dim = latent_dim
        if act == 'elu':
            self.act = nn.ELU() 
        else:
            raise NotImplementedError()

        self.ch = 128
        self.initial_size = 8

        self.linear = nn.Linear(self.latent_dim, self.initial_size**2 * self.ch, bias=False)
        self.convnet = nn.Sequential(
                UpBlock(up_scale=2, in_ch1=self.ch, in_ch2=self.ch, out_ch=self.ch, kern=(3,3), stride=(1,1), pad=(1,1), act=act),
                UpBlock(up_scale=4, in_ch1=2*self.ch, in_ch2=self.ch, out_ch=self.ch, kern=(3,3), stride=(1,1), pad=(1,1), act=act),
                UpBlock(up_scale=8, in_ch1=2*self.ch, in_ch2=self.ch, out_ch=self.ch, kern=(3,3), stride=(1,1), pad=(1,1), act=act),
                UpBlock(with_skip=False,
                    in_ch1=2*self.ch, in_ch2=self.ch, out_ch=self.ch, kern=(3,3), stride=(1,1), pad=(1,1), act=act),
                )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.ch, self.initial_size, self.initial_size)
        x = self.convnet((x, x))
        return self.act(x)

class UpBlock(nn.Module):
    def __init__(self, with_skip=True, up_scale=None, in_ch1=None, in_ch2=None, **kwargs):
        super().__init__()
        self.with_skip = with_skip
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(in_ch = in_ch1, **kwargs), # <- should expect 256 in, produces 128
            ConvBlock(in_ch = in_ch2, **kwargs)) # <- only gets 128 in
        if self.with_skip:
            self.skip = nn.Upsample(scale_factor=up_scale)

    def forward(self, conv_input_skip_input):
        conv_input, skip_input = conv_input_skip_input
        conv_output = self.conv(conv_input)
        if self.with_skip:
            skip_output = self.skip(skip_input)
            return torch.cat((conv_output, skip_output), dim=1), skip_input
        else:
            return conv_output

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kern, stride, pad, act='elu'):
        super().__init__()
        if act == 'elu':
            self.act = nn.ELU
        else:
            raise NotImplementedError()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kern, stride, pad, bias=False),
            # TODO - add batchnorm
#            nn.BatchNorm2d(out_ch),
            self.act())

    def forward(self, x):
        return self.net(x)


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
