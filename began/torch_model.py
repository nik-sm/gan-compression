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
    def __init__(self, latent_dim, num_filters, image_size, num_ups, output_activ='elu'):
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
        if output_activ == 'elu':
            self.output_activ = nn.ELU()
        elif output_activ == 'sigmoid':
            self.output_activ = nn.Sigmoid()
        else:
            raise NotImplementedError()

    def forward(self, inp):

        k = self.num_filters
        s = self.initial_size
        fmaps = self.initial_fmaps(inp).view(-1, k, s, s)
        chain = self.conv[0](fmaps)
        for conv, skip in zip(self.conv[1:], self.skip):
            chain = conv( torch.cat((chain, skip(fmaps)), dim=1 ))

        return self.output_activ(self.output_fmap(chain) )

class SizedDiscriminator(nn.Module):
    def __init__(self, latent_dim, num_filters, image_size, num_dns, output_activ='elu'):
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
        self.decoder = SizedGenerator(latent_dim, num_filters, image_size, num_dns, output_activ)

    def forward(self, inp):

        fmaps = self.initial_activ(self.initial_fmaps(inp))
        chain = fmaps
        for conv in self.conv:
            chain = conv(chain)
        latent = self.output(chain.view((-1, (self.num_dns + 1) * self.num_filters * (self.initial_size**2))))
        return self.decoder.forward(latent)

class AdjustedSizedGenerator(nn.Module):
    def __init__(self, latent_dim, num_filters, image_size, num_ups, n_layers_skipped=0, output_activ='elu'):
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
        if output_activ == 'elu':
            self.output_activ = nn.ELU()
        elif output_activ == 'sigmoid':
            self.output_activ = nn.Sigmoid()
        else:
            raise NotImplementedError()

    def forward(self, inp):

        k = self.num_filters
        s = self.initial_size
        fmaps = self.initial_fmaps(inp).view(-1, k, s, s)
        chain = self.conv[0](fmaps)
        for conv, skip in zip(self.conv[1:], self.skip):
            chain = conv( torch.cat((chain, skip(fmaps)), dim=1 ))

        return self.output_activ(self.output_fmap(chain) )
