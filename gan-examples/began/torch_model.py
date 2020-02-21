import torch.nn as nn
import torch
import math


class Residual(nn.Module):
    def __init__(self, weight, layer):
        super(Residual, self).__init__()
        self.weight = weight
        self.l = layer
    def forward(self, inp):
        return self.weight * inp + (1 - self.weight) * self.l(inp)

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

# Generates 64x64 images
class Generator(nn.Module):
    def __init__(self, latent_dim, num_filters, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.initial_size = 8

        self.initial_fmaps = nn.Linear(latent_dim,
                                       self.initial_size**2 * num_filters,
                                       bias=False)


        # input is [k x 8 x 8]
        self.ups1 = upsampling_layer(num_filters, with_skip=False)
        self.skip1 = nn.Upsample(scale_factor=2, mode="nearest")


        # input is [k x 16 x 16]
        self.ups2 = upsampling_layer(num_filters, with_skip=True)
        self.skip2 = nn.Upsample(scale_factor=4, mode="nearest")

        # input is [k x 32 x 32]
        self.ups3 = upsampling_layer(num_filters, with_skip=True)

        # input is [k x 64 x 64]
        self.output_fmap = nn.Conv2d(num_filters, 3, kernel_size=(3, 3), padding=1, bias=False)
        self.output_activ = nn.ELU()

    def forward(self, input):
        k = self.num_filters
        s = self.initial_size
        fmaps = self.initial_fmaps(input).view(-1, k, s, s)
        ups1 = self.ups1(fmaps)
        ups2 = self.ups2(torch.cat((ups1, self.skip1(fmaps)), dim=1))
        ups3 = self.ups3(torch.cat((ups2, self.skip2(fmaps)), dim=1))
        return self.output_activ(self.output_fmap(ups3))

class Discriminator(nn.Module):
    def __init__(self, latent_dim, num_filters, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.latent_dim = latent_dim
        self.num_filters = num_filters


        self.initial_fmaps = nn.Conv2d(3, num_filters, kernel_size=(3, 3), padding=1)
        self.initial_activ = nn.ELU()


        # input is [k x 64 x 64]
        self.dns1 = downsampling_layer(num_filters, l=1)

        # input is [k x 32 x 32]
        self.dns2 = downsampling_layer(num_filters, l=2)

        # input is [k x 16 x 16]
        self.dns3 = downsampling_layer(num_filters, l=3)

        # input is [k x 8 x 8]
        self.output = nn.Linear(num_filters * 8 * 8, latent_dim, bias=False)

        # used for autoencoder loss
        self.decoder = Generator(latent_dim, num_filters, ngpu)

    def forward(self, input):
        k = self.num_filters
        fmaps = self.initial_activ(self.initial_fmaps(input))
        dns1 = self.dns1(fmaps)
        dns2 = self.dns2(dns1)
        dns3 = self.dns3(dns2)
        latent = self.output(dns3.view((-1, k * 8 * 8)))
        return self.decoder.forward(latent)

class SizedGenerator(nn.Module):
    def __init__(self, latent_dim, num_filters, image_size, num_ups, ngpu):
        super(SizedGenerator, self).__init__()
        self.ngpu = ngpu

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
        self.output_activ = nn.ELU()

    def forward(self, inp):

        k = self.num_filters
        s = self.initial_size
        fmaps = self.initial_fmaps(inp).view(-1, k, s, s)
        chain = self.conv[0](fmaps)
        for conv, skip in zip(self.conv[1:], self.skip):
            chain = conv( torch.cat((chain, skip(fmaps)), dim=1 ))

        return self.output_activ(self.output_fmap(chain) )

class SizedDiscriminator(nn.Module):
    def __init__(self, latent_dim, num_filters, image_size, num_dns, ngpu):
        super(SizedDiscriminator, self).__init__()
        self.ngpu = ngpu

        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.output_size = image_size
        self.initial_size = int(image_size / (2**num_dns))
        print(self.initial_size)
        self.num_dns = num_dns

        self.initial_fmaps = nn.Conv2d(3, self.num_filters, kernel_size=(3, 3), padding=1)
        self.initial_activ = nn.ELU()

        self.carry = torch.tensor(1, dtype=torch.float, requires_grad=False)

        conv = [downsampling_layer(self.num_filters, i+1) for i in range(num_dns)]
        self.conv = nn.ModuleList(conv)

        self.output = nn.Linear((num_dns+1) * num_filters * (self.initial_size**2), latent_dim, bias=False)
        self.decoder = SizedGenerator(latent_dim, num_filters, image_size, num_dns, ngpu)

    def forward(self, inp):

        fmaps = self.initial_activ(self.initial_fmaps(inp))
        chain = fmaps
        for conv in self.conv:
            chain = conv(chain)
        latent = self.output(chain.view((-1, (self.num_dns + 1) * self.num_filters * (self.initial_size**2))))
        return self.decoder.forward(latent)