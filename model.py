import torch.nn as nn
import torch
import math

class SimpleDiscriminator(nn.Module):
    def __init__(self, latent_dim, num_filters, image_size, num_dns, act='elu'):
        super().__init__()
        if act == 'elu':
            self.act = nn.ELU()
        else:
            raise NotImplementedError()

        self.image_size = image_size
        reduced_img_size = int(self.image_size / (2**num_dns))
        self.shape_after_conv = reduced_img_size**2 * \
            ((num_dns+1) * num_filters)  # N_pixels x N_channels

        convs = [DownBlock(num_filters, 1+i, act) for i in range(num_dns)]

        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=(3, 3),
                      padding=1),  # TODO - bias=False?
            self.act,
            *convs,
            Flatten())
        self.linear = nn.Linear(self.shape_after_conv, latent_dim, bias=False)
        self.decoder = SimpleGenerator(latent_dim, self.image_size, act)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        x = self.decoder(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class DownBlock(nn.Module):
    def __init__(self, num_filters, multiplier, act='elu'):
        super().__init__()
        if act == 'elu':
            self.act = nn.ELU()
        else:
            raise NotImplementedError()

        self.conv_net = nn.Sequential(
            nn.Conv2d(num_filters * multiplier, num_filters*multiplier,
                      kernel_size=(3, 3), padding=1, bias=False),
            self.act,
            nn.Conv2d(num_filters * multiplier, num_filters*(multiplier+1),
                      kernel_size=(3, 3), padding=1, stride=2, bias=False),
            self.act)

    def forward(self, x):
        return self.conv_net(x)


class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim, image_size, act='elu'):
        """
        Works for images of size 128x128
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        if math.floor(math.log2(image_size)) != math.ceil(math.log2(image_size)):
            raise ValueError(f"Image size must be power of 2: {image_size}")

        if act == 'elu':
            self.act = nn.ELU()
        else:
            raise NotImplementedError()

        self.ch = 128
        self.initial_size = 8

        self.linear = nn.Linear(
            self.latent_dim, self.initial_size**2 * self.ch, bias=False)

        # first_conv is applied simply to the input
        self.first_conv = UpBlock(self.ch, self.ch, kernel_size=(3,3), stride=(1,1), padding=(1,1), act=self.act)
        self.first_skip = nn.Upsample(scale_factor=2, mode='nearest')

        # other_convs are applied to cat(prev_layer, skip_input), so they need to accept 2*self.ch
        n_layers = int(math.log2(self.image_size/self.initial_size))
        self.other_convs = nn.ModuleList()
        self.other_skips = nn.ModuleList()
        for i in range(1, n_layers - 1):
            self.other_convs.append(UpBlock(2*self.ch, self.ch, kernel_size=(3,3), stride=(1,1), padding=(1,1), act=self.act))
            self.other_skips.append(nn.Upsample(scale_factor=2**(i+1), mode='nearest'))

        # final_conv also applied to cat(..., skip_input), but its output will NOT be catted with anything
        self.final_conv = nn.Sequential(
            UpBlock(2*self.ch, self.ch, kernel_size=(3,3), stride=(1,1), padding=(1,1), act=self.act),
            nn.Conv2d(self.ch, 3, kernel_size=(3, 3), padding=1, bias=False),
            self.act)

    def forward(self, x, skip_linear_layer=False):
        if not skip_linear_layer:
            x = self.linear(x)
        x_0 = x.view(-1, self.ch, self.initial_size, self.initial_size)

        conv_output = self.first_conv(x_0)
        skip_output = self.first_skip(x_0)

        for conv, skip in zip(self.other_convs, self.other_skips):
            conv_output = conv(torch.cat((conv_output, skip_output), dim=1))
            skip_output = skip(x_0)

        return self.final_conv(torch.cat((conv_output, skip_output), dim=1))


class UpBlock(nn.Module):
    def __init__(self, in_ch1=None, in_ch2=None, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(in_ch=in_ch1, out_ch=in_ch2, **kwargs), # <- should expect 256 in, produces 128
            ConvBlock(in_ch=in_ch2, out_ch=in_ch2, **kwargs))  # <- only gets 128 in

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act='elu', **kwargs):
        super().__init__()
        self.act = act
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
            # TODO - add batchnorm
            #            nn.BatchNorm2d(out_ch),
            self.act)

    def forward(self, x):
        return self.net(x)
