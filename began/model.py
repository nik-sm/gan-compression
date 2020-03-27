import torch.nn as nn
import torch


class SimpleDiscriminator(nn.Module):
    def __init__(self, latent_dim, num_filters, image_size, num_dns, act='elu'):
        super().__init__()
        if act == 'elu':
            self.act = nn.ELU()
        else:
            raise NotImplementedError()

        reduced_img_size = int(image_size / (2**num_dns))
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
        self.decoder = SimpleGenerator(latent_dim, act)

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

        self.linear = nn.Linear(
            self.latent_dim, self.initial_size**2 * self.ch, bias=False)
        self.conv_net = nn.Sequential(
            UpBlock(up_scale=2, in_ch1=self.ch, in_ch2=self.ch, out_ch=self.ch, kern=(3, 3), stride=(
                1, 1), pad=(1, 1), act=act, passthrough=True),  # NOTE - want this one to NOT cat(conv(x), skip(x))
            UpBlock(up_scale=4, in_ch1=2*self.ch, in_ch2=self.ch,
                    out_ch=self.ch, kern=(3, 3), stride=(1, 1), pad=(1, 1), act=act),
            UpBlock(up_scale=8, in_ch1=2*self.ch, in_ch2=self.ch,
                    out_ch=self.ch, kern=(3, 3), stride=(1, 1), pad=(1, 1), act=act),
            UpBlock(with_skip=False,
                    in_ch1=2*self.ch, in_ch2=self.ch, out_ch=self.ch, kern=(3, 3), stride=(1, 1), pad=(1, 1), act=act),
            nn.Conv2d(self.ch, 3, kernel_size=(3, 3), padding=1, bias=False),
            self.act)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.ch, self.initial_size, self.initial_size)
        x = self.conv_net((x, x))
        return x


class UpBlock(nn.Module):
    def __init__(self, with_skip=True, up_scale=None, in_ch1=None, in_ch2=None, passthrough=False, **kwargs):
        super().__init__()
        self.with_skip = with_skip
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # <- should expect 256 in, produces 128
            ConvBlock(in_ch=in_ch1, **kwargs),
            ConvBlock(in_ch=in_ch2, **kwargs))  # <- only gets 128 in

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
