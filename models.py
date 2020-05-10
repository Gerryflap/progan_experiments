import math

import torch
from torch import nn
import torch.nn.functional as F

from util import Conv2dNormalizedLR, local_response_normalization, LinearNormalizedLR


class ProGANUpBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, upsample=True, local_response_norm=True):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv_1 = Conv2dNormalizedLR(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv_2 = Conv2dNormalizedLR(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv_rgb = Conv2dNormalizedLR(output_channels, 3, kernel_size=1)
        self.upsample = upsample
        self.lrn = local_response_norm

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)

        x = self.conv_1(x)
        if self.lrn:
            x = local_response_normalization(x)
        x = F.leaky_relu(x)

        x = self.conv_2(x)
        if self.lrn:
            x = local_response_normalization(x)
        x = F.leaky_relu(x)

        rgb = self.conv_rgb(x)
        rgb = torch.sigmoid(rgb)

        return x, rgb


class ProGANGenerator(torch.nn.Module):
    def __init__(self, latent_size, n_upscales, output_h_size, local_response_norm=True):
        super().__init__()
        self.n_upscales = n_upscales
        self.output_h_size = output_h_size
        self.initial_size = output_h_size ** n_upscales
        self.lrn = local_response_norm

        self.inp_layer = LinearNormalizedLR(latent_size, self.initial_size * 4 * 4)
        self.init_block = ProGANUpBlock(self.initial_size, self.initial_size, upsample=False,
                                        local_response_norm=local_response_norm)

        self.layer_list = []
        for i in range(n_upscales):
            inp_channels = output_h_size ** (n_upscales - i)
            outp_channels = output_h_size ** (n_upscales - i - 1)
            self.layer_list.append(ProGANUpBlock(inp_channels, outp_channels, local_response_norm=local_response_norm))
        self.layers = torch.nn.ModuleList(self.layer_list)

    def forward(self, x, phase=None):
        if phase is None:
            phase = self.n_upscales

        n_upscales = min(int(phase), self.n_upscales)
        alpha = phase - (n_upscales)

        if alpha == 0.0 and n_upscales >= 1:
            alpha += 1.0

        x = self.inp_layer(x)
        x = x.view(-1, self.initial_size, 4, 4)
        if self.lrn:
            x = local_response_normalization(x)
        x = F.leaky_relu(x)

        x, rgb = self.init_block(x)

        if alpha == 0.0 and n_upscales == 0:
            return rgb

        next_x, next_rgb = self.layers[0](x)

        n_actual_upscales = n_upscales
        if 0 < alpha < 1:
            n_actual_upscales += 1

        for i in range(1, min(self.n_upscales, n_actual_upscales)):
            x, rgb = next_x, next_rgb
            next_x, next_rgb = self.layers[i](x)

        if alpha == 1.0 and n_upscales > 0:
            return next_rgb

        out_rgb = (1 - alpha) * F.interpolate(rgb, scale_factor=2) + alpha * next_rgb
        return out_rgb


class ProGANDownBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, downsample=True, local_response_norm=False, progan_var_input=False):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.progran_var_input = progan_var_input

        self.conv_1 = Conv2dNormalizedLR(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv_2 = Conv2dNormalizedLR(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv_rgb = Conv2dNormalizedLR(3, (input_channels - 1) if progan_var_input else input_channels, kernel_size=1)
        self.downsample = downsample
        self.lrn = local_response_norm

    def forward(self, x):
        x = self.conv_1(x)
        if self.lrn:
            x = local_response_normalization(x)
        x = F.leaky_relu(x)

        x = self.conv_2(x)
        if self.lrn:
            x = local_response_normalization(x)
        x = F.leaky_relu(x)

        if self.downsample:
            x = F.interpolate(x, scale_factor=0.5)
        return x

    def from_rgb(self, x):
        # Generated an input for this network from RGB
        x = self.conv_rgb(x)
        return x


class ProGANDiscriminator(torch.nn.Module):
    def __init__(self, n_downscales, full_res_h_size):
        super().__init__()
        self.n_downscales = n_downscales
        self.h_size = full_res_h_size

        self.deepest_channels = full_res_h_size * (2 ** (n_downscales))

        self.outp_layer_1 = LinearNormalizedLR(self.deepest_channels * 4 * 4, self.deepest_channels)
        self.outp_layer_2 = LinearNormalizedLR(self.deepest_channels, 1)
        outp_block = ProGANDownBlock(self.deepest_channels + 1, self.deepest_channels, downsample=False,
                                        local_response_norm=False, progan_var_input=True)

        self.layer_list = [outp_block]
        for i in range(n_downscales):
            inp_channels = full_res_h_size * (2 ** (n_downscales - i - 1))
            outp_channels = full_res_h_size * (2 ** (n_downscales - i))
            self.layer_list.append(ProGANDownBlock(inp_channels, outp_channels, local_response_norm=False))
        self.layers = torch.nn.ModuleList(self.layer_list)


    def forward(self, x, phase=None):
        if phase is None:
            phase = self.n_upscales

        n_downscales = min(int(phase), self.n_downscales)
        alpha = phase - n_downscales

        if alpha == 0.0:
            x = self.layers[n_downscales].from_rgb(x)
        else:
            x1 = self.layers[n_downscales + 1].from_rgb(x)
            x1 = self.layers[n_downscales + 1](x1)

            x2 = F.interpolate(x, scale_factor=0.5)
            x2 = self.layers[n_downscales].from_rgb(x2)

            x = alpha * x1 + (1-alpha) * x2
        x = F.leaky_relu(x)

        for i in range(n_downscales + 1):
            if i == n_downscales:
                # Apply the ProGAN mbatch stddev trick
                stddevs = x.std(dim=0, keepdim=True)
                stddev = stddevs.mean()
                feature_map = torch.zeros_like(x[:, :1]) + stddev
                x = torch.cat([x, feature_map], dim=1)
            layer = self.layers[n_downscales - i]
            x = layer(x)

        x = x.view(-1, self.deepest_channels * 4 * 4)

        x = self.outp_layer_1(x)
        x = F.leaky_relu(x)

        x = self.outp_layer_2(x)
        x = F.leaky_relu(x)

        return x


if __name__ == "__main__":
    G = ProGANGenerator(256, 4, 4)
    D = ProGANDiscriminator(4, 4)

    for phase in [0, 0.5, 1, 2, 3, 3.5, 4]:
        z = torch.normal(0, 1, (1, 256))
        x_gen = G(z, phase=phase)
        print("G out: ", x_gen.size())
        d_out = D(x_gen, phase=phase)
        print(d_out.size())
