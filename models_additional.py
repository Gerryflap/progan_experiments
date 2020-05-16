import math

import torch
from torch import nn
import torch.nn.functional as F

from util import Conv2dNormalizedLR, local_response_normalization, LinearNormalizedLR, Conv2dTransposeNormalizedLR


class ProGANUpBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, upsample=True, local_response_norm=True):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv_1 = Conv2dTransposeNormalizedLR(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv_2 = Conv2dTransposeNormalizedLR(output_channels, output_channels, kernel_size=3, padding=1)
        self.conv_rgb = Conv2dNormalizedLR(output_channels, 3, kernel_size=1)
        self.upsample = upsample
        self.lrn = local_response_norm

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)

        x = self.conv_1(x)
        if self.lrn:
            x = local_response_normalization(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv_2(x)
        if self.lrn:
            x = local_response_normalization(x)
        x = F.leaky_relu(x, 0.2)

        rgb = self.conv_rgb(x)

        return x, rgb


class ProGANAdditiveGenerator(torch.nn.Module):
    def __init__(self, latent_size, n_upscales, output_h_size, local_response_norm=True, scaling_factor=2):
        super().__init__()
        self.n_upscales = n_upscales
        self.output_h_size = output_h_size
        self.scaling_factor = scaling_factor
        self.initial_size = int(output_h_size * self.scaling_factor ** (n_upscales))
        self.lrn = local_response_norm

        self.inp_layer = LinearNormalizedLR(latent_size, self.initial_size * 4 * 4)
        self.init_layer = Conv2dTransposeNormalizedLR(self.initial_size, self.initial_size, kernel_size=3, padding=1)
        self.init_rgb = Conv2dNormalizedLR(self.initial_size, 3, kernel_size=1)

        self.layer_list = []
        for i in range(n_upscales):
            inp_channels = int(output_h_size * self.scaling_factor ** (n_upscales - i))
            outp_channels = int(output_h_size * self.scaling_factor ** (n_upscales - i - 1))
            self.layer_list.append(ProGANUpBlock(inp_channels, outp_channels, local_response_norm=local_response_norm))
        self.layers = torch.nn.ModuleList(self.layer_list)

    def forward(self, x, phase=None):

        # Project latent vectors onto hypersphere
        x_divisor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True)) + 1e-8
        x = x/x_divisor

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
        x = F.leaky_relu(x, 0.2)

        x = self.init_layer(x)
        if self.lrn:
            x = local_response_normalization(x)
        x = F.leaky_relu(x, 0.2)

        rgb = self.init_rgb(x)

        if alpha == 0.0 and n_upscales == 0:
            return torch.sigmoid(rgb)

        next_x, next_rgb = self.layers[0](x)
        next_rgb = F.interpolate(rgb, scale_factor=2, mode="bicubic") + next_rgb

        n_actual_upscales = n_upscales
        if 0 < alpha < 1:
            n_actual_upscales += 1

        for i in range(1, min(self.n_upscales, n_actual_upscales)):
            x, rgb = next_x,  next_rgb
            next_x, next_rgb = self.layers[i](x)
            next_rgb = F.interpolate(rgb, scale_factor=2, mode="bicubic") + next_rgb

        if alpha == 1.0 and n_upscales > 0:
            return torch.sigmoid(F.interpolate(rgb, scale_factor=2, mode="bicubic") + next_rgb)

        out_rgb = (1 - alpha) * F.interpolate(rgb, scale_factor=2, mode="bicubic") + alpha * next_rgb
        return torch.sigmoid(out_rgb)

