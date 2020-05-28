import math

import torch
from torch import nn
import torch.nn.functional as F

from util import Conv2dNormalizedLR, local_response_normalization, LinearNormalizedLR, Conv2dTransposeNormalizedLR


class ProGANUpBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels, upsample=True, local_response_norm=True, weight_norm=False):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.weight_norm = weight_norm

        self.conv_1 = Conv2dTransposeNormalizedLR(input_channels, output_channels, kernel_size=3, padding=1, weight_norm=self.weight_norm)
        self.conv_2 = Conv2dTransposeNormalizedLR(output_channels, output_channels, kernel_size=3, padding=1, weight_norm=self.weight_norm)
        # Weight Norm is always disabled here because we don't want to normalize the RGB output
        self.conv_rgb = Conv2dNormalizedLR(output_channels, 3, kernel_size=1, weight_norm=False)
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
        rgb = torch.sigmoid(rgb)

        return x, rgb


class ProGANGenerator(torch.nn.Module):
    def __init__(self, latent_size, n_upscales, output_h_size, local_response_norm=True, scaling_factor=2,
                 hypersphere_latent=True, max_h_size: int = 1e10, weight_norm=False):
        super().__init__()
        self.n_upscales = n_upscales
        self.output_h_size = output_h_size
        self.scaling_factor = scaling_factor
        self.weight_norm = weight_norm
        self.initial_size = min(int(output_h_size * self.scaling_factor ** (n_upscales)), max_h_size)
        self.lrn = local_response_norm
        self.hypersphere_latent = hypersphere_latent

        self.inp_layer = LinearNormalizedLR(latent_size, self.initial_size * 4 * 4, weight_norm=self.weight_norm)
        self.init_layer = Conv2dTransposeNormalizedLR(self.initial_size, self.initial_size, kernel_size=3, padding=1, weight_norm=self.weight_norm)
        self.init_rgb = Conv2dNormalizedLR(self.initial_size, 3, kernel_size=1, weight_norm=self.weight_norm)

        self.layer_list = []
        for i in range(n_upscales):
            inp_channels = min(int(output_h_size * self.scaling_factor ** (n_upscales - i)), max_h_size)
            outp_channels = min(int(output_h_size * self.scaling_factor ** (n_upscales - i - 1)), max_h_size)
            self.layer_list.append(ProGANUpBlock(inp_channels, outp_channels, local_response_norm=local_response_norm, weight_norm=self.weight_norm))
        self.layers = torch.nn.ModuleList(self.layer_list)

    def forward(self, x, phase=None):
        if self.hypersphere_latent:
            x_divisor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + 1e-8
            x = x / x_divisor

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
    def __init__(self, input_channels, output_channels, downsample=True, local_response_norm=False,
                 progan_var_input=False, last_layer=False):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.progran_var_input = progan_var_input
        self.last_layer = last_layer

        conv_1_input_channels = input_channels + (1 if progan_var_input else 0)
        # According to the ProGAN paper appendix, the "hidden" number of channels should be the same as the input size
        conv_1_output_channels = output_channels if self.last_layer else input_channels
        self.conv_1 = Conv2dNormalizedLR(conv_1_input_channels, conv_1_output_channels,
                                         kernel_size=3, padding=1)
        if not self.last_layer:
            self.conv_2 = Conv2dNormalizedLR(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv_rgb = Conv2dNormalizedLR(3, input_channels, kernel_size=1)
        self.downsample = downsample
        self.lrn = local_response_norm

    def forward(self, x):
        if self.progran_var_input:
            # Apply the ProGAN mbatch stddev trick
            stddevs = x.std(dim=0, keepdim=True)
            stddev = stddevs.mean()
            feature_map = torch.zeros_like(x[:, :1]) + stddev
            x = torch.cat([x, feature_map], dim=1)
        x = self.conv_1(x)
        if self.lrn:
            x = local_response_normalization(x)
        x = F.leaky_relu(x, 0.2)

        if not self.last_layer:
            x = self.conv_2(x)
            if self.lrn:
                x = local_response_normalization(x)
            x = F.leaky_relu(x, 0.2)

        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def from_rgb(self, x):
        # Generated an input for this network from RGB
        x = self.conv_rgb(x)
        x = F.leaky_relu(x, 0.2)
        return x


class ProGANDiscriminator(torch.nn.Module):
    down_block = ProGANDownBlock
    def __init__(self, n_downscales, full_res_h_size, scaling_factor=2, max_h_size: int = 1e10):
        super().__init__()
        self.n_downscales = n_downscales
        self.h_size = full_res_h_size
        self.scaling_factor = scaling_factor

        self.deepest_channels = min(int(full_res_h_size * (self.scaling_factor ** (n_downscales))), max_h_size)

        self.outp_layer_1 = LinearNormalizedLR(self.deepest_channels * 4 * 4, self.deepest_channels)
        self.outp_layer_2 = LinearNormalizedLR(self.deepest_channels, 1)
        outp_block = self.down_block(self.deepest_channels, self.deepest_channels, downsample=False,
                                     local_response_norm=False, progan_var_input=True, last_layer=True)

        self.layer_list = [outp_block]
        for i in range(n_downscales):
            inp_channels = min(int(full_res_h_size * (self.scaling_factor ** (n_downscales - i - 1))), max_h_size)
            outp_channels = min(int(full_res_h_size * (self.scaling_factor ** (n_downscales - i))), max_h_size)
            self.layer_list.append(self.down_block(inp_channels, outp_channels, local_response_norm=False))
        self.layers = torch.nn.ModuleList(self.layer_list)
        self.dis_l = None


    def forward(self, x, phase=None):
        if phase is None:
            phase = self.n_downscales

        n_downscales = min(int(phase), self.n_downscales)
        alpha = phase - n_downscales

        if alpha == 0.0:
            x = self.layers[n_downscales].from_rgb(x)
        else:
            x1 = self.layers[n_downscales + 1].from_rgb(x)
            x1 = self.layers[n_downscales + 1](x1)

            x2 = F.avg_pool2d(x, 2)
            x2 = self.layers[n_downscales].from_rgb(x2)

            x = alpha * x1 + (1 - alpha) * x2

        for i in range(0, n_downscales + 1):
            if i == n_downscales:
                # Save dis_l
                self.dis_l = x
            layer = self.layers[n_downscales - i]
            x = layer(x)

        x = x.view(-1, self.deepest_channels * 4 * 4)

        x = self.outp_layer_1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.outp_layer_2(x)

        return x


if __name__ == "__main__":
    def compute_n_params(model):
        total = 0
        for p in model.parameters():
            n_params = 1
            for d in p.size():
                n_params *= d
            total += n_params
        return total


    G = ProGANGenerator(128, 4, 256, scaling_factor=1)
    D = ProGANDiscriminator(4, 256, scaling_factor=1)

    for phase in [0, 0.5, 1, 2, 3, 3.5, 4]:
        z = torch.normal(0, 1, (1, 128))
        x_gen = G(z, phase=phase)
        print("G out: ", x_gen.size())
        d_out = D(x_gen, phase=phase)
        print(d_out.size())

    print("G_params: ", compute_n_params(G))
    print("D_params: ", compute_n_params(D))
