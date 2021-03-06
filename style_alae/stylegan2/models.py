import torch
from torch import nn

import util
from style_alae.models_and_components import instance_norm_with_style_out, ALAEEncoder, StyleALAEEncoderBlock
from util import Conv2dNormalizedLR, Conv2dTransposeNormalizedLR
import torch.nn.functional as F


class Conv2dTransposeNormalizedLRStyleGAN2(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, stride=1, padding=0, bias=True, weight_norm=True, gain=util.lrelu_gain):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight_norm = weight_norm
        self.he_constant = gain / (float(in_channels * kernel_size * kernel_size) ** 0.5)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.gain = gain

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size, kernel_size))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, inp, style):
        weight = self.weight * self.he_constant
        weight = weight.view(1, self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)
        batch_size = style.size(0)
        style = style.view(batch_size, self.in_channels, 1, 1, 1)
        weight = style * weight
        weight = util.apply_weight_norm(weight, input_dims=(1, 3, 4))

        weight = weight.view(self.in_channels * batch_size, self.out_channels, self.kernel_size, self.kernel_size)

        # print(weight.size())
        # print("Inp: ", inp.size())
        inp = inp.view(1, self.in_channels * batch_size, inp.size(2), inp.size(3))
        x = torch.nn.functional.conv_transpose2d(inp, weight, self.bias, self.stride, self.padding, groups=batch_size)
        # x = x.view( self.out_channels, batch_size, inp.size(2), inp.size(3))
        x = x.view(batch_size, self.out_channels, inp.size(2), inp.size(3))
        x = x * self.gain
        return x

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0.0, 1.0)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)


class Stylev2ALAEGeneratorBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, w_size, is_start=False):
        super().__init__()
        self.is_start = is_start
        self.out_size = out_size

        self.conv1 = Conv2dTransposeNormalizedLRStyleGAN2(in_size, out_size, 3, padding=1, bias=False)
        self.bias1 = torch.nn.Parameter(torch.zeros((1, out_size, 1, 1)))

        if is_start:
            self.start = torch.nn.Parameter(torch.ones((1, out_size, 4, 4)), requires_grad=True)

        self.conv2 = Conv2dTransposeNormalizedLRStyleGAN2(out_size, out_size, 3, padding=1, bias=False)
        self.bias2 = torch.nn.Parameter(torch.zeros((1, out_size, 1, 1)))


        self.Aaff1 = Conv2dNormalizedLR(w_size, in_size, kernel_size=1, gain=1.0)
        self.Baff1 = Conv2dNormalizedLR(1, out_size, kernel_size=1, gain=1.0, bias=False)

        self.Aaff2 = Conv2dNormalizedLR(w_size, out_size, kernel_size=1, gain=1.0)

        self.Baff2 = Conv2dNormalizedLR(1, out_size, kernel_size=1, gain=1.0, bias=False)

        self.rgb = Conv2dNormalizedLR(out_size, 3, 1, gain=0.03)
        self.reset_parameters()

    def to_rgb(self, x):
        x = self.rgb(x)
        # x = F.sigmoid(x)
        return x

    def forward(self, x, w):
        if self.is_start:
            x = torch.cat([self.start]*w.size(0), dim=0)
        else:
            x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
        style1_aff = self.Aaff1(w)
        x = self.conv1(x, style1_aff)
        noise = torch.normal(0, 1, (w.size(0), 1, x.size(2), x.size(3)), device=w.device)
        noise_y = self.Baff1(noise)
        x = x + noise_y
        x = x + self.bias1
        x = F.leaky_relu(x, 0.2)

        style2_aff = self.Aaff2(w)
        x = self.conv2(x, style2_aff)

        noise = torch.normal(0, 1, (w.size(0), 1, x.size(2), x.size(3)), device=w.device)
        noise_y = self.Baff2(noise)
        x = x + noise_y
        x = x + self.bias2
        x = F.leaky_relu(x, 0.2)

        return x, self.to_rgb(x)

    def reset_parameters(self):
        if self.is_start:
            torch.nn.init.ones_(self.start)
        else:
            self.conv1.reset_parameters()
            torch.nn.init.zeros_(self.Baff1.weight)
            self.Aaff1.reset_parameters()

        torch.nn.init.zeros_(self.Baff2.weight)


        for layer in [self.conv2, self.rgb, self.Aaff2]:
            layer.reset_parameters()


class ALAEGeneratorStyleGAN2(torch.nn.Module):
    def __init__(self, w_size, n_upscales, output_h_size, scaling_factor=2, max_h_size: int = 1e10):
        super().__init__()
        self.n_upscales = n_upscales
        self.output_h_size = output_h_size
        self.scaling_factor = scaling_factor
        self.initial_size = min(int(output_h_size * self.scaling_factor ** (n_upscales)), max_h_size)
        self.w_size = w_size

        self.init_layer = Stylev2ALAEGeneratorBlock(self.initial_size, self.initial_size, w_size, is_start=True)

        self.layer_list = []
        for i in range(n_upscales):
            inp_channels = min(int(output_h_size * self.scaling_factor ** (n_upscales - i)), max_h_size)
            outp_channels = min(int(output_h_size * self.scaling_factor ** (n_upscales - i - 1)), max_h_size)
            self.layer_list.append(Stylev2ALAEGeneratorBlock(inp_channels, outp_channels, w_size))
        self.layers = torch.nn.ModuleList(self.layer_list)
        self.reset_parameters()

    def forward(self, w, phase=None):
        if phase is None:
            phase = self.n_upscales

        n_upscales = min(int(phase), self.n_upscales)
        alpha = phase - (n_upscales)

        if alpha == 0.0 and n_upscales >= 1:
            alpha += 1.0

        x, rgb = self.init_layer(None, w)

        if alpha == 0.0 and n_upscales == 0:
            return rgb

        next_x, next_rgb = self.layers[0](x, w)

        n_actual_upscales = n_upscales
        if 0 < alpha < 1:
            n_actual_upscales += 1

        for i in range(1, min(self.n_upscales, n_actual_upscales)):
            rgb = F.upsample_bilinear(rgb, scale_factor=2) + next_rgb
            x = next_x
            next_x, next_rgb = self.layers[i](x, w)

        if alpha == 1.0 and n_upscales > 0:
            return F.upsample_bilinear(rgb, scale_factor=2) + next_rgb

        out_rgb = F.upsample_bilinear(rgb, scale_factor=2) + alpha * next_rgb
        return out_rgb

    def reset_parameters(self):
        for layer in self.layer_list + [self.init_layer]:
            layer.reset_parameters()


class StyleALAEResidualEncoderBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, w_size):
        super().__init__()
        self.conv1 = Conv2dNormalizedLR(in_size, out_size, 3, padding=1)
        self.conv2 = Conv2dNormalizedLR(out_size, out_size, 3, padding=1)
        if in_size != out_size:
            self.conv_down = Conv2dNormalizedLR(in_size, out_size, 1)
        else:
            self.conv_down = None
        self.rgb = Conv2dNormalizedLR(3, in_size, 1)

        self.aff1 = Conv2dNormalizedLR(out_size*2, w_size, kernel_size=1, gain=1.0)
        self.aff2 = Conv2dNormalizedLR(out_size*2, w_size, kernel_size=1, gain=1.0)

    def forward(self, x):
        carry = x
        if self.conv_down is not None:
            carry = self.conv_down(carry)
        carry = torch.nn.functional.avg_pool2d(carry, 2)


        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)

        x, style1m, style1s = instance_norm_with_style_out(x)
        style1 = self.aff1(torch.cat([style1m, style1s], dim=1))

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)

        x = torch.nn.functional.avg_pool2d(x, 2)
        x, style2m, style2s = instance_norm_with_style_out(x)
        style2 = self.aff2(torch.cat([style2m, style2s], dim=1))
        x = x + carry
        x = x / (2.0**0.5)
        return x, style1, style2

    def reset_parameters(self):
        for layer in [self.conv1, self.conv2, self.rgb, self.aff1, self.aff2]:
            layer.reset_parameters()

    def from_rgb(self, x):
        x = self.rgb(x)
        x = F.leaky_relu(x, 0.2)
        return x


class ALAEResidualEncoder(ALAEEncoder):
    EncBlock = StyleALAEResidualEncoderBlock


# All of this below might not work yet
class StyleALAEAltEncoderBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, w_size):
        super().__init__()
        self.conv1 = Conv2dNormalizedLR(in_size, out_size, 3, padding=1)
        self.conv2 = Conv2dNormalizedLR(out_size, out_size, 3, padding=1)
        self.rgb = Conv2dNormalizedLR(3, in_size, 1)

        self.aff1 = Conv2dNormalizedLR(out_size*2, w_size, kernel_size=1, gain=1.0)
        self.aff2 = Conv2dNormalizedLR(out_size*2, w_size, kernel_size=1, gain=1.0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)

        x, style1m, style1s = instance_norm_with_style_out(x)
        style1 = self.aff1(torch.cat([style1m, style1s], dim=1))

        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)

        x = torch.nn.functional.avg_pool2d(x, 2)
        x, style2m, style2s = instance_norm_with_style_out(x)
        style2 = self.aff2(torch.cat([style2m, style2s], dim=1))
        return x, style1, style2

    def reset_parameters(self):
        for layer in [self.conv1, self.conv2, self.rgb, self.aff1, self.aff2]:
            layer.reset_parameters()

    def from_rgb(self, x):
        x = self.rgb(x)
        return x


class ALAESkipEncoder(torch.nn.Module):
    EncBlock = StyleALAEAltEncoderBlock
    def __init__(self, w_size, n_downscales, full_res_h_size, scaling_factor=2, max_h_size: int = 1e10):
        super().__init__()
        self.n_downscales = n_downscales
        self.h_size = full_res_h_size
        self.scaling_factor = scaling_factor
        self.w_size = w_size

        self.deepest_channels = min(int(full_res_h_size * (self.scaling_factor ** (n_downscales))), max_h_size)

        outp_block = self.EncBlock(self.deepest_channels, self.deepest_channels, w_size)

        self.layer_list = [outp_block]
        for i in range(n_downscales):
            inp_channels = min(int(full_res_h_size * (self.scaling_factor ** (n_downscales - i - 1))), max_h_size)
            outp_channels = min(int(full_res_h_size * (self.scaling_factor ** (n_downscales - i))), max_h_size)
            self.layer_list.append(self.EncBlock(inp_channels, outp_channels, w_size))
        self.layers = torch.nn.ModuleList(self.layer_list)
        self.reset_parameters()

    def forward(self, x, phase=None):
        if phase is None:
            phase = self.n_downscales

        n_downscales = min(int(phase), self.n_downscales)
        alpha = phase - n_downscales

        img = x

        w = None
        if alpha == 0.0:
            x = self.layers[n_downscales].from_rgb(x)
        else:
            x1 = self.layers[n_downscales + 1].from_rgb(x)
            x1, w1, w2 = self.layers[n_downscales + 1](x1)

            x2 = F.avg_pool2d(x, 2)
            x2 = self.layers[n_downscales].from_rgb(x2)

            x = alpha * x1 + (1 - alpha) * x2
            w = alpha * (w1 + w2)

        for i in range(0, n_downscales + 1):
            layer = self.layers[n_downscales - i]
            x = x + layer.from_rgb(F.interpolate(img, size=x.size(2)))
            x = x /(2**0.5)
            x, w1, w2 = layer(x)
            if w is None:
                w = w1 + w2
            else:
                w += w1 + w2
        return w

    def reset_parameters(self):
        for layer in self.layer_list:
            layer.reset_parameters()