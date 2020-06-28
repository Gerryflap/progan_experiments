import torch

from util import Conv2dNormalizedLR, LinearNormalizedLR, Conv2dTransposeNormalizedLR, Reshape
import torch.nn.functional as F


def instance_norm_with_style_out(x):
    # Outputs instance normed x and the means and stddevs of the style
    means = x.mean(dim=(2, 3), keepdim=True)
    stddevs = torch.sqrt(torch.mean((x - means)**2, dim=(2, 3), keepdim=True) + 1e-8)

    x = (x - means)/(stddevs)
    return x, means, stddevs


def adaIN(x, style):
    # Performs adaptive instance norm. Style should be a tuple of two (B,C,1,1) tensors for ys and yb
    ys, yb = style
    x_means = x.mean(dim=(2, 3), keepdim=True)
    x_std = torch.sqrt(torch.mean((x - x_means)**2, dim=(2, 3), keepdim=True) + 1e-8)
    x = ys * ((x - x_means)/x_std) + yb
    return x


class StyleALAEEncoderBlock(torch.nn.Module):
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
        x = F.leaky_relu(x, 0.2)
        return x

class StyleALAEGeneratorBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, w_size, is_start=False):
        super().__init__()
        self.is_start = is_start
        self.out_size = out_size

        if not is_start:
            self.conv1 = Conv2dTransposeNormalizedLR(in_size, out_size, 3, padding=1)
        else:
            self.start = torch.nn.Parameter(torch.ones((1, out_size, 4, 4)), requires_grad=True)

        self.conv2 = Conv2dTransposeNormalizedLR(out_size, out_size, 3, padding=1)

        self.Aaff1 = Conv2dNormalizedLR(w_size, out_size * 2, kernel_size=1, gain=1.0)
        self.Aaff2 = Conv2dNormalizedLR(w_size, out_size * 2, kernel_size=1, gain=1.0)

        self.Baff1s = Conv2dNormalizedLR(1, out_size, kernel_size=1, gain=1.0)
        self.Baff2s = Conv2dNormalizedLR(1, out_size, kernel_size=1, gain=1.0)
        self.Baff1b = Conv2dNormalizedLR(1, out_size, kernel_size=1, gain=1.0)
        self.Baff2b = Conv2dNormalizedLR(1, out_size, kernel_size=1, gain=1.0)

        self.rgb = Conv2dNormalizedLR(out_size, 3, 1, gain=0.03)
        self.reset_parameters()

    def to_rgb(self, x):
        x = self.rgb(x)
        # x = F.sigmoid(x)
        return x

    def forward(self, x, w):
        if self.is_start:
            x = self.start
        else:
            x = torch.nn.functional.upsample_bilinear(x, scale_factor=2)
            x = self.conv1(x)
        noise = torch.normal(0, 1, (w.size(0), 1, x.size(2), x.size(3)), device="cuda")
        noise_ys = self.Baff1s(noise)
        noise_yb = self.Baff1b(noise)
        x = x * noise_ys + noise_yb
        x = F.leaky_relu(x, 0.2)


        style1_aff = self.Aaff1(w)
        ys, yb = style1_aff[:, :self.out_size], style1_aff[:, self.out_size:]
        x = adaIN(x, (ys, yb))

        x = self.conv2(x)
        noise = torch.normal(0, 1, (w.size(0), 1, x.size(2), x.size(3)), device="cuda")
        noise_ys = self.Baff2s(noise)
        noise_yb = self.Baff2b(noise)
        x = x * noise_ys + noise_yb
        x = F.leaky_relu(x, 0.2)


        style2_aff = self.Aaff2(w)
        ys, yb = style2_aff[:, :self.out_size], style2_aff[:, self.out_size:]
        x = adaIN(x, (ys, yb))
        return x, self.to_rgb(x)

    def reset_parameters(self):
        if self.is_start:
            torch.nn.init.ones_(self.start)
        else:
            self.conv1.reset_parameters()

        torch.nn.init.zeros_(self.Baff1s.weight)
        torch.nn.init.zeros_(self.Baff2s.weight)
        torch.nn.init.ones_(self.Baff1s.bias)
        torch.nn.init.ones_(self.Baff2s.bias)
        torch.nn.init.zeros_(self.Baff1b.weight)
        torch.nn.init.zeros_(self.Baff2b.weight)
        torch.nn.init.zeros_(self.Baff1b.bias)
        torch.nn.init.zeros_(self.Baff2b.bias)

        for layer in [self.conv2, self.rgb, self.Aaff1, self.Aaff2]:
            layer.reset_parameters()


class ALAEEncoder(torch.nn.Module):
    def __init__(self, w_size, n_downscales, full_res_h_size, scaling_factor=2, max_h_size: int = 1e10):
        super().__init__()
        self.n_downscales = n_downscales
        self.h_size = full_res_h_size
        self.scaling_factor = scaling_factor
        self.w_size = w_size

        self.deepest_channels = min(int(full_res_h_size * (self.scaling_factor ** (n_downscales))), max_h_size)

        outp_block = StyleALAEEncoderBlock(self.deepest_channels, self.deepest_channels, w_size)

        self.layer_list = [outp_block]
        for i in range(n_downscales):
            inp_channels = min(int(full_res_h_size * (self.scaling_factor ** (n_downscales - i - 1))), max_h_size)
            outp_channels = min(int(full_res_h_size * (self.scaling_factor ** (n_downscales - i))), max_h_size)
            self.layer_list.append(StyleALAEEncoderBlock(inp_channels, outp_channels, w_size))
        self.layers = torch.nn.ModuleList(self.layer_list)
        self.reset_parameters()

    def forward(self, x, phase=None):
        if phase is None:
            phase = self.n_downscales

        n_downscales = min(int(phase), self.n_downscales)
        alpha = phase - n_downscales

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
            x, w1, w2 = layer(x)
            if w is None:
                w = w1 + w2
            else:
                w += w1 + w2
        return w

    def reset_parameters(self):
        for layer in self.layer_list:
            layer.reset_parameters()


class ALAEGenerator(torch.nn.Module):
    def __init__(self, w_size, n_upscales, output_h_size, scaling_factor=2, max_h_size: int = 1e10):
        super().__init__()
        self.n_upscales = n_upscales
        self.output_h_size = output_h_size
        self.scaling_factor = scaling_factor
        self.initial_size = min(int(output_h_size * self.scaling_factor ** (n_upscales)), max_h_size)
        self.w_size = w_size

        self.init_layer = StyleALAEGeneratorBlock(self.initial_size, self.initial_size, w_size, is_start=True)

        self.layer_list = []
        for i in range(n_upscales):
            inp_channels = min(int(output_h_size * self.scaling_factor ** (n_upscales - i)), max_h_size)
            outp_channels = min(int(output_h_size * self.scaling_factor ** (n_upscales - i - 1)), max_h_size)
            self.layer_list.append(StyleALAEGeneratorBlock(inp_channels, outp_channels, w_size))
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
            x, rgb = next_x, next_rgb
            next_x, next_rgb = self.layers[i](x, w)

        if alpha == 1.0 and n_upscales > 0:
            return next_rgb

        out_rgb = (1 - alpha) * F.interpolate(rgb, scale_factor=2, mode='bilinear') + alpha * next_rgb
        return out_rgb

    def reset_parameters(self):
        for layer in self.layer_list + [self.init_layer]:
            layer.reset_parameters()


def init_F_net(latent_size, n_layers):
    layers = []
    for i in range(n_layers):
        last_layer = i == n_layers - 1
        layers.append(LinearNormalizedLR(latent_size, latent_size))
        if not last_layer:
            layers.append(torch.nn.LeakyReLU(0.2))
        else:
            layers.append(Reshape((-1, latent_size, 1, 1)))
    for layer in layers:
        if isinstance(layer, LinearNormalizedLR):
            layer.reset_parameters()
    return torch.nn.Sequential(*layers)


def init_D_net(latent_size, n_layers):
    layers = [Reshape((-1, latent_size))]
    for i in range(n_layers):
        last_layer = i == n_layers - 1
        if last_layer:
            out = 1
        else:
            out = latent_size
        layers.append(LinearNormalizedLR(latent_size, out))
        if not last_layer:
            layers.append(torch.nn.LeakyReLU(0.2))

    for layer in layers:
        if isinstance(layer, LinearNormalizedLR):
            layer.reset_parameters()
    return torch.nn.Sequential(*layers)
