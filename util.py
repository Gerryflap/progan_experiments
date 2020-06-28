import torch
from torch import nn

lrelu_gain = (2.0/(1+0.2**2))**0.5

def weights_init(m):
    # This was taken from the PyTorch DCGAN tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    classname = m.__class__.__name__

    if classname.find('LinearNormalizedLR') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif classname.find('Conv2dNormalizedLR') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('Conv2dTransposeNormalizedLR') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class Conv2dNormalizedLR(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, stride=1, padding=0, bias=True, weight_norm=False, gain=lrelu_gain):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight_norm = weight_norm
        self.he_constant = gain / (float(in_channels * kernel_size * kernel_size) ** 0.5)

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, inp):
        weight = self.weight * self.he_constant
        if self.weight_norm:
            weight = apply_weight_norm(weight, input_dims=(1, 2, 3))
        x = torch.nn.functional.conv2d(inp, weight, self.bias, self.stride, self.padding)
        return x

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0.0, 1.0)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)


class Conv2dTransposeNormalizedLR(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, stride=1, padding=0, bias=True, weight_norm=False, gain=lrelu_gain):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight_norm = weight_norm
        # In the ProGAN source code the kernel**2 is also included.
        # I don't understand why, since the input of conv2d transpose is 1x1 as far as I'm aware, but okay.
        self.he_constant = gain / (float(in_channels * kernel_size * kernel_size) ** 0.5)

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size, kernel_size))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, inp):
        weight = self.weight * self.he_constant
        if self.weight_norm:
            weight = apply_weight_norm(weight, input_dims=(0, 2, 3))
        x = torch.nn.functional.conv_transpose2d(inp, weight, self.bias, self.stride, self.padding)
        return x

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0.0, 1.0)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)

class LinearNormalizedLR(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias=True, weight_norm=False, gain=lrelu_gain):
        super().__init__()
        self.he_constant = gain / (float(in_channels)**0.5)

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.weight_norm = weight_norm

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, inp):
        weight = self.weight * self.he_constant
        if self.weight_norm:
            weight = apply_weight_norm(weight, input_dims=1)
        x = torch.nn.functional.linear(inp, weight, self.bias)
        return x

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0.0, 1.0)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)


def local_response_normalization(x, eps=1e-8):
    """
    Implements the variant of LRN used in ProGAN https://arxiv.org/pdf/1710.10196.pdf
    :param eps: Epsilon is a small number added to the divisor to avoid division by zero
    :param x: Output of convolutional layer (or any other tensor with channels on axis 1)
    :return: Normalized x
    """
    divisor = (torch.pow(x, 2).mean(dim=1, keepdim=True) + eps).sqrt()
    b = x / divisor
    return b


class LocalResponseNorm(torch.nn.Module):
    def __init__(self, eps=1e-8):
        """
        Implements the variant of LRN used in ProGAN https://arxiv.org/pdf/1710.10196.pdf
        :param eps: Epsilon is a small number added to the divisor to avoid division by zero
        """
        super().__init__()
        self.eps = eps

    def forward(self, inp):
        return local_response_normalization(inp, self.eps)


def update_output_network(G_out, G, factor=0.999):
    for (p_out, p_train) in zip(G_out.parameters(), G.parameters()):
        p_out.data = p_out.data * factor + p_train.data * (1.0 - factor)


def save_checkpoint(folder_path, G, G_out, D, optim_G, optim_D, info_obj, enc=None, enc_opt=None):
    torch.save(
        {
            "G": G.state_dict(),
            "G_out": G_out.state_dict(),
            "D": D.state_dict(),
            "optim_G": optim_G.state_dict(),
            "optim_D": optim_D.state_dict(),
            "info": info_obj,
            "enc": enc.state_dict() if enc is not None else None,
            "enc_opt": enc_opt.state_dict() if enc_opt is not None else None,
        },
        folder_path
    )


def load_checkpoint(folder_path, G, G_out, D, optim_G, optim_D, enc=None, enc_opt=None):
    """
    Loads state dict into Modules
    :param path: Path to checkpoint
    :return Info object created by other methods
    """
    checkpoint = torch.load(folder_path)
    G.load_state_dict(checkpoint["G"])
    G_out.load_state_dict(checkpoint["G_out"])
    D.load_state_dict(checkpoint["D"])
    optim_G.load_state_dict(checkpoint["optim_G"])
    optim_D.load_state_dict(checkpoint["optim_D"])
    if enc is not None:
        enc.load_state_dict(checkpoint["enc"])
    if enc_opt is not None:
        enc_opt.load_state_dict(checkpoint["enc_opt"])
    return checkpoint["info"]


def apply_weight_norm(w, input_dims=(1, 2, 3), eps=1e-8):
    """
    Applies the "demodulation" operation from StyleGAN2 as a form of normalization.
    :param w: Weights
    :return: Normed weights
    """
    divisor = torch.rsqrt(torch.square(w).sum(dim=input_dims, keepdim=True) + eps)
    return w * divisor

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class ToColorTransform(object):
    def __call__(self, img):
        out = torch.cat([img]*3, dim=0)
        return out

def pixel_norm(x, epsilon=1e-8):
    # This function is taken from ALAE (https://github.com/podgorskiy/ALAE/)
    return x * torch.rsqrt(torch.mean(x.pow(2.0), dim=1, keepdim=True) + epsilon)

if __name__ == "__main__":
    layers = [Conv2dNormalizedLR(10, 10, 3, padding=1) for i in range(10)]
    for layer in layers:
        layer.reset_parameters()
    out = torch.normal(0, 1, (100, 10, 5, 5))
    for layer in layers:
        out = torch.nn.functional.leaky_relu(layer(out), 0.2)
    print(out.mean(), out.std(dim=0).mean())
