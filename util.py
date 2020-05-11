import torch
from torch import nn


def weights_init(m):
    # This was taken from the PyTorch DCGAN tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    # The value for stddev has been altered to be equal to the ALI value
    classname = m.__class__.__name__

    if classname.find('LinearNormalizedLR') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif classname.find('Conv2dNormalizedLR') != -1:
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.he_constant = (2.0/float(in_channels*kernel_size*kernel_size))**0.5

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, inp):
        weight = self.weight * self.he_constant
        x = torch.nn.functional.conv2d(inp, weight, self.bias, self.stride, self.padding)
        return x

    def reset_parameters(self):
        self.apply(weights_init)

class Conv2dTransposeNormalizedLR(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.he_constant = (2.0/float(in_channels))**0.5

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size, kernel_size))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, inp):
        weight = self.weight * self.he_constant
        x = torch.nn.functional.conv_transpose2d(inp, weight, self.bias, self.stride, self.padding)
        return x

    def reset_parameters(self):
        self.apply(weights_init)


class LinearNormalizedLR(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias=True):
        super().__init__()
        self.he_constant = (2.0/float(in_channels))**0.5

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, inp):
        weight = self.weight * self.he_constant
        x = torch.nn.functional.linear(inp, weight, self.bias)
        return x

    def reset_parameters(self):
        self.apply(weights_init)


def local_response_normalization(x, eps=1e-8):
    """
    Implements the variant of LRN used in ProGAN https://arxiv.org/pdf/1710.10196.pdf
    :param eps: Epsilon is a small number added to the divisor to avoid division by zero
    :param x: Output of convolutional layer (or any other tensor with channels on axis 1)
    :return: Normalized x
    """
    divisor = (torch.pow(x, 2).mean(dim=1, keepdim=True) + eps).sqrt()
    b = x/divisor
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
