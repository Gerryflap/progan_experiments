"""
    This file measures the contribution of each resolution to the output image for the additive generator model.
"""
import argparse
import torch

parser = argparse.ArgumentParser(description="Layer influence computation")
parser.add_argument("--path", action="store", type=str, required=True, help="Path to generator model")
args = parser.parse_args()

model = torch.load(args.path)

max_phase = model.n_upscales

layer_stddevs = []
carry = None
z_shape = int(model.inp_layer.weight.data.size(1))
zs = torch.randn((16, z_shape)).cuda()
for phase in range(max_phase):
    output = model(zs, phase=float(phase))

    if carry is not None:
        layer_stddevs.append((output - carry).std().detach().item())
    else:
        layer_stddevs.append((output).std().detach().item())
    carry = torch.nn.functional.interpolate(output, scale_factor=2, mode='bilinear')

print(layer_stddevs)