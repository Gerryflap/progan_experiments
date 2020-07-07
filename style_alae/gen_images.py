import argparse
import math

import torch
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description="Image generator")
parser.add_argument("--F_path", action="store", type=str, help="Path to F model .pt file", required=True)
parser.add_argument("--G_path", action="store", type=str, help="Path to G model .pt file", required=True)
parser.add_argument("--phase", action="store", type=float, help="Model phase", default=4.0)
parser.add_argument("--truncation_psi", action="store", type=float, help="Truncation psi is used trade diversity for quality. 1.0 is max diversity, 0.0 is least diversity.", default=1.0)
parser.add_argument("--n_images", action="store", type=int, help="Number of images to generate", default=16)
parser.add_argument("--nrow", action="store", type=int, help="nrow parameter of the torchvision save_image method", default=4)
args = parser.parse_args()

resolution = 4 * (2**math.ceil(args.phase))

F = torch.load(args.F_path, map_location=torch.device('cuda'))
G = torch.load(args.G_path, map_location=torch.device('cuda'))

w_size = G.w_size

z = torch.randn((args.n_images, w_size), device="cuda")
w = F(z)

if args.truncation_psi != 1.0:
    samples = torch.normal(0, 1, (1000, w_size), device="cuda")
    mean_w = F(samples).mean(dim=0, keepdim=True)
    w = mean_w + args.truncation_psi * (w - mean_w)



x = G(w)

save_image(x, "grid_output.png", nrow=args.nrow)
