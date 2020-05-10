import math

import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

from models import ProGANDiscriminator, ProGANGenerator

steps_per_phase = 10000
n_static_steps = 10000
batch_size = 16
latent_size = 32
h_size = 8
lr = 0.0001
gamma = 750.0
max_upscales = 4


dataset = CelebA("/run/media/gerben/LinuxData/data/", download=False,
                 transform=transforms.Compose([
                     transforms.CenterCrop(178),
                     transforms.Resize(64),
                     transforms.ToTensor()
                 ])
                 )

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
print(dataset[0][0].size())


# Algo
n_static_steps_taken = 10000
n_shifting_steps_taken = 10000
static = True

G = ProGANGenerator(latent_size, max_upscales, 4, local_response_norm=False)
D = ProGANDiscriminator(max_upscales, h_size)

G = G.cuda()
D = D.cuda()

G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0, 0.9))
D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0, 0.9))

first_print = True

while True:
    for batch in loader:
        if static:
            n_static_steps_taken += 1
        else:
            n_shifting_steps_taken += 1

        phase = min(n_shifting_steps_taken/steps_per_phase, max_upscales)

        x, _ = batch
        x = x.cuda()
        x = F.interpolate(x, 4 * (2 ** (math.ceil(phase))))

        # =========== Train D ========
        D_opt.zero_grad()

        d_real_outputs = D(x, phase=phase)

        z = torch.normal(0, 1, (batch_size, latent_size), device="cuda")
        fake_batch = G(z, phase=phase)

        # Compute outputs for fake images
        d_fake_outputs = D(fake_batch, phase=phase)

        # Compute losses
        d_loss = (d_fake_outputs - d_real_outputs).mean()

        size = [s if i == 0 else 1 for i, s in enumerate(fake_batch.size())]
        eps = torch.rand(size).cuda()
        x_hat = eps * x + (1.0 - eps) * fake_batch.detach()
        x_hat.requires_grad = True
        dis_out = D(x_hat, phase=phase)
        grad_outputs = torch.ones_like(dis_out)
        grad = torch.autograd.grad(dis_out, x_hat, create_graph=True, only_inputs=True, grad_outputs=grad_outputs)[0]
        grad_norm = grad.norm(2, dim=list(range(1, len(grad.size()))))
        d_grad_loss = (torch.pow(grad_norm - gamma, 2)/(gamma**2)).mean()

        drift_loss = (d_real_outputs ** 2).mean() + (d_fake_outputs ** 2).mean()
        d_loss_total = d_loss + 10.0 * d_grad_loss + 0.0001 * drift_loss

        d_loss_total.backward()

        # Update weights
        D_opt.step()

        # ======== Train G ========
        # Make gradients for G zero
        G_opt.zero_grad()

        # Generate a batch of fakes
        z = torch.normal(0, 1, (batch_size, latent_size), device="cuda")
        fake_batch = G(z, phase=phase)

        # Compute loss for G, images should become more 'real' to the discriminator
        g_loss = -D(fake_batch, phase=phase).mean()
        g_loss.backward()

        G_opt.step()

        switched = False
        if static and (n_static_steps_taken % n_static_steps == 0):
            print("Switching to shift")
            static = False
            switched = True
        elif (not static) and (n_shifting_steps_taken % steps_per_phase == 0):
            print("Switching to static")
            static = True
            switched = True

        if (n_shifting_steps_taken + n_static_steps_taken)%1000 == 0:
            if first_print:
                torchvision.utils.save_image(x, "results/reals.png")
                first_print = False
            print("D loss: ", d_loss.detach().cpu().item())
            print("D loss total: ", d_loss_total.detach().cpu().item())
            print("G loss: ", g_loss.detach().cpu().item())
            print()

            # with open("results/results_%d_%d_%.3f.png"%(n_static_steps_taken, n_shifting_steps_taken, phase), "w") as f:
            #     torchvision.utils.save_image(fake_batch, f)
            torchvision.utils.save_image(fake_batch, "results/results_%d_%d_%.3f.png"%(n_static_steps_taken, n_shifting_steps_taken, phase))