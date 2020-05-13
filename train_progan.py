import math

import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

from models import ProGANDiscriminator, ProGANGenerator


# Algo
def train(
        dataset,
        n_shifting_steps=20000,
        n_static_steps=20000,
        batch_size=16,
        latent_size=256,
        h_size=8,
        lr=0.001,
        gamma=750.0,
        max_upscales=4,
        network_scaling_factor=2.0,
        lrn_in_G=True,
        start_at=0,
        num_workers=0,
        progress_bar=False,
        shuffle=True
):
    if num_workers == 0:
        print("Using num_workers = 0. It might be useful to add more workers if your machine allows for it.")
    n_static_steps_taken = start_at
    n_shifting_steps_taken = start_at
    static = True

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

    G = ProGANGenerator(latent_size, max_upscales, 4, local_response_norm=lrn_in_G, scaling_factor=network_scaling_factor)
    D = ProGANDiscriminator(max_upscales, h_size, scaling_factor=network_scaling_factor)

    G = G.cuda()
    D = D.cuda()

    G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0, 0.99))
    D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0, 0.99))

    first_print = True

    test_z = torch.normal(0, 1, (16, latent_size), device="cuda")

    while True:
        for batch in loader:
            if static:
                n_static_steps_taken += 1
            else:
                n_shifting_steps_taken += 1

            phase = min(n_shifting_steps_taken / n_shifting_steps, max_upscales)

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
            grad = torch.autograd.grad(dis_out, x_hat, create_graph=True, only_inputs=True, grad_outputs=grad_outputs)[
                0]
            grad_norm = grad.norm(2, dim=list(range(1, len(grad.size())))) ** 2
            d_grad_loss = (torch.pow(grad_norm - gamma, 2) / (gamma ** 2)).mean()

            drift_loss = (d_real_outputs ** 2).mean() + (d_fake_outputs ** 2).mean()
            d_loss_total = d_loss + 10.0 * d_grad_loss + 0.001 * drift_loss

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
            elif (not static) and (n_shifting_steps_taken % n_shifting_steps == 0):
                print("Switching to static")
                static = True
                switched = True

            if progress_bar:
                percent = ((n_shifting_steps_taken + n_static_steps_taken)%1000)/10.0
                print("%03d %% till image generation..." % int(percent), end="\r", flush=True)

            if (n_shifting_steps_taken + n_static_steps_taken) % 1000 == 0:
                if first_print:
                    torchvision.utils.save_image(x, "results/reals.png")
                    first_print = False
                print("D loss: ", d_loss.detach().cpu().item())
                print("D loss total: ", d_loss_total.detach().cpu().item())
                print("G loss: ", g_loss.detach().cpu().item())
                print()

                test_batch = G(test_z, phase=phase)
                torchvision.utils.save_image(test_batch, "results/results_%d_%d_%.3f.png" % (
                    n_static_steps_taken, n_shifting_steps_taken, phase))


if __name__ == "__main__":
    from torchvision.datasets import CelebA
    # dataset = CelebA("/run/media/gerben/LinuxData/data/", download=False,
    #                  transform=transforms.Compose([
    #                      transforms.CenterCrop(178),
    #                      transforms.Resize(128),
    #                      transforms.ToTensor()
    #                  ])
    #                  )

    from frgc_cropped import FRGCCropped
    dataset = FRGCCropped("/run/media/gerben/LinuxData/data/frgc_cropped",
                     transform=transforms.Compose([
                         transforms.ToTensor()
                     ])
                     )

    train(dataset,
          n_shifting_steps=4000,
          n_static_steps=4000,
          batch_size=16,
          latent_size=256,
          h_size=4,
          lr=0.001,
          gamma=750.0,
          max_upscales=4,
          network_scaling_factor=1.5,
          lrn_in_G=False,
          start_at=0,
          progress_bar=True,
          num_workers=4
          )
