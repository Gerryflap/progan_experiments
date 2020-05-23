import math

import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

import util
from models import ProGANDiscriminator, ProGANGenerator

# Algo
from models_additional import ProGANAdditiveGenerator


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
        shuffle=True,
        n_steps_per_output=1000,
        use_special_output_network=False,  # When true: use exponential running avg of weights for G output
        use_additive_net=False,            # Use a network that adds the output of rgb layers together
        occasional_regularization=False,    # Only apply regularization every 10 steps, but 10x as strongly
        max_h_size=None                     # The maximum size of a hidden later output. None will default to 1e10, which is basically infinite
):
    if max_h_size is None:
        max_h_size = int(1e10)
    if num_workers == 0:
        print("Using num_workers = 0. It might be useful to add more workers if your machine allows for it.")
    n_static_steps_taken = start_at
    n_shifting_steps_taken = start_at
    static = True

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

    if use_additive_net:
        G = ProGANAdditiveGenerator(latent_size, max_upscales, 4, local_response_norm=lrn_in_G,
                                    scaling_factor=network_scaling_factor, max_h_size=max_h_size)
    else:
        G = ProGANGenerator(latent_size, max_upscales, 4, local_response_norm=lrn_in_G,
                            scaling_factor=network_scaling_factor, max_h_size=max_h_size)

    D = ProGANDiscriminator(max_upscales, h_size, scaling_factor=network_scaling_factor, max_h_size=max_h_size)

    G = G.cuda()
    D = D.cuda()

    if use_special_output_network:
        if use_additive_net:
            G_out = ProGANAdditiveGenerator(latent_size, max_upscales, 4, local_response_norm=lrn_in_G,
                                            scaling_factor=network_scaling_factor, max_h_size=max_h_size)
        else:
            G_out = ProGANGenerator(latent_size, max_upscales, 4, local_response_norm=lrn_in_G,
                                    scaling_factor=network_scaling_factor, max_h_size=max_h_size)
        G_out = G_out.cuda()
        # Set the weights of G_out to those of G
        util.update_output_network(G_out, G, factor=0.0)
    else:
        G_out = G

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

            if (not occasional_regularization) or (n_static_steps_taken % n_static_steps == 0)%10 == 0:
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

                d_reg = 10.0 * d_grad_loss

                drift_loss = (d_real_outputs ** 2).mean() + (d_fake_outputs ** 2).mean()
                d_reg += 0.001 * drift_loss

                if occasional_regularization:
                    d_reg = 10.0 * d_reg
                d_loss_total = d_loss + d_reg
            else:
                d_loss_total = d_loss

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

            if use_special_output_network:
                util.update_output_network(G_out, G)

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
                percent = ((n_shifting_steps_taken + n_static_steps_taken) % n_steps_per_output) / (
                            n_steps_per_output / 100.0)
                print("%03d %% till image generation..." % int(percent), end="\r", flush=True)

            if (n_shifting_steps_taken + n_static_steps_taken) % n_steps_per_output == 0:
                if first_print:
                    torchvision.utils.save_image(x, "results/reals.png")
                    first_print = False
                print("D loss: ", d_loss.detach().cpu().item())
                print("D loss total: ", d_loss_total.detach().cpu().item())
                print("G loss: ", g_loss.detach().cpu().item())
                print()

                test_batch = G_out(test_z, phase=phase)
                torchvision.utils.save_image(test_batch, "results/results_%d_%d_%.3f.png" % (
                    n_static_steps_taken, n_shifting_steps_taken, phase))

                torch.save(G, "G.pt")
                torch.save(G_out, "G_out.pt")
                torch.save(D, "D.pt")


if __name__ == "__main__":
    from torchvision.datasets import CelebA

    dataset = CelebA("/run/media/gerben/LinuxData/data/", download=False,
                     transform=transforms.Compose([
                         transforms.CenterCrop(178),
                         transforms.Resize(128),
                         transforms.ToTensor()
                     ])
                     )

    from image_dataset import ImageDataset

    dataset2 = ImageDataset("/run/media/gerben/LinuxData/data/frgc_cropped",
                            transform=transforms.Compose([
                               transforms.ToTensor()
                           ])
                            )

    dataset3 = ImageDataset("/run/media/gerben/LinuxData/data/ffhq_thumbnails/thumbnails128x128",
                            transform=transforms.Compose([
                               transforms.ToTensor()
                           ])
                            )

    train(dataset3,
          n_shifting_steps=5000,
          n_static_steps=5000,
          batch_size=16,
          latent_size=256,
          h_size=8,
          lr=0.01,
          gamma=750.0,
          max_upscales=5,
          network_scaling_factor=2.0,
          lrn_in_G=True,
          start_at=0,
          progress_bar=True,
          num_workers=4,
          n_steps_per_output=1000,
          use_special_output_network=True,
          use_additive_net=True,
          max_h_size=256
          )
