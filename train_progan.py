import math
import os
import time
from datetime import date, datetime

import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

import util
from models import ProGANDiscriminator, ProGANGenerator

# Algo
from models_additional import ProGANAdditiveGenerator, ProGANResDiscriminator


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
        start_phase=0,                        # Can be used to start at a certain resolution. Only works well for whole numbers.
        num_workers=0,
        progress_bar=False,
        shuffle=True,
        n_steps_per_output=1000,
        use_special_output_network=False,  # When true: use exponential running avg of weights for G output
        use_additive_net=False,            # Use a network that adds the output of rgb layers together
        use_residual_discriminator=False,
        occasional_regularization=False,    # Only apply regularization every 10 steps, but 10x as strongly
        max_h_size=None,                     # The maximum size of a hidden later output. None will default to 1e10, which is basically infinite,
        load_path=None,                    # Path to experiment folder. Can be used to load a checkpoint. It currently only sets the parameters, not hyperparameters!
):

    if max_h_size is None:
        max_h_size = int(1e10)
    if num_workers == 0:
        print("Using num_workers = 0. It might be useful to add more workers if your machine allows for it.")

    n_static_steps_taken = 0
    n_shifting_steps_taken = 0
    static = True


    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

    if use_additive_net:
        G = ProGANAdditiveGenerator(latent_size, max_upscales, 4, local_response_norm=lrn_in_G,
                                    scaling_factor=network_scaling_factor, max_h_size=max_h_size)
    else:
        G = ProGANGenerator(latent_size, max_upscales, 4, local_response_norm=lrn_in_G,
                            scaling_factor=network_scaling_factor, max_h_size=max_h_size)

    if use_residual_discriminator:
        D = ProGANResDiscriminator(max_upscales, h_size, scaling_factor=network_scaling_factor, max_h_size=max_h_size)
    else:
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

    if load_path is not None:
        info = util.load_checkpoint(os.path.join(load_path, "checkpoint.pt"), G, G_out, D, G_opt, D_opt)
        n_static_steps_taken = info["n_stat"]
        n_shifting_steps_taken = info["n_shift"]
        static = info["static"]
        output_path = load_path
    else:
        now = datetime.now()

        output_path = os.path.join("results", "exp_%s"%(now.strftime("%Y%m%d%H%M")))
        os.mkdir(output_path)
        info = None

    first_print = True

    if info is None or info["test_z"] is None:
        test_z = torch.normal(0, 1, (16, latent_size), device="cuda")
    else:
        test_z = info["test_z"]
    last_print = None

    while True:
        for batch in loader:
            if static:
                n_static_steps_taken += 1
            else:
                n_shifting_steps_taken += 1

            phase = min(start_phase + (n_shifting_steps_taken / n_shifting_steps), max_upscales)

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



            if progress_bar:
                percent = ((n_shifting_steps_taken + n_static_steps_taken) % n_steps_per_output) / (
                            n_steps_per_output / 100.0)
                print("%03d %% till image generation..." % int(percent), end="\r", flush=True)

            if (n_shifting_steps_taken + n_static_steps_taken) % n_steps_per_output == 0:
                if progress_bar:
                    print(" "*50, end="\r", flush=True)
                print("Print at ", n_static_steps_taken, n_shifting_steps_taken, phase)
                if first_print:
                    torchvision.utils.save_image(x, os.path.join(output_path, "reals.png"))
                    first_print = False
                    last_print = time.time()
                else:
                    current_time = time.time()
                    diff = current_time - last_print
                    per_step = diff/n_steps_per_output
                    last_print = current_time
                    print("Seconds since last print: %.2f, seconds per step: %.5f"%(diff, per_step))

                print("D loss: ", d_loss.detach().cpu().item())
                print("D loss total: ", d_loss_total.detach().cpu().item())
                print("G loss: ", g_loss.detach().cpu().item())

                print()

                test_batch = G_out(test_z, phase=phase)
                torchvision.utils.save_image(test_batch, os.path.join(output_path, "results_%d_%d_%.3f.png" % (
                    n_static_steps_taken, n_shifting_steps_taken, phase)))

                torch.save(G, os.path.join(output_path, "G.pt"))
                info = {
                    "n_stat": n_static_steps_taken,
                    "n_shift": n_shifting_steps_taken,
                    "static": static,
                    "test_z": test_z
                }
                util.save_checkpoint(os.path.join(output_path, "checkpoint.pt"), G, G_out, D, G_opt, D_opt, info)

            if static and (n_static_steps_taken % n_static_steps == 0):
                print("Switching to shift")
                static = False
            elif (not static) and (n_shifting_steps_taken % n_shifting_steps == 0):
                print("Switching to static")
                static = True


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
          n_shifting_steps=15000,
          n_static_steps=15000,
          batch_size=16,
          latent_size=512,
          h_size=4,
          lr=0.003,
          gamma=750.0,
          max_upscales=4,
          network_scaling_factor=2.0,
          lrn_in_G=True,
          start_phase=0,
          progress_bar=True,
          num_workers=4,
          n_steps_per_output=1000,
          use_special_output_network=True,
          use_additive_net=True,
          use_residual_discriminator=True,
          max_h_size=128,
          load_path="results/exp_202005251652"
          )
