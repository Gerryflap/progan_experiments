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

# Algo
from style_alae.models_and_components import ALAEGenerator, ALAEEncoder,  Fnetwork, Discriminator
from style_alae.stylegan2.models import ALAEGeneratorStyleGAN2


def train(
        dataset,
        n_shifting_steps=20000,
        n_static_steps=20000,
        batch_size=16,
        latent_size=256,
        h_size=8,
        lr=0.001,
        gamma=10.0,
        max_upscales=4,
        network_scaling_factor=2.0,
        start_phase=0,  # Can be used to start at a certain resolution. Only works well for whole numbers.
        num_workers=0,
        progress_bar=False,
        shuffle=True,
        n_steps_per_output=1000,
        max_h_size=None,
        # The maximum size of a hidden later output. None will default to 1e10, which is basically infinite,
        load_path=None,
        # Path to experiment folder. Can be used to load a checkpoint. It currently only sets the parameters, not hyperparameters!
        use_stylegan2_gen=False
):
    if max_h_size is None:
        max_h_size = int(1e10)
    if num_workers == 0:
        print("Using num_workers = 0. It might be useful to add more workers if your machine allows for it.")

    n_static_steps_taken = 0
    n_shifting_steps_taken = 0
    static = True

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

    if not use_stylegan2_gen:
        G = ALAEGenerator(latent_size, max_upscales, h_size, scaling_factor=network_scaling_factor, max_h_size=max_h_size)
    else:
        G = ALAEGeneratorStyleGAN2(latent_size, max_upscales, h_size, scaling_factor=network_scaling_factor, max_h_size=max_h_size)

    E = ALAEEncoder(latent_size, max_upscales, h_size, scaling_factor=network_scaling_factor, max_h_size=max_h_size)

    Fnet = Fnetwork(latent_size, 8)

    D = Discriminator(latent_size, 3)

    G = G.cuda()
    D = D.cuda()
    Fnet = Fnet.cuda()
    E = E.cuda()

    G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0, 0.99))
    D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0, 0.99))
    E_opt = torch.optim.Adam(E.parameters(), lr=lr, betas=(0, 0.99))
    F_opt = torch.optim.Adam(Fnet.parameters(), lr=lr * 0.01, betas=(0, 0.99))
    # F_opt = torch.optim.Adam(Fnet.parameters(), lr=lr*0.1, betas=(0, 0.99))

    if load_path is not None:
        print("ERROR: load path not supported")
        exit()
        # info = util.load_checkpoint(os.path.join(load_path, "checkpoint.pt"), G, G_out, D, G_opt, D_opt, encoder, enc_opt)
        # n_static_steps_taken = info["n_stat"]
        # n_shifting_steps_taken = info["n_shift"]
        # static = info["static"]
        #
        # if n_static_steps_taken - n_static_steps >= n_shifting_steps_taken:
        #     static = False
        #     print("Switching to shift")
        #
        #
        # output_path = load_path
        info = None
    else:
        now = datetime.now()

        output_path = os.path.join("results", "exp_%s" % (now.strftime("%Y%m%d%H%M")))
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, "encoding"))
        info = None

    first_print = True

    if info is None or info["test_z"] is None:
        test_z = torch.normal(0, 1, (16, latent_size), device="cuda")
    else:
        test_z = info["test_z"]

    if info is None or info["test_x"] is None:
        test_x = None
    else:
        test_x = info["test_x"]
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

            if test_x is None:
                test_x = x[:16]

            x = F.interpolate(x, 4 * (2 ** (math.ceil(phase))), mode='bilinear')
            # =========== Train D ========
            D_opt.zero_grad()
            E_opt.zero_grad()

            x.requires_grad = True
            d_real_outputs = D(E(x, phase=phase))

            grad_outputs = torch.ones_like(d_real_outputs)
            grad = \
            torch.autograd.grad(d_real_outputs, x, create_graph=True, only_inputs=True, grad_outputs=grad_outputs)[
                0]
            grad_norm = torch.sum(grad.pow(2.0), dim=[1, 2, 3])
            d_grad_loss = grad_norm.mean()

            d_reg = (gamma / 2.0) * d_grad_loss

            z = torch.normal(0, 1, (batch_size, latent_size), device="cuda")
            w_fake = Fnet(z)
            fake_batch = G(w_fake, phase=phase)

            # Compute outputs for fake images
            d_fake_outputs = D(E(fake_batch, phase=phase))

            # Compute losses
            d_loss = F.softplus(d_fake_outputs).mean() + F.softplus(-d_real_outputs).mean() + d_reg

            d_loss.backward()

            # Update weights
            D_opt.step()
            E_opt.step()

            # ======== Train G ========
            # Make gradients for G zero
            G_opt.zero_grad()
            F_opt.zero_grad()

            # Generate a batch of fakes
            z = torch.normal(0, 1, (batch_size, latent_size), device="cuda")
            w_fake = Fnet(z)
            fake_batch = G(w_fake, phase=phase)

            # Compute loss for G, images should become more 'real' to the discriminator
            g_loss = F.softplus(-D(E(fake_batch, phase=phase))).mean()
            g_loss.backward()

            G_opt.step()
            F_opt.step()

            # Step 3: Update E and G
            E_opt.zero_grad()
            G_opt.zero_grad()
            with torch.no_grad():
                z = torch.normal(0, 1, (batch_size, latent_size), device="cuda")
                w = Fnet(z)
            w_recon = E(G(w, phase=phase), phase=phase)
            loss = F.mse_loss(w_recon, w)
            loss.backward()
            E_opt.step()
            G_opt.step()

            if progress_bar:
                percent = ((n_shifting_steps_taken + n_static_steps_taken) % n_steps_per_output) / (
                        n_steps_per_output / 100.0)
                print("%03d %% till image generation..." % int(percent), end="\r", flush=True)

            if (n_shifting_steps_taken + n_static_steps_taken) % n_steps_per_output == 0:
                if progress_bar:
                    print(" " * 50, end="\r", flush=True)
                print("Print at ", n_static_steps_taken, n_shifting_steps_taken, phase)
                if first_print:
                    torchvision.utils.save_image(x, os.path.join(output_path, "reals.png"))
                    first_print = False
                    last_print = time.time()
                else:
                    current_time = time.time()
                    diff = current_time - last_print
                    per_step = diff / n_steps_per_output
                    last_print = current_time
                    print("Seconds since last print: %.2f, seconds per step: %.5f" % (diff, per_step))

                print("D loss: ", d_loss.detach().cpu().item())
                print("G loss: ", g_loss.detach().cpu().item())
                print("w loss", loss.detach().cpu().item())

                print()

                test_batch = torch.clamp(G(Fnet(test_z), phase=phase).detach(), 0, 1)
                torchvision.utils.save_image(test_batch, os.path.join(output_path, "results_%d_%d_%.3f.png" % (
                    n_static_steps_taken, n_shifting_steps_taken, phase)))

                x_res = F.interpolate(test_x, 4 * (2 ** (math.ceil(phase))), mode='bilinear')
                test_recons = torch.clamp(G(E(x_res, phase=phase), phase=phase).detach(), 0, 1)
                torchvision.utils.save_image(torch.cat([x_res, test_recons], dim=0),
                                             os.path.join(output_path, "encoding", "results_%d_%d_%.3f.png" % (
                                             n_static_steps_taken, n_shifting_steps_taken, phase)),
                                             nrow=min(16, batch_size))

                torch.save(G, os.path.join(output_path, "G.pt"))
                torch.save(E, os.path.join(output_path, "E.pt"))
                torch.save(Fnet, os.path.join(output_path, "F.pt"))
                info = {
                    "n_stat": n_static_steps_taken,
                    "n_shift": n_shifting_steps_taken,
                    "static": static,
                    "test_z": test_z,
                    "test_x": test_x,
                }
                # util.save_checkpoint(os.path.join(output_path, "checkpoint.pt"), G, G_out, D, G_opt, D_opt, info, enc=encoder, enc_opt=enc_opt)

            if static and (n_static_steps_taken % n_static_steps == 0):
                print("Switching to shift")
                static = False
            elif (not static) and (n_shifting_steps_taken % n_shifting_steps == 0):
                print("Switching to static")
                static = True


if __name__ == "__main__":
    from torchvision.datasets import CelebA, MNIST

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

    dataset4 = ImageDataset("/run/media/gerben/LinuxData/data/celeba/cropped_faces64",
                            transform=transforms.Compose([
                                transforms.ToTensor()
                            ])
                            )

    mnist = MNIST("mnist_data", transform=transforms.Compose([
                                transforms.Resize(32),
                                transforms.ToTensor(),
                                util.ToColorTransform()
                            ]), download=True)

    train(dataset3,
          n_shifting_steps=10000,
          n_static_steps=10000,
          batch_size=16,
          latent_size=256,
          h_size=24,
          lr=0.002,
          gamma=10.0,
          max_upscales=4,
          network_scaling_factor=2.0,
          start_phase=4,
          progress_bar=False,
          num_workers=4,
          n_steps_per_output=1000,
          max_h_size=256,
          use_stylegan2_gen=True
          )
