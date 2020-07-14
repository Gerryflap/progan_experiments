from torchvision.transforms import transforms

from image_dataset import ImageDataset
from style_alae.train_ALAE import train

dataset = ImageDataset("../master_thesis/data/celeba_cropped/img_align",
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])
                        )
dataset2 = ImageDataset("data/aligned64",
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])
                        )

train(dataset2,
      n_shifting_steps=50000,
      n_static_steps=50000,
      batch_size=16,
      latent_size=256,
      h_size=128,
      lr=0.002,
      gamma=10.0,
      max_upscales=4,
      network_scaling_factor=2.0,
      start_phase=4,
      progress_bar=False,
      num_workers=0,
      n_steps_per_output=2000,
      max_h_size=256,
      use_stylegan2_gen=True,
      reg_every_n_steps=1
      )

