from torchvision.transforms import transforms

from image_dataset import ImageDataset
from style_alae.train_ALAE import train

dataset = ImageDataset("../master_thesis/data/celeba_cropped/img_align",
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])
                        )

train(dataset,
      n_shifting_steps=50000,
      n_static_steps=50000,
      batch_size=16,
      latent_size=512,
      h_size=32,
      lr=0.001,
      gamma=10.0,
      max_upscales=4,
      network_scaling_factor=2.0,
      start_phase=0,
      progress_bar=False,
      num_workers=0,
      n_steps_per_output=5000,
      max_h_size=512,
      )

