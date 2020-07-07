# Progressive growing GANs in PyTorch (WIP)

This repository is an exploration of progressive growing GANs like ProGAN, StyleGAN and StyleALAE.
At this moment it is mostly focussed on StyleALAE and all other algorithms will probably contain bugs or won't work anyways.


## Current results
The current StyleALAE implementation does appear to yield okay results. 
A version trained with 30,000 static and 30,000 shifting batches, 
which means it does 30,000 batches per resolution and then takes 30,000 batches to slowly shift to the next resolution,
manages to generate the following images when trained up to 64x64 FFHQ images (using Ψ = 0.85) :

![There should be an image here](results/grid_output.png)
