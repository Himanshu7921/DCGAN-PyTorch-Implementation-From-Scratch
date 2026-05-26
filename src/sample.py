"""
--------------------------------------------------- Sampling Pipeline --------------------------------------------------- 

This module implements the inference and visualization
pipeline for the trained DCGAN Generator

The generator maps latent vectors sampled from:
    z ~ N(0, I)

into structured RGB image manifolds through
deep convolutional feature synthesis

Utilities included:
    - Pretrained generator loading
    - Latent space sampling
    - Synthetic image generation
    - Generated image visualization

The module is primarily used for:
    - Qualitative generator evaluation
    - Latent manifold inspection
    - Adversarial convergence monitoring
    - Synthetic face generation

Paper:
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    https://arxiv.org/pdf/1511.06434
"""


import torch
import matplotlib.pyplot as plt
from config import config
from utils import load_generator

def sample_images(device, generator, n_images: int = 120, n_epochs: int = 1):
    LATENT_DIM = config["latent_space_size"]
    z = torch.randn(n_images, LATENT_DIM, 1, 1).to(device)
    with torch.no_grad():
        fake_imgs = generator(z)
        fake_imgs = fake_imgs.view(
            n_images,
            config["n_channels"],
            config["img_size"],
            config["img_size"]
        )

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            img = fake_imgs[i].cpu()
            img = img.permute(1, 2, 0)
            img = (img + 1) / 2 # Convert from [-1,1] → [0,1]
            img = img.clamp(0, 1)
            ax.imshow(img)
            ax.axis("off")

    fig.suptitle(
        f"Sampled images after epoch = {n_epochs}",
        fontsize=16
    )
    plt.tight_layout()
    return fig

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator =  load_generator(device)
    fig = sample_images(device, generator, n_images = 120)
    plt.show()

if __name__ == "__main__":
    main()