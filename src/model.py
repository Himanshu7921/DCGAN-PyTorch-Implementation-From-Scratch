"""
--------------------------------------------------- Implementation Details --------------------------------------------------- 
This implements the complete GAN Architecture using:
    - Generator      (src/generator.py)
    - Discriminator  (src/discriminator.py)

The model learns a minimax adversarial game between
the Generator G and the Discriminator D

Generator:
    G : Z → X

Discriminator:
    D : X → [0, 1]

where:
    Z ~ N(0, I)
    X ~ p_data(x)

The generator attempts to synthesize realistic samples
from a latent distribution while the discriminator attempts
to distinguish real samples from generated samples

The adversarial objective is defined as:

    min_G max_D V(G, D)

Through iterative adversarial optimization the generator
progressively approximates the true data distribution

--------------------------------------------------- Generator Transformation Pipeline --------------------------------------------------- 

Latent Vector Sampling:
    z ~ N(0, I)

Input Noise Tensor:
    z ∈ R^{B × 128 × 1 × 1}

Generator Transformation:
    [B, 128, 1, 1]
        ↓
    [B, 512, 4, 4]
        ↓
    [B, 256, 8, 8]
        ↓
    [B, 128, 16, 16]
        ↓
    [B, 64, 32, 32]
        ↓
    [B, 3, 64, 64]

The generator progressively upsamples low dimensional latent
representations into high dimensional structured image manifolds
using fractionally strided convolutions

--------------------------------------------------- Discriminator Transformation Pipeline --------------------------------------------------- 

Input Image:
    x ∈ R^{B × 3 × 64 × 64}

Discriminator Transformation:
    [B, 3, 64, 64]
        ↓
    [B, 64, 32, 32]
        ↓
    [B, 128, 16, 16]
        ↓
    [B, 256, 8, 8]
        ↓
    [B, 512, 4, 4]
        ↓
    [B, 1, 1, 1]

The discriminator progressively compresses spatial dimensions
while increasing feature depth allowing the network to learn
high level semantic representations useful for adversarial classification

--------------------------------------------------- Design Objectives --------------------------------------------------- 

> Stable adversarial optimization
> Deep hierarchical feature synthesis
> Learnable spatial upsampling and downsampling
> Preservation of spatial feature locality
> Controlled discriminator gradient propagation
> High dimensional manifold approximation

--------------------------------------------------- Optimization Details --------------------------------------------------- 

> Optimizer:
    Adam

> Generator Optimizer:
    β1 = 0.5
    β2 = 0.999

> Discriminator Optimizer:
    β1 = 0.5
    β2 = 0.999

> Weight Initialization:
    Conv / ConvTranspose:
        Normal(0, 0.02)

    BatchNorm:
        γ ~ Normal(1, 0.02)
        β = 0

> Spectral Normalization:
    Applied to discriminator layers for stabilizing
    adversarial training and constraining the spectral norm
    of discriminator weight matrices

--------------------------------------------------- Device Configuration --------------------------------------------------- 

> Training Device:
    CUDA if available otherwise CPU

--------------------------------------------------- Paper Details --------------------------------------------------- 

Paper name:
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

Paper link:
    https://arxiv.org/pdf/1511.06434
"""

import torch.nn as nn
from config import config
from generator import Generator
from discriminator import Discriminator

class DCGAN(nn.Module):
    def __init__(self, latent_space_size: int = config["latent_space_size"],
                out_channels: int = config["n_channels"],
                feature_mapG: int = config["feature_mapG"],
                n_layersG: int = config["n_layersG"],
                g_kernel_size: int = config["g_kernel_size"],
                g_stride: int = config["g_stride"],
                g_padding: int = config["g_padding"],
                in_channels: int = config["n_channels"],
                feature_mapD: int = config["feature_mapD"],
                n_layersD: int = config["n_layersD"],
                d_kernel_size: int = config["d_kernel_size"],
                d_stride: int = config["d_stride"],
                d_padding: int = config["d_padding"]
                ):
        super(DCGAN, self).__init__()
        self.generator = Generator(
            latent_space_size = latent_space_size,
            out_channels = out_channels,
            feature_mapG = feature_mapG,
            n_layers = n_layersG,
            g_kernel_size = g_kernel_size,
            g_padding = g_padding,
            g_stride = g_stride
        )
        self.discriminator = Discriminator(
            in_channels = in_channels,
            feature_mapD = feature_mapD,
            n_layers = n_layersD,
            d_kernel_size = d_kernel_size,
            d_padding = d_padding,
            d_stride = d_stride
        )