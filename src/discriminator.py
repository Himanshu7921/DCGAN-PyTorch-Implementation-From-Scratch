"""
--------------------------------------------------- Implementation Details --------------------------------------------------- 
The discriminator learns a nonlinear mapping:
    D : X → [0, 1]

where:
    X ~ p_data(x) or X ~ p_g(x)

The objective is to classify whether a given sample
belongs to the real data distribution or is generated
by the generator network through adversarial optimization

> This Implements the Discriminator of the GAN Architecture,
  each Discriminator is built from core Discriminator Blocks
  stacked on top of each other

> Design Objectives:
    - Stable adversarial optimization
    - Deep hierarchical feature extraction
    - Progressive spatial downsampling
    - Preservation of spatial information
    - Robust real vs fake classification
    - Controlled discriminator Lipschitz continuity

> Spectral Normalization:
    Spectral normalization is applied to convolutional layers
    in order to stabilize discriminator training by constraining
    the spectral norm of the weight matrices

    W_sn = W / σ(W)

    where:
        σ(W) = largest singular value of W

    This prevents uncontrolled discriminator growth and improves
    gradient stability during adversarial optimization

> Weight Initialization:
    Conv / ConvTranspose:
        Normal(0, 0.02)

    BatchNorm:
        γ ~ Normal(1, 0.02)
        β = 0

> Architectural Constraints:
    - No max pooling layers
    - No fully connected layers
    - Strided convolutions for downsampling
    - BatchNorm avoided at input layer for stability purposes
    - LeakyReLU(0.2) used for stable gradient propagation
    - Spectral normalization applied to discriminator layers

Input:
    x ∈ R^{B × 3 × 64 × 64}

Output:
    D(x) ∈ R^{B × 1 × 1 × 1}

--------------------------------------------------- Feature Hierarchy --------------------------------------------------- 
Input Image:
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

The discriminator progressively compresses the spatial dimensions
while increasing the feature depth allowing the network to learn
high level semantic representations useful for adversarial classification

--------------------------------------------------- Paper Details --------------------------------------------------- 
Paper name: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
Paper link: https://arxiv.org/pdf/1511.06434
"""

import torch.nn as nn
import torch
from torch.nn.utils import spectral_norm
from config import config
from blocks import DiscriminatorBlock


class Discriminator(nn.Module):
    """
    Discriminator: Classify the given image as fake or real
        fake_img = 0
        real_img = 1
    > Get's the input from Generator (batch_size, 3, img_size, img_size) [batch_size, 3, 64, 64]
    Discriminator(x):
        - input: image of size [batch_size, 3, 64, 64]
        - Process: conv2d --> LeakyReLU(0.2) --> [conv2d() --> BatchNorm() --> LeakyReLU(0.2)] x n --> conv2d() --> flatten --> FullyConnected --> scalar score
    """
    def __init__(self,
                in_channels: int = config["n_channels"],
                feature_mapD: int = config["feature_mapD"],
                n_layers: int = config["n_layersD"],
                d_kernel_size: int = config["d_kernel_size"],
                d_stride: int = config["d_stride"],
                d_padding: int = config["d_padding"]):
        super(Discriminator, self).__init__()

        # Not using BatchNorm in the input layer of the Discriminator
        self.discriminator = [
            spectral_norm(nn.Conv2d(in_channels = in_channels, out_channels = feature_mapD, kernel_size = d_kernel_size, stride = d_stride, padding = d_padding, bias = False)),
            nn.LeakyReLU(negative_slope = 0.2)
        ]

        prev_feature_mapD = feature_mapD

        # Stacking Multiple Layers of DiscriminatorBlock
        for _ in range(n_layers - 1):
            self.discriminator.append(
                DiscriminatorBlock(
                    in_channels = prev_feature_mapD,
                    out_channels = prev_feature_mapD * 2,
                    d_kernel_size = d_kernel_size,
                    d_stride = d_stride,
                    d_padding = d_padding
                )
            )

            prev_feature_mapD = prev_feature_mapD * 2
        
        # Apply last layers for the Discriminator
        self.discriminator.append(
            nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels = prev_feature_mapD, out_channels = 1, kernel_size = d_kernel_size, stride= 1, padding = 0, bias = False))
                # NOTE: I'm not using nn.Sigmoid() here because later in training i'll use 'nn.BCEWithLogitsLoss()'
            )
        )
        self.discriminator = nn.Sequential(*self.discriminator)

    
    def forward(self, x: torch.Tensor):
        return self.discriminator(x)
        # for layer in self.discriminator:
        #     x = layer(x)
        #     print(f"{type(layer)} | x.shape = {x.shape}")
        # return x