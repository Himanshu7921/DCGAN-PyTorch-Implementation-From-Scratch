"""
--------------------------------------------------- Implementation Details --------------------------------------------------- 
The generator learns a nonlinear mapping:
    G : Z → X

where:
    Z ~ N(0, I)
    X ~ p_data(x)

The objective is to approximate the true data distribution
through adversarial optimization.

> This Implements the Generator of the GAN Architecture,
  each Generator is built from core Generator Blocks (src/blocks.py/GeneratorBlock) stacked on top of each other

> Design Objectives:
    - Stable adversarial optimization
    - Deep hierarchical feature synthesis
    - Elimination of fully connected layers
    - Learnable spatial upsampling
    - Preservation of spatial feature locality

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
    - Fractionally strided convolutions for upsampling

Input:
    z ∈ R^{B × 128 × 1 × 1}

Output:
    x̂ ∈ R^{B × 3 × 64 × 64}

--------------------------------------------------- Paper Details --------------------------------------------------- 
Paper name: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
Paper link: https://arxiv.org/pdf/1511.06434
"""

import torch
import torch.nn as nn

from blocks import GeneratorBlock
from config import config

class Generator(nn.Module):
    """
    This class Implements the Generator Architecture discussed in the paper:
        UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS
        paper link: https://arxiv.org/pdf/1511.06434
    
    > Things to keep in mind (Mentioned in the Paper):
        > Historical attempts to scale up GANs using CNNs to model images have been unsuccessful.
        > However, after extensive model exploration they have identified a family of architectures that
        resulted in stable training across a range of datasets and allowed for training higher
        resolution and deeper generative models.
        
        1. Instead of using spatial pooling functions (maxpolling), I'll use strided convolutions,
           this will allow the model to learn it's own spatial upsampling (Generator) and downsampling (Discriminator)

        2. Eliminating fully connected layers and Global Max Polling because:
           - Global Avg pooling improves the model stability but hurt the convergence speed
           - Traditionals CNN Architectures: ConvLayers --> ConvLayers --> ConvLayers --> Flatten --> Fully Connected Layers --> output 
                > Why Fully Connected layers are issue here?
                    - Fully Connected Linear Layer: Huge Number of Parameters and Spatial Information Gets Destroyed
           - Solution:
                1. We progressively increase the resolution of a low dim image (latent vector z) and decreases the channel dim from (latent_vector_size) 128 --> 3 (image_channel)
                2. We then feed this image to the Discriminator
                    - Generator: z ~ N(0, 1) [64, 128, 1, 1] --> [64, 512, 4, 4] --> [64, 256, 8, 8] --> [64, 128, 16, 16] --> [64, 64, 32, 32] --> [64, 3, 64, 64]

        3. Batch Normalization is used for training deep models, but it was seen that Batch Norm hurts when it is applied to,
           the generator output layer and the discriminator input layer, therefore this was avoided by not applying
           batchnorm to the generator output layer and the discriminator input layer.

        4. The ReLU Activation function is used in the Generator and Tanh in the final layer of Generator,
           for Discriminator I used LeakyReLU(negative_slope = 0.2), as mentioned in the paper
    """
    def __init__(self, latent_space_size: int = config["latent_space_size"],
                out_channels: int = config["n_channels"],
                feature_mapG: int = config["feature_mapG"],
                n_layers: int = config["n_layersG"],
                g_kernel_size: int = config["g_kernel_size"],
                g_stride: int = config["g_stride"],
                g_padding: int = config["g_padding"]):
        super(Generator, self).__init__()
        self.generator = [GeneratorBlock(in_channels = latent_space_size,
                                        out_channels = feature_mapG,
                                        g_kernel_size = g_kernel_size,
                                        g_stride  = 1,
                                        g_padding = 0)] # 1st Layer is already inserted
        prev_feature_mapG = feature_mapG
        # prev_feature_mapG = latent_space_size

        # Stack the Layers of GeneratorBlock
        for _ in range(n_layers-1):
            self.generator.append(
                GeneratorBlock(in_channels = prev_feature_mapG,
                               out_channels =  prev_feature_mapG // 2,
                               g_kernel_size = g_kernel_size,
                               g_stride  = g_stride,
                               g_padding = g_padding)
            )
            prev_feature_mapG = prev_feature_mapG // 2
        
        # Stack last layers: using 'nn.Tanh()' as last layer
        self.generator.append(
            nn.Sequential(
            nn.ConvTranspose2d(in_channels = prev_feature_mapG,
                                        out_channels = out_channels,
                                        kernel_size = g_kernel_size,
                                        stride  = g_stride,
                                        padding = g_padding,
                                        bias = False),
            nn.Tanh()
            )
        )

        self.generator = nn.Sequential(*self.generator)
    
    
    def forward(self, x: torch.Tensor):
        return self.generator(x)
    
if __name__ == "__main__":

    print("=" * 80)
    print("DCGAN GENERATOR FORWARD PASS VALIDATION")
    print("=" * 80)

    # ------------------------------ Configuration ------------------------------
    B = config["batch_size"]
    Z_DIM = config["latent_space_size"]
    IMG_SIZE = config["img_size"]

    print(f"[Config] Batch Size          : {B}")
    print(f"[Config] Latent Dimension    : {Z_DIM}")
    print(f"[Config] Target Image Size   : {IMG_SIZE}x{IMG_SIZE}")
    print("-" * 80)

    # ------------------------------ Latent Sampling ------------------------------
    z = torch.randn(B, Z_DIM, 1, 1)

    print("[Latent Space Sampling]")
    print(f"z ~ N(0, I)")
    print(f"Latent Tensor Shape         : {z.shape}")
    print("-" * 80)

    # ------------------------------ Generator Initialization ------------------------------
    print("[Model Initialization]")
    generator = Generator()

    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)

    print(f"Total Parameters            : {total_params:,}")
    print(f"Trainable Parameters        : {trainable_params:,}")
    print("-" * 80)

    # ------------------------------ Forward Pass ------------------------------
    print("[Generator Forward Pass]")

    with torch.no_grad():
        fake_images = generator(z)

    print(f"G(z) Output Shape           : {fake_images.shape}")
    print(f"Output Tensor Range         : "
          f"[{fake_images.min():.4f}, {fake_images.max():.4f}]")

    print("-" * 80)

    # ------------------------------ Validation ------------------------------
    assert fake_images.shape == (B, 3, IMG_SIZE, IMG_SIZE), "Generator output shape mismatch"

    print("[Status] Forward pass successful")
    print("=" * 80)