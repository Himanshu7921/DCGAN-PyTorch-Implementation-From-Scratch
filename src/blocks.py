"""
This contains Generator and Discriminator's core building blocks

Each Generator Block contains:
    - Convolution Transpose 2d
    - Batch Norm 2d
    - ReLU Actvation

"""
import torch
import torch.nn as nn
from config import config
from torch.nn.utils import spectral_norm

class GeneratorBlock(nn.Module):
    """
    This represents one block of the Generator Architecture
    also in the final layer of the Generator we will use `nn.Tanh()` and not `nn.ReLU()` because we need the final tensor to be in image range
    """
    def __init__(self, in_channels: int, out_channels: int, g_kernel_size: int = config["g_kernel_size"], g_stride: int = config["g_stride"], g_padding: int = config["g_padding"]):
        super(GeneratorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = g_kernel_size, stride = g_stride, padding = g_padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor):
        return self.block(x)


class DiscriminatorBlock(nn.Module):
    """
    This represents one block of the Discriminator Architecture
    """
    def __init__(self, in_channels: int,
                out_channels: int,
                d_kernel_size: int = config["d_kernel_size"],
                d_stride: int = config["d_stride"],
                d_padding: int = config["d_padding"]):
        super(DiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = d_kernel_size, stride = d_stride, padding = d_padding, bias = False)),
            # Adding SpectralNorm for GAN Stability
            nn.LeakyReLU(negative_slope = 0.2, inplace=True)            
        )
    
    def forward(self, x: torch.Tensor):
        return self.block(x)