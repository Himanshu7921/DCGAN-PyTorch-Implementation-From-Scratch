"""
--------------------------------------------------- Utility Functions --------------------------------------------------- 

This module contains utility functions used throughout
the DCGAN training and inference pipeline

Utilities included:
    - Weight initialization
    - Dataset loading
    - Checkpoint saving
    - Training checkpoint restoration
    - Generator loading for inference

The module supports:
    - Stable adversarial training
    - Checkpoint based experiment recovery
    - Latent space sampling
    - Reproducible training workflows

Paper:
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    https://arxiv.org/pdf/1511.06434
"""

import wandb
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import config
from model import DCGAN

def weights_init(m):
    # I've refered this 'weights_init' from Official implementation of GAN
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(
            m.weight.data,
            0.0,
            0.02
        )

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(
            m.weight.data,
            1.0,
            0.02
        )
        nn.init.constant_(
            m.bias.data,
            0
        )

def get_loader():
    """
    Loads the images to train on
    """
    transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    dataset = datasets.ImageFolder(
        root = config["DATA_PATH"],
        transform=transform
    )

    loader = DataLoader(dataset, batch_size = config["batch_size"], shuffle=True)
    return loader

# Saving Model Script
def save_model(generator, discriminator, g_optimizer, d_optimizer, epoch):

    # Save GAN Checkpoint
    checkpoint = {
        # Generator
        "generator_state_dict": generator.state_dict(),
        "generator_optimizer_state_dict": g_optimizer.state_dict(),
        "generator_lr": g_optimizer.param_groups[0]["lr"],

        # Discriminator
        "discriminator_state_dict": discriminator.state_dict(),
        "discriminator_optimizer_state_dict": d_optimizer.state_dict(),
        "discriminator_lr": d_optimizer.param_groups[0]["lr"],

        # -------------------------------------- Generator's setting -------------------------------------- 
        "g_kernel_size": config["g_kernel_size"],
        "g_stride": config["g_stride"],
        "g_padding": config["g_padding"],
        "feature_mapG": config["feature_mapG"],
        "g_lr": config["g_lr"],
        "n_layersG": config["n_layersG"],

        # -------------------------------------- Discriminator's setting -------------------------------------- 
        "d_kernel_size": config["d_kernel_size"],
        "d_stride": config["d_stride"],
        "d_padding": config["d_padding"],
        "feature_mapD": config["feature_mapD"],
        "d_lr": config["d_lr"],
        "n_layersD": config["n_layersD"],

        # -------------------------------------- Model's setting -------------------------------------- 
        "k": config["k"],
        "batch_size": config["batch_size"],
        "img_size": config["img_size"],
        "n_channels": config["n_channels"],
        "latent_space_size": config["latent_space_size"],
        "epochs": config["epochs"],
        "beta_1": config["beta_1"],
        "beta_2": config["beta_2"],
        "device": config["device"],

        # Training Metadata
        "epoch": epoch
    }

    # Save Checkpoint
    torch.save(
        checkpoint,
        config["SAVE_PATH"]
    )

    print("Checkpoint Saved Successfully!")

    # Add the model's checkpoint to weight and bias
    artifact = wandb.Artifact(
        "dcgan-checkpoint",
        type="model"
    )
    artifact.add_file(config["SAVE_PATH"])
    wandb.log_artifact(artifact)

# Script for loading the model for resuming training
def load_training_checkpoint(device, generator, discriminator, g_optimizer, d_optimizer):
    """
    Load Generator and Discriminator for Training from the saved Checkpoint
    """
    # Load Checkpoint
    checkpoint = torch.load(
        config["SAVE_PATH"],
        map_location=device
    )

    # -------------------------------------- Load Generator -------------------------------------- 
    generator.load_state_dict(checkpoint["generator_state_dict"])
    g_optimizer.load_state_dict(checkpoint["generator_optimizer_state_dict"])

    # -------------------------------------- Load Discriminator -------------------------------------- 
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    d_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])

    # -------------------------------------- Restore Config Values -------------------------------------- 
    config["g_kernel_size"] = checkpoint["g_kernel_size"]
    config["g_stride"] = checkpoint["g_stride"]
    config["g_padding"] = checkpoint["g_padding"]
    config["feature_mapG"] = checkpoint["feature_mapG"]
    config["g_lr"] = checkpoint["g_lr"]
    config["n_layersG"] = checkpoint["n_layersG"]

    config["d_kernel_size"] = checkpoint["d_kernel_size"]
    config["d_stride"] = checkpoint["d_stride"]
    config["d_padding"] = checkpoint["d_padding"]
    config["feature_mapD"] = checkpoint["feature_mapD"]
    config["d_lr"] = checkpoint["d_lr"]
    config["n_layersD"] = checkpoint["n_layersD"]

    config["k"] = checkpoint["k"]
    config["batch_size"] = checkpoint["batch_size"]
    config["img_size"] = checkpoint["img_size"]
    config["n_channels"] = checkpoint["n_channels"]
    config["latent_space_size"] = checkpoint["latent_space_size"]
    config["epochs"] = checkpoint["epochs"]
    config["beta_1"] = checkpoint["beta_1"]
    config["beta_2"] = checkpoint["beta_2"]
    config["device"] = checkpoint["device"]


    start_epoch = checkpoint["epoch"] + 1
    
    generator.train()
    discriminator.train()
    print(f"Checkpoint Loaded Successfully from Epoch [{start_epoch}]")

# Script for loading the model for sampling purpose
def load_generator(device):
    """Load Generator for sampling"""
    model = DCGAN()
    generator = model.generator.to(device)
    checkpoint = torch.load(
        config["SAVE_PATH"],
        map_location=device
    )
    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )
    generator.eval()
    return generator

def initialize_wandb(generator: DCGAN, discriminator: DCGAN, g_optimizer: torch.optim.Adam, d_optimizer: torch.optim.Adam, loader):
    generator_arch = repr(generator)
    discriminator_arch = repr(discriminator)

    wandb.init(
        project="DCGAN-celeba",
        name=f"run_lr{g_optimizer.param_groups[0]['lr']}",
        config={
            "g_lr": g_optimizer.param_groups[0]['lr'],
            "d_lr": d_optimizer.param_groups[0]['lr'],
            "epochs": config["epochs"],
            "batch_size": loader.batch_size,
            "latent_dims": config["latent_space_size"],
            "K": config["k"],

            "generator_parameters":
                sum(p.numel() for p in generator.parameters()),

            "discriminator_parameters":
                sum(p.numel() for p in discriminator.parameters()),

            "generator_architecture": generator_arch,
            "discriminator_architecture": discriminator_arch
        }
    )

    with open("generator_architecture.txt", "w") as f:
        f.write(repr(generator))

    with open("discriminator_architecture.txt", "w") as f:
        f.write(repr(discriminator))

    wandb.save("generator_architecture.txt", policy="now")
    wandb.save("discriminator_architecture.txt", policy="now")