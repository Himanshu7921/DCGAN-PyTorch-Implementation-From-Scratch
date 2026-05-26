"""
--------------------------------------------------- Training Pipeline --------------------------------------------------- 

This module implements the complete adversarial training pipeline
for Deep Convolutional Generative Adversarial Networks (DCGAN)

The training procedure optimizes a minimax game between:
    - Generator      G(z)
    - Discriminator  D(x)

through iterative adversarial optimization

Generator Objective:
    min_G log(1 - D(G(z)))

Discriminator Objective:
    max_D log(D(x)) + log(1 - D(G(z)))

where:
    z ~ N(0, I)
    x ~ p_data(x)

The generator attempts to synthesize realistic image samples
from a latent distribution while the discriminator attempts
to classify whether a sample belongs to the true data distribution
or the generated distribution

--------------------------------------------------- Training Dynamics --------------------------------------------------- 

Discriminator Optimization:
    - Real images are classified as real
    - Generated images are classified as fake
    - Binary cross entropy objective is optimized

Generator Optimization:
    - Latent vectors are sampled from Gaussian distribution
    - Generator synthesizes fake samples
    - Generator attempts to fool discriminator
    - Adversarial gradients are propagated through D(G(z))

The optimization alternates between:
    1. Discriminator updates
    2. Generator updates

for stable adversarial convergence

--------------------------------------------------- Latent Space Modeling --------------------------------------------------- 

Latent Vector:
    z ∈ R^{B × 128 × 1 × 1}

Latent Prior:
    z ~ N(0, I)

Generator Mapping:
    G : Z → X

Discriminator Mapping:
    D : X → [0, 1]

--------------------------------------------------- Training Objectives --------------------------------------------------- 

> Stable adversarial optimization
> Deep convolutional representation learning
> High dimensional manifold approximation
> Robust discriminator gradient propagation
> Controlled generator-discriminator equilibrium
> Structured image synthesis from latent representations

--------------------------------------------------- Optimization Details --------------------------------------------------- 

Loss Function:
    BCEWithLogitsLoss

Generator Optimizer:
    Adam(
        lr = 2e-4,
        β1 = 0.5,
        β2 = 0.999
    )

Discriminator Optimizer:
    Adam(
        lr = 1e-4,
        β1 = 0.5,
        β2 = 0.999
    )

Weight Initialization:
    Conv / ConvTranspose:
        Normal(0, 0.02)

    BatchNorm:
        γ ~ Normal(1, 0.02)
        β = 0

Spectral Normalization:
    Applied to discriminator layers for stabilizing
    adversarial optimization and constraining
    discriminator weight matrices

--------------------------------------------------- Metrics Tracking --------------------------------------------------- 

Tracked Metrics:
    - Generator Loss
    - Discriminator Loss
    - D(x)
    - D(G(z))
    - Learning Rates

The training pipeline integrates:
    - tqdm based progress monitoring
    - wandb experiment tracking
    - qualitative image generation monitoring
    - epoch wise adversarial statistics logging

--------------------------------------------------- Device Configuration --------------------------------------------------- 

Training Device:
    CUDA if available otherwise CPU

Mixed precision and distributed training support
can be integrated in future extensions

--------------------------------------------------- Paper Details --------------------------------------------------- 

Paper name:
    Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

Paper link:
    https://arxiv.org/pdf/1511.06434
"""


import torch
import wandb
import argparse
import torch.nn as nn
from tqdm.auto import tqdm

from config import config
from utils import weights_init
from sample import generate_images
from model import DCGAN
from utils import get_loader, save_model


def parse_args():

    parser = argparse.ArgumentParser(
        description="DCGAN Training Pipeline"
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=config["latent_space_size"],
        help="Latent vector dimension"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=config["batch_size"],
        help="Training batch size"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=config["epochs"],
        help="Number of training epochs"
    )

    parser.add_argument(
        "--g_lr",
        type=float,
        default=config["g_lr"],
        help="Generator learning rate"
    )

    parser.add_argument(
        "--d_lr",
        type=float,
        default=config["d_lr"],
        help="Discriminator learning rate"
    )

    parser.add_argument(
    "--feature_map_g",
    type=int,
    default=config["feature_mapG"],
    help="Generator feature map size"
)

    parser.add_argument(
        "--feature_map_d",
        type=int,
        default=config["feature_mapD"],
        help="Discriminator feature map size"
)

    return parser.parse_args()

def train(generator, discriminator, device, loader, d_optimizer, g_optimizer):
    LATENT_DIM = config["latent_space_size"]

    loss_fn = nn.BCEWithLogitsLoss()

    # fake_label = 0.0, real_label = 1.0 ---> for reference

    epochs = config["epochs"]
    print(f"Training on device = [{device}]")

    pbar = tqdm(range(epochs), desc="GAN Training Path", unit="epoch", dynamic_ncols=True)

    generator.train()
    discriminator.train()

    for epoch in pbar:
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_dx = 0
        epoch_dgz = 0
        total_batches = 0
        for images, _  in loader:

            images = images.to(device)


            # Handle last incomplete batch properly
            current_batch_size = images.shape[0]

            # --------------------------------------------------------- Train Discriminator --------------------------------------------------------- 
            # Update the Discriminator network: Maximize log(D(x)) + log(1- D(G(z)))

            for _ in range(config["k"]):

                # Sample random latent vector from Standard Gaussian z ~ N(0, 1)
                z = torch.randn(current_batch_size, LATENT_DIM, 1, 1).to(device)

                # ------------------------------ Adding Noise for stabilizing GANs ------------------------------------------------

                # Pass this latent vector in Generator
                fake_img = generator(z) # I don't want to update the Generator's parameters while backpropogating through Discriminator's network

                # fake_img_score = discriminator(fake_img).view(fake_img.shape[0], -1) # Let's see what the discriminator is thinking about this fake generated image
                # real_img_score = discriminator(images).view(images.shape[0], -1)# Make sure the discriminator know's how the real image looks like

                fake_img_score = discriminator(fake_img.detach()).view(-1) # Let's see what the discriminator is thinking about this fake generated image
                real_img_score = discriminator(images).view(-1) # Make sure the discriminator know's how the real image looks like
                
                # Optimize the discriminator
                d_loss = (
                    loss_fn(fake_img_score, torch.zeros_like(fake_img_score))
                    +
                    loss_fn(real_img_score, torch.ones_like(real_img_score))
                )

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            # --------------------------------------------------------- Train Generator --------------------------------------------------------- 
            # Update the Generator's network: Maximize log(D(G(z)))

            # Sample random latent vector from Standard Gaussian z ~ N(0, 1)
            z = torch.randn(current_batch_size, LATENT_DIM, 1, 1).to(device)

            fake_img = generator(z)

            # g_fake_img_score = discriminator(fake_img).view(fake_img.shape[0], -1)
            g_fake_img_score = discriminator(fake_img).view(-1)

            g_loss = loss_fn(
                g_fake_img_score,
                torch.ones_like(g_fake_img_score)
            )

            # Optimize the generator
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --------------------------------------------------------- Metrics Collection --------------------------------------------------------- 

            with torch.no_grad():

                dx = torch.sigmoid(real_img_score).mean().item()
                dgz = torch.sigmoid(g_fake_img_score).mean().item()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_dx += dx
            epoch_dgz += dgz

            total_batches += 1

        # --------------------------------------------------------- Epoch Metrics --------------------------------------------------------- 

        avg_d_loss = epoch_d_loss / total_batches
        avg_g_loss = epoch_g_loss / total_batches
        avg_dx = epoch_dx / total_batches
        avg_dgz = epoch_dgz / total_batches

        # Update advanced progress bar metrics every single epoch with fresh evaluations
        pbar.set_postfix({
            "D_Loss": f"{avg_d_loss:.3f}",
            "G_Loss": f"{avg_g_loss:.3f}",
            "D(x)": f"{avg_dx:.3f}",
            "D(G(z))": f"{avg_dgz:.3f}"
        })

        # Log metrics to wandb every epoch
        wandb.log({
            "epoch": epoch,

            # Losses
            "Loss/Discriminator": avg_d_loss,
            "Loss/Generator": avg_g_loss,

            # GAN scores
            "Scores/D(x)": avg_dx,
            "Scores/D(G(z))": avg_dgz,

            # Learning rates
            "LR/Generator": g_optimizer.param_groups[0]["lr"],
            "LR/Discriminator": d_optimizer.param_groups[0]["lr"],
        })

        # Track metrics at the end of the epoch
        if epoch % 2 == 0:
            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"d_loss: {avg_d_loss:.4f} | "
                f"g_loss: {avg_g_loss:.4f} | "
                f"D(x): {avg_dx:.4f} | "
                f"D(G(z)): {avg_dgz:.4f}"
            )
            generator.eval()

            with torch.no_grad():
                generated_imgs = generate_images(n_epochs=epoch)

            generator.train()

            wandb.log({
                "Generated Images": wandb.Image(generated_imgs)
            })

def main():
    # get the arg parser
    args = parse_args()

    # Override config dynamically
    config["latent_space_size"] = args.latent_dim
    config["batch_size"] = args.batch_size
    config["epochs"] = args.epochs
    config["g_lr"] = args.g_lr
    config["d_lr"] = args.d_lr
    config["feature_map_g"] = args.feature_map_g
    config["feature_map_d"] = args.feature_map_d

    # Get loader and device to train on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_loader()

    # Build the Architecture
    model = DCGAN()
    generator = model.generator.to(device)
    discriminator = model.discriminator.to(device)

    # Proper Initializations of parameters
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Define Optimzers for both Generator and Discriminator
    g_optimizer = torch.optim.Adam(generator.parameters(), config["g_lr"], betas = (config["beta_1"], config["beta_2"]))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), config["d_lr"], betas = (config["beta_1"], config["beta_2"]))

    # train the model
    train(loader = loader,
        generator = generator,
        discriminator = discriminator,
        device = device,
        d_optimizer = d_optimizer,
        g_optimizer = g_optimizer)
    
    # finally save the model after training 
    save_model(generator = generator,
               discriminator = discriminator,
               g_optimizer = g_optimizer,
               d_optimizer = d_optimizer,
               epoch = config["epochs"])


if __name__ == "__main__":
    main()