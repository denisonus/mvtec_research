"""
models.py – AutoEncoder (AE) and Variational AutoEncoder (VAE) definitions
for patch-based anomaly detection on MVTec AD2.

Architecture overview:
  • ConvBackbone: 4-layer CNN encoder (stride-2 convolutions → 16× spatial
    down-sampling) mirrored by a 4-layer transpose-conv decoder.
    Each conv block uses Conv → BatchNorm → LeakyReLU for stable training
    and better gradient flow.
  • AutoEncoder: wraps the backbone – forward pass = encode → decode.
  • VariationalAutoEncoder: adds μ/log-σ² heads after the encoder, applies
    the reparameterisation trick, then decodes.

The models are intentionally lightweight so they train quickly on a single
GPU (or even CPU) for benchmarking / comparison purposes.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

# ── module-level logger ────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── shared encoder/decoder backbone ───────────────────────────────────


class ConvBackbone(nn.Module):
    """Shared convolutional backbone used by both AE and VAE.

    Encoder:  [3, H, W] → [latent_channels, H/16, W/16]
    Decoder:  [latent_channels, H/16, W/16] → [3, H, W]

    Each stage halves (encoder) or doubles (decoder) the spatial resolution
    via stride-2 (transposed) convolutions with 4×4 kernels, BatchNorm, and
    LeakyReLU.  The final decoder layer uses Sigmoid to constrain output
    to [0, 1].
    """

    def __init__(self, latent_channels: int):
        super().__init__()
        # Encoder: progressively increase channels while halving resolution
        #   3 → 32 → 64 → 128 → latent_channels   (spatial: H → H/16)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Decoder: mirror of encoder – up-sample back to original resolution
        #   latent_channels → 128 → 64 → 32 → 3   (spatial: H/16 → H)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # output pixels in [0, 1] to match normalised inputs
        )


# ── deterministic autoencoder ─────────────────────────────────────────


class AutoEncoder(nn.Module):
    """Plain autoencoder: encode → bottleneck → decode.

    The anomaly signal comes from the reconstruction error: regions the
    model cannot faithfully reconstruct are likely anomalous.
    """

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.backbone = ConvBackbone(latent_channels=latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone.encoder(x)  # compress to latent space
        recon = self.backbone.decoder(z)  # reconstruct from latent
        return recon


# ── variational autoencoder ───────────────────────────────────────────


class VariationalAutoEncoder(nn.Module):
    """VAE variant: encoder outputs μ and log-σ², then we sample z via
    the reparameterisation trick before decoding.

    The training loss adds a KL-divergence term (weighted by β) that
    regularises the latent space toward a standard normal distribution.
    """

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.backbone = ConvBackbone(latent_channels=latent_channels)

        # Two 1×1 conv heads project the encoder features to μ and log-σ²
        self.mu_head = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + ε·σ  (ε ~ N(0,1)).  Gradients flow through μ and σ
        but not through the random noise ε."""
        std = torch.exp(0.5 * logvar)  # σ = exp(log-σ² / 2)
        eps = torch.randn_like(std)  # random noise
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone.encoder(x)  # shared encoder
        mu = self.mu_head(features)  # mean of latent distribution
        logvar = self.logvar_head(features)  # log-variance of latent
        z = self.reparameterize(mu, logvar)  # stochastic sampling
        recon = self.backbone.decoder(z)  # decode sampled latent
        return recon, mu, logvar


# ── loss functions ────────────────────────────────────────────────────


def ae_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple pixel-wise MSE reconstruction loss for the autoencoder."""
    return torch.mean((recon - target) ** 2)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss = reconstruction MSE + β · KL-divergence.

    The KL term pushes the approximate posterior q(z|x) toward the prior
    N(0, I).  β < 1 produces a β-VAE that leans toward better reconstructions.

    Returns (total_loss, recon_loss, kld).
    """
    recon_loss = torch.mean((recon - target) ** 2)
    # Closed-form KL for two Gaussians  (appendix B of Kingma & Welling 2014)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kld
    return total, recon_loss, kld


# ── factory function ──────────────────────────────────────────────────


def build_model(model_type: str, latent_channels: int) -> nn.Module:
    """Instantiate an AE or VAE model and log its parameter count."""
    if model_type == "ae":
        model = AutoEncoder(latent_channels=latent_channels)
    elif model_type == "vae":
        model = VariationalAutoEncoder(latent_channels=latent_channels)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Count total trainable parameters for a quick sanity check
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Built %s model  (latent_channels=%d, trainable_params=%s)",
        model_type.upper(),
        latent_channels,
        f"{n_params:,}",
    )
    return model
