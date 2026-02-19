from __future__ import annotations

import torch
from torch import nn


class ConvBackbone(nn.Module):
    def __init__(self, latent_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )


class AutoEncoder(nn.Module):
    def __init__(self, latent_channels: int = 256):
        super().__init__()
        self.backbone = ConvBackbone(latent_channels=latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone.encoder(x)
        recon = self.backbone.decoder(z)
        return recon


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_channels: int = 256):
        super().__init__()
        self.backbone = ConvBackbone(latent_channels=latent_channels)
        self.mu_head = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone.encoder(x)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        z = self.reparameterize(mu, logvar)
        recon = self.backbone.decoder(z)
        return recon, mu, logvar


def ae_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((recon - target) ** 2)


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = torch.mean((recon - target) ** 2)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kld
    return total, recon_loss, kld


def build_model(model_type: str, latent_channels: int) -> nn.Module:
    if model_type == "ae":
        return AutoEncoder(latent_channels=latent_channels)
    if model_type == "vae":
        return VariationalAutoEncoder(latent_channels=latent_channels)
    raise ValueError(f"Unsupported model_type: {model_type}")