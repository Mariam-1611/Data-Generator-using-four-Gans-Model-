"""
Standard Variational Autoencoder (VAE) for date generation.

Difference from CVAE
---------------------
• The condition vector is NOT fed to the encoder.
• The decoder receives ONLY the latent z + condition — forcing the latent
  space to be a pure Gaussian prior that the decoder must combine with the
  condition to produce the date.
• This is the "semi-supervised" flavour often called a "conditional decoder
  VAE" or simply VAE with a conditional decoder.

Architecture
------------
Encoder : digit_embeddings  → μ, log_σ²   (no condition input)
Decoder : [z | condition]   → digit logits (B, 8, 10)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from model.utils.tokenizer import COND_DIM, VOCAB_SIZE, DATE_DIGIT_LEN


class VAEEncoder(nn.Module):
    """Encodes only the date digits — no condition."""

    def __init__(self, embed_dim: int = 16, hidden: int = 256, latent: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        in_dim = DATE_DIGIT_LEN * embed_dim          # no condition here
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head     = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)

    def forward(self, digits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """digits (B,8) → μ (B,latent), logvar (B,latent)"""
        emb = self.embed(digits).view(digits.size(0), -1)
        h   = self.net(emb)
        return self.mu_head(h), self.logvar_head(h)


class VAEDecoder(nn.Module):
    """Decodes z + condition into digit logits."""

    def __init__(self, latent: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent + COND_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, DATE_DIGIT_LEN * VOCAB_SIZE),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, cond], dim=-1)).view(-1, DATE_DIGIT_LEN, VOCAB_SIZE)


class VAE(nn.Module):
    """
    Standard VAE with a conditional decoder.

    Key differences vs CVAE
    ------------------------
    CVAE : condition → encoder AND decoder
    VAE  : condition → decoder ONLY  (encoder is unconditional)
    """

    def __init__(
        self,
        embed_dim: int = 16,
        hidden:    int = 256,
        latent:    int = 64,
    ) -> None:
        super().__init__()
        self.latent  = latent
        self.encoder = VAEEncoder(embed_dim, hidden, latent)
        self.decoder = VAEDecoder(latent, hidden)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(
        self, digits: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(digits)            # encoder ignores cond
        z          = self.reparameterise(mu, logvar)
        logits     = self.decoder(z, cond)           # decoder uses cond
        return logits, mu, logvar

    @staticmethod
    def loss(
        logits:  torch.Tensor,
        targets: torch.Tensor,
        mu:      torch.Tensor,
        logvar:  torch.Tensor,
        beta:    float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = logits.size(0)
        recon = F.cross_entropy(
            logits.view(B * DATE_DIGIT_LEN, VOCAB_SIZE),
            targets.view(B * DATE_DIGIT_LEN),
        )
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl, recon, kl

    @torch.no_grad()
    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(0,I), decode with condition. Returns (B,8)."""
        z      = torch.randn(cond.size(0), self.latent, device=cond.device)
        logits = self.decoder(z, cond)
        return logits.argmax(dim=-1)
