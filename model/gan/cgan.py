"""
Conditional GAN (CGAN) for date generation.

Generator  : noise z + condition → digit logits (8 positions × 10 classes)
Discriminator : digit embeddings + condition → real/fake scalar

We use the Gumbel-Softmax trick to pass discrete tokens differentiably
through the discriminator during generator training.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from model.utils.tokenizer import COND_DIM, VOCAB_SIZE, DATE_DIGIT_LEN


class Generator(nn.Module):
    """Maps (noise, condition) → (B, 8, 10) logits."""

    def __init__(self, noise_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim + COND_DIM, hidden),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, DATE_DIGIT_LEN * VOCAB_SIZE),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x      = torch.cat([z, cond], dim=-1)
        logits = self.net(x)
        return logits.view(-1, DATE_DIGIT_LEN, VOCAB_SIZE)   # (B, 8, 10)

    @torch.no_grad()
    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        """Return argmax digit indices (B, 8)."""
        z      = torch.randn(cond.size(0), self.noise_dim, device=cond.device)
        logits = self.forward(z, cond)
        return logits.argmax(dim=-1)


class Discriminator(nn.Module):
    """
    Maps (digit embeddings, condition) → real/fake logit.
    Receives either hard one-hot vectors (real) or Gumbel-Softmax soft
    samples (fake), both of shape (B, 8, 10).
    """

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        in_dim = DATE_DIGIT_LEN * VOCAB_SIZE + COND_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, digit_repr: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        digit_repr : (B, 8, 10) float — one-hot or soft sample
        cond       : (B, COND_DIM)
        Returns    : (B, 1) logit
        """
        x = digit_repr.view(digit_repr.size(0), -1)   # (B, 80)
        x = torch.cat([x, cond], dim=-1)
        return self.net(x)


class ConditionalGAN(nn.Module):
    """Container that holds G and D together for convenience."""

    def __init__(self, noise_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.generator     = Generator(noise_dim, hidden)
        self.discriminator = Discriminator(hidden)

    # ── loss functions ────────────────────────────────────────────────────────

    @staticmethod
    def d_loss(
        real_logit: torch.Tensor,
        fake_logit: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy GAN discriminator loss."""
        real_loss = F.binary_cross_entropy_with_logits(
            real_logit, torch.ones_like(real_logit)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_logit, torch.zeros_like(fake_logit)
        )
        return (real_loss + fake_loss) * 0.5

    @staticmethod
    def g_loss(fake_logit: torch.Tensor) -> torch.Tensor:
        """Generator loss: fool the discriminator."""
        return F.binary_cross_entropy_with_logits(
            fake_logit, torch.ones_like(fake_logit)
        )
