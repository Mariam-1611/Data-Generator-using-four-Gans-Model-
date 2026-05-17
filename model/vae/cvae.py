"""
Conditional Variational Autoencoder (CVAE) for date generation.

Architecture:
  Encoder : [date_digits_embedding + condition] → μ, log_σ²  (latent z)
  Decoder : [z + condition] → digit logits for each of the 8 positions

The date is represented as 8 digit tokens (0-9), one per character of ddmmyyyy.
The condition vector is concatenated at both encoder and decoder input.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from model.utils.tokenizer import COND_DIM, VOCAB_SIZE, DATE_DIGIT_LEN


class CVAEEncoder(nn.Module):
    def __init__(self, embed_dim: int = 16, hidden: int = 256, latent: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        in_dim = DATE_DIGIT_LEN * embed_dim + COND_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head     = nn.Linear(hidden, latent)
        self.logvar_head = nn.Linear(hidden, latent)

    def forward(
        self, digits: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        digits : (B, 8)  LongTensor
        cond   : (B, COND_DIM) float
        Returns μ, log_σ² each of shape (B, latent)
        """
        emb = self.embed(digits).view(digits.size(0), -1)   # (B, 8*embed_dim)
        x   = torch.cat([emb, cond], dim=-1)
        h   = self.net(x)
        return self.mu_head(h), self.logvar_head(h)


class CVAEDecoder(nn.Module):
    def __init__(self, latent: int = 64, hidden: int = 256) -> None:
        super().__init__()
        in_dim = latent + COND_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, DATE_DIGIT_LEN * VOCAB_SIZE),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        z    : (B, latent)
        cond : (B, COND_DIM)
        Returns logits : (B, 8, 10)
        """
        x      = torch.cat([z, cond], dim=-1)
        logits = self.net(x)
        return logits.view(-1, DATE_DIGIT_LEN, VOCAB_SIZE)


class CVAE(nn.Module):
    def __init__(
        self,
        embed_dim: int = 16,
        hidden:    int = 256,
        latent:    int = 64,
    ) -> None:
        super().__init__()
        self.latent   = latent
        self.encoder  = CVAEEncoder(embed_dim, hidden, latent)
        self.decoder  = CVAEDecoder(latent, hidden)

    # ── reparameterisation ────────────────────────────────────────────────────
    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ── forward (train) ───────────────────────────────────────────────────────
    def forward(
        self, digits: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(digits, cond)
        z          = self.reparameterise(mu, logvar)
        logits     = self.decoder(z, cond)
        return logits, mu, logvar

    # ── loss ──────────────────────────────────────────────────────────────────
    @staticmethod
    def loss(
        logits:  torch.Tensor,
        targets: torch.Tensor,
        mu:      torch.Tensor,
        logvar:  torch.Tensor,
        beta:    float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        logits  : (B, 8, 10)
        targets : (B, 8)  LongTensor
        Returns total_loss, recon_loss, kl_loss
        """
        B = logits.size(0)
        recon = F.cross_entropy(
            logits.view(B * DATE_DIGIT_LEN, VOCAB_SIZE),
            targets.view(B * DATE_DIGIT_LEN),
        )
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl, recon, kl

    # ── generation (inference) ────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Sample from the prior and decode.
        cond : (B, COND_DIM)
        Returns digit indices : (B, 8)
        """
        z      = torch.randn(cond.size(0), self.latent, device=cond.device)
        logits = self.decoder(z, cond)                      # (B, 8, 10)
        return logits.argmax(dim=-1)                        # (B, 8)
