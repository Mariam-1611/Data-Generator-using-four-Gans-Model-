"""
Conditional Autoencoder (AE) for date generation.

A deterministic encoder-decoder with NO stochastic latent space.
Unlike the VAE, there is no KL term — the encoder maps directly to a
fixed latent vector which the decoder turns back into digit logits.

At generation time we feed a zero latent vector (the "centre" of the
learned latent space) together with the condition, letting the decoder
rely entirely on the condition to reconstruct a valid date.

Architecture
------------
Encoder : [digit_embeddings | condition] → z  (deterministic)
Decoder : [z | condition]               → digit logits  (B, 8, 10)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.tokenizer import COND_DIM, VOCAB_SIZE, DATE_DIGIT_LEN


class AEEncoder(nn.Module):
    def __init__(self, embed_dim: int = 16, hidden: int = 256, latent: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim)
        in_dim = DATE_DIGIT_LEN * embed_dim + COND_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )

    def forward(self, digits: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """digits (B,8), cond (B,COND_DIM) → z (B, latent)"""
        emb = self.embed(digits).view(digits.size(0), -1)
        return self.net(torch.cat([emb, cond], dim=-1))


class AEDecoder(nn.Module):
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
        """z (B, latent), cond (B, COND_DIM) → logits (B, 8, 10)"""
        return self.net(torch.cat([z, cond], dim=-1)).view(-1, DATE_DIGIT_LEN, VOCAB_SIZE)


class ConditionalAE(nn.Module):
    """
    Conditional Autoencoder (deterministic).

    Key difference from CVAE
    -------------------------
    • No reparameterisation, no KL loss — purely reconstruction.
    • At inference: latent z is set to zeros so the decoder must rely
      on the condition vector alone to produce a valid date.
    """

    def __init__(self, embed_dim: int = 16, hidden: int = 256, latent: int = 64) -> None:
        super().__init__()
        self.latent  = latent
        self.encoder = AEEncoder(embed_dim, hidden, latent)
        self.decoder = AEDecoder(latent, hidden)

    def forward(self, digits: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, 8, 10)."""
        z = self.encoder(digits, cond)
        return self.decoder(z, cond)

    @staticmethod
    def loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy reconstruction loss only (no KL)."""
        B = logits.size(0)
        return F.cross_entropy(
            logits.view(B * DATE_DIGIT_LEN, VOCAB_SIZE),
            targets.view(B * DATE_DIGIT_LEN),
        )

    @torch.no_grad()
    def generate(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Generate digit indices conditioned on `cond`.
        Uses zero latent vector — the decoder must leverage the condition.
        cond : (B, COND_DIM) → returns (B, 8)
        """
        z      = torch.zeros(cond.size(0), self.latent, device=cond.device)
        logits = self.decoder(z, cond)
        return logits.argmax(dim=-1)
