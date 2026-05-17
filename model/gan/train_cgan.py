"""
Train the Conditional GAN (CGAN) for date generation.

Usage:
    python -m model.gan.train_cgan \
        --data  data/data.txt \
        --epochs 50 \
        --batch  512 \
        --lr_g  2e-4 \
        --lr_d  1e-4 \
        --noise_dim 64 \
        --seed  42
"""

from __future__ import annotations
import argparse
import os
import random
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.utils.dataset    import DatesDataset
from model.utils.tokenizer  import decode_date
from model.utils.validation import condition_accuracy
from model.gan.cgan         import ConditionalGAN


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def digits_to_onehot(digits: torch.Tensor) -> torch.Tensor:
    """
    digits : (B, 8) LongTensor
    Returns (B, 8, 10) float one-hot
    """
    return F.one_hot(digits, num_classes=10).float()


def evaluate_condition_acc(
    model: ConditionalGAN,
    raw_lines: list[str],
    cond_tensors: torch.Tensor,
    device: torch.device,
    n: int = 500,
) -> float:
    model.generator.eval()
    indices   = random.sample(range(len(raw_lines)), min(n, len(raw_lines)))
    lines_sub = [raw_lines[i] for i in indices]
    conds_sub = cond_tensors[indices].to(device)

    generated = model.generator.generate(conds_sub)
    dates     = [decode_date(generated[i]) for i in range(generated.size(0))]
    return condition_accuracy(lines_sub, dates)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CGAN] device={device}")

    # ── data ──────────────────────────────────────────────────────────────────
    dataset = DatesDataset(args.data)
    n_train = int(0.9 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, _ = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)

    # cache for validation metric
    all_lines        = []
    all_cond_tensors = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)
    all_cond_tensors = torch.stack([dataset.samples[i][0] for i in range(len(dataset))])

    # ── model + optimisers ────────────────────────────────────────────────────
    model    = ConditionalGAN(noise_dim=args.noise_dim).to(device)
    opt_g    = optim.Adam(model.generator.parameters(),     lr=args.lr_g, betas=(0.5, 0.999))
    opt_d    = optim.Adam(model.discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    history  = {"g_loss": [], "d_loss": [], "cond_acc": []}

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.generator.train()
        model.discriminator.train()

        g_total, d_total, n_batches = 0.0, 0.0, 0

        for cond, digits in loader:
            cond, digits = cond.to(device), digits.to(device)
            B = cond.size(0)

            real_oh = digits_to_onehot(digits)   # (B, 8, 10)

            # ── Discriminator step ─────────────────────────────────────────
            z           = torch.randn(B, args.noise_dim, device=device)
            fake_logits = model.generator(z, cond)                        # (B, 8, 10)
            # Gumbel-Softmax → differentiable discrete-like sample
            fake_soft   = F.gumbel_softmax(fake_logits, tau=1.0, hard=False, dim=-1)

            real_score  = model.discriminator(real_oh, cond)
            fake_score  = model.discriminator(fake_soft.detach(), cond)
            loss_d      = ConditionalGAN.d_loss(real_score, fake_score)

            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            # ── Generator step ─────────────────────────────────────────────
            z           = torch.randn(B, args.noise_dim, device=device)
            fake_logits = model.generator(z, cond)
            fake_soft   = F.gumbel_softmax(fake_logits, tau=1.0, hard=False, dim=-1)
            fake_score  = model.discriminator(fake_soft, cond)
            loss_g      = ConditionalGAN.g_loss(fake_score)

            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            g_total  += loss_g.item()
            d_total  += loss_d.item()
            n_batches += 1

        g_avg = g_total / n_batches
        d_avg = d_total / n_batches
        acc   = evaluate_condition_acc(model, all_lines, all_cond_tensors, device)

        history["g_loss"].append(g_avg)
        history["d_loss"].append(d_avg)
        history["cond_acc"].append(acc)

        print(
            f"[CGAN] Epoch {epoch:03d}/{args.epochs} | "
            f"G={g_avg:.4f}  D={d_avg:.4f}  cond_acc={acc:.3f}"
        )

    # ── save weights ──────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "cgan_weights.pt"))
    print(f"[CGAN] Saved weights → {args.out_dir}/cgan_weights.pt")

    with open(os.path.join(args.out_dir, "cgan_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── plots ─────────────────────────────────────────────────────────────────
    epochs_range = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, history["g_loss"], label="G Loss")
    axes[0].plot(epochs_range, history["d_loss"], label="D Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("CGAN – Generator & Discriminator Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["cond_acc"], color="green")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Condition Accuracy")
    axes[1].set_title("CGAN – Condition Satisfaction Rate")

    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "cgan_training_curves.png"), dpi=150)
    print("[CGAN] Saved training curves.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CGAN for Dates Generation")
    parser.add_argument("--data",      default="data/data.txt")
    parser.add_argument("--out_dir",   default="model/gan")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--batch",     type=int,   default=512)
    parser.add_argument("--lr_g",      type=float, default=2e-4)
    parser.add_argument("--lr_d",      type=float, default=1e-4)
    parser.add_argument("--noise_dim", type=int,   default=64)
    parser.add_argument("--seed",      type=int,   default=42)
    main(parser.parse_args())
