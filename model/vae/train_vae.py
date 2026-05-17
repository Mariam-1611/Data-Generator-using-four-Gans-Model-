"""
Train the standard VAE (conditional decoder only) for date generation.

Usage:
    python -m model.vae.train_vae \
        --data   data/data.txt \
        --epochs 30 \
        --batch  512 \
        --lr     1e-3 \
        --latent 64 \
        --beta   1.0 \
        --seed   42
"""

from __future__ import annotations
import argparse
import os
import random
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.utils.dataset    import DatesDataset
from model.utils.tokenizer  import decode_date
from model.utils.validation import condition_accuracy
from model.vae.vae          import VAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_condition_acc(model, raw_lines, cond_tensors, device, n=500) -> float:
    model.eval()
    indices   = random.sample(range(len(raw_lines)), min(n, len(raw_lines)))
    lines_sub = [raw_lines[i] for i in indices]
    conds_sub = cond_tensors[indices].to(device)
    generated = model.generate(conds_sub)
    dates     = [decode_date(generated[i]) for i in range(generated.size(0))]
    return condition_accuracy(lines_sub, dates)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VAE] device={device}")

    dataset = DatesDataset(args.data)
    n_train = int(0.9 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    all_lines = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)
    all_cond_tensors = torch.stack([dataset.samples[i][0] for i in range(len(dataset))])

    model     = VAE(embed_dim=16, hidden=256, latent=args.latent).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "cond_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for cond, digits in train_loader:
            cond, digits = cond.to(device), digits.to(device)
            logits, mu, logvar = model(digits, cond)
            loss, _, _ = VAE.loss(logits, digits, mu, logvar, beta=args.beta)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * cond.size(0)
        train_loss = total_loss / n_train

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for cond, digits in val_loader:
                cond, digits = cond.to(device), digits.to(device)
                logits, mu, logvar = model(digits, cond)
                l, _, _ = VAE.loss(logits, digits, mu, logvar, beta=args.beta)
                val_sum += l.item() * cond.size(0)
        val_loss = val_sum / n_val

        acc = evaluate_condition_acc(model, all_lines, all_cond_tensors, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["cond_acc"].append(acc)
        print(f"[VAE] Epoch {epoch:03d}/{args.epochs} | train={train_loss:.4f}  val={val_loss:.4f}  cond_acc={acc:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "vae_weights.pt"))
    with open(os.path.join(args.out_dir, "vae_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    epochs_r = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_r, history["train_loss"], label="Train")
    axes[0].plot(epochs_r, history["val_loss"],   label="Val")
    axes[0].set_title("VAE – ELBO Loss"); axes[0].legend()
    axes[1].plot(epochs_r, history["cond_acc"], color="green")
    axes[1].set_title("VAE – Condition Accuracy")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "vae_training_curves.png"), dpi=150)
    print(f"[VAE] Done. Weights → {args.out_dir}/vae_weights.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/data.txt")
    parser.add_argument("--out_dir", default="model/vae")
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--batch",   type=int,   default=512)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--latent",  type=int,   default=64)
    parser.add_argument("--beta",    type=float, default=1.0)
    parser.add_argument("--seed",    type=int,   default=42)
    main(parser.parse_args())
