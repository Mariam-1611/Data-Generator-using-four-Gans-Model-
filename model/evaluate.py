"""
Evaluate all four trained models on the held-out validation split.

Metrics:
  • Condition Satisfaction Rate — fraction of dates satisfying ALL 4 conditions.
  • Per-condition accuracy     — individual satisfaction for day/month/leap/decade.

Usage:
    python -m model.evaluate --data data/data.txt --n_eval 2000 --seed 42
"""

from __future__ import annotations
import argparse
import os
import re
import random
import json
import sys

import torch
from torch.utils.data import random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.utils.dataset    import DatesDataset
from model.utils.tokenizer  import decode_date
from model.utils.validation import validate, condition_accuracy, parse_date_safe, is_leap_year, DAY_MAP, MONTH_MAP


# ── model loaders ─────────────────────────────────────────────────────────────

def load_model(name: str, device: torch.device):
    CKPTS = {
        "ae":   "model/ae/ae_weights.pt",
        "vae":  "model/vae/vae_weights.pt",
        "cvae": "model/vae/cvae_weights.pt",
        "cgan": "model/gan/cgan_weights.pt",
    }
    ckpt = CKPTS[name]
    if not os.path.exists(ckpt):
        return None

    if name == "ae":
        from model.ae.ae import ConditionalAE
        m = ConditionalAE().to(device)
    elif name == "vae":
        from model.vae.vae import VAE
        m = VAE().to(device)
    elif name == "cvae":
        from model.vae.cvae import CVAE
        m = CVAE().to(device)
    elif name == "cgan":
        from model.gan.cgan import ConditionalGAN
        m = ConditionalGAN().to(device)

    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_model(name, model, raw_lines, cond_tensors, device, n) -> dict:
    indices   = random.sample(range(len(raw_lines)), min(n, len(raw_lines)))
    lines_sub = [raw_lines[i] for i in indices]
    conds_sub = cond_tensors[indices].to(device)

    if name == "cgan":
        digits = model.generator.generate(conds_sub)
    else:
        digits = model.generate(conds_sub)

    dates = [decode_date(digits[i]) for i in range(digits.size(0))]
    acc   = condition_accuracy(lines_sub, dates)

    per = {"day": 0, "month": 0, "leap": 0, "decade": 0}
    for line, date in zip(lines_sub, dates):
        tokens = re.findall(r'\[([^\]]+)\]', line)
        if len(tokens) < 4:
            continue
        d = parse_date_safe(date)
        if d is None:
            continue
        if d.weekday() == DAY_MAP.get(tokens[0], -1):   per["day"]    += 1
        if d.month == MONTH_MAP.get(tokens[1], -1):      per["month"]  += 1
        if is_leap_year(d.year) == (tokens[2] == "True"): per["leap"]  += 1
        ds = int(tokens[3]) * 10
        if ds <= d.year <= ds + 9:                        per["decade"] += 1

    total = len(lines_sub)
    per   = {k: v / total for k, v in per.items()}

    print(f"\n── {name.upper()} ──")
    print(f"  All-4 accuracy : {acc:.3f}")
    for k, v in per.items():
        print(f"    {k:8s}     : {v:.3f}")
    print("  Sample outputs (first 5):")
    for line, date in zip(lines_sub[:5], dates[:5]):
        ok = "✓" if validate(line, date) else "✗"
        print(f"    {ok}  {line}  →  {date}")

    return {"overall": acc, **per}


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DatesDataset(args.data)
    n_train = int(0.9 * len(dataset))
    n_val   = len(dataset) - n_train
    _, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    all_lines = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)

    val_lines     = [all_lines[i] for i in val_ds.indices]
    val_cond_tens = torch.stack([dataset.samples[i][0] for i in val_ds.indices])

    results = {}
    for name in ["ae", "vae", "cvae", "cgan"]:
        model = load_model(name, device)
        if model is None:
            print(f"[eval] Skipping {name} — weights not found.")
            continue
        results[name] = eval_model(name, model, val_lines, val_cond_tens, device, args.n_eval)

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n\n══════════════════════════════════ SUMMARY ══════════════════════════════════")
    header = f"{'Model':<10} {'All-4':>8} {'Day':>8} {'Month':>8} {'Leap':>8} {'Decade':>8}"
    print(header)
    print("─" * len(header))
    for name, r in results.items():
        print(
            f"{name:<10} {r['overall']:>8.3f} "
            f"{r.get('day',0):>8.3f} {r.get('month',0):>8.3f} "
            f"{r.get('leap',0):>8.3f} {r.get('decade',0):>8.3f}"
        )

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved → outputs/eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/data.txt")
    parser.add_argument("--n_eval", type=int, default=2000)
    parser.add_argument("--seed",   type=int, default=42)
    main(parser.parse_args())
