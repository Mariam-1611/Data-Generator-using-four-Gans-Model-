"""
Inference entry-point for the Dates Generator assignment.

Models available:
    ae   – Conditional Autoencoder (AE)
    vae  – Variational Autoencoder (VAE)
    cvae – Conditional VAE (CVAE)
    cgan – Conditional GAN (CGAN)

Usage:
    python model/predict.py -i data/example_input.txt \
                            -o outputs/predictions.txt \
                            --model cvae
"""

from __future__ import annotations
import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.utils.dataset   import ConditionOnlyDataset
from model.utils.tokenizer import decode_date


# ── model loaders ─────────────────────────────────────────────────────────────

def load_ae(device):
    from model.ae.ae import ConditionalAE
    m = ConditionalAE(embed_dim=16, hidden=256, latent=64).to(device)
    m.load_state_dict(torch.load("model/ae/ae_weights.pt", map_location=device))
    m.eval(); return m

def load_vae(device):
    from model.vae.vae import VAE
    m = VAE(embed_dim=16, hidden=256, latent=64).to(device)
    m.load_state_dict(torch.load("model/vae/vae_weights.pt", map_location=device))
    m.eval(); return m

def load_cvae(device):
    from model.vae.cvae import CVAE
    m = CVAE(embed_dim=16, hidden=256, latent=64).to(device)
    m.load_state_dict(torch.load("model/vae/cvae_weights.pt", map_location=device))
    m.eval(); return m

def load_cgan(device):
    from model.gan.cgan import ConditionalGAN
    m = ConditionalGAN(noise_dim=64, hidden=256).to(device)
    m.load_state_dict(torch.load("model/gan/cgan_weights.pt", map_location=device))
    m.eval(); return m


LOADERS = {
    "ae":   load_ae,
    "vae":  load_vae,
    "cvae": load_cvae,
    "cgan": load_cgan,
}


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model_name: str, input_path: str, output_path: str, batch_size: int = 256) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] model={model_name}  device={device}")

    if model_name not in LOADERS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(LOADERS)}")
    model = LOADERS[model_name](device)

    ds = ConditionOnlyDataset(input_path)
    print(f"[predict] {len(ds)} conditions to process.")

    results: list[str] = []
    for start in range(0, len(ds), batch_size):
        end         = min(start + batch_size, len(ds))
        batch_lines = [ds[i][0] for i in range(start, end)]
        batch_conds = torch.stack([ds[i][1] for i in range(start, end)]).to(device)

        if model_name == "cgan":
            digits = model.generator.generate(batch_conds)
        else:
            digits = model.generate(batch_conds)

        for i, line in enumerate(batch_lines):
            results.append(f"{line} {decode_date(digits[i])}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(results) + "\n")
    print(f"[predict] Wrote {len(results)} predictions → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dates Generator – Inference")
    parser.add_argument("-i", "--input",  required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--model", default="cvae", choices=list(LOADERS))
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    run_inference(args.model, args.input, args.output, args.batch_size)
