# DSAI 490 – Assignment 2: Dates Generator

## Models

| # | Model | Full Name | Key Idea |
|---|-------|-----------|----------|
| 1 | **AE** | Conditional Autoencoder | Deterministic encoder → fixed z; decoder uses z + condition. At inference z = **zeros** |
| 2 | **VAE** | Variational Autoencoder | Encoder takes digits only (no condition); decoder takes z + condition |
| 3 | **CVAE** | Conditional VAE | Encoder takes digits + condition; decoder takes z + condition — fully conditional |
| 4 | **CGAN** | Conditional GAN | Generator maps noise + condition → digits; Discriminator judges real/fake |

### AE vs VAE vs CVAE — the key difference

```
AE   : Encoder(digits)           → z       → Decoder(z + cond)   [deterministic]
VAE  : Encoder(digits)           → μ, σ    → Decoder(z + cond)   [encoder ignores cond]
CVAE : Encoder(digits + cond)    → μ, σ    → Decoder(z + cond)   [fully conditional]
```

---

## Project Structure

```
dates_generator/
├── data/
│   ├── data.txt                    # full training dataset
│   └── example_input.txt           # inference example (conditions only)
├── model/
│   ├── utils/
│   │   ├── tokenizer.py            # condition & date encoding/decoding
│   │   ├── dataset.py              # PyTorch Dataset classes
│   │   └── validation.py           # condition-satisfaction checker
│   ├── ae/
│   │   ├── ae.py                   # AE architecture
│   │   └── train_ae.py             # AE training script
│   ├── vae/
│   │   ├── vae.py                  # VAE architecture
│   │   ├── train_vae.py            # VAE training script
│   │   ├── cvae.py                 # CVAE architecture
│   │   └── train_cvae.py           # CVAE training script
│   ├── gan/
│   │   ├── cgan.py                 # CGAN architecture
│   │   └── train_cgan.py           # CGAN training script
│   ├── evaluate.py                 # Multi-model evaluation
│   └── predict.py                  # Inference entry-point
├── outputs/                        # Predictions, plots, eval results
├── environment.yml
└── README.md
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate dates_generator
```

---

## Training

Run all commands from the `dates_generator/` root directory.

```bash
# AE
python -m model.ae.train_ae --epochs 30

# VAE
python -m model.vae.train_vae --epochs 30

# CVAE
python -m model.vae.train_cvae --epochs 30

# CGAN
python -m model.gan.train_cgan --epochs 50
```

Each script saves:
- Model weights (`.pt` file)
- Training curves plot (`.png`)
- Training history (`.json`)

---

## Inference

```bash
python model/predict.py \
    -i data/example_input.txt \
    -o outputs/predictions.txt \
    --model cvae        # ae | vae | cvae | cgan
```

Output format mirrors `data.txt`:
```
[DAY] [MONTH] [LEAP] [DECADE] dd-m-yyyy
```

---

## Evaluation

```bash
python -m model.evaluate --n_eval 2000
```

Prints a summary table:

```
Model       All-4      Day    Month     Leap   Decade
──────────────────────────────────────────────────────
ae          0.XXX    0.XXX   0.XXX    0.XXX   0.XXX
vae         0.XXX    ...
cvae        0.XXX    ...
cgan        0.XXX    ...
```

Results saved to `outputs/eval_results.json`.

---

## Key Design Notes

**Tokenizer** — Condition encoded as a 62-dim one-hot vector (7 days + 12 months + 2 leap + 41 decades). Date encoded as 8 digit tokens `[d1 d2 m1 m2 y1 y2 y3 y4]` ∈ {0…9}.

**Primary metric** — Condition Satisfaction Rate: fraction of generated dates satisfying all four conditions simultaneously. Computed live during training.

**CGAN** — uses Gumbel-Softmax to pass discrete digit outputs differentiably to the discriminator during generator training.
