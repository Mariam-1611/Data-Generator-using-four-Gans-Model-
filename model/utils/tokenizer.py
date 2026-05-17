"""
Custom tokenizer for the Dates Generator assignment.
Converts condition strings and date strings into numeric tensors and back.
"""

from __future__ import annotations
import re
import torch
from typing import List, Tuple, Optional


# ── Vocabulary definitions ────────────────────────────────────────────────────

DAY_TOKENS   = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTH_TOKENS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
LEAP_TOKENS  = ["False", "True"]
DECADE_TOKENS = [str(d) for d in range(180, 221)]   # 180 … 220

DAY_TO_IDX    = {t: i for i, t in enumerate(DAY_TOKENS)}
MONTH_TO_IDX  = {t: i for i, t in enumerate(MONTH_TOKENS)}
LEAP_TO_IDX   = {t: i for i, t in enumerate(LEAP_TOKENS)}
DECADE_TO_IDX = {t: i for i, t in enumerate(DECADE_TOKENS)}

# Condition vector length = 7 + 12 + 2 + 41 = 62  (one-hot concatenation)
COND_DIM = len(DAY_TOKENS) + len(MONTH_TOKENS) + len(LEAP_TOKENS) + len(DECADE_TOKENS)

# ── Date digit vocabulary ──────────────────────────────────────────────────────
# We represent a date as a sequence of tokens: DD MM YYYY  (digit by digit)
# Digits: 0-9  → indices 0-9
# Separator '-' → index 10
# PAD         → index 11
DIGIT_PAD   = 11
DIGIT_SEP   = 10
DATE_SEQ_LEN = 10   # e.g. "0 3 - 1 2 - 1 9 6 2"  →  10 tokens (no sep stored, just digits)
# We store only the 8 digits: d1 d2 m1 m2 y1 y2 y3 y4
DATE_DIGIT_LEN = 8
VOCAB_SIZE     = 10   # digits 0-9


# ── Condition encoder ─────────────────────────────────────────────────────────

def encode_condition(line: str) -> torch.Tensor:
    """
    Parse one condition line (with or without the trailing date) and return
    a float32 one-hot condition vector of shape (COND_DIM,).

    Accepted formats:
        "[WED] [JAN] [False] [180]"
        "[WED] [JAN] [False] [180] 3-12-1962"
    """
    tokens = re.findall(r'\[([^\]]+)\]', line)
    if len(tokens) < 4:
        raise ValueError(f"Cannot parse condition from: {line!r}")

    day_str, month_str, leap_str, decade_str = tokens[:4]

    vec = torch.zeros(COND_DIM, dtype=torch.float32)
    base = 0

    # day one-hot
    vec[base + DAY_TO_IDX[day_str]] = 1.0
    base += len(DAY_TOKENS)

    # month one-hot
    vec[base + MONTH_TO_IDX[month_str]] = 1.0
    base += len(MONTH_TOKENS)

    # leap one-hot
    vec[base + LEAP_TO_IDX[leap_str]] = 1.0
    base += len(LEAP_TOKENS)

    # decade one-hot
    vec[base + DECADE_TO_IDX[decade_str]] = 1.0

    return vec


# ── Date encoder/decoder ──────────────────────────────────────────────────────

def encode_date(date_str: str) -> torch.Tensor:
    """
    Convert "dd-mm-yyyy" to a LongTensor of 8 digits.
    e.g. "03-12-1962" → tensor([0,3, 1,2, 1,9,6,2])
    """
    parts = date_str.strip().split('-')
    dd, mm, yyyy = parts
    dd   = dd.zfill(2)
    mm   = mm.zfill(2)
    yyyy = yyyy.zfill(4)
    digits = [int(c) for c in dd + mm + yyyy]
    return torch.tensor(digits, dtype=torch.long)   # shape (8,)


def decode_date(digits: torch.Tensor) -> str:
    """
    Convert an (8,) tensor of digit indices back to "d-m-yyyy" string.
    Strips leading zeros from day and month for the canonical format.
    """
    d = digits.tolist()
    dd   = int(f"{d[0]}{d[1]}")
    mm   = int(f"{d[2]}{d[3]}")
    yyyy = int(f"{d[4]}{d[5]}{d[6]}{d[7]}")
    return f"{dd}-{mm}-{yyyy}"


# ── Full-line parser ──────────────────────────────────────────────────────────

def parse_line(line: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse a full data line and return (condition_vector, date_digits).
    """
    line = line.strip()
    # Split off the trailing date
    match = re.match(
        r'(\[.+?\]\s+\[.+?\]\s+\[.+?\]\s+\[.+?\])\s+(\S+)$',
        line
    )
    if not match:
        raise ValueError(f"Cannot parse line: {line!r}")
    cond_str, date_str = match.group(1), match.group(2)
    return encode_condition(cond_str), encode_date(date_str)


def parse_condition_line(line: str) -> torch.Tensor:
    """Parse a condition-only line (no trailing date)."""
    return encode_condition(line.strip())
