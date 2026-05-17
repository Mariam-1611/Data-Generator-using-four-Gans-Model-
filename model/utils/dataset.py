"""
PyTorch Dataset for the Dates Generator problem.
"""

from __future__ import annotations
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from model.utils.tokenizer import parse_line, parse_condition_line


class DatesDataset(Dataset):
    """
    Loads data.txt and exposes (condition_vector, date_digits) pairs.

    condition_vector : float32 tensor of shape (COND_DIM,)
    date_digits      : int64  tensor of shape (8,)   — digits of dd mm yyyy
    """

    def __init__(self, filepath: str) -> None:
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cond, digits = parse_line(line)
                self.samples.append((cond, digits))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class ConditionOnlyDataset(Dataset):
    """
    Loads an input-only file (no dates), exposes condition vectors.
    Used during inference with predict.py.
    """

    def __init__(self, filepath: str) -> None:
        self.lines: List[str] = []
        self.conditions: List[torch.Tensor] = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.lines.append(line)
                self.conditions.append(parse_condition_line(line))

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self.lines[idx], self.conditions[idx]
