"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> torch.Generator:
    """Seed Python, NumPy, and PyTorch and return a DataLoader generator."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed one DataLoader worker from PyTorch's initial seed."""

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

