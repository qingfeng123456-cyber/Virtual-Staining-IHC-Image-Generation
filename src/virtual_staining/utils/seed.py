"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True, benchmark: bool | None = None) -> torch.Generator:
    """Seed Python, NumPy, and PyTorch and return a DataLoader generator.

    When ``benchmark`` is None (default), cuDNN benchmark is set to the inverse
    of ``deterministic`` (legacy behaviour): deterministic mode disables the
    auto-tuner, non-deterministic mode enables it. Pass an explicit bool to
    override, e.g. ``benchmark=True`` to enable the auto-tuner while keeping
    deterministic algorithms on (the auto-tuner's choice may itself be
    non-deterministic, so this is only safe when input sizes are fixed, as they
    are for this project's fixed 256x256 patches on the RTX 5090).
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if benchmark is None:
        benchmark = not deterministic
    torch.backends.cudnn.benchmark = bool(benchmark)
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

