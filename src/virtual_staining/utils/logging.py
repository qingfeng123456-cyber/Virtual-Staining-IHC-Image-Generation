"""Consistent console and file logging."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: str | Path | None = None, verbose: bool = False) -> logging.Logger:
    """Configure an idempotent project logger."""

    logger = logging.getLogger("virtual_staining")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

