"""Canonical names and stable project constants."""

from __future__ import annotations

import re
from pathlib import Path

CANONICAL_MARKERS: tuple[str, ...] = ("DAPI", "HLA-DR", "CD45RO", "Vimentin", "CD68")
TARGET_MARKERS: tuple[str, ...] = ("HLA-DR", "CD45RO", "Vimentin", "CD68")
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
DEFAULT_SEED = 2026
DEFAULT_IMAGE_SIZE = 256

_MARKER_LOOKUP = {
    "dapi": "DAPI",
    "hladr": "HLA-DR",
    "cd45ro": "CD45RO",
    "vimentin": "Vimentin",
    "cd68": "CD68",
}


def normalize_marker(value: str | Path) -> str:
    """Return a canonical marker name despite case or punctuation differences."""

    key = re.sub(r"[^a-z0-9]", "", Path(str(value)).name.lower())
    if key not in _MARKER_LOOKUP:
        raise ValueError(f"Unsupported marker name: {value!s}")
    return _MARKER_LOOKUP[key]


def marker_slug(marker: str) -> str:
    """Return the stable snake-case identifier used in manifests."""

    canonical = normalize_marker(marker)
    return canonical.lower().replace("-", "_")


def normalize_stem(filename: str | Path) -> str:
    """Normalize a source or prediction filename to its original sample stem."""

    stem = Path(filename).stem
    while stem.lower().endswith("_fake"):
        stem = stem[:-5]
    return stem

