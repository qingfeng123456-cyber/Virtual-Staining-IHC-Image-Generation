"""Lossless-in-memory image conversion and controlled competition JPEG output."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class ImageSpec:
    """Audited storage and model-facing properties for an image stream."""

    width: int
    height: int
    storage_channels: int
    logical_channels: int
    mode: str
    dtype: str = "uint8"
    value_min: float = 0.0
    value_max: float = 255.0
    save_format: str = "JPEG"
    jpeg_quality: int = 100
    jpeg_subsampling: int = 0
    grayscale_rgb: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ImageSpec:
        return cls(**value)


def inspect_image(path: str | Path, logical_channels: int | None = None) -> ImageSpec:
    """Inspect one image without silently changing its storage mode."""

    source = Path(path)
    with Image.open(source) as image:
        image.load()
        mode = image.mode
        width, height = image.size
        channels = len(image.getbands())
        fmt = image.format or source.suffix.lstrip(".").upper()
    logical = logical_channels if logical_channels is not None else channels
    return ImageSpec(width, height, channels, logical, mode, save_format=fmt)


def read_image_array(path: str | Path) -> tuple[np.ndarray, str]:
    """Read an image as HWC uint8 RGB or HW uint8 grayscale."""

    with Image.open(Path(path)) as image:
        if image.mode in {"1", "L", "I;16", "I", "F"}:
            converted = image.convert("L")
        else:
            converted = image.convert("RGB")
        array = np.asarray(converted, dtype=np.uint8).copy()
        return array, converted.mode


def load_image_tensor(path: str | Path, logical_channels: int | None = None) -> torch.Tensor:
    """Load a CHW float32 tensor in [0, 1]."""

    array, _ = read_image_array(path)
    if array.ndim == 2:
        array = array[..., None]
    if logical_channels == 1 and array.shape[-1] == 3:
        array = np.rint(array.astype(np.float32).mean(axis=-1, keepdims=True)).astype(np.uint8)
    elif logical_channels == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().div_(255.0)
    return tensor


def _tensor_to_uint8(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    value = tensor.detach().cpu().float().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError("Only a batch of one image can be saved at once")
        value = value[0]
    if value.ndim == 3 and value.shape[0] in {1, 3}:
        value = value.transpose(1, 2, 0)
    value = np.clip(value.astype(np.float32), 0.0, 1.0)
    value = np.rint(value * 255.0).astype(np.uint8)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    return value


def save_image_tensor(
    tensor: torch.Tensor | np.ndarray,
    path: str | Path,
    spec: ImageSpec | None = None,
    *,
    quality: int = 100,
    subsampling: int = 0,
) -> Path:
    """Save one prediction once, preserving the audited storage mode."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    array = _tensor_to_uint8(tensor)
    target_mode = spec.mode if spec is not None else ("L" if array.ndim == 2 else "RGB")
    if target_mode == "RGB" and array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if target_mode == "L" and array.ndim == 3:
        array = np.rint(array.astype(np.float32).mean(axis=-1)).astype(np.uint8)
    image = Image.fromarray(array, mode="L" if array.ndim == 2 else "RGB")
    suffix = output.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output, format="JPEG", quality=quality, subsampling=subsampling, optimize=False)
    else:
        image.save(output)
    return output


def sha1_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Calculate a full SHA-1 digest without loading the file into memory."""

    digest = hashlib.sha1()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
