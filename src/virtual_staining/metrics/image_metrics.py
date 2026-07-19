"""Per-image reference metrics backed by scikit-image.

The functions in this module deliberately operate on one image at a time.  They
perform no resizing, blurring, colour conversion, or other metric-inflating
post-processing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _as_numpy_image(image: Any) -> np.ndarray:
    """Convert a tensor/array into a finite 2-D or HWC floating point image."""
    try:
        import torch

        if isinstance(image, torch.Tensor):
            array = image.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            array = np.asarray(image)
    except ImportError:
        array = np.asarray(image)

    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("Per-image metrics do not accept a batch with more than one image")
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a 2-D grayscale or 3-D HWC image, got shape {array.shape}")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Metric inputs must not contain NaN or Inf")
    return array


def _prepare_pair(prediction: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    pred = _as_numpy_image(prediction)
    ref = _as_numpy_image(target)
    if pred.shape != ref.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} != {ref.shape}")
    return pred, ref


def _ssim_window(shape: tuple[int, ...], requested: int | None) -> int:
    spatial_min = min(shape[0], shape[1])
    if spatial_min < 3:
        raise ValueError("SSIM requires both spatial dimensions to be at least 3 pixels")
    maximum = min(7, spatial_min)
    if maximum % 2 == 0:
        maximum -= 1
    if requested is None:
        return maximum
    window = min(int(requested), maximum)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        raise ValueError("SSIM window size must be an odd integer of at least 3")
    return window


def calculate_ssim(
    prediction: Any,
    target: Any,
    *,
    data_range: float = 1.0,
    win_size: int | None = None,
) -> float:
    """Calculate scikit-image SSIM for one grayscale or RGB image."""
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    pred, ref = _prepare_pair(prediction, target)
    channel_axis = -1 if pred.ndim == 3 else None
    value = structural_similarity(
        ref,
        pred,
        data_range=float(data_range),
        channel_axis=channel_axis,
        win_size=_ssim_window(pred.shape, win_size),
    )
    return float(value)


def calculate_psnr(
    prediction: Any,
    target: Any,
    *,
    data_range: float = 1.0,
    cap: float | None = None,
) -> float:
    """Calculate per-image PSNR, optionally capping positive infinity."""
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    pred, ref = _prepare_pair(prediction, target)
    # Exact matches legitimately produce positive infinity.  Preserve that
    # value without leaking a divide-by-zero warning into every validation run.
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(peak_signal_noise_ratio(ref, pred, data_range=float(data_range)))
    if cap is not None:
        if cap <= 0:
            raise ValueError("PSNR cap must be positive")
        value = min(value, float(cap))
    return value


def calculate_image_metrics(
    prediction: Any,
    target: Any,
    *,
    data_range: float = 1.0,
    psnr_cap: float | None = None,
    ssim_win_size: int | None = None,
) -> dict[str, float]:
    """Return reference SSIM and PSNR for a single image."""
    return {
        "ssim": calculate_ssim(
            prediction, target, data_range=data_range, win_size=ssim_win_size
        ),
        "psnr": calculate_psnr(prediction, target, data_range=data_range, cap=psnr_cap),
    }
