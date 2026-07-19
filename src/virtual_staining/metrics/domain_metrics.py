"""Three-domain image metrics and ROI-grouped paired comparisons.

Validation predictions are scored before quantization, after uint8 rounding,
and after the exact competition JPEG round trip.  JPEG encoding is applied to
the prediction only: a validation target has already been decoded from its
official file and must not be compressed a second time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from virtual_staining.metrics.image_metrics import calculate_image_metrics
from virtual_staining.utils.image_io import ImageSpec, _tensor_to_uint8

DOMAIN_NAMES = ("float", "uint8", "jpg")


@dataclass(frozen=True)
class BootstrapDifference:
    """Paired ROI-bootstrap summary for candidate minus baseline."""

    metric: str
    difference: float
    ci_low: float
    ci_high: float
    confidence: float
    roi_count: int
    bootstrap_samples: int
    win_count: int
    tie_count: int
    loss_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with explicit direction."""
        return {
            "metric": self.metric,
            "direction": "candidate_minus_baseline",
            "difference": self.difference,
            "mean_difference": self.difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "roi_count": self.roi_count,
            "bootstrap_samples": self.bootstrap_samples,
            "win_count": self.win_count,
            "tie_count": self.tie_count,
            "loss_count": self.loss_count,
            "win_tie_loss": {
                "win": self.win_count,
                "tie": self.tie_count,
                "loss": self.loss_count,
            },
        }


def _as_float_image(image: Any) -> np.ndarray:
    """Convert one tensor/array in ``[0, 1]`` to HW or HWC float32."""
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
            raise ValueError("Domain metrics accept only one image at a time")
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a HW or HWC image, got shape {array.shape}")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Domain metric inputs must not contain NaN or Inf")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("Domain metric inputs must lie in [0, 1]")
    return array


def _round_to_uint8(image: np.ndarray) -> np.ndarray:
    """Use the same round-and-clamp conversion as competition image output."""
    if image.ndim == 3:
        chw = np.moveaxis(image, -1, 0)
        return _tensor_to_uint8(chw)
    return _tensor_to_uint8(image)


def _storage_array(array: np.ndarray, mode: str) -> np.ndarray:
    if mode == "RGB":
        if array.ndim == 2:
            return np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] != 3:
            raise ValueError(f"RGB JPEG requires one or three channels, got {array.shape}")
        return array
    if mode == "L":
        if array.ndim == 3:
            return np.rint(array.astype(np.float32).mean(axis=-1)).astype(np.uint8)
        return array
    raise ValueError(f"JPEG round trip supports storage mode L or RGB, got {mode!r}")


def _match_reference_shape(array: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Restore logical channels after decoding the storage representation."""
    if array.shape == reference.shape:
        return array
    if reference.ndim == 2 and array.ndim == 3:
        return np.rint(array.astype(np.float32).mean(axis=-1)).astype(np.uint8)
    if reference.ndim == 3 and array.ndim == 2 and reference.shape[-1] == 3:
        return np.repeat(array[..., None], 3, axis=-1)
    raise ValueError(
        f"Decoded JPEG shape {array.shape} cannot match reference shape {reference.shape}"
    )


def jpeg_round_trip(
    prediction: Any,
    *,
    image_spec: ImageSpec | None = None,
    reference_shape: Sequence[int] | None = None,
    quality: int = 100,
    subsampling: int = 0,
) -> np.ndarray:
    """Encode and decode one prediction with final competition JPEG settings.

    The returned array is float32 in ``[0, 1]``.  ``reference_shape`` controls
    the logical channel representation after decoding, while ``image_spec``
    controls the on-disk storage mode.
    """
    if not 1 <= int(quality) <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    if int(subsampling) not in (0, 1, 2):
        raise ValueError("JPEG subsampling must be 0, 1, or 2")
    float_image = _as_float_image(prediction)
    rounded = _round_to_uint8(float_image)
    inferred_mode = "L" if rounded.ndim == 2 else "RGB"
    mode = image_spec.mode if image_spec is not None else inferred_mode
    stored = _storage_array(rounded, mode)
    pil_image = Image.fromarray(stored, mode=mode)
    encoded = BytesIO()
    pil_image.save(
        encoded,
        format="JPEG",
        quality=int(quality),
        subsampling=int(subsampling),
        optimize=False,
    )
    encoded.seek(0)
    with Image.open(encoded) as decoded_image:
        decoded_image.load()
        decoded = np.asarray(decoded_image.convert(mode), dtype=np.uint8).copy()
    if reference_shape is not None:
        logical_reference = np.empty(tuple(int(value) for value in reference_shape), dtype=np.uint8)
        decoded = _match_reference_shape(decoded, logical_reference)
    return decoded.astype(np.float32) / 255.0


def calculate_three_domain_metrics(
    prediction: Any,
    target: Any,
    *,
    image_spec: ImageSpec | None = None,
    psnr_cap: float | None = None,
    ssim_win_size: int | None = None,
    jpeg_quality: int = 100,
    jpeg_subsampling: int = 0,
) -> dict[str, dict[str, float]]:
    """Calculate per-image float, uint8, and final-JPEG SSIM/PSNR."""
    pred = _as_float_image(prediction)
    ref = _as_float_image(target)
    if pred.shape != ref.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} != {ref.shape}")
    pred_uint8 = _round_to_uint8(pred)
    ref_uint8 = _round_to_uint8(ref)
    pred_quantized = pred_uint8.astype(np.float32) / 255.0
    ref_quantized = ref_uint8.astype(np.float32) / 255.0
    pred_jpg = jpeg_round_trip(
        pred,
        image_spec=image_spec,
        reference_shape=ref.shape,
        quality=jpeg_quality,
        subsampling=jpeg_subsampling,
    )

    def score(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
        return calculate_image_metrics(
            candidate,
            reference,
            data_range=1.0,
            psnr_cap=psnr_cap,
            ssim_win_size=ssim_win_size,
        )

    return {
        "float": score(pred, ref),
        "uint8": score(pred_quantized, ref_quantized),
        "jpg": score(pred_jpg, ref_quantized),
    }


def calculate_domain_metrics(*args: Any, **kwargs: Any) -> dict[str, dict[str, float]]:
    """Alias for :func:`calculate_three_domain_metrics`."""
    return calculate_three_domain_metrics(*args, **kwargs)


def flatten_domain_metrics(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Flatten a three-domain metric mapping for CSV/JSON records."""
    missing = [domain for domain in DOMAIN_NAMES if domain not in metrics]
    if missing:
        raise ValueError(f"Missing metric domains: {missing}")
    return {
        f"{domain}_{metric}": float(metrics[domain][metric])
        for domain in DOMAIN_NAMES
        for metric in ("ssim", "psnr")
    }


def calculate_domain_metric_record(*args: Any, **kwargs: Any) -> dict[str, float]:
    """Calculate and flatten three-domain metrics for a per-image record."""
    return flatten_domain_metrics(calculate_three_domain_metrics(*args, **kwargs))


def percentile_ranks(values: Sequence[float]) -> np.ndarray:
    """Return average-tie percentile ranks in ``[0, 1]`` (higher is better)."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Percentile ranks require a non-empty one-dimensional sequence")
    if np.isnan(array).any():
        raise ValueError("Percentile rank values must not contain NaN")
    if array.size == 1:
        return np.ones(1, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_zero_based_rank = (start + stop - 1) / 2.0
        ranks[order[start:stop]] = average_zero_based_rank / (array.size - 1)
        start = stop
    return ranks


def calculate_rank_proxy(
    ssim_values: Sequence[float],
    psnr_values: Sequence[float],
    *,
    ssim_weight: float = 0.7,
) -> np.ndarray:
    """Calculate ``0.7*SSIM rank + 0.3*PSNR rank`` for validation samples."""
    if not 0.0 <= float(ssim_weight) <= 1.0:
        raise ValueError("ssim_weight must lie in [0, 1]")
    if len(ssim_values) != len(psnr_values):
        raise ValueError("SSIM and PSNR sequences must have the same length")
    ssim_rank = percentile_ranks(ssim_values)
    psnr_rank = percentile_ranks(psnr_values)
    return float(ssim_weight) * ssim_rank + (1.0 - float(ssim_weight)) * psnr_rank


def attach_rank_proxy(
    records: Iterable[Mapping[str, Any]],
    *,
    ssim_key: str = "jpg_ssim",
    psnr_key: str = "jpg_psnr",
    output_key: str = "rank_proxy",
) -> list[dict[str, Any]]:
    """Copy records and attach validation-set percentile-rank proxies."""
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("Cannot rank an empty record sequence")
    try:
        proxies = calculate_rank_proxy(
            [float(row[ssim_key]) for row in rows],
            [float(row[psnr_key]) for row in rows],
        )
    except KeyError as error:
        raise ValueError(f"Rank-proxy record is missing key {error.args[0]!r}") from error
    for row, proxy in zip(rows, proxies, strict=True):
        row[output_key] = float(proxy)
    return rows


def _roi_metric_means(
    records: Iterable[Mapping[str, Any]], metric: str, group_key: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        if group_key not in record:
            raise ValueError(f"Metric record is missing group key {group_key!r}")
        if metric not in record:
            raise ValueError(f"Metric record is missing metric {metric!r}")
        roi_id = str(record[group_key])
        value = float(record[metric])
        if not np.isfinite(value):
            raise ValueError("ROI bootstrap metrics must be finite")
        grouped.setdefault(roi_id, []).append(value)
    if not grouped:
        raise ValueError("ROI bootstrap requires at least one record")
    return {roi_id: float(np.mean(values)) for roi_id, values in grouped.items()}


def roi_bootstrap_difference(
    candidate_records: Iterable[Mapping[str, Any]],
    baseline_records: Iterable[Mapping[str, Any]],
    *,
    metric: str = "jpg_ssim",
    group_key: str = "roi_id",
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 2026,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Bootstrap a paired candidate-minus-baseline difference by ROI.

    Patch metrics are averaged within each ROI before resampling, so large ROIs
    cannot dominate the comparison merely by containing more patches.
    """
    if int(bootstrap_samples) < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if float(tie_tolerance) < 0.0:
        raise ValueError("tie_tolerance must be nonnegative")
    candidate = _roi_metric_means(candidate_records, metric, group_key)
    baseline = _roi_metric_means(baseline_records, metric, group_key)
    candidate_rois = set(candidate)
    baseline_rois = set(baseline)
    if candidate_rois != baseline_rois:
        missing_candidate = sorted(baseline_rois - candidate_rois)
        missing_baseline = sorted(candidate_rois - baseline_rois)
        raise ValueError(
            "Candidate and baseline ROI sets differ: "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )
    roi_ids = sorted(candidate_rois)
    differences = np.asarray(
        [candidate[roi_id] - baseline[roi_id] for roi_id in roi_ids], dtype=np.float64
    )
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, differences.size, size=(int(bootstrap_samples), differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    ci_low, ci_high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    tolerance = float(tie_tolerance)
    summary = BootstrapDifference(
        metric=metric,
        difference=float(differences.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=float(confidence),
        roi_count=len(roi_ids),
        bootstrap_samples=int(bootstrap_samples),
        win_count=int(np.count_nonzero(differences > tolerance)),
        tie_count=int(np.count_nonzero(np.abs(differences) <= tolerance)),
        loss_count=int(np.count_nonzero(differences < -tolerance)),
    )
    return summary.to_dict()
