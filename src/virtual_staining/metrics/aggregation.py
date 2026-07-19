"""Aggregation of per-image SSIM/PSNR records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _finite_mean(values: list[float], *, preserve_positive_inf: bool = True) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    if np.isposinf(array).any() and preserve_positive_inf:
        return float("inf")
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_median(values: list[float], *, preserve_positive_inf: bool = True) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    if np.isposinf(array).all() and preserve_positive_inf:
        return float("inf")
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("inf")


def _worst_fraction(values: list[float], fraction: float = 0.10) -> float:
    if not 0 < fraction <= 1:
        raise ValueError("Worst-sample fraction must be in (0, 1]")
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return float("nan")
    count = max(1, int(np.ceil(array.size * fraction)))
    return float(np.mean(np.sort(array)[:count]))


def _proxy(ssim: float, psnr: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("psnr_norm_max must be greater than psnr_norm_min")
    normalized = float(np.clip((psnr - low) / (high - low), 0.0, 1.0))
    return float(0.7 * ssim + 0.3 * normalized)


def aggregate_metric_records(
    records: Iterable[Mapping[str, Any]],
    *,
    psnr_norm_min: float = 15.0,
    psnr_norm_max: float = 40.0,
    group_key: str = "roi_id",
    worst_fraction: float = 0.10,
) -> dict[str, Any]:
    """Aggregate already-computed per-image metric records.

    Infinite PSNR is retained in the reported image-level mean when an exact
    match exists.  The local proxy safely clips it to its configured upper
    bound.  Group metrics are means within each group followed by a macro mean.
    """
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("Cannot aggregate an empty metric record sequence")
    ssims = [float(row["ssim"]) for row in rows]
    psnrs = [float(row["psnr"]) for row in rows]
    group_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"ssim": [], "psnr": []}
    )
    for index, row in enumerate(rows):
        group = str(row.get(group_key) or f"sample-{index}")
        group_values[group]["ssim"].append(float(row["ssim"]))
        group_values[group]["psnr"].append(float(row["psnr"]))
    group_ssims = [_finite_mean(values["ssim"]) for values in group_values.values()]
    group_psnrs = [_finite_mean(values["psnr"]) for values in group_values.values()]
    mean_ssim = _finite_mean(ssims)
    mean_psnr = _finite_mean(psnrs)
    result: dict[str, Any] = {
        "count": len(rows),
        "group_count": len(group_values),
        "mean_ssim": mean_ssim,
        "median_ssim": _finite_median(ssims),
        "mean_psnr": mean_psnr,
        "median_psnr": _finite_median(psnrs),
        "roi_ssim": _finite_mean(group_ssims),
        "roi_psnr": _finite_mean(group_psnrs),
        "worst_10pct_ssim": _worst_fraction(ssims, worst_fraction),
        "worst_10pct_psnr": _worst_fraction(psnrs, worst_fraction),
        "local_proxy_score": _proxy(
            mean_ssim, mean_psnr, float(psnr_norm_min), float(psnr_norm_max)
        ),
        "psnr_infinite_count": int(np.isposinf(np.asarray(psnrs)).sum()),
    }
    return result


def aggregate_target_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    target_key: str = "target",
    psnr_norm_min: float = 15.0,
    psnr_norm_max: float = 40.0,
    group_key: str = "roi_id",
) -> dict[str, Any]:
    """Aggregate records per target and add macro averages."""
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("Cannot aggregate an empty metric record sequence")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(target_key, "output"))].append(row)
    per_target = {
        target: aggregate_metric_records(
            target_rows,
            psnr_norm_min=psnr_norm_min,
            psnr_norm_max=psnr_norm_max,
            group_key=group_key,
        )
        for target, target_rows in sorted(grouped.items())
    }
    macro_fields = (
        "mean_ssim",
        "median_ssim",
        "mean_psnr",
        "median_psnr",
        "roi_ssim",
        "roi_psnr",
        "worst_10pct_ssim",
        "worst_10pct_psnr",
        "local_proxy_score",
    )
    macro = {
        field: _finite_mean([float(metrics[field]) for metrics in per_target.values()])
        for field in macro_fields
    }
    return {"per_target": per_target, "macro": macro, "count": len(rows)}


def aggregate_stratified_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    strata_keys: Iterable[str] = (
        "organ",
        "activity_bin",
        "context_availability",
        "image_mean_bin",
        "border_class",
    ),
    psnr_norm_min: float = 15.0,
    psnr_norm_max: float = 40.0,
) -> dict[str, dict[str, Any]]:
    """Aggregate metrics for every declared validation stratum.

    Missing labels are represented explicitly as ``"unknown"``.  This keeps
    local-only and non-coordinate baselines comparable without pretending that
    border or context status was observed.
    """

    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("Cannot stratify an empty metric record sequence")
    result: dict[str, dict[str, Any]] = {}
    for key in strata_keys:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            label = str(row.get(key, "unknown") or "unknown")
            grouped[label].append(row)
        result[str(key)] = {
            label: aggregate_target_metrics(
                values,
                psnr_norm_min=psnr_norm_min,
                psnr_norm_max=psnr_norm_max,
            )
            for label, values in sorted(grouped.items())
        }
    return result


def weighted_organ_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    psnr_norm_min: float = 15.0,
    psnr_norm_max: float = 40.0,
) -> dict[str, Any] | None:
    """Return weighted organ metrics only when every configured organ exists."""

    rows = [dict(record) for record in records]
    normalized_weights = {
        str(organ).casefold(): float(weight) for organ, weight in weights.items()
    }
    if not normalized_weights:
        return None
    if any(weight < 0.0 for weight in normalized_weights.values()):
        raise ValueError("Organ weights must be nonnegative")
    total = sum(normalized_weights.values())
    if total <= 0.0:
        raise ValueError("Organ weights must contain a positive value")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("organ", "unknown")).casefold()].append(row)
    if not set(normalized_weights).issubset(grouped):
        return None
    normalized_weights = {
        organ: weight / total for organ, weight in normalized_weights.items()
    }
    per_organ = {
        organ: aggregate_target_metrics(
            grouped[organ],
            psnr_norm_min=psnr_norm_min,
            psnr_norm_max=psnr_norm_max,
        )["macro"]
        for organ in normalized_weights
    }
    fields = ("mean_ssim", "mean_psnr", "roi_ssim", "roi_psnr", "local_proxy_score")
    weighted = {
        field: float(
            sum(normalized_weights[organ] * float(per_organ[organ][field]) for organ in per_organ)
        )
        for field in fields
    }
    return {
        "weights": normalized_weights,
        "per_organ": per_organ,
        "weighted": weighted,
    }
