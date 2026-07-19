"""Validation-only nonnegative and arithmetic prediction ensembles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor


def normalize_nonnegative_weights(weights: Sequence[float], count: int | None = None) -> list[float]:
    """Validate and normalize nonnegative model weights to sum to one."""
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Ensemble weights must be a non-empty one-dimensional sequence")
    if count is not None and array.size != count:
        raise ValueError(f"Expected {count} weights, received {array.size}")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Ensemble weights must be finite and nonnegative")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("At least one ensemble weight must be positive")
    return (array / total).tolist()


def validation_score_weights(scores: Sequence[float], *, temperature: float = 0.02) -> list[float]:
    """Turn validation SSIM scores into nonnegative softmax weights."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Validation scores must be a finite one-dimensional sequence")
    logits = (array - array.max()) / temperature
    weights = np.exp(logits)
    return normalize_nonnegative_weights(weights)


def fit_nonnegative_weights(predictions: Sequence[Any], target: Any) -> list[float]:
    """Fit validation-only least-squares weights and project them onto a simplex."""
    if not predictions:
        raise ValueError("At least one validation prediction is required")
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in predictions]
    reference = np.asarray(target, dtype=np.float64).reshape(-1)
    if any(array.shape != reference.shape for array in arrays):
        raise ValueError("All validation predictions must match the target shape")
    design = np.stack(arrays, axis=1)
    solution, _, _, _ = np.linalg.lstsq(design, reference, rcond=None)
    solution = np.clip(solution, 0.0, None)
    if float(solution.sum()) <= 1e-12:
        solution = np.ones(len(predictions), dtype=np.float64)
    return normalize_nonnegative_weights(solution)


def ensemble_tensors(
    predictions: Sequence[Tensor], weights: Sequence[float] | None = None
) -> Tensor:
    """Average same-shaped tensors and clamp the legal image range."""
    if not predictions:
        raise ValueError("At least one prediction tensor is required")
    shape = predictions[0].shape
    if any(prediction.shape != shape for prediction in predictions):
        raise ValueError("All ensemble prediction tensors must have identical shapes")
    normalized = (
        normalize_nonnegative_weights(weights, len(predictions))
        if weights is not None
        else [1.0 / len(predictions)] * len(predictions)
    )
    result = torch.zeros_like(predictions[0], dtype=torch.float32)
    for weight, prediction in zip(normalized, predictions, strict=True):
        result.add_(prediction.float(), alpha=float(weight))
    return result.clamp(0.0, 1.0)


def ensemble_predictions(
    members: Sequence[Mapping[str, Tensor]], weights: Sequence[float] | None = None
) -> dict[str, Tensor]:
    """Ensemble task dictionaries, requiring every member to expose the same tasks."""
    if not members:
        raise ValueError("At least one prediction member is required")
    keys = set(members[0])
    if any(set(member) != keys for member in members):
        raise KeyError("All ensemble members must contain the same task keys")
    return {
        key: ensemble_tensors([member[key] for member in members], weights)
        for key in sorted(keys)
    }
