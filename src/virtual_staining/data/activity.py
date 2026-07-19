"""Training-target-only activity features for optional stratified sampling."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from virtual_staining.constants import normalize_marker
from virtual_staining.utils.image_io import load_image_tensor

from .dataset import marker_path_column

_TRAIN_SPLITS = {"train", "official_train", "final_train"}


def _resolve_image_path(value: str, data_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def _features(image: np.ndarray) -> dict[str, float]:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 3:
        values = values.mean(axis=0)
    horizontal = (
        np.abs(values[:, 1:] - values[:, :-1])
        if values.shape[1] > 1
        else np.zeros((1, 1))
    )
    vertical = (
        np.abs(values[1:, :] - values[:-1, :])
        if values.shape[0] > 1
        else np.zeros((1, 1))
    )
    mean = float(values.mean())
    std = float(values.std())
    p95 = float(np.quantile(values, 0.95))
    gradient_energy = float((horizontal.mean() + vertical.mean()) / 2.0)
    foreground_fraction = float(np.mean(values > mean))
    activity = float(std + gradient_energy + 0.25 * foreground_fraction + 0.25 * p95)
    return {
        "target_mean": mean,
        "target_std": std,
        "target_p95": p95,
        "gradient_energy": gradient_energy,
        "foreground_fraction": foreground_fraction,
        "activity": activity,
    }


def compute_training_activity(
    rows: Sequence[Mapping[str, Any]],
    data_root: str | Path,
    *,
    target: str,
    activity_key: str = "target_activity",
    output_csv: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach deterministic activity values using training labels only."""

    canonical_target = normalize_marker(target)
    root = Path(data_root).expanduser().resolve()
    if not rows:
        raise ValueError("Activity features require at least one training row")
    enriched: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        row = dict(source)
        split = str(row.get("split", "")).strip().casefold()
        if split not in _TRAIN_SPLITS:
            raise ValueError(
                "Activity features accept training rows only; "
                f"row {index} has split={split!r}"
            )
        column = marker_path_column(canonical_target)
        if not str(row.get(column, "")).strip():
            raise ValueError(f"Activity row {index} lacks target path column {column!r}")
        path = _resolve_image_path(str(row[column]), root)
        tensor = load_image_tensor(path, logical_channels=1)
        features = _features(tensor.detach().cpu().numpy())
        row[activity_key] = features["activity"]
        enriched.append(row)
        feature_rows.append(
            {
                "source_index": index,
                "canonical_key": str(row.get("canonical_key", index)),
                "split": split,
                "target": canonical_target,
                "target_path": str(path),
                **features,
            }
        )
    destination: Path | None = None
    if output_csv is not None:
        destination = Path(output_csv).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
            writer.writeheader()
            writer.writerows(feature_rows)
        temporary.replace(destination)
    report = {
        "count": len(enriched),
        "target": canonical_target,
        "activity_key": activity_key,
        "uses_target_labels": True,
        "allowed_splits": sorted(_TRAIN_SPLITS),
        "output_csv": str(destination) if destination is not None else None,
    }
    return enriched, report
