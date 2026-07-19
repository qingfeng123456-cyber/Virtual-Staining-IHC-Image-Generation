"""Streaming dataset audit with bounded correlation and alignment diagnostics."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from virtual_staining.constants import CANONICAL_MARKERS, TARGET_MARKERS
from virtual_staining.utils.image_io import read_image_array

from .discovery import DiscoveryResult, discover_data_root
from .manifest import MARKER_PATH_COLUMNS, ManifestBuildResult, build_manifests, read_manifest


@dataclass(frozen=True)
class AuditResult:
    data_root: Path
    audit_json: Path
    audit_markdown: Path
    figures_dir: Path
    resolved_config: Path
    train_count: int
    figure_count: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in result.items()}


class StreamingImageStatistics:
    """Numerically stable scalar and histogram statistics for uint8 images."""

    def __init__(self) -> None:
        self.image_count = 0
        self.value_count = 0
        self.value_sum = 0.0
        self.value_square_sum = 0.0
        self.minimum = 255
        self.maximum = 0
        self.histogram = np.zeros(256, dtype=np.int64)
        self.shapes: Counter[str] = Counter()
        self.modes: Counter[str] = Counter()
        self.channel_counts: Counter[int] = Counter()
        self.channel_difference_sum = 0.0
        self.channel_difference_count = 0
        self.channel_difference_max = 0.0
        self.background_count = 0
        self.edge_sum = 0.0
        self.edge_count = 0

    def update(self, array: np.ndarray, mode: str) -> None:
        if array.dtype != np.uint8:
            raise ValueError(f"Audit expects uint8 input, got {array.dtype}")
        self.image_count += 1
        self.shapes[str(tuple(array.shape))] += 1
        self.modes[mode] += 1
        self.channel_counts[1 if array.ndim == 2 else int(array.shape[-1])] += 1
        flat = array.reshape(-1)
        self.value_count += flat.size
        self.value_sum += float(flat.sum(dtype=np.float64))
        self.value_square_sum += float(np.square(flat.astype(np.float64)).sum())
        self.minimum = min(self.minimum, int(flat.min()))
        self.maximum = max(self.maximum, int(flat.max()))
        self.histogram += np.bincount(flat, minlength=256)
        grayscale = _to_grayscale(array)
        self.background_count += int(np.count_nonzero(grayscale <= 5))
        dx = np.abs(np.diff(grayscale.astype(np.float32), axis=1))
        dy = np.abs(np.diff(grayscale.astype(np.float32), axis=0))
        self.edge_sum += float(dx.sum(dtype=np.float64) + dy.sum(dtype=np.float64))
        self.edge_count += dx.size + dy.size
        if array.ndim == 3 and array.shape[-1] >= 3:
            channels = array[..., :3].astype(np.int16)
            differences = np.maximum.reduce(
                (
                    np.abs(channels[..., 0] - channels[..., 1]),
                    np.abs(channels[..., 0] - channels[..., 2]),
                    np.abs(channels[..., 1] - channels[..., 2]),
                )
            )
            self.channel_difference_sum += float(differences.sum(dtype=np.float64))
            self.channel_difference_count += differences.size
            self.channel_difference_max = max(self.channel_difference_max, float(differences.max()))

    def _quantile(self, probability: float) -> float:
        if self.value_count == 0:
            return math.nan
        rank = probability * (self.value_count - 1)
        index = int(np.searchsorted(np.cumsum(self.histogram), rank + 1, side="left"))
        return index / 255.0

    def summary(self) -> dict[str, Any]:
        if self.value_count == 0:
            return {"count": 0}
        mean = self.value_sum / self.value_count
        variance = max(0.0, self.value_square_sum / self.value_count - mean * mean)
        channel_mean = (
            self.channel_difference_sum / self.channel_difference_count
            if self.channel_difference_count
            else 0.0
        )
        grayscale_rgb = (
            self.channel_difference_count > 0
            and self.channel_difference_max <= 2.0
            and channel_mean <= 0.25
        )
        storage_channels = max(self.channel_counts, default=1)
        return {
            "count": self.image_count,
            "shape_counts": dict(self.shapes),
            "mode_counts": dict(self.modes),
            "min": self.minimum / 255.0,
            "max": self.maximum / 255.0,
            "mean": mean / 255.0,
            "std": math.sqrt(variance) / 255.0,
            "quantiles": {
                "p01": self._quantile(0.01),
                "p05": self._quantile(0.05),
                "p50": self._quantile(0.50),
                "p95": self._quantile(0.95),
                "p99": self._quantile(0.99),
            },
            "background_fraction_lte_5": self.background_count / max(1, self.image_count * _pixels_per_image(self.shapes)),
            "mean_edge_strength": self.edge_sum / max(1, self.edge_count) / 255.0,
            "storage_channels": storage_channels,
            "channel_difference_mean": channel_mean / 255.0,
            "channel_difference_max": self.channel_difference_max / 255.0,
            "grayscale_rgb": grayscale_rgb,
            "logical_channels": 1 if grayscale_rgb or storage_channels == 1 else storage_channels,
        }


def _pixels_per_image(shapes: Counter[str]) -> float:
    total_pixels = 0
    total_images = 0
    for shape_text, count in shapes.items():
        dimensions = [int(value) for value in shape_text.strip("()").split(",") if value.strip()]
        if len(dimensions) >= 2:
            total_pixels += dimensions[0] * dimensions[1] * count
            total_images += count
    return total_pixels / max(1, total_images)


def _to_grayscale(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.shape[-1] == 1:
        return array[..., 0]
    rgb = array[..., :3].astype(np.float32)
    return np.rint(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.uint8)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray, spearman: bool = False) -> float:
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    if left_flat.size != right_flat.size or left_flat.size < 2:
        return math.nan
    if spearman:
        left_flat = _rankdata(left_flat)
        right_flat = _rankdata(right_flat)
    left_centered = left_flat - left_flat.mean()
    right_centered = right_flat - right_flat.mean()
    denominator = math.sqrt(float(np.square(left_centered).sum() * np.square(right_centered).sum()))
    return float(np.dot(left_centered, right_centered) / denominator) if denominator > 0.0 else math.nan


def _edge_map(array: np.ndarray) -> np.ndarray:
    gray = _to_grayscale(array).astype(np.float32) / 255.0
    edge = np.zeros_like(gray)
    edge[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    edge[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])
    return edge


def _overlap_for_shift(
    left: np.ndarray, right: np.ndarray, dx: int, dy: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = left.shape
    left_x0 = max(0, dx)
    left_x1 = min(width, width + dx)
    right_x0 = max(0, -dx)
    right_x1 = min(width, width - dx)
    left_y0 = max(0, dy)
    left_y1 = min(height, height + dy)
    right_y0 = max(0, -dy)
    right_y1 = min(height, height - dy)
    return (
        left[left_y0:left_y1, left_x0:left_x1],
        right[right_y0:right_y1, right_x0:right_x1],
    )


def _best_alignment(left: np.ndarray, right: np.ndarray, radius: int = 4) -> dict[str, Any]:
    left_edge = _edge_map(left)
    right_edge = _edge_map(right)
    zero = _correlation(left_edge, right_edge)
    best = (-math.inf, 0, 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            left_overlap, right_overlap = _overlap_for_shift(left_edge, right_edge, dx, dy)
            correlation = _correlation(left_overlap, right_overlap)
            comparable = correlation if math.isfinite(correlation) else -math.inf
            candidate = (comparable, -abs(dx) - abs(dy), -abs(dx), dx, dy)
            current = (best[0], -abs(best[1]) - abs(best[2]), -abs(best[1]), best[1], best[2])
            if candidate > current:
                best = (comparable, dx, dy)
    best_correlation = best[0] if math.isfinite(best[0]) else math.nan
    improvement = best_correlation - zero if math.isfinite(best_correlation) and math.isfinite(zero) else math.nan
    return {
        "dx": best[1],
        "dy": best[2],
        "zero_correlation": zero,
        "best_correlation": best_correlation,
        "improvement": improvement,
    }


def _resolve_paths(row: Mapping[str, str], root: Path) -> dict[str, Path]:
    paths = {"DAPI": root / row["dapi_path"]}
    for marker, column in MARKER_PATH_COLUMNS.items():
        value = row.get(column, "")
        if value:
            paths[marker] = root / value
    return paths


def _sample_indices(length: int, count: int, seed: int) -> list[int]:
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return sorted(indices[: min(length, count)])


def _render_pair_figure(
    row: Mapping[str, str], root: Path, destination: Path
) -> None:
    tiles: list[tuple[str, Image.Image]] = []
    for marker in CANONICAL_MARKERS:
        path = root / (row["dapi_path"] if marker == "DAPI" else row[MARKER_PATH_COLUMNS[marker]])
        with Image.open(path) as image:
            tiles.append((marker, image.convert("RGB").copy()))
    tile_width = max(image.width for _, image in tiles)
    tile_height = max(image.height for _, image in tiles)
    label_height = 28
    canvas = Image.new("RGB", (tile_width * len(tiles), tile_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (marker, image) in enumerate(tiles):
        canvas.paste(image, (index * tile_width, label_height))
        draw.text((index * tile_width + 6, 7), marker, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")


def _finite_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None}
    return {"count": int(finite.size), "mean": float(finite.mean()), "median": float(np.median(finite))}


def _alignment_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0, "classification": "not_audited", "enable_small_shift_augmentation": False}
    dx = np.asarray([int(record["dx"]) for record in records])
    dy = np.asarray([int(record["dy"]) for record in records])
    improvements = np.asarray(
        [float(record["improvement"]) for record in records if math.isfinite(float(record["improvement"]))]
    )
    median_dx = int(np.rint(np.median(dx)))
    median_dy = int(np.rint(np.median(dy)))
    median_magnitude = float(np.median(np.maximum(np.abs(dx), np.abs(dy))))
    median_improvement = float(np.median(improvements)) if improvements.size else math.nan
    enable_shift = 1.0 <= median_magnitude <= 2.0 and median_improvement >= 0.05
    if median_magnitude <= 0.5 or median_improvement < 0.02:
        classification = "approximately_aligned"
    elif median_magnitude <= 2.0:
        classification = "small_offset"
    else:
        classification = "severe_or_inconsistent_mismatch"
    return {
        "count": len(records),
        "median_dx": median_dx,
        "median_dy": median_dy,
        "median_max_axis_offset": median_magnitude,
        "median_correlation_improvement": median_improvement,
        "classification": classification,
        "enable_small_shift_augmentation": enable_shift,
    }


def _write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Data audit",
        "",
        f"- Data root: `{payload['data_root']}`",
        f"- Audited training images: {payload['train_count']}",
        f"- ROI leakage verifiable from filenames: {payload['roi_leakage_verifiable']}",
        f"- Official test status: {payload['official_test_status']}",
        "",
        "## Marker storage and value statistics",
        "",
        "| Marker | Count | Modes | Logical channels | Mean | Std | Essentially grayscale RGB |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for marker in CANONICAL_MARKERS:
        stats = payload["markers"].get(marker, {"count": 0})
        lines.append(
            "| {marker} | {count} | {modes} | {channels} | {mean} | {std} | {gray} |".format(
                marker=marker,
                count=stats.get("count", 0),
                modes=", ".join(stats.get("mode_counts", {})),
                channels=stats.get("logical_channels", "n/a"),
                mean=_display_float(stats.get("mean")),
                std=_display_float(stats.get("std")),
                gray=stats.get("grayscale_rgb", "n/a"),
            )
        )
    lines.extend(
        [
            "",
            "## Pairing and leakage",
            "",
            "The manifest was paired by normalized stem, not directory traversal order. "
            "When real ROI identifiers were absent, consecutive numeric stems were grouped into "
            "explicit surrogate blocks; this reduces adjacent-patch leakage but cannot prove true ROI separation.",
            "",
            "## Alignment",
            "",
        ]
    )
    for marker, summary in payload["alignment"].items():
        lines.append(
            f"- {marker}: {summary['classification']}; median offset "
            f"({summary.get('median_dx', 'n/a')}, {summary.get('median_dy', 'n/a')}), "
            f"shift augmentation={summary['enable_small_shift_augmentation']}."
        )
    lines.extend(
        [
            "",
            "No automatic registration was applied. Alignment diagnostics are reporting and "
            "augmentation guidance only; labels remain byte-for-byte untouched.",
            "",
            "## Test isolation",
            "",
            "Official test inputs are excluded from all statistics, normalization, model selection, "
            "and visualizations. The smoke manifest is held-out validation input and is explicitly non-official.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _display_float(value: Any) -> str:
    return f"{float(value):.5f}" if isinstance(value, (int, float)) and math.isfinite(float(value)) else "n/a"


def _write_resolved_config(payload: Mapping[str, Any], path: Path) -> None:
    logical_channels = {
        marker: int(stats.get("logical_channels", stats.get("storage_channels", 1)))
        for marker, stats in payload["markers"].items()
    }
    max_shift = max(
        (2 if summary.get("enable_small_shift_augmentation") else 0 for summary in payload["alignment"].values()),
        default=0,
    )
    root_path = Path(str(payload["data_root"])).resolve()
    project_path = path.resolve().parent.parent
    try:
        serialized_root = root_path.relative_to(project_path).as_posix()
    except ValueError:
        serialized_root = str(root_path)
    resolved = {
        "data": {
            "root": serialized_root,
            "input_channels": logical_channels.get("DAPI", 1),
            "target_channels": {marker: logical_channels.get(marker, 1) for marker in TARGET_MARKERS},
            "num_workers": 0,
        },
        "augmentation": {"max_translation": max_shift},
        "audit": {
            "roi_leakage_verifiable": payload["roi_leakage_verifiable"],
            "official_test_status": payload["official_test_status"],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        serialized = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    except ImportError:
        serialized = json.dumps(resolved, ensure_ascii=False, indent=2)
    path.write_text(serialized, encoding="utf-8")


def audit_data(
    data_root: str | Path | DiscoveryResult = "AUTO",
    *,
    manifest: str | Path | None = None,
    workspace: str | Path | None = None,
    output_path: str | Path = "artifacts/data_audit.json",
    markdown_path: str | Path = "docs/DATA_AUDIT.md",
    figures_dir: str | Path = "artifacts/figures/data_pairs",
    resolved_config_path: str | Path = "configs/resolved_local.yaml",
    correlation_sample_count: int = 64,
    alignment_sample_count: int = 32,
    figure_count: int = 16,
    seed: int = 2026,
) -> AuditResult:
    """Audit only supervised training data; official test inputs never enter statistics."""

    workspace_path = Path(workspace or Path.cwd()).expanduser().resolve()
    if isinstance(data_root, DiscoveryResult):
        root = data_root.selected_root
    elif str(data_root).casefold() == "auto":
        root = discover_data_root("AUTO", workspace=workspace_path).selected_root
    else:
        root = Path(data_root)
        if not root.is_absolute():
            root = workspace_path / root
        root = root.resolve()
    if manifest is None:
        build_result: ManifestBuildResult = build_manifests(root, workspace=workspace_path)
        manifest_path = build_result.train_manifest
        metadata_path = build_result.metadata_path
        val_rows = read_manifest(build_result.val_manifest)
    else:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = workspace_path / manifest_path
        metadata_path = manifest_path.parent / "manifest_metadata.json"
        val_path = manifest_path.parent / "val_manifest.csv"
        val_rows = read_manifest(val_path) if val_path.is_file() else []
    rows = read_manifest(manifest_path)
    supervised_rows = [row for row in [*rows, *val_rows] if row.get("split") != "test"]
    if not supervised_rows:
        raise ValueError(f"No supervised rows available for data audit: {manifest_path}")
    marker_stats = {marker: StreamingImageStatistics() for marker in CANONICAL_MARKERS}
    roi_counts: Counter[str] = Counter()
    for row in supervised_rows:
        roi_counts[str(row.get("group_id") or row.get("roi_id"))] += 1
        for marker, path in _resolve_paths(row, root).items():
            array, mode = read_image_array(path)
            marker_stats[marker].update(array, mode)

    correlation_indices = _sample_indices(len(supervised_rows), correlation_sample_count, seed)
    correlations: dict[str, dict[str, list[float]]] = {
        marker: {"pearson": [], "spearman": []} for marker in TARGET_MARKERS
    }
    for index in correlation_indices:
        row = supervised_rows[index]
        dapi_array, _ = read_image_array(root / row["dapi_path"])
        dapi_gray = _to_grayscale(dapi_array)[::4, ::4]
        for marker, column in MARKER_PATH_COLUMNS.items():
            target_array, _ = read_image_array(root / row[column])
            target_gray = _to_grayscale(target_array)[::4, ::4]
            correlations[marker]["pearson"].append(_correlation(dapi_gray, target_gray))
            correlations[marker]["spearman"].append(_correlation(dapi_gray, target_gray, spearman=True))

    alignment_indices = _sample_indices(len(supervised_rows), alignment_sample_count, seed + 1)
    alignment_records: dict[str, list[dict[str, Any]]] = {marker: [] for marker in TARGET_MARKERS}
    for index in alignment_indices:
        row = supervised_rows[index]
        dapi_array, _ = read_image_array(root / row["dapi_path"])
        for marker, column in MARKER_PATH_COLUMNS.items():
            target_array, _ = read_image_array(root / row[column])
            alignment_records[marker].append(_best_alignment(dapi_array, target_array))

    figure_destination = Path(figures_dir)
    if not figure_destination.is_absolute():
        figure_destination = workspace_path / figure_destination
    rendered = 0
    for index in _sample_indices(len(supervised_rows), figure_count, seed + 2):
        row = supervised_rows[index]
        safe_name = str(row["canonical_key"]).replace("/", "__").replace("\\", "__")
        _render_pair_figure(row, root, figure_destination / f"{safe_name}.png")
        rendered += 1

    manifest_metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        manifest_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "data_root": str(root),
        "train_count": len(supervised_rows),
        "train_manifest": str(manifest_path),
        "official_test_status": manifest_metadata.get("official_test_status", "unknown"),
        "official_test_excluded_from_statistics": True,
        "roi_id_sources": manifest_metadata.get("roi_id_source", []),
        "roi_leakage_verifiable": manifest_metadata.get("roi_leakage_verifiable", False),
        "markers": {marker: statistics.summary() for marker, statistics in marker_stats.items()},
        "correlations": {
            marker: {
                "pearson": _finite_summary(values["pearson"]),
                "spearman": _finite_summary(values["spearman"]),
            }
            for marker, values in correlations.items()
        },
        "alignment": {
            marker: _alignment_summary(records) for marker, records in alignment_records.items()
        },
        "roi_patch_counts": dict(sorted(roi_counts.items(), key=lambda item: item[0].casefold())),
        "figure_count": rendered,
        "notes": [
            "No image was modified, deleted, registered, or re-encoded during audit.",
            "Numeric 32-patch groups are surrogates when true ROI metadata is absent.",
            "Local statistics do not constitute an official competition score.",
        ],
    }
    audit_destination = Path(output_path)
    markdown_destination = Path(markdown_path)
    resolved_destination = Path(resolved_config_path)
    if not audit_destination.is_absolute():
        audit_destination = workspace_path / audit_destination
    if not markdown_destination.is_absolute():
        markdown_destination = workspace_path / markdown_destination
    if not resolved_destination.is_absolute():
        resolved_destination = workspace_path / resolved_destination
    audit_destination.parent.mkdir(parents=True, exist_ok=True)
    audit_destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(payload, markdown_destination)
    _write_resolved_config(payload, resolved_destination)
    return AuditResult(
        data_root=root,
        audit_json=audit_destination,
        audit_markdown=markdown_destination,
        figures_dir=figure_destination,
        resolved_config=resolved_destination,
        train_count=len(supervised_rows),
        figure_count=rendered,
        payload=payload,
    )


def audit_dataset(*args: Any, **kwargs: Any) -> AuditResult:
    """Compatibility alias for CLI and pipeline callers."""

    return audit_data(*args, **kwargs)
