"""Verified ROI coordinate parsing, indexing, and boundary-continuity auditing.

Numeric stems and image-content matches are deliberately excluded from coordinate
parsing.  They may be useful audit evidence, but they are not authoritative grid
coordinates and therefore cannot enable a promotable context experiment.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch

from virtual_staining.utils.image_io import load_image_tensor

_COORDINATE_PATTERN = re.compile(
    r"^(?P<roi>ROI\d+)_(?P<row>\d+)_(?P<col>\d+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class PatchCoordinate:
    """Authoritative two-dimensional position parsed from a competition filename."""

    roi_id: str
    row: int
    col: int

    def __post_init__(self) -> None:
        if not re.fullmatch(r"ROI\d+", self.roi_id, flags=re.IGNORECASE):
            raise ValueError(f"Invalid ROI identifier: {self.roi_id!r}")
        if self.row < 0 or self.col < 0:
            raise ValueError("Patch row and column must be nonnegative")
        object.__setattr__(self, "roi_id", self.roi_id.upper())


@dataclass(frozen=True)
class CoordinateParseResult:
    """Explicit parse status; failure never falls back to an inferred coordinate."""

    value: str
    stem: str
    coordinate: PatchCoordinate | None
    source: str
    reason: str | None

    @property
    def parsed(self) -> bool:
        return self.coordinate is not None


@dataclass(frozen=True)
class ROIGridSummary:
    """Coordinate coverage for one organ/split/ROI grid."""

    organ: str
    split: str
    roi_id: str
    patch_count: int
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    missing_coordinates: tuple[tuple[int, int], ...]
    duplicate_coordinates: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BoundaryContinuityAudit:
    """Aggregate evidence for coordinate orientation and edge continuity."""

    horizontal_pair_count: int
    vertical_pair_count: int
    expected_horizontal_mae: float | None
    expected_vertical_mae: float | None
    orientation_scores: dict[str, float]
    best_orientation: str | None
    expected_orientation_margin: float | None
    direction_verified: bool
    continuity_verified: bool
    border_width: int


@dataclass(frozen=True)
class ROIGridAudit:
    """Serializable ROI-grid audit and strict context-gating decision."""

    total_rows: int
    parsed_rows: int
    parse_fraction: float
    unparsed_examples: tuple[dict[str, str], ...]
    grids: tuple[ROIGridSummary, ...]
    duplicate_coordinates: tuple[dict[str, Any], ...]
    train_val_shared_rois: tuple[str, ...]
    cross_split_adjacent_pairs: tuple[dict[str, str], ...]
    boundary: BoundaryContinuityAudit
    filename_grid_verified: bool
    context_enabled: bool
    context_gate_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _filename_stem(value: str | Path) -> str:
    text = str(value).strip()
    if not text:
        return ""
    name = Path(text).name
    suffix = Path(name).suffix
    return name[: -len(suffix)] if suffix else name


def parse_patch_coordinate_status(value: str | Path) -> CoordinateParseResult:
    """Parse ``ROI<digits>_<row>_<col>`` and return an explicit status.

    The expression is anchored.  Numeric stems, prefixes, suffixes, signed values,
    and partial matches are rejected rather than guessed.
    """

    text = str(value)
    stem = _filename_stem(value)
    match = _COORDINATE_PATTERN.fullmatch(stem)
    if match is None:
        reason = "empty_filename" if not stem else "filename_does_not_match_roi_row_col"
        return CoordinateParseResult(
            value=text,
            stem=stem,
            coordinate=None,
            source="unparsed",
            reason=reason,
        )
    coordinate = PatchCoordinate(
        roi_id=match.group("roi"),
        row=int(match.group("row")),
        col=int(match.group("col")),
    )
    return CoordinateParseResult(
        value=text,
        stem=stem,
        coordinate=coordinate,
        source="filename_regex",
        reason=None,
    )


def parse_patch_coordinate(value: str | Path) -> PatchCoordinate | None:
    """Return a verified filename coordinate, or ``None`` without inference."""

    return parse_patch_coordinate_status(value).coordinate


def coordinate_value_from_row(row: Mapping[str, Any]) -> str:
    """Choose the filename-bearing manifest field used for strict parsing."""

    for field in ("stem", "patch_id", "dapi_path"):
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def _row_label(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("canonical_key") or row.get("patch_id") or f"row_{index}")


class ROIIndex:
    """Index rows by organ, split, verified ROI, row, and column.

    Duplicate coordinates are rejected because silently selecting one patch would
    make the context source ambiguous.  Unparseable rows remain addressable as
    centers, but have no coordinate and can never acquire inferred neighbors.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        require_verified: bool = False,
    ) -> None:
        self.rows = tuple(dict(row) for row in rows)
        self.coordinates: tuple[PatchCoordinate | None, ...] = tuple(
            parse_patch_coordinate(coordinate_value_from_row(row)) for row in self.rows
        )
        by_key: dict[tuple[str, str, str, int, int], int] = {}
        duplicates: list[str] = []
        for index, (row, coordinate) in enumerate(zip(self.rows, self.coordinates, strict=True)):
            if coordinate is None:
                continue
            key = self.key(row, coordinate)
            if key in by_key:
                duplicates.append(
                    f"{key}: {_row_label(self.rows[by_key[key]], by_key[key])}, "
                    f"{_row_label(row, index)}"
                )
            else:
                by_key[key] = index
        if duplicates:
            raise ValueError("Duplicate ROI grid coordinate(s): " + "; ".join(duplicates))
        self._by_key = by_key
        self.verified = bool(self.rows) and all(value is not None for value in self.coordinates)
        if require_verified and not self.verified:
            missing = [
                _row_label(row, index)
                for index, (row, coordinate) in enumerate(
                    zip(self.rows, self.coordinates, strict=True)
                )
                if coordinate is None
            ]
            raise ValueError(
                "Context requires verified ROI_row_col filenames; unverified rows: "
                + ", ".join(missing[:10])
            )

    @staticmethod
    def key(
        row: Mapping[str, Any], coordinate: PatchCoordinate
    ) -> tuple[str, str, str, int, int]:
        return (
            str(row.get("organ", "unknown")).casefold(),
            str(row.get("split", "")).casefold(),
            coordinate.roi_id.casefold(),
            coordinate.row,
            coordinate.col,
        )

    def coordinate_for(self, index: int) -> PatchCoordinate | None:
        return self.coordinates[index]

    def neighbor_index(self, center_index: int, row_offset: int, col_offset: int) -> int | None:
        coordinate = self.coordinates[center_index]
        if coordinate is None:
            return None
        center = self.rows[center_index]
        neighbor_row = coordinate.row + row_offset
        neighbor_col = coordinate.col + col_offset
        if neighbor_row < 0 or neighbor_col < 0:
            return None
        neighbor = PatchCoordinate(
            roi_id=coordinate.roi_id,
            row=neighbor_row,
            col=neighbor_col,
        )
        return self._by_key.get(self.key(center, neighbor))

    def neighborhood_indices(
        self,
        center_index: int,
        *,
        grid_size: int = 3,
    ) -> tuple[tuple[tuple[int, int], int | None], ...]:
        if grid_size < 1 or grid_size % 2 == 0:
            raise ValueError(f"grid_size must be a positive odd integer, got {grid_size}")
        radius = grid_size // 2
        return tuple(
            ((row_offset, col_offset), self.neighbor_index(center_index, row_offset, col_offset))
            for row_offset in range(-radius, radius + 1)
            for col_offset in range(-radius, radius + 1)
        )


def _resolve_dapi_path(row: Mapping[str, Any], data_root: Path) -> Path:
    value = str(row.get("dapi_path", "")).strip()
    if not value:
        raise ValueError(f"Manifest row has no DAPI path: {row!r}")
    path = Path(value)
    return (path if path.is_absolute() else data_root / path).resolve()


def _strip_mae(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.shape != right.shape or left.numel() == 0:
        return None
    return float(torch.mean(torch.abs(left.float() - right.float())).item())


def _edge_errors(
    first: torch.Tensor,
    second: torch.Tensor,
    border_width: int,
) -> dict[str, float | None]:
    width = min(border_width, first.shape[-1], second.shape[-1])
    height = min(border_width, first.shape[-2], second.shape[-2])
    return {
        "right_left": _strip_mae(first[..., -width:], second[..., :width]),
        "left_right": _strip_mae(first[..., :width], second[..., -width:]),
        "bottom_top": _strip_mae(first[..., -height:, :], second[..., :height, :]),
        "top_bottom": _strip_mae(first[..., :height, :], second[..., -height:, :]),
    }


def _median(values: Sequence[float]) -> float | None:
    return float(median(values)) if values else None


def _orientation_score(
    horizontal: Sequence[dict[str, float | None]],
    vertical: Sequence[dict[str, float | None]],
    horizontal_edge: str,
    vertical_edge: str,
) -> float | None:
    values = [
        float(value)
        for errors in horizontal
        if (value := errors[horizontal_edge]) is not None
    ]
    values.extend(
        float(value)
        for errors in vertical
        if (value := errors[vertical_edge]) is not None
    )
    return _median(values)


def _boundary_audit(
    indexed_groups: Mapping[
        tuple[str, str, str], Mapping[tuple[int, int], tuple[int, Mapping[str, Any]]]
    ],
    data_root: Path,
    *,
    border_width: int,
    min_pairs_per_axis: int,
    min_direction_margin: float,
    max_boundary_mae: float,
) -> BoundaryContinuityAudit:
    cache: dict[int, torch.Tensor] = {}

    def image(index: int, row: Mapping[str, Any]) -> torch.Tensor:
        if index not in cache:
            cache[index] = load_image_tensor(_resolve_dapi_path(row, data_root))
        return cache[index]

    horizontal: list[dict[str, float | None]] = []
    vertical: list[dict[str, float | None]] = []
    for entries in indexed_groups.values():
        for (row_number, col_number), (index, row) in entries.items():
            right = entries.get((row_number, col_number + 1))
            if right is not None:
                right_index, right_row = right
                horizontal.append(
                    _edge_errors(image(index, row), image(right_index, right_row), border_width)
                )
            below = entries.get((row_number + 1, col_number))
            if below is not None:
                below_index, below_row = below
                vertical.append(
                    _edge_errors(image(index, row), image(below_index, below_row), border_width)
                )

    candidates = {
        "row_down_col_right": ("right_left", "bottom_top"),
        "row_down_col_left": ("left_right", "bottom_top"),
        "row_up_col_right": ("right_left", "top_bottom"),
        "row_up_col_left": ("left_right", "top_bottom"),
        "transposed_row_down_col_right": ("bottom_top", "right_left"),
        "transposed_row_down_col_left": ("top_bottom", "right_left"),
        "transposed_row_up_col_right": ("bottom_top", "left_right"),
        "transposed_row_up_col_left": ("top_bottom", "left_right"),
    }
    scores = {
        name: score
        for name, (horizontal_edge, vertical_edge) in candidates.items()
        if (score := _orientation_score(horizontal, vertical, horizontal_edge, vertical_edge))
        is not None
    }
    ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]))
    best_orientation = ordered[0][0] if ordered else None
    expected_score = scores.get("row_down_col_right")
    alternatives = [score for name, score in ordered if name != "row_down_col_right"]
    runner_up = min(alternatives) if alternatives else None
    margin = None
    if expected_score is not None and runner_up is not None:
        margin = (runner_up - expected_score) / max(runner_up, 1e-12)
    enough_pairs = (
        len(horizontal) >= min_pairs_per_axis and len(vertical) >= min_pairs_per_axis
    )
    direction_verified = bool(
        enough_pairs
        and best_orientation == "row_down_col_right"
        and margin is not None
        and margin >= min_direction_margin
    )
    continuity_verified = bool(
        direction_verified and expected_score is not None and expected_score <= max_boundary_mae
    )
    return BoundaryContinuityAudit(
        horizontal_pair_count=len(horizontal),
        vertical_pair_count=len(vertical),
        expected_horizontal_mae=_median(
            [float(value) for errors in horizontal if (value := errors["right_left"]) is not None]
        ),
        expected_vertical_mae=_median(
            [float(value) for errors in vertical if (value := errors["bottom_top"]) is not None]
        ),
        orientation_scores=scores,
        best_orientation=best_orientation,
        expected_orientation_margin=margin,
        direction_verified=direction_verified,
        continuity_verified=continuity_verified,
        border_width=border_width,
    )


def audit_roi_grid(
    rows: Sequence[Mapping[str, Any]],
    data_root: str | Path,
    *,
    border_width: int = 8,
    min_pairs_per_axis: int = 1,
    min_direction_margin: float = 0.05,
    max_boundary_mae: float = 0.15,
) -> ROIGridAudit:
    """Audit filename coordinates, leakage, holes, and DAPI boundary orientation."""

    if border_width < 1:
        raise ValueError("border_width must be positive")
    if min_pairs_per_axis < 1:
        raise ValueError("min_pairs_per_axis must be positive")
    root = Path(data_root).expanduser().resolve()
    indexed: dict[
        tuple[str, str, str], dict[tuple[int, int], tuple[int, Mapping[str, Any]]]
    ] = defaultdict(dict)
    coordinate_rows: list[tuple[int, Mapping[str, Any], PatchCoordinate]] = []
    unparsed: list[dict[str, str]] = []
    duplicates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        result = parse_patch_coordinate_status(coordinate_value_from_row(row))
        coordinate = result.coordinate
        if coordinate is None:
            unparsed.append(
                {
                    "canonical_key": _row_label(row, index),
                    "value": result.value,
                    "reason": str(result.reason),
                }
            )
            continue
        coordinate_rows.append((index, row, coordinate))
        group_key = (
            str(row.get("organ", "unknown")).casefold(),
            str(row.get("split", "")).casefold(),
            coordinate.roi_id.casefold(),
        )
        location = (coordinate.row, coordinate.col)
        if location in indexed[group_key]:
            previous_index, previous_row = indexed[group_key][location]
            duplicates.append(
                {
                    "organ": group_key[0],
                    "split": group_key[1],
                    "roi_id": coordinate.roi_id,
                    "row": coordinate.row,
                    "col": coordinate.col,
                    "first": _row_label(previous_row, previous_index),
                    "duplicate": _row_label(row, index),
                }
            )
        else:
            indexed[group_key][location] = (index, row)

    summaries: list[ROIGridSummary] = []
    for (organ, split, roi_id), entries in sorted(indexed.items()):
        row_values = [location[0] for location in entries]
        col_values = [location[1] for location in entries]
        row_min, row_max = min(row_values), max(row_values)
        col_min, col_max = min(col_values), max(col_values)
        missing = tuple(
            (row_number, col_number)
            for row_number in range(row_min, row_max + 1)
            for col_number in range(col_min, col_max + 1)
            if (row_number, col_number) not in entries
        )
        group_duplicates = tuple(
            (int(item["row"]), int(item["col"]))
            for item in duplicates
            if item["organ"] == organ and item["split"] == split and item["roi_id"].casefold() == roi_id
        )
        summaries.append(
            ROIGridSummary(
                organ=organ,
                split=split,
                roi_id=roi_id.upper(),
                patch_count=len(entries),
                row_min=row_min,
                row_max=row_max,
                col_min=col_min,
                col_max=col_max,
                missing_coordinates=missing,
                duplicate_coordinates=group_duplicates,
            )
        )

    splits_by_roi: dict[tuple[str, str], set[str]] = defaultdict(set)
    combined_coordinates: dict[
        tuple[str, str], dict[tuple[int, int], list[tuple[str, str]]]
    ] = defaultdict(lambda: defaultdict(list))
    for index, row, coordinate in coordinate_rows:
        organ = str(row.get("organ", "unknown")).casefold()
        split = str(row.get("split", "")).casefold()
        key = (organ, coordinate.roi_id.casefold())
        splits_by_roi[key].add(split)
        combined_coordinates[key][(coordinate.row, coordinate.col)].append(
            (split, _row_label(row, index))
        )
    shared = tuple(
        f"{organ}/{roi_id.upper()}"
        for (organ, roi_id), splits in sorted(splits_by_roi.items())
        if "train" in splits and "val" in splits
    )
    cross_split_adjacent: list[dict[str, str]] = []
    for (organ, roi_id), entries in sorted(combined_coordinates.items()):
        for (row_number, col_number), sources in entries.items():
            for direction, neighbor_location in (
                ("right", (row_number, col_number + 1)),
                ("down", (row_number + 1, col_number)),
            ):
                for left_split, left_label in sources:
                    for right_split, right_label in entries.get(neighbor_location, []):
                        if left_split != right_split:
                            cross_split_adjacent.append(
                                {
                                    "organ": organ,
                                    "roi_id": roi_id.upper(),
                                    "direction": direction,
                                    "first": left_label,
                                    "first_split": left_split,
                                    "second": right_label,
                                    "second_split": right_split,
                                }
                            )

    boundary = _boundary_audit(
        indexed,
        root,
        border_width=border_width,
        min_pairs_per_axis=min_pairs_per_axis,
        min_direction_margin=min_direction_margin,
        max_boundary_mae=max_boundary_mae,
    )
    filename_verified = bool(rows) and len(coordinate_rows) == len(rows) and not duplicates
    reasons: list[str] = []
    if not rows:
        reasons.append("empty_manifest")
    if unparsed:
        reasons.append("unverified_filename_coordinates")
    if duplicates:
        reasons.append("duplicate_coordinates")
    if shared:
        reasons.append("train_val_roi_overlap")
    if cross_split_adjacent:
        reasons.append("cross_split_adjacent_patches")
    if not boundary.direction_verified:
        reasons.append("coordinate_direction_not_verified")
    if not boundary.continuity_verified:
        reasons.append("boundary_continuity_not_verified")
    context_enabled = filename_verified and not shared and not cross_split_adjacent and not reasons
    return ROIGridAudit(
        total_rows=len(rows),
        parsed_rows=len(coordinate_rows),
        parse_fraction=len(coordinate_rows) / max(1, len(rows)),
        unparsed_examples=tuple(unparsed[:50]),
        grids=tuple(summaries),
        duplicate_coordinates=tuple(duplicates),
        train_val_shared_rois=shared,
        cross_split_adjacent_pairs=tuple(cross_split_adjacent),
        boundary=boundary,
        filename_grid_verified=filename_verified,
        context_enabled=context_enabled,
        context_gate_reasons=tuple(dict.fromkeys(reasons)),
    )


def assess_context_eligibility(
    rows: Sequence[Mapping[str, Any]],
    data_root: str | Path,
    **kwargs: Any,
) -> ROIGridAudit:
    """Compatibility name emphasizing that the audit result is the context gate."""

    return audit_roi_grid(rows, data_root, **kwargs)
