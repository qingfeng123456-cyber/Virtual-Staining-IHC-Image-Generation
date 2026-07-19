"""Authoritative ROI-grouped inner folds for official training rows only."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .roi_index import PatchCoordinate, coordinate_value_from_row, parse_patch_coordinate

_AUTHORITATIVE_TRAIN_SPLITS = {"train", "official_train", "final_train"}
_FILENAME_SOURCES = {"filename_regex", "filename_coordinate"}


def _normalized_split(value: Any) -> str:
    return str(value).strip().casefold()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class InnerFoldAssignment:
    """One patch assignment with its authoritative coordinate provenance."""

    source_index: int
    canonical_key: str
    organ: str
    roi_id: str
    row: int
    col: int
    fold: int


@dataclass(frozen=True)
class ROIGroupedFoldResult:
    """Deterministic fold assignments and balance diagnostics."""

    fold_count: int
    seed: int
    assignments: tuple[InnerFoldAssignment, ...]
    fold_patch_counts: tuple[int, ...]
    fold_organ_patch_counts: tuple[dict[str, int], ...]

    @property
    def assignment_sha256(self) -> str:
        """Return a path-independent hash of the complete fold contract."""

        payload = {
            "schema_version": 1,
            "fold_count": self.fold_count,
            "seed": self.seed,
            "assignments": self.to_rows(),
            "fold_patch_counts": list(self.fold_patch_counts),
            "fold_organ_patch_counts": [
                dict(sorted(counts.items())) for counts in self.fold_organ_patch_counts
            ],
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def assignment_by_index(self) -> dict[int, int]:
        return {assignment.source_index: assignment.fold for assignment in self.assignments}

    def validation_indices(self, fold: int) -> tuple[int, ...]:
        self._validate_fold(fold)
        return tuple(
            assignment.source_index
            for assignment in self.assignments
            if assignment.fold == fold
        )

    def training_indices(self, fold: int) -> tuple[int, ...]:
        self._validate_fold(fold)
        return tuple(
            assignment.source_index
            for assignment in self.assignments
            if assignment.fold != fold
        )

    def _validate_fold(self, fold: int) -> None:
        if fold < 0 or fold >= self.fold_count:
            raise ValueError(f"fold must be in [0, {self.fold_count - 1}], got {fold}")

    def to_rows(self) -> list[dict[str, Any]]:
        return [asdict(assignment) for assignment in self.assignments]

    def write_csv(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        fields = ("source_index", "canonical_key", "organ", "roi_id", "row", "col", "fold")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.to_rows())
        temporary.replace(destination)
        return destination


def _validated_coordinate(
    row: Mapping[str, Any],
    index: int,
) -> tuple[str, str, PatchCoordinate]:
    split = _normalized_split(row.get("split"))
    source_split = _normalized_split(row.get("source_split"))
    if split not in _AUTHORITATIVE_TRAIN_SPLITS:
        raise ValueError(f"Row {index} is not a training row: split={split!r}")
    if source_split not in _AUTHORITATIVE_TRAIN_SPLITS:
        raise ValueError(
            f"Row {index} is not from an authoritative official train source: "
            f"source_split={source_split!r}"
        )
    if _truthy(row.get("is_smoke", False)):
        raise ValueError(f"Row {index} is a smoke row and cannot enter grouped folds")
    coordinate = parse_patch_coordinate(coordinate_value_from_row(row))
    if coordinate is None:
        raise ValueError(f"Row {index} has no authoritative filename PatchCoordinate")
    coordinate_source = str(row.get("roi_id_source", "")).strip().casefold()
    if coordinate_source and coordinate_source not in _FILENAME_SOURCES:
        raise ValueError(
            f"Row {index} has non-authoritative roi_id_source={coordinate_source!r}"
        )
    organ = str(row.get("organ", "")).strip()
    canonical_key = str(row.get("canonical_key", "")).strip()
    if not organ or not canonical_key:
        raise ValueError(f"Row {index} requires non-empty organ and canonical_key")
    return organ.casefold(), canonical_key, coordinate


def _seeded_group_key(seed: int, organ: str, roi_id: str) -> str:
    payload = f"{seed}:{organ.casefold()}:{roi_id.casefold()}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_authoritative_roi_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold_count: int = 5,
    seed: int = 2026,
    source_indices: Sequence[int] | None = None,
) -> ROIGroupedFoldResult:
    """Balance indivisible organ-scoped ROI groups across deterministic inner folds."""

    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
        raise ValueError("fold_count must be an integer of at least two")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not rows:
        raise ValueError("Official train rows cannot be empty")
    if source_indices is None:
        resolved_source_indices = tuple(range(len(rows)))
    else:
        if len(source_indices) != len(rows):
            raise ValueError("source_indices must have the same length as rows")
        resolved_source_indices = tuple(source_indices)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in resolved_source_indices
        ):
            raise ValueError("source_indices must contain nonnegative integers")
        if len(set(resolved_source_indices)) != len(resolved_source_indices):
            raise ValueError("source_indices must be unique")

    groups: dict[tuple[str, str], list[tuple[int, str, PatchCoordinate]]] = defaultdict(list)
    seen_coordinates: dict[tuple[str, str, int, int], int] = {}
    seen_canonical_keys: dict[str, int] = {}
    for row, source_index in zip(rows, resolved_source_indices, strict=True):
        organ, canonical_key, coordinate = _validated_coordinate(row, source_index)
        normalized_key = canonical_key.casefold()
        if normalized_key in seen_canonical_keys:
            raise ValueError(
                "Duplicate canonical_key at rows "
                f"{seen_canonical_keys[normalized_key]} and {source_index}: {canonical_key}"
            )
        seen_canonical_keys[normalized_key] = source_index
        coordinate_key = (
            organ.casefold(),
            coordinate.roi_id.casefold(),
            coordinate.row,
            coordinate.col,
        )
        if coordinate_key in seen_coordinates:
            raise ValueError(
                "Duplicate authoritative ROI coordinate at rows "
                f"{seen_coordinates[coordinate_key]} and {source_index}: {coordinate_key}"
            )
        seen_coordinates[coordinate_key] = source_index
        groups[(organ, coordinate.roi_id)].append(
            (source_index, canonical_key, coordinate)
        )
    if len(groups) < fold_count:
        raise ValueError(
            f"Need at least {fold_count} authoritative ROI groups, found {len(groups)}"
        )

    organ_totals: dict[str, int] = defaultdict(int)
    for (organ, _), members in groups.items():
        organ_totals[organ] += len(members)
    total_patches = sum(organ_totals.values())
    total_target = total_patches / fold_count
    organ_targets = {organ: count / fold_count for organ, count in organ_totals.items()}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            _seeded_group_key(seed, item[0][0], item[0][1]),
            item[0][0].casefold(),
            item[0][1].casefold(),
        ),
    )

    fold_totals = [0] * fold_count
    fold_organs: list[dict[str, int]] = [defaultdict(int) for _ in range(fold_count)]
    group_folds: dict[tuple[str, str], int] = {}
    for (organ, roi_id), members in ordered_groups:
        size = len(members)
        seeded_start = int(_seeded_group_key(seed, organ, roi_id)[:8], 16) % fold_count

        def score(
            fold: int,
            current_organ: str = organ,
            tie_start: int = seeded_start,
        ) -> tuple[float, int, int, int]:
            organ_load = fold_organs[fold][current_organ]
            combined_load = (
                organ_load / max(organ_targets[current_organ], 1.0)
                + fold_totals[fold] / max(total_target, 1.0)
            )
            return (
                combined_load,
                organ_load,
                fold_totals[fold],
                (fold - tie_start) % fold_count,
            )

        selected = min(range(fold_count), key=score)
        group_folds[(organ, roi_id)] = selected
        fold_totals[selected] += size
        fold_organs[selected][organ] += size

    assignments: list[InnerFoldAssignment] = []
    for (organ, roi_id), members in groups.items():
        fold = group_folds[(organ, roi_id)]
        assignments.extend(
            InnerFoldAssignment(
                source_index=index,
                canonical_key=canonical_key,
                organ=organ,
                roi_id=coordinate.roi_id,
                row=coordinate.row,
                col=coordinate.col,
                fold=fold,
            )
            for index, canonical_key, coordinate in members
        )
    assignments.sort(key=lambda value: value.source_index)
    return ROIGroupedFoldResult(
        fold_count=fold_count,
        seed=seed,
        assignments=tuple(assignments),
        fold_patch_counts=tuple(fold_totals),
        fold_organ_patch_counts=tuple(dict(sorted(counts.items())) for counts in fold_organs),
    )
