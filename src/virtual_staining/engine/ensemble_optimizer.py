"""Leakage-guarded deterministic optimization of prediction-ensemble weights.

The public optimizer accepts only arrays accompanied by verified sidecar metadata.
This prevents a caller from turning a training/test array into an apparently safe
input merely by passing ``source="validation"``.  The lower-level coordinate search
remains available for deterministic numerical optimization once provenance has been
established by the caller.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from virtual_staining.constants import normalize_marker
from virtual_staining.data.manifest import MARKER_PATH_COLUMNS
from virtual_staining.data.roi_index import (
    coordinate_value_from_row,
    parse_patch_coordinate,
)

from .ensemble import normalize_nonnegative_weights

ScoreFunction = Callable[[np.ndarray, np.ndarray], float]

_ALLOWED_SOURCES = {
    "val": "validation",
    "validation": "validation",
    "oof": "oof",
    "out_of_fold": "oof",
    "out-of-fold": "oof",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SIDECAR_SCHEMA_VERSION = 2
_JPG_DOMAIN = "jpg_roundtrip"
_VALIDATION_SPLITS = {
    "val",
    "validation",
    "official_val",
    "official_validation",
    "final_val",
}
_TRAIN_SPLITS = {"train", "official_train", "final_train"}
_AUTHORITATIVE_ROI_SOURCES = {"filename_regex", "filename_coordinate"}


def _normalized_text(value: Any) -> str:
    return str(value).strip().casefold()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalized_text(value) in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class EnsembleManifestAnchor:
    """Externally read manifest/fold truth used to verify array sidecars."""

    source: str
    manifest_path: str
    manifest_sha256: str
    fold_assignment_path: str | None
    fold_assignment_sha256: str | None
    roi_audit_path: str | None
    roi_audit_sha256: str | None
    roi_audit_total_rows: int
    roi_audit_parsed_rows: int
    audited_manifests: tuple[tuple[str, str, int], ...]
    audited_manifest_row_count: int
    sample_keys: tuple[str, ...]
    sample_organs: tuple[str, ...]
    manifest_splits: tuple[str, ...]
    group_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    fold_count: int
    target_marker: str
    metric_domain: str
    roi_authority: str
    unsafe_engineering_override_used: bool
    unsafe_reasons: tuple[str, ...]

    @property
    def sample_keys_sha256(self) -> str:
        return _sample_keys_sha256(self.sample_keys)

    @property
    def sample_organs_sha256(self) -> str:
        return _sample_keys_sha256(self.sample_organs)

    @property
    def audited_manifests_sha256(self) -> str:
        payload = json.dumps(
            [list(value) for value in self.audited_manifests],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "fold_assignment_path": self.fold_assignment_path,
            "fold_assignment_sha256": self.fold_assignment_sha256,
            "roi_audit_path": self.roi_audit_path,
            "roi_audit_sha256": self.roi_audit_sha256,
            "roi_audit_total_rows": self.roi_audit_total_rows,
            "roi_audit_parsed_rows": self.roi_audit_parsed_rows,
            "audited_manifests": [
                {"path": path, "sha256": digest, "row_count": row_count}
                for path, digest, row_count in self.audited_manifests
            ],
            "audited_manifest_row_count": self.audited_manifest_row_count,
            "audited_manifests_sha256": self.audited_manifests_sha256,
            "sample_count": len(self.sample_keys),
            "sample_keys_sha256": self.sample_keys_sha256,
            "sample_organs_sha256": self.sample_organs_sha256,
            "manifest_splits": sorted(set(self.manifest_splits)),
            "organs": sorted(set(self.sample_organs)),
            "fold_count": self.fold_count,
            "group_count": len(set(self.group_ids)),
            "target_marker": self.target_marker,
            "metric_domain": self.metric_domain,
            "roi_authority": self.roi_authority,
            "unsafe_engineering_override_used": self.unsafe_engineering_override_used,
            "unsafe_reasons": list(self.unsafe_reasons),
        }


@dataclass(frozen=True)
class ArrayProvenance:
    """Validated provenance for one array artifact."""

    array_path: str
    sidecar_path: str
    role: str
    source: str
    array_sha256: str
    content_sha256: str
    manifest_sha256: str
    fold_assignment_sha256: str | None
    roi_audit_sha256: str | None
    audited_manifests_sha256: str
    sample_keys_sha256: str
    sample_organs_sha256: str
    sample_keys: tuple[str, ...]
    sample_organs: tuple[str, ...]
    fold_ids: tuple[int, ...]
    group_ids: tuple[str, ...]
    fold_count: int
    target_marker: str
    metric_domain: str
    roi_authority: str
    unsafe_engineering_override_used: bool
    array_shape: tuple[int, ...]
    array_dtype: str
    artifact_id: str

    def summary(self) -> dict[str, Any]:
        """Return compact provenance suitable for an experiment artifact."""

        return {
            "array_path": self.array_path,
            "sidecar_path": self.sidecar_path,
            "role": self.role,
            "source": self.source,
            "array_sha256": self.array_sha256,
            "manifest_sha256": self.manifest_sha256,
            "fold_assignment_sha256": self.fold_assignment_sha256,
            "roi_audit_sha256": self.roi_audit_sha256,
            "audited_manifests_sha256": self.audited_manifests_sha256,
            "sample_keys_sha256": self.sample_keys_sha256,
            "sample_organs_sha256": self.sample_organs_sha256,
            "sample_count": len(self.sample_keys),
            "manifest_sample_count": len(self.sample_keys),
            "manifest_coverage_complete": True,
            "fold_count": self.fold_count,
            "group_count": len(set(self.group_ids)),
            "target_marker": self.target_marker,
            "metric_domain": self.metric_domain,
            "roi_authority": self.roi_authority,
            "unsafe_engineering_override_used": (
                self.unsafe_engineering_override_used
            ),
            "array_shape": list(self.array_shape),
            "array_dtype": self.array_dtype,
            "artifact_id": self.artifact_id,
        }


@dataclass(frozen=True)
class EnsembleInputProvenance:
    """Cross-array provenance contract verified before weight fitting."""

    source: str
    manifest_sha256: str
    sample_keys_sha256: str
    sample_keys: tuple[str, ...]
    fold_ids: tuple[int, ...]
    group_ids: tuple[str, ...]
    fold_count: int
    predictions: tuple[ArrayProvenance, ...]
    target: ArrayProvenance
    anchor: EnsembleManifestAnchor

    def summary(self) -> dict[str, Any]:
        """Return an auditable, bounded-size representation of the contract."""

        return {
            "schema_version": _SIDECAR_SCHEMA_VERSION,
            "source": self.source,
            "manifest_sha256": self.manifest_sha256,
            "sample_keys_sha256": self.sample_keys_sha256,
            "sample_count": len(self.sample_keys),
            "manifest_sample_count": len(self.sample_keys),
            "manifest_coverage_complete": True,
            "fold_count": self.fold_count,
            "group_count": len(set(self.group_ids)),
            "grouped_oof_verified": self.source == "oof",
            "external_manifest_anchor": self.anchor.summary(),
            "unsafe_engineering_override_used": (
                self.anchor.unsafe_engineering_override_used
            ),
            "predictions": [item.summary() for item in self.predictions],
            "target": self.target.summary(),
        }


@dataclass(frozen=True)
class CrossValidationFoldResult:
    """Held-out result for one grouped OOF fold."""

    fold_id: int
    train_samples: int
    held_out_samples: int
    held_out_groups: int
    weights: tuple[float, ...]
    score: float
    uniform_score: float
    evaluations: int
    optimizer: str = "coordinate"
    fallback_reason: str | None = None


@dataclass(frozen=True)
class EnsembleOptimizationResult:
    """Auditable result of fitting one global simplex weight vector."""

    weights: tuple[float, ...]
    score: float
    uniform_score: float
    source: str
    evaluations: int
    used_learned_weights: bool
    optimizer: str = "coordinate"
    fallback_reason: str | None = None
    cross_validated: bool = False
    cross_validated_score: float | None = None
    cross_validated_uniform_score: float | None = None
    fold_results: tuple[CrossValidationFoldResult, ...] = ()


@dataclass(frozen=True)
class _ROIAuditBinding:
    path: str | None
    sha256: str | None
    total_rows: int
    parsed_rows: int
    audited_manifests: tuple[tuple[str, str, int], ...]
    audited_row_count: int
    reasons: tuple[str, ...]


def validate_prediction_source(source: str) -> str:
    """Normalize and validate the only prediction sources permitted for fitting."""

    if not isinstance(source, str):
        raise TypeError("Prediction source must be an explicit string")
    normalized = _ALLOWED_SOURCES.get(source.strip().lower())
    if normalized is None:
        raise ValueError(
            "Ensemble weights may be fitted only from explicit validation or OOF predictions"
        )
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_keys_sha256(sample_keys: Sequence[str]) -> str:
    payload = json.dumps(
        list(sample_keys), ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _array_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def default_sidecar_path(array_path: str | Path) -> Path:
    """Return the required ``<array filename>.meta.json`` sidecar path."""

    resolved = Path(array_path).expanduser().resolve()
    return resolved.with_name(f"{resolved.name}.meta.json")


def _read_csv_rows(path: Path, *, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{label} has no CSV header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{label} must contain at least one row: {path}")
    return rows


def _manifest_row_scope_reasons(
    row: Mapping[str, Any],
    *,
    source: str,
    index: int,
) -> list[str]:
    split = _normalized_text(row.get("split"))
    source_split = _normalized_text(row.get("source_split"))
    if "test" in split or "test" in source_split:
        raise ValueError(
            f"Manifest row {index} belongs to test data and can never fit ensemble weights"
        )
    expected = _VALIDATION_SPLITS if source == "validation" else _TRAIN_SPLITS
    reasons: list[str] = []
    if split not in expected:
        reasons.append(f"row_{index}:non_authoritative_split:{split or 'missing'}")
    if source_split not in expected:
        reasons.append(
            f"row_{index}:non_authoritative_source_split:{source_split or 'missing'}"
        )
    if (
        "smoke" in split
        or "smoke" in source_split
        or _truthy(row.get("is_smoke"))
    ):
        reasons.append(f"row_{index}:smoke_or_tiny_subset")
    scope = _normalized_text(
        row.get("manifest_scope")
        or row.get("data_scope")
        or row.get("subset_reason")
    )
    if scope in {"smoke", "tiny", "subset", "truncated", "debug"}:
        reasons.append(f"row_{index}:non_full_manifest_scope:{scope}")
    if _truthy(row.get("is_truncated")):
        reasons.append(f"row_{index}:truncated_manifest")
    return reasons


def _require_int_field(
    row: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> int:
    value = str(row.get(field, "")).strip()
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{label}: {field} must be an integer, found {value!r}") from error
    if parsed < 0:
        raise ValueError(f"{label}: {field} must be nonnegative")
    return parsed


def _audit_integer(audit: Mapping[str, Any], field: str, reasons: list[str]) -> int:
    value = audit.get(field)
    if isinstance(value, bool):
        reasons.append(f"invalid_roi_audit_field:{field}")
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        reasons.append(f"invalid_roi_audit_field:{field}")
        return 0
    if parsed < 0:
        reasons.append(f"invalid_roi_audit_field:{field}")
        return 0
    return parsed


def _build_roi_audit_binding(
    target_manifest: Path,
    *,
    source: str,
    roi_audit_path: str | Path | None,
    audited_manifest_paths: Sequence[str | Path] | None,
) -> _ROIAuditBinding:
    reasons: list[str] = []
    if audited_manifest_paths is None:
        audit_manifest_values: tuple[str | Path, ...] = (target_manifest,)
        reasons.append("missing_train_val_audited_manifest_pair")
    else:
        audit_manifest_values = tuple(audited_manifest_paths)
        if len(audit_manifest_values) != 2:
            reasons.append("audited_manifest_paths_must_be_train_and_val")

    audited: list[tuple[str, str, int]] = []
    seen_paths: set[Path] = set()
    seen_keys: set[str] = set()
    roles_by_path: dict[Path, str] = {}
    groups_by_role: dict[str, set[str]] = {"train": set(), "validation": set()}
    coordinates_by_group: dict[
        str, dict[tuple[int, int], tuple[str, str]]
    ] = {}
    audited_row_count = 0
    for value in audit_manifest_values:
        path = Path(value).expanduser().resolve()
        if path in seen_paths:
            reasons.append("duplicate_audited_manifest_path")
            continue
        seen_paths.add(path)
        rows = _read_csv_rows(path, label="Audited train/val manifest")
        audited.append((str(path), _sha256_file(path), len(rows)))
        audited_row_count += len(rows)
        splits = {_normalized_text(row.get("split")) for row in rows}
        source_splits = {_normalized_text(row.get("source_split")) for row in rows}
        if any("test" in split for split in splits | source_splits):
            raise ValueError("Audited manifests must never include test data")
        if splits.issubset(_TRAIN_SPLITS) and source_splits.issubset(_TRAIN_SPLITS):
            role = "train"
        elif splits.issubset(_VALIDATION_SPLITS) and source_splits.issubset(
            _VALIDATION_SPLITS
        ):
            role = "validation"
        else:
            role = "unverified"
            reasons.append(f"audited_manifest_has_unverified_split:{path.name}")
        roles_by_path[path] = role

        for index, row in enumerate(rows):
            key = str(row.get("canonical_key", "")).strip()
            organ = _normalized_text(row.get("organ"))
            if not key or not organ:
                reasons.append(f"audited_manifest_missing_identity:{path.name}:{index}")
                continue
            normalized_key = key.casefold()
            if normalized_key in seen_keys:
                reasons.append(f"audited_manifest_canonical_key_overlap:{key}")
            seen_keys.add(normalized_key)
            coordinate = parse_patch_coordinate(coordinate_value_from_row(row))
            roi_source = _normalized_text(row.get("roi_id_source"))
            if coordinate is None:
                reasons.append(
                    f"audited_manifest_missing_coordinate:{path.name}:{index}"
                )
                continue
            if roi_source not in _AUTHORITATIVE_ROI_SOURCES:
                reasons.append(
                    f"audited_manifest_non_authoritative_roi:{path.name}:{index}"
                )
            group = f"{organ}/{coordinate.roi_id.casefold()}"
            if role in groups_by_role:
                groups_by_role[role].add(group)
            locations = coordinates_by_group.setdefault(group, {})
            location = (coordinate.row, coordinate.col)
            if location in locations:
                reasons.append(f"audited_manifest_duplicate_coordinate:{group}:{location}")
            locations[location] = (role, key)

    if set(roles_by_path.values()) != {"train", "validation"}:
        reasons.append("audited_manifests_do_not_form_train_val_pair")
    target_role = "validation" if source == "validation" else "train"
    if roles_by_path.get(target_manifest) != target_role:
        reasons.append("ensemble_manifest_not_bound_to_matching_audited_split")
    shared_groups = groups_by_role["train"].intersection(groups_by_role["validation"])
    if shared_groups:
        reasons.append("audited_train_val_roi_overlap")
    for group, locations in coordinates_by_group.items():
        for (row, col), (first_role, _first_key) in locations.items():
            for neighbor in ((row, col + 1), (row + 1, col)):
                second = locations.get(neighbor)
                if second is not None and second[0] != first_role:
                    reasons.append(f"audited_cross_split_adjacent_patch:{group}")

    if roi_audit_path is None:
        reasons.append("missing_roi_audit_path")
        return _ROIAuditBinding(
            path=None,
            sha256=None,
            total_rows=0,
            parsed_rows=0,
            audited_manifests=tuple(audited),
            audited_row_count=audited_row_count,
            reasons=tuple(dict.fromkeys(reasons)),
        )
    audit_path = Path(roi_audit_path).expanduser().resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(f"ROI audit does not exist: {audit_path}")
    raw_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(raw_audit, Mapping):
        raise ValueError(f"ROI audit root must be a JSON object: {audit_path}")
    total_rows = _audit_integer(raw_audit, "total_rows", reasons)
    parsed_rows = _audit_integer(raw_audit, "parsed_rows", reasons)
    if total_rows <= 0:
        reasons.append("empty_roi_audit")
    if total_rows != audited_row_count:
        reasons.append("roi_audit_manifest_row_count_mismatch")
    if parsed_rows != total_rows:
        reasons.append("roi_audit_does_not_parse_every_row")
    if raw_audit.get("filename_grid_verified") is not True:
        reasons.append("roi_audit_filename_grid_not_verified")
    for field, reason in (
        ("duplicate_coordinates", "roi_audit_duplicate_coordinates"),
        ("train_val_shared_rois", "roi_audit_train_val_roi_overlap"),
        ("cross_split_adjacent_pairs", "roi_audit_cross_split_adjacent_patches"),
    ):
        if field not in raw_audit:
            reasons.append(f"roi_audit_missing_field:{field}")
        elif raw_audit[field]:
            reasons.append(reason)
    boundary = raw_audit.get("boundary")
    if not isinstance(boundary, Mapping):
        reasons.append("roi_audit_missing_boundary")
    else:
        if boundary.get("direction_verified") is not True:
            reasons.append("roi_audit_direction_not_verified")
        if boundary.get("continuity_verified") is not True:
            reasons.append("roi_audit_continuity_not_verified")
    gate_reasons = raw_audit.get("context_gate_reasons")
    if not isinstance(gate_reasons, Sequence) or isinstance(
        gate_reasons, (str, bytes)
    ):
        reasons.append("roi_audit_invalid_context_gate_reasons")
    elif gate_reasons:
        reasons.extend(f"roi_audit_gate:{value}" for value in gate_reasons)
    if raw_audit.get("context_enabled") is not True:
        reasons.append("roi_audit_context_gate_not_passed")
    return _ROIAuditBinding(
        path=str(audit_path),
        sha256=_sha256_file(audit_path),
        total_rows=total_rows,
        parsed_rows=parsed_rows,
        audited_manifests=tuple(audited),
        audited_row_count=audited_row_count,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def build_ensemble_manifest_anchor(
    manifest_path: str | Path,
    *,
    source: str,
    target_marker: str,
    metric_domain: str = _JPG_DOMAIN,
    fold_assignment_path: str | Path | None = None,
    roi_audit_path: str | Path | None = None,
    audited_manifest_paths: Sequence[str | Path] | None = None,
    allow_unsafe_engineering_manifest: bool = False,
) -> EnsembleManifestAnchor:
    """Build array truth from actual CSV files rather than sidecar assertions.

    Strict validation accepts only fully labelled official validation rows with
    filename coordinates. Strict OOF fitting additionally requires an exact
    authoritative ROI-grouped fold-assignment CSV. A surrogate/debug manifest can
    be opened only through the explicit engineering override; the resulting anchor
    remains permanently marked and cannot later be loaded through the strict path.
    Test rows are never accepted, including through the unsafe override.
    """

    normalized_source = validate_prediction_source(source)
    normalized_domain = _normalized_text(metric_domain)
    if normalized_domain != _JPG_DOMAIN:
        raise ValueError(
            "Ensemble fitting is gated on final JPG round-trip metrics; "
            f"metric_domain must be {_JPG_DOMAIN!r}"
        )
    marker = normalize_marker(target_marker)
    marker_column = MARKER_PATH_COLUMNS[marker]
    manifest = Path(manifest_path).expanduser().resolve()
    rows = _read_csv_rows(manifest, label="Ensemble manifest")
    audit_binding = _build_roi_audit_binding(
        manifest,
        source=normalized_source,
        roi_audit_path=roi_audit_path,
        audited_manifest_paths=audited_manifest_paths,
    )

    keys: list[str] = []
    organs: list[str] = []
    splits: list[str] = []
    coordinates: list[Any] = []
    derived_groups: list[str] = []
    unsafe_reasons: list[str] = []
    seen_keys: set[str] = set()
    seen_coordinates: set[tuple[str, str, int, int]] = set()
    for index, row in enumerate(rows):
        key = str(row.get("canonical_key", "")).strip()
        organ = _normalized_text(row.get("organ"))
        if not key or not organ:
            raise ValueError(
                f"Manifest row {index} requires nonempty canonical_key and organ"
            )
        normalized_key = key.casefold()
        if normalized_key in seen_keys:
            raise ValueError(f"Manifest canonical_key is duplicated: {key!r}")
        seen_keys.add(normalized_key)
        if not str(row.get(marker_column, "")).strip():
            raise ValueError(
                f"Manifest row {index} has no {marker} target path in {marker_column}"
            )
        if "is_paired" in row and not _truthy(row.get("is_paired")):
            raise ValueError(
                f"Manifest row {index} is not paired and cannot provide ensemble targets"
            )
        if str(row.get("missing_targets", "")).strip():
            raise ValueError(f"Manifest row {index} declares missing target labels")
        unsafe_reasons.extend(
            _manifest_row_scope_reasons(row, source=normalized_source, index=index)
        )

        coordinate = parse_patch_coordinate(coordinate_value_from_row(row))
        roi_source = _normalized_text(row.get("roi_id_source"))
        if coordinate is None:
            unsafe_reasons.append(f"row_{index}:missing_filename_coordinate")
            fallback = str(row.get("group_id") or row.get("roi_id") or key).strip()
            group = f"{organ}/{fallback.casefold()}"
        else:
            if roi_source not in _AUTHORITATIVE_ROI_SOURCES:
                unsafe_reasons.append(
                    f"row_{index}:non_authoritative_roi_source:{roi_source or 'missing'}"
                )
            coordinate_key = (
                organ,
                coordinate.roi_id.casefold(),
                coordinate.row,
                coordinate.col,
            )
            if coordinate_key in seen_coordinates:
                raise ValueError(
                    "Manifest contains a duplicate organ-scoped ROI coordinate: "
                    f"{coordinate_key}"
                )
            seen_coordinates.add(coordinate_key)
            group = f"{organ}/{coordinate.roi_id.casefold()}"
        keys.append(key)
        organs.append(organ)
        splits.append(_normalized_text(row.get("split")))
        coordinates.append(coordinate)
        derived_groups.append(group)

    assignment: Path | None = None
    assignment_hash: str | None = None
    fold_ids: list[int]
    group_ids: list[str]
    if normalized_source == "validation":
        if fold_assignment_path is not None:
            raise ValueError("Validation anchors must not supply an OOF fold assignment")
        fold_ids = [0] * len(rows)
        group_ids = derived_groups
        if len(set(group_ids)) < 2:
            unsafe_reasons.append("validation_has_fewer_than_two_roi_groups")
    else:
        if fold_assignment_path is None:
            raise ValueError("OOF anchors require an external fold-assignment CSV")
        assignment = Path(fold_assignment_path).expanduser().resolve()
        assignment_rows = _read_csv_rows(assignment, label="OOF fold assignment")
        if len(assignment_rows) != len(rows):
            raise ValueError(
                "OOF fold assignment must cover every manifest row exactly once"
            )
        by_index: dict[int, Mapping[str, Any]] = {}
        for assignment_row in assignment_rows:
            source_index = _require_int_field(
                assignment_row,
                "source_index",
                label="OOF fold assignment",
            )
            if source_index >= len(rows) or source_index in by_index:
                raise ValueError(
                    "OOF fold assignment source_index values must uniquely cover the manifest"
                )
            by_index[source_index] = assignment_row
        if set(by_index) != set(range(len(rows))):
            raise ValueError(
                "OOF fold assignment source_index values must uniquely cover the manifest"
            )

        fold_ids = []
        group_ids = []
        for index, (_row, coordinate) in enumerate(
            zip(rows, coordinates, strict=True)
        ):
            assignment_row = by_index[index]
            label = f"OOF fold assignment row for source_index={index}"
            assigned_key = str(assignment_row.get("canonical_key", "")).strip()
            assigned_organ = _normalized_text(assignment_row.get("organ"))
            if assigned_key != keys[index] or assigned_organ != organs[index]:
                raise ValueError(
                    f"{label} does not match manifest canonical_key/organ order"
                )
            assigned_roi = str(assignment_row.get("roi_id", "")).strip().upper()
            assigned_row = _require_int_field(assignment_row, "row", label=label)
            assigned_col = _require_int_field(assignment_row, "col", label=label)
            fold = _require_int_field(assignment_row, "fold", label=label)
            if not re.fullmatch(r"ROI\d+", assigned_roi, flags=re.IGNORECASE):
                raise ValueError(f"{label}: roi_id is not authoritative ROI<digits>")
            if coordinate is None:
                unsafe_reasons.append(
                    f"row_{index}:assignment_coordinate_not_confirmed_by_filename"
                )
            elif (
                assigned_roi != coordinate.roi_id
                or assigned_row != coordinate.row
                or assigned_col != coordinate.col
            ):
                raise ValueError(f"{label} coordinate does not match the manifest filename")
            fold_ids.append(fold)
            group_ids.append(f"{organs[index]}/{assigned_roi.casefold()}")
        assignment_hash = _sha256_file(assignment)

    observed_folds = set(fold_ids)
    fold_count = max(observed_folds) + 1
    if observed_folds != set(range(fold_count)):
        raise ValueError("Fold IDs must fully cover contiguous folds [0, fold_count)")
    if normalized_source == "oof" and fold_count < 2:
        raise ValueError("OOF fitting requires at least two fully covered folds")
    group_to_fold: dict[str, int] = {}
    for group_id, fold_id in zip(group_ids, fold_ids, strict=True):
        previous = group_to_fold.setdefault(group_id, fold_id)
        if previous != fold_id:
            raise ValueError(f"ROI group {group_id!r} crosses OOF fold boundaries")

    unsafe_reasons.extend(audit_binding.reasons)
    if allow_unsafe_engineering_manifest and not unsafe_reasons:
        unsafe_reasons.append("explicit_unsafe_engineering_override")
    unique_reasons = tuple(dict.fromkeys(unsafe_reasons))
    if unique_reasons and not allow_unsafe_engineering_manifest:
        examples = "; ".join(unique_reasons[:5])
        raise ValueError(
            "Manifest is not an authoritative full ROI grid; strict ensemble fitting "
            f"is blocked ({examples})"
        )
    unsafe = bool(unique_reasons)
    return EnsembleManifestAnchor(
        source=normalized_source,
        manifest_path=str(manifest),
        manifest_sha256=_sha256_file(manifest),
        fold_assignment_path=str(assignment) if assignment is not None else None,
        fold_assignment_sha256=assignment_hash,
        roi_audit_path=audit_binding.path,
        roi_audit_sha256=audit_binding.sha256,
        roi_audit_total_rows=audit_binding.total_rows,
        roi_audit_parsed_rows=audit_binding.parsed_rows,
        audited_manifests=audit_binding.audited_manifests,
        audited_manifest_row_count=audit_binding.audited_row_count,
        sample_keys=tuple(keys),
        sample_organs=tuple(organs),
        manifest_splits=tuple(splits),
        group_ids=tuple(group_ids),
        fold_ids=tuple(fold_ids),
        fold_count=fold_count,
        target_marker=marker,
        metric_domain=normalized_domain,
        roi_authority=(
            "unsafe_engineering_manifest" if unsafe else "authoritative_filename_grid"
        ),
        unsafe_engineering_override_used=unsafe,
        unsafe_reasons=unique_reasons,
    )


def write_ensemble_array_sidecar(
    array_path: str | Path,
    *,
    role: str,
    anchor: EnsembleManifestAnchor,
    artifact_id: str,
    output_path: str | Path | None = None,
) -> Path:
    """Write a schema-v2 sidecar derived only from an external anchor."""

    path = Path(array_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Ensemble array does not exist: {path}")
    normalized_role = _normalized_text(role)
    if normalized_role not in {"prediction", "target"}:
        raise ValueError("role must be 'prediction' or 'target'")
    stable_id = str(artifact_id).strip()
    if not stable_id:
        raise ValueError("artifact_id must be a nonempty stable identifier")
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.ndim < 1:
        raise ValueError("Ensemble arrays must have a sample axis")
    if int(array.shape[0]) != len(anchor.sample_keys):
        raise ValueError("Array sample axis does not fully cover the external manifest")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Ensemble arrays must have a numeric dtype")
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_sidecar_path(path)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SIDECAR_SCHEMA_VERSION,
        "role": normalized_role,
        "split": anchor.source,
        "array_filename": path.name,
        "array_sha256": _sha256_file(path),
        "array_shape": list(array.shape),
        "array_dtype": str(array.dtype),
        "sample_keys": list(anchor.sample_keys),
        "sample_keys_sha256": anchor.sample_keys_sha256,
        "sample_organs": list(anchor.sample_organs),
        "sample_organs_sha256": anchor.sample_organs_sha256,
        "manifest_path": anchor.manifest_path,
        "manifest_sha256": anchor.manifest_sha256,
        "manifest_sample_count": len(anchor.sample_keys),
        "manifest_sample_keys_sha256": anchor.sample_keys_sha256,
        "fold_assignment_path": anchor.fold_assignment_path,
        "fold_assignment_sha256": anchor.fold_assignment_sha256,
        "roi_audit_path": anchor.roi_audit_path,
        "roi_audit_sha256": anchor.roi_audit_sha256,
        "roi_audit_total_rows": anchor.roi_audit_total_rows,
        "roi_audit_parsed_rows": anchor.roi_audit_parsed_rows,
        "audited_manifests": [
            {"path": path, "sha256": digest, "row_count": row_count}
            for path, digest, row_count in anchor.audited_manifests
        ],
        "audited_manifest_row_count": anchor.audited_manifest_row_count,
        "audited_manifests_sha256": anchor.audited_manifests_sha256,
        "fold_ids": list(anchor.fold_ids),
        "group_ids": list(anchor.group_ids),
        "fold_count": anchor.fold_count,
        "target_marker": anchor.target_marker,
        "metric_domain": anchor.metric_domain,
        "roi_authority": anchor.roi_authority,
        "unsafe_engineering_override_used": (
            anchor.unsafe_engineering_override_used
        ),
        "unsafe_reasons": list(anchor.unsafe_reasons),
        "artifact_id": stable_id,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _require_sha256(value: Any, field: str, sidecar: Path) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{sidecar}: {field} must be a lowercase SHA256 digest")
    return normalized


def _require_string_list(value: Any, field: str, count: int, sidecar: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{sidecar}: {field} must contain exactly {count} entries")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{sidecar}: {field} must contain strings")
    entries = tuple(item.strip() for item in value)
    if any(not item for item in entries):
        raise ValueError(f"{sidecar}: {field} must not contain empty values")
    return entries


def _require_fold_ids(value: Any, count: int, sidecar: Path) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{sidecar}: fold_ids must contain exactly {count} entries")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{sidecar}: fold_ids must contain integers")
    fold_ids = tuple(int(item) for item in value)
    if any(item < 0 for item in fold_ids):
        raise ValueError(f"{sidecar}: fold_ids contains an uncovered OOF sample")
    return fold_ids


def _load_array_provenance(
    array_path: str | Path,
    sidecar_path: str | Path | None,
    *,
    expected_role: str,
    anchor: EnsembleManifestAnchor,
) -> tuple[np.ndarray, ArrayProvenance]:
    path = Path(array_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Ensemble array does not exist: {path}")
    metadata_path = (
        Path(sidecar_path).expanduser().resolve()
        if sidecar_path is not None
        else default_sidecar_path(path)
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Required ensemble provenance sidecar does not exist: {metadata_path}"
        )
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{metadata_path}: sidecar root must be a JSON object")
    if raw.get("schema_version") != _SIDECAR_SCHEMA_VERSION:
        raise ValueError(
            f"{metadata_path}: schema_version must be {_SIDECAR_SCHEMA_VERSION}"
        )
    role = str(raw.get("role", "")).strip().casefold()
    if role != expected_role:
        raise ValueError(f"{metadata_path}: expected role={expected_role!r}, found {role!r}")
    source = validate_prediction_source(str(raw.get("split", "")))
    if source != anchor.source:
        raise ValueError(
            f"{metadata_path}: split does not match the external manifest anchor"
        )
    if str(raw.get("array_filename", "")) != path.name:
        raise ValueError(f"{metadata_path}: array_filename does not match {path.name!r}")
    expected_file_hash = _require_sha256(raw.get("array_sha256"), "array_sha256", metadata_path)
    actual_file_hash = _sha256_file(path)
    if actual_file_hash != expected_file_hash:
        raise ValueError(f"{metadata_path}: array SHA256 does not match {path}")

    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.ndim < 1 or array.shape[0] < 1:
        raise ValueError(f"{path}: ensemble arrays must have a nonempty sample axis")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{path}: ensemble arrays must have a numeric dtype")
    declared_shape = raw.get("array_shape")
    if not isinstance(declared_shape, list) or declared_shape != list(array.shape):
        raise ValueError(f"{metadata_path}: array_shape does not match the array")
    if str(raw.get("array_dtype", "")) != str(array.dtype):
        raise ValueError(f"{metadata_path}: array_dtype does not match the array")

    sample_count = int(array.shape[0])
    sample_keys = _require_string_list(
        raw.get("sample_keys"), "sample_keys", sample_count, metadata_path
    )
    if len(set(sample_keys)) != sample_count:
        raise ValueError(f"{metadata_path}: sample_keys must be unique and ordered")
    declared_keys_hash = _require_sha256(
        raw.get("sample_keys_sha256"), "sample_keys_sha256", metadata_path
    )
    if _sample_keys_sha256(sample_keys) != declared_keys_hash:
        raise ValueError(f"{metadata_path}: sample_keys_sha256 does not match sample_keys")
    manifest_hash = _require_sha256(
        raw.get("manifest_sha256"), "manifest_sha256", metadata_path
    )
    if manifest_hash != anchor.manifest_sha256:
        raise ValueError(
            f"{metadata_path}: manifest_sha256 does not match the actual manifest"
        )
    declared_manifest_path = Path(
        str(raw.get("manifest_path", ""))
    ).expanduser().resolve()
    if declared_manifest_path != Path(anchor.manifest_path):
        raise ValueError(
            f"{metadata_path}: manifest_path does not match the external anchor"
        )
    manifest_sample_count = raw.get("manifest_sample_count")
    if (
        isinstance(manifest_sample_count, bool)
        or not isinstance(manifest_sample_count, int)
        or manifest_sample_count != sample_count
    ):
        raise ValueError(
            f"{metadata_path}: array does not fully cover manifest_sample_count"
        )
    manifest_keys_hash = _require_sha256(
        raw.get("manifest_sample_keys_sha256"),
        "manifest_sample_keys_sha256",
        metadata_path,
    )
    if manifest_keys_hash != declared_keys_hash:
        raise ValueError(
            f"{metadata_path}: ordered sample keys do not fully cover the bound manifest"
        )
    if sample_keys != anchor.sample_keys:
        raise ValueError(
            f"{metadata_path}: ordered sample keys do not match the actual manifest"
        )
    sample_organs = _require_string_list(
        raw.get("sample_organs"), "sample_organs", sample_count, metadata_path
    )
    sample_organs_hash = _require_sha256(
        raw.get("sample_organs_sha256"), "sample_organs_sha256", metadata_path
    )
    if _sample_keys_sha256(sample_organs) != sample_organs_hash:
        raise ValueError(
            f"{metadata_path}: sample_organs_sha256 does not match sample_organs"
        )
    if sample_organs != anchor.sample_organs:
        raise ValueError(
            f"{metadata_path}: sample organs do not match the actual manifest"
        )
    fold_ids = _require_fold_ids(raw.get("fold_ids"), sample_count, metadata_path)
    group_ids = _require_string_list(
        raw.get("group_ids"), "group_ids", sample_count, metadata_path
    )
    fold_count_value = raw.get("fold_count")
    if isinstance(fold_count_value, bool) or not isinstance(fold_count_value, int):
        raise ValueError(f"{metadata_path}: fold_count must be an integer")
    fold_count = int(fold_count_value)
    observed_folds = set(fold_ids)
    if observed_folds != set(range(fold_count)):
        raise ValueError(
            f"{metadata_path}: fold_ids must fully cover contiguous folds [0, fold_count)"
        )
    if source == "validation" and fold_count != 1:
        raise ValueError(f"{metadata_path}: validation artifacts must describe one fold")
    if source == "oof" and fold_count < 2:
        raise ValueError(f"{metadata_path}: OOF artifacts require at least two covered folds")
    group_to_fold: dict[str, int] = {}
    for group_id, fold_id in zip(group_ids, fold_ids, strict=True):
        prior = group_to_fold.setdefault(group_id, fold_id)
        if prior != fold_id:
            raise ValueError(
                f"{metadata_path}: group {group_id!r} crosses OOF fold boundaries"
            )
    if fold_ids != anchor.fold_ids or group_ids != anchor.group_ids:
        raise ValueError(
            f"{metadata_path}: fold/group assignments do not match the external anchor"
        )
    if fold_count != anchor.fold_count:
        raise ValueError(
            f"{metadata_path}: fold_count does not match the external anchor"
        )
    declared_fold_path = raw.get("fold_assignment_path")
    declared_fold_hash = raw.get("fold_assignment_sha256")
    if anchor.fold_assignment_path is None:
        if declared_fold_path is not None or declared_fold_hash is not None:
            raise ValueError(
                f"{metadata_path}: validation sidecar must not assert a fold assignment"
            )
        fold_assignment_hash = None
    else:
        resolved_fold_path = Path(str(declared_fold_path or "")).expanduser().resolve()
        if resolved_fold_path != Path(anchor.fold_assignment_path):
            raise ValueError(
                f"{metadata_path}: fold_assignment_path does not match the external anchor"
            )
        fold_assignment_hash = _require_sha256(
            declared_fold_hash, "fold_assignment_sha256", metadata_path
        )
        if fold_assignment_hash != anchor.fold_assignment_sha256:
            raise ValueError(
                f"{metadata_path}: fold_assignment_sha256 does not match the actual CSV"
            )
    declared_audit_path = raw.get("roi_audit_path")
    declared_audit_hash = raw.get("roi_audit_sha256")
    if anchor.roi_audit_path is None:
        if declared_audit_path is not None or declared_audit_hash is not None:
            raise ValueError(
                f"{metadata_path}: sidecar asserts an ROI audit absent from the anchor"
            )
        roi_audit_hash = None
    else:
        resolved_audit_path = Path(
            str(declared_audit_path or "")
        ).expanduser().resolve()
        if resolved_audit_path != Path(anchor.roi_audit_path):
            raise ValueError(
                f"{metadata_path}: roi_audit_path does not match the external anchor"
            )
        roi_audit_hash = _require_sha256(
            declared_audit_hash, "roi_audit_sha256", metadata_path
        )
        if roi_audit_hash != anchor.roi_audit_sha256:
            raise ValueError(
                f"{metadata_path}: roi_audit_sha256 does not match the actual audit"
            )
    for field, expected in (
        ("roi_audit_total_rows", anchor.roi_audit_total_rows),
        ("roi_audit_parsed_rows", anchor.roi_audit_parsed_rows),
        ("audited_manifest_row_count", anchor.audited_manifest_row_count),
    ):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValueError(
                f"{metadata_path}: {field} does not match the external anchor"
            )
    raw_audited = raw.get("audited_manifests")
    expected_audited = [
        {"path": path, "sha256": digest, "row_count": row_count}
        for path, digest, row_count in anchor.audited_manifests
    ]
    if raw_audited != expected_audited:
        raise ValueError(
            f"{metadata_path}: audited_manifests do not match the external anchor"
        )
    audited_hash = _require_sha256(
        raw.get("audited_manifests_sha256"),
        "audited_manifests_sha256",
        metadata_path,
    )
    if audited_hash != anchor.audited_manifests_sha256:
        raise ValueError(
            f"{metadata_path}: audited_manifests_sha256 does not match the anchor"
        )
    target_marker = str(raw.get("target_marker", "")).strip()
    if target_marker != anchor.target_marker:
        raise ValueError(
            f"{metadata_path}: target_marker does not match the external anchor"
        )
    metric_domain = _normalized_text(raw.get("metric_domain"))
    if metric_domain != anchor.metric_domain or metric_domain != _JPG_DOMAIN:
        raise ValueError(
            f"{metadata_path}: metric_domain must be bound to final JPG round-trip"
        )
    roi_authority = str(raw.get("roi_authority", "")).strip()
    if roi_authority != anchor.roi_authority:
        raise ValueError(
            f"{metadata_path}: roi_authority does not match the external anchor"
        )
    unsafe_value = raw.get("unsafe_engineering_override_used")
    if not isinstance(unsafe_value, bool):
        raise ValueError(
            f"{metadata_path}: unsafe_engineering_override_used must be boolean"
        )
    if unsafe_value != anchor.unsafe_engineering_override_used:
        raise ValueError(
            f"{metadata_path}: unsafe engineering marker does not match the anchor"
        )
    unsafe_reasons = raw.get("unsafe_reasons")
    if (
        not isinstance(unsafe_reasons, list)
        or tuple(str(value) for value in unsafe_reasons) != anchor.unsafe_reasons
    ):
        raise ValueError(
            f"{metadata_path}: unsafe_reasons do not match the external anchor"
        )
    artifact_id = str(raw.get("artifact_id", "")).strip()
    if not artifact_id:
        raise ValueError(f"{metadata_path}: artifact_id must be a nonempty stable identifier")
    provenance = ArrayProvenance(
        array_path=str(path),
        sidecar_path=str(metadata_path),
        role=role,
        source=source,
        array_sha256=actual_file_hash,
        content_sha256=_array_content_sha256(array),
        manifest_sha256=manifest_hash,
        fold_assignment_sha256=fold_assignment_hash,
        roi_audit_sha256=roi_audit_hash,
        audited_manifests_sha256=audited_hash,
        sample_keys_sha256=declared_keys_hash,
        sample_organs_sha256=sample_organs_hash,
        sample_keys=sample_keys,
        sample_organs=sample_organs,
        fold_ids=fold_ids,
        group_ids=group_ids,
        fold_count=fold_count,
        target_marker=target_marker,
        metric_domain=metric_domain,
        roi_authority=roi_authority,
        unsafe_engineering_override_used=unsafe_value,
        array_shape=tuple(int(value) for value in array.shape),
        array_dtype=str(array.dtype),
        artifact_id=artifact_id,
    )
    return array, provenance


def load_verified_ensemble_inputs(
    prediction_paths: Sequence[str | Path],
    target_path: str | Path,
    *,
    prediction_sidecars: Sequence[str | Path] | None = None,
    target_sidecar: str | Path | None = None,
    expected_source: str | None = None,
    manifest_path: str | Path | None = None,
    target_marker: str | None = None,
    metric_domain: str = _JPG_DOMAIN,
    fold_assignment_path: str | Path | None = None,
    roi_audit_path: str | Path | None = None,
    audited_manifest_paths: Sequence[str | Path] | None = None,
    allow_unsafe_engineering_manifest: bool = False,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, EnsembleInputProvenance]:
    """Load arrays against externally read manifest and fold-assignment truth."""

    if not prediction_paths:
        raise ValueError("At least one prediction array is required")
    if expected_source is None:
        raise ValueError("expected_source is required; a sidecar cannot choose its own source")
    if manifest_path is None:
        raise ValueError("manifest_path is required as an external ensemble anchor")
    if target_marker is None:
        raise ValueError("target_marker is required as an external ensemble anchor")
    anchor = build_ensemble_manifest_anchor(
        manifest_path,
        source=expected_source,
        target_marker=target_marker,
        metric_domain=metric_domain,
        fold_assignment_path=fold_assignment_path,
        roi_audit_path=roi_audit_path,
        audited_manifest_paths=audited_manifest_paths,
        allow_unsafe_engineering_manifest=allow_unsafe_engineering_manifest,
    )
    if prediction_sidecars is not None and len(prediction_sidecars) != len(prediction_paths):
        raise ValueError("prediction_sidecars must match prediction_paths one-for-one")
    sidecars: Sequence[str | Path | None] = (
        tuple(prediction_sidecars)
        if prediction_sidecars is not None
        else (None,) * len(prediction_paths)
    )
    loaded = [
        _load_array_provenance(
            path,
            sidecar,
            expected_role="prediction",
            anchor=anchor,
        )
        for path, sidecar in zip(prediction_paths, sidecars, strict=True)
    ]
    target, target_provenance = _load_array_provenance(
        target_path,
        target_sidecar,
        expected_role="target",
        anchor=anchor,
    )
    prediction_arrays = tuple(item[0] for item in loaded)
    prediction_provenance = tuple(item[1] for item in loaded)
    reference = target_provenance
    for item in prediction_provenance:
        for field in (
            "source",
            "manifest_sha256",
            "fold_assignment_sha256",
            "roi_audit_sha256",
            "audited_manifests_sha256",
            "sample_keys_sha256",
            "sample_organs_sha256",
            "sample_keys",
            "sample_organs",
            "fold_ids",
            "group_ids",
            "fold_count",
            "target_marker",
            "metric_domain",
            "roi_authority",
            "unsafe_engineering_override_used",
        ):
            if getattr(item, field) != getattr(reference, field):
                raise ValueError(
                    f"Ensemble provenance mismatch for {field}: {item.sidecar_path} "
                    f"does not match {reference.sidecar_path}"
                )
    artifact_ids = [item.artifact_id for item in prediction_provenance]
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("Prediction sidecars must use distinct artifact_id values")
    contract = EnsembleInputProvenance(
        source=reference.source,
        manifest_sha256=reference.manifest_sha256,
        sample_keys_sha256=reference.sample_keys_sha256,
        sample_keys=reference.sample_keys,
        fold_ids=reference.fold_ids,
        group_ids=reference.group_ids,
        fold_count=reference.fold_count,
        predictions=prediction_provenance,
        target=target_provenance,
        anchor=anchor,
    )
    return prediction_arrays, target, contract


def _prepare_arrays(
    predictions: Sequence[Any],
    target: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if not predictions:
        raise ValueError("At least one validation/OOF prediction is required")
    target_array = np.asarray(target, dtype=np.float64)
    if target_array.size == 0:
        raise ValueError("The ensemble target must not be empty")
    prediction_arrays = [np.asarray(prediction, dtype=np.float64) for prediction in predictions]
    if any(prediction.shape != target_array.shape for prediction in prediction_arrays):
        raise ValueError("All predictions must have exactly the same shape as the target")
    stacked = np.stack(prediction_arrays, axis=0)
    if not np.isfinite(stacked).all() or not np.isfinite(target_array).all():
        raise ValueError("Predictions and targets must contain only finite values")
    return stacked, target_array


def blend_predictions(predictions: Sequence[Any], weights: Sequence[float]) -> np.ndarray:
    """Apply one global nonnegative unit-sum weight vector to model predictions."""

    if not predictions:
        raise ValueError("At least one prediction is required")
    arrays = [np.asarray(prediction, dtype=np.float64) for prediction in predictions]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("All prediction arrays must have identical shapes")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Prediction arrays must contain only finite values")
    normalized = np.asarray(
        normalize_nonnegative_weights(weights, len(arrays)),
        dtype=np.float64,
    )
    return np.tensordot(normalized, np.stack(arrays, axis=0), axes=(0, 0))


def _default_score(target: np.ndarray, prediction: np.ndarray) -> float:
    difference = prediction - target
    return -float(np.mean(difference * difference, dtype=np.float64))


def _score(
    target: np.ndarray,
    stacked_predictions: np.ndarray,
    weights: np.ndarray,
    score_function: ScoreFunction,
) -> float:
    blended = np.tensordot(weights, stacked_predictions, axes=(0, 0))
    value = float(score_function(target, blended))
    if not math.isfinite(value):
        raise ValueError(f"Ensemble score function returned a non-finite value: {value}")
    return value


def _canonical_steps(step_sizes: Sequence[float]) -> tuple[float, ...]:
    values = sorted({float(value) for value in step_sizes}, reverse=True)
    if not values or any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in values):
        raise ValueError("step_sizes must contain finite values in (0, 1]")
    return tuple(values)


def coordinate_search_weights(
    predictions: Sequence[Any],
    target: Any,
    *,
    source: str,
    score_function: ScoreFunction | None = None,
    initial_weights: Sequence[float] | None = None,
    step_sizes: Sequence[float] = (0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001),
    max_passes_per_step: int = 100,
    min_improvement: float = 1e-12,
    min_gain_over_uniform: float = 0.0,
    uniform_shrinkage: float = 0.0,
) -> EnsembleOptimizationResult:
    """Fit deterministic simplex weights by pairwise coordinate transfers.

    This is the numerical primitive.  Use :func:`optimize_ensemble_weights` for
    experiment fitting so sidecar provenance is mandatory.
    """

    normalized_source = validate_prediction_source(source)
    stacked, target_array = _prepare_arrays(predictions, target)
    model_count = stacked.shape[0]
    if max_passes_per_step < 1:
        raise ValueError("max_passes_per_step must be positive")
    if not math.isfinite(min_improvement) or min_improvement < 0.0:
        raise ValueError("min_improvement must be finite and nonnegative")
    if not math.isfinite(min_gain_over_uniform) or min_gain_over_uniform < 0.0:
        raise ValueError("min_gain_over_uniform must be finite and nonnegative")
    if not math.isfinite(uniform_shrinkage) or not 0.0 <= uniform_shrinkage <= 1.0:
        raise ValueError("uniform_shrinkage must be in [0, 1]")
    steps = _canonical_steps(step_sizes)
    metric = score_function or _default_score
    uniform = np.full(model_count, 1.0 / model_count, dtype=np.float64)
    uniform_score = _score(target_array, stacked, uniform, metric)
    if initial_weights is None:
        current = uniform.copy()
    else:
        current = np.asarray(
            normalize_nonnegative_weights(initial_weights, model_count),
            dtype=np.float64,
        )
    current_score = _score(target_array, stacked, current, metric)
    evaluations = 2 if initial_weights is not None else 1

    for step in steps:
        for _ in range(max_passes_per_step):
            best_score = current_score
            best_weights: np.ndarray | None = None
            for donor in range(model_count):
                transfer = min(step, float(current[donor]))
                if transfer <= 0.0:
                    continue
                for receiver in range(model_count):
                    if receiver == donor:
                        continue
                    candidate = current.copy()
                    candidate[donor] -= transfer
                    candidate[receiver] += transfer
                    candidate_score = _score(target_array, stacked, candidate, metric)
                    evaluations += 1
                    if candidate_score > best_score + min_improvement:
                        best_score = candidate_score
                        best_weights = candidate
            if best_weights is None:
                break
            current = best_weights
            current_score = best_score

    if uniform_shrinkage > 0.0:
        current = (1.0 - uniform_shrinkage) * current + uniform_shrinkage * uniform
        current = np.asarray(normalize_nonnegative_weights(current, model_count))
        current_score = _score(target_array, stacked, current, metric)
        evaluations += 1
    use_learned = current_score >= uniform_score + min_gain_over_uniform
    if not use_learned:
        current = uniform
        current_score = uniform_score
    normalized_result = tuple(normalize_nonnegative_weights(current, model_count))
    return EnsembleOptimizationResult(
        weights=normalized_result,
        score=current_score,
        uniform_score=uniform_score,
        source=normalized_source,
        evaluations=evaluations,
        used_learned_weights=use_learned and not np.allclose(current, uniform, atol=1e-12),
    )


def validate_optimizer_name(optimizer: str) -> str:
    """Validate the explicitly supported ensemble optimization backends."""

    normalized = str(optimizer).strip().casefold()
    if normalized not in {"coordinate", "slsqp"}:
        raise ValueError("Ensemble optimizer must be coordinate or slsqp")
    return normalized


def _validate_failure_policy(failure_policy: str) -> str:
    normalized = str(failure_policy).strip().casefold()
    if normalized not in {"uniform", "error"}:
        raise ValueError("optimizer_failure_policy must be uniform or error")
    return normalized


def _slsqp_failure(
    *,
    reason: str,
    failure_policy: str,
    model_count: int,
    uniform_score: float,
    source: str,
    evaluations: int,
) -> EnsembleOptimizationResult:
    if failure_policy == "error":
        raise RuntimeError(f"SLSQP ensemble optimization failed: {reason}")
    uniform = tuple(1.0 / model_count for _ in range(model_count))
    return EnsembleOptimizationResult(
        weights=uniform,
        score=uniform_score,
        uniform_score=uniform_score,
        source=source,
        evaluations=evaluations,
        used_learned_weights=False,
        optimizer="slsqp",
        fallback_reason=reason,
    )


def slsqp_weights(
    predictions: Sequence[Any],
    target: Any,
    *,
    source: str,
    score_function: ScoreFunction | None = None,
    initial_weights: Sequence[float] | None = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-12,
    min_gain_over_uniform: float = 0.0,
    uniform_shrinkage: float = 0.0,
    failure_policy: str = "uniform",
) -> EnsembleOptimizationResult:
    """Fit nonnegative unit-sum weights with deterministic SciPy SLSQP.

    Numerical solver failures never produce unchecked weights.  They either return
    the auditable uniform vector or raise, according to ``failure_policy``.
    """

    normalized_source = validate_prediction_source(source)
    normalized_failure_policy = _validate_failure_policy(failure_policy)
    stacked, target_array = _prepare_arrays(predictions, target)
    model_count = int(stacked.shape[0])
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(min_gain_over_uniform) or min_gain_over_uniform < 0.0:
        raise ValueError("min_gain_over_uniform must be finite and nonnegative")
    if not math.isfinite(uniform_shrinkage) or not 0.0 <= uniform_shrinkage <= 1.0:
        raise ValueError("uniform_shrinkage must be in [0, 1]")
    metric = score_function or _default_score
    uniform = np.full(model_count, 1.0 / model_count, dtype=np.float64)
    uniform_score = _score(target_array, stacked, uniform, metric)
    if model_count == 1:
        return EnsembleOptimizationResult(
            weights=(1.0,),
            score=uniform_score,
            uniform_score=uniform_score,
            source=normalized_source,
            evaluations=1,
            used_learned_weights=False,
            optimizer="slsqp",
        )
    initial = (
        uniform.copy()
        if initial_weights is None
        else np.asarray(
            normalize_nonnegative_weights(initial_weights, model_count), dtype=np.float64
        )
    )

    def objective(weights: np.ndarray) -> float:
        return -_score(target_array, stacked, weights, metric)

    try:
        solver_result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * model_count,
            constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
            options={"ftol": tolerance, "maxiter": max_iterations, "disp": False},
        )
    except Exception as error:
        return _slsqp_failure(
            reason=f"exception:{type(error).__name__}:{error}",
            failure_policy=normalized_failure_policy,
            model_count=model_count,
            uniform_score=uniform_score,
            source=normalized_source,
            evaluations=1,
        )
    evaluations = int(getattr(solver_result, "nfev", 0)) + 2
    if not bool(solver_result.success):
        return _slsqp_failure(
            reason=f"solver:{solver_result.message}",
            failure_policy=normalized_failure_policy,
            model_count=model_count,
            uniform_score=uniform_score,
            source=normalized_source,
            evaluations=evaluations,
        )
    raw_weights = np.asarray(solver_result.x, dtype=np.float64)
    if (
        raw_weights.shape != (model_count,)
        or not np.isfinite(raw_weights).all()
        or np.any(raw_weights < -1e-8)
        or np.any(raw_weights > 1.0 + 1e-8)
        or not math.isclose(float(raw_weights.sum()), 1.0, abs_tol=1e-6)
    ):
        return _slsqp_failure(
            reason="solver_returned_invalid_simplex",
            failure_policy=normalized_failure_policy,
            model_count=model_count,
            uniform_score=uniform_score,
            source=normalized_source,
            evaluations=evaluations,
        )
    current = np.asarray(
        normalize_nonnegative_weights(np.clip(raw_weights, 0.0, 1.0), model_count),
        dtype=np.float64,
    )
    if uniform_shrinkage > 0.0:
        current = (1.0 - uniform_shrinkage) * current + uniform_shrinkage * uniform
        current = np.asarray(normalize_nonnegative_weights(current, model_count))
    current_score = _score(target_array, stacked, current, metric)
    if current_score < uniform_score + min_gain_over_uniform:
        current = uniform
        current_score = uniform_score
    normalized_result = tuple(normalize_nonnegative_weights(current, model_count))
    return EnsembleOptimizationResult(
        weights=normalized_result,
        score=current_score,
        uniform_score=uniform_score,
        source=normalized_source,
        evaluations=evaluations,
        used_learned_weights=not np.allclose(current, uniform, atol=1e-12),
        optimizer="slsqp",
    )


def _fit_weights(
    predictions: Sequence[Any],
    target: Any,
    *,
    optimizer: str,
    source: str,
    score_function: ScoreFunction | None,
    initial_weights: Sequence[float] | None,
    step_sizes: Sequence[float],
    max_passes_per_step: int,
    min_improvement: float,
    min_gain_over_uniform: float,
    uniform_shrinkage: float,
    optimizer_failure_policy: str,
) -> EnsembleOptimizationResult:
    if optimizer == "coordinate":
        return coordinate_search_weights(
            predictions,
            target,
            source=source,
            score_function=score_function,
            initial_weights=initial_weights,
            step_sizes=step_sizes,
            max_passes_per_step=max_passes_per_step,
            min_improvement=min_improvement,
            min_gain_over_uniform=min_gain_over_uniform,
            uniform_shrinkage=uniform_shrinkage,
        )
    return slsqp_weights(
        predictions,
        target,
        source=source,
        score_function=score_function,
        initial_weights=initial_weights,
        max_iterations=max_passes_per_step,
        min_gain_over_uniform=min_gain_over_uniform,
        uniform_shrinkage=uniform_shrinkage,
        failure_policy=optimizer_failure_policy,
    )


def _verify_in_memory_arrays(
    predictions: Sequence[Any],
    target: Any,
    provenance: EnsembleInputProvenance,
) -> None:
    if len(predictions) != len(provenance.predictions):
        raise ValueError("Prediction count no longer matches verified sidecar provenance")
    pairs = [*zip(predictions, provenance.predictions, strict=True), (target, provenance.target)]
    for array, record in pairs:
        actual = np.asarray(array)
        if tuple(actual.shape) != record.array_shape or str(actual.dtype) != record.array_dtype:
            raise ValueError(f"In-memory array no longer matches {record.sidecar_path}")
        if _array_content_sha256(actual) != record.content_sha256:
            raise ValueError(f"In-memory array content changed after verifying {record.sidecar_path}")


def _cross_validate_oof_weights(
    stacked: np.ndarray,
    target: np.ndarray,
    provenance: EnsembleInputProvenance,
    *,
    score_function: ScoreFunction,
    min_improvement: float,
    uniform_shrinkage: float,
    step_sizes: Sequence[float],
    max_passes_per_step: int,
    optimizer: str,
    optimizer_failure_policy: str,
) -> tuple[tuple[CrossValidationFoldResult, ...], float, float, int]:
    fold_ids = np.asarray(provenance.fold_ids, dtype=np.int64)
    group_ids = np.asarray(provenance.group_ids, dtype=object)
    model_count = int(stacked.shape[0])
    uniform = np.full(model_count, 1.0 / model_count, dtype=np.float64)
    results: list[CrossValidationFoldResult] = []
    weighted_score = 0.0
    weighted_uniform_score = 0.0
    total_samples = 0
    evaluations = 0
    for fold_id in range(provenance.fold_count):
        held_out = fold_ids == fold_id
        training = ~held_out
        held_count = int(np.count_nonzero(held_out))
        train_count = int(np.count_nonzero(training))
        if held_count < 1 or train_count < 1:
            raise ValueError(f"OOF fold {fold_id} lacks training or held-out samples")
        fitted = _fit_weights(
            [item[training] for item in stacked],
            target[training],
            optimizer=optimizer,
            source="oof",
            score_function=score_function,
            initial_weights=None,
            step_sizes=step_sizes,
            max_passes_per_step=max_passes_per_step,
            min_improvement=min_improvement,
            min_gain_over_uniform=0.0,
            uniform_shrinkage=uniform_shrinkage,
            optimizer_failure_policy=optimizer_failure_policy,
        )
        held_stacked = stacked[:, held_out]
        held_target = target[held_out]
        held_score = _score(
            held_target, held_stacked, np.asarray(fitted.weights), score_function
        )
        held_uniform = _score(held_target, held_stacked, uniform, score_function)
        results.append(
            CrossValidationFoldResult(
                fold_id=fold_id,
                train_samples=train_count,
                held_out_samples=held_count,
                held_out_groups=len(set(group_ids[held_out].tolist())),
                weights=fitted.weights,
                score=held_score,
                uniform_score=held_uniform,
                evaluations=fitted.evaluations + 2,
                optimizer=fitted.optimizer,
                fallback_reason=fitted.fallback_reason,
            )
        )
        weighted_score += held_score * held_count
        weighted_uniform_score += held_uniform * held_count
        total_samples += held_count
        evaluations += fitted.evaluations + 2
    return (
        tuple(results),
        weighted_score / total_samples,
        weighted_uniform_score / total_samples,
        evaluations,
    )


def optimize_ensemble_weights(
    predictions: Sequence[Any],
    target: Any,
    *,
    source: str,
    provenance: EnsembleInputProvenance | None = None,
    cross_validate_weights: bool = False,
    optimizer: str = "coordinate",
    optimizer_failure_policy: str = "uniform",
    score_function: ScoreFunction | None = None,
    initial_weights: Sequence[float] | None = None,
    step_sizes: Sequence[float] = (0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001),
    max_passes_per_step: int = 100,
    min_improvement: float = 1e-12,
    min_gain_over_uniform: float = 0.0,
    uniform_shrinkage: float = 0.0,
) -> EnsembleOptimizationResult:
    """Fit weights only after sidecars establish validation/OOF provenance.

    ``source`` is an assertion checked against the sidecars, never the authority for
    deciding whether the arrays are safe.  With ``cross_validate_weights`` enabled,
    grouped OOF folds are used for leave-one-fold-out weight fitting and the learned
    global vector is retained only when held-out score clears the uniform guard.
    """

    if provenance is None:
        raise ValueError(
            "Verified sidecar provenance is required; a source string alone is insufficient"
        )
    normalized_source = validate_prediction_source(source)
    if normalized_source != provenance.source:
        raise ValueError(
            f"Source assertion {normalized_source!r} does not match sidecar provenance "
            f"{provenance.source!r}"
        )
    _verify_in_memory_arrays(predictions, target, provenance)
    if not isinstance(cross_validate_weights, bool):
        raise TypeError("cross_validate_weights must be boolean")
    if cross_validate_weights and provenance.source != "oof":
        raise ValueError("cross_validate_weights=true requires grouped OOF sidecars")
    normalized_optimizer = validate_optimizer_name(optimizer)
    normalized_failure_policy = _validate_failure_policy(optimizer_failure_policy)

    result = _fit_weights(
        predictions,
        target,
        optimizer=normalized_optimizer,
        source=provenance.source,
        score_function=score_function,
        initial_weights=initial_weights,
        step_sizes=step_sizes,
        max_passes_per_step=max_passes_per_step,
        min_improvement=min_improvement,
        min_gain_over_uniform=min_gain_over_uniform,
        uniform_shrinkage=uniform_shrinkage,
        optimizer_failure_policy=normalized_failure_policy,
    )
    if not cross_validate_weights:
        return result

    stacked, target_array = _prepare_arrays(predictions, target)
    metric = score_function or _default_score
    folds, cv_score, cv_uniform_score, cv_evaluations = _cross_validate_oof_weights(
        stacked,
        target_array,
        provenance,
        score_function=metric,
        min_improvement=min_improvement,
        uniform_shrinkage=uniform_shrinkage,
        step_sizes=step_sizes,
        max_passes_per_step=max_passes_per_step,
        optimizer=normalized_optimizer,
        optimizer_failure_policy=normalized_failure_policy,
    )
    cv_solver_fallback = next(
        (fold.fallback_reason for fold in folds if fold.fallback_reason is not None),
        None,
    )
    passes_cv_guard = (
        cv_solver_fallback is None
        and cv_score >= cv_uniform_score + min_gain_over_uniform
    )
    if not passes_cv_guard:
        uniform = tuple(1.0 / len(predictions) for _ in predictions)
        result = replace(
            result,
            weights=uniform,
            score=result.uniform_score,
            used_learned_weights=False,
            fallback_reason=(
                f"cross_validation:{cv_solver_fallback}"
                if cv_solver_fallback is not None
                else result.fallback_reason
            ),
        )
    return replace(
        result,
        evaluations=result.evaluations + cv_evaluations,
        cross_validated=True,
        cross_validated_score=cv_score,
        cross_validated_uniform_score=cv_uniform_score,
        fold_results=folds,
    )
