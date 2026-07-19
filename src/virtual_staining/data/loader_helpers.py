"""Composable split and sampler plans for the training CLI.

The helpers in this module deliberately perform no image I/O.  Authoritative
folds consume an already completed ROI-grid audit, while activity sampling
consumes scalar values that were precomputed from training targets only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .grouped_folds import ROIGroupedFoldResult, build_authoritative_roi_folds
from .roi_index import ROIGridAudit
from .samplers import ActivityStratifiedSampler


def _source_indices(
    rows: Sequence[Mapping[str, Any]],
    source_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    resolved = tuple(range(len(rows))) if source_indices is None else tuple(source_indices)
    if len(resolved) != len(rows):
        raise ValueError("source_indices must have the same length as rows")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in resolved
    ):
        raise ValueError("source_indices must contain nonnegative integers")
    if len(set(resolved)) != len(resolved):
        raise ValueError("source_indices must be unique")
    return resolved


def _audit_mapping(audit: ROIGridAudit | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(audit, ROIGridAudit):
        return audit.to_dict()
    if not isinstance(audit, Mapping):
        raise TypeError("roi_audit must be an ROIGridAudit or mapping")
    return audit


def require_verified_filename_grid(
    roi_audit: ROIGridAudit | Mapping[str, Any],
    *,
    minimum_rows: int = 1,
) -> None:
    """Reject an audit that cannot authorize ROI-grouped model selection.

    Coordinate orientation is a separate context gate.  Grouped folds only
    require complete, unique filename coordinates and proof that the audited
    outer splits do not share an ROI or an adjacent patch.
    """

    audit = _audit_mapping(roi_audit)
    total_rows = int(audit.get("total_rows", 0))
    parsed_rows = int(audit.get("parsed_rows", 0))
    failures: list[str] = []
    if total_rows < int(minimum_rows):
        failures.append("audit_does_not_cover_requested_rows")
    if total_rows <= 0:
        failures.append("empty_audit")
    if parsed_rows != total_rows:
        failures.append("incomplete_filename_coordinates")
    if not bool(audit.get("filename_grid_verified", False)):
        failures.append("unverified_filename_coordinates")
    if audit.get("duplicate_coordinates"):
        failures.append("duplicate_coordinates")
    if audit.get("train_val_shared_rois"):
        failures.append("train_val_roi_overlap")
    if audit.get("cross_split_adjacent_pairs"):
        failures.append("cross_split_adjacent_patches")
    if failures:
        reasons = ", ".join(dict.fromkeys(failures))
        raise ValueError(
            "Authoritative grouped folds require a verified filename-coordinate "
            f"audit without outer-split leakage: {reasons}"
        )


@dataclass(frozen=True)
class AuthoritativeFoldSplit:
    """One deterministic inner-fold view over authoritative training rows."""

    fold: int
    source_indices: tuple[int, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_rows: tuple[Mapping[str, Any], ...]
    validation_rows: tuple[Mapping[str, Any], ...]
    assignments: ROIGroupedFoldResult
    assignment_sha256: str
    assignment_csv: Path | None = None
    assignment_hash_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return provenance without duplicating manifest row contents."""

        return {
            "fold": self.fold,
            "fold_count": self.assignments.fold_count,
            "seed": self.assignments.seed,
            "source_count": len(self.source_indices),
            "train_count": len(self.train_indices),
            "validation_count": len(self.validation_indices),
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "assignment_sha256": self.assignment_sha256,
            "assignment_csv": str(self.assignment_csv) if self.assignment_csv else None,
            "assignment_hash_file": (
                str(self.assignment_hash_file) if self.assignment_hash_file else None
            ),
        }


def _write_assignment_hash(csv_path: Path, digest: str) -> Path:
    destination = csv_path.with_suffix(f"{csv_path.suffix}.sha256")
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(f"{digest}  {csv_path.name}\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def prepare_authoritative_fold_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    fold: int,
    roi_audit: ROIGridAudit | Mapping[str, Any],
    fold_count: int = 5,
    seed: int = 2026,
    source_indices: Sequence[int] | None = None,
    output_csv: str | Path | None = None,
) -> AuthoritativeFoldSplit:
    """Build an official-train-only fold and optionally persist its contract."""

    row_values = tuple(rows)
    resolved_indices = _source_indices(row_values, source_indices)
    require_verified_filename_grid(roi_audit, minimum_rows=len(row_values))
    assignments = build_authoritative_roi_folds(
        row_values,
        fold_count=fold_count,
        seed=seed,
        source_indices=resolved_indices,
    )
    train_indices = assignments.training_indices(fold)
    validation_indices = assignments.validation_indices(fold)
    by_source_index = dict(zip(resolved_indices, row_values, strict=True))
    train_rows = tuple(by_source_index[index] for index in train_indices)
    validation_rows = tuple(by_source_index[index] for index in validation_indices)

    assignment_csv: Path | None = None
    assignment_hash_file: Path | None = None
    if output_csv is not None:
        assignment_csv = assignments.write_csv(output_csv)
        assignment_hash_file = _write_assignment_hash(
            assignment_csv, assignments.assignment_sha256
        )
    return AuthoritativeFoldSplit(
        fold=int(fold),
        source_indices=resolved_indices,
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_rows=train_rows,
        validation_rows=validation_rows,
        assignments=assignments,
        assignment_sha256=assignments.assignment_sha256,
        assignment_csv=assignment_csv,
        assignment_hash_file=assignment_hash_file,
    )


@dataclass(frozen=True)
class ActivitySamplingPlan:
    """Feature-flagged DataLoader sampler arguments and resumable state."""

    source_indices: tuple[int, ...]
    sampler: ActivityStratifiedSampler | None

    @property
    def enabled(self) -> bool:
        return self.sampler is not None

    @property
    def shuffle(self) -> bool:
        return self.sampler is None

    def dataloader_kwargs(self) -> dict[str, Any]:
        """Return mutually compatible ``shuffle`` and ``sampler`` arguments."""

        if self.sampler is None:
            return {"shuffle": True}
        return {"shuffle": False, "sampler": self.sampler}

    def set_epoch(self, epoch: int) -> None:
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source_indices": list(self.source_indices),
            "sampler": self.sampler.state_dict() if self.sampler is not None else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        enabled = bool(state.get("enabled", False))
        if enabled != self.enabled:
            raise ValueError("Activity sampler enabled state does not match checkpoint")
        saved_indices = tuple(int(value) for value in state.get("source_indices", ()))
        if saved_indices != self.source_indices:
            raise ValueError("Activity sampler source indices do not match checkpoint")
        sampler_state = state.get("sampler")
        if self.sampler is None:
            if sampler_state is not None:
                raise ValueError("Disabled activity sampler received sampler state")
            return
        if not isinstance(sampler_state, Mapping):
            raise ValueError("Enabled activity sampler checkpoint lacks sampler state")
        self.sampler.load_state_dict(sampler_state)


def prepare_activity_sampling(
    rows: Sequence[Mapping[str, Any]],
    *,
    enabled: bool = False,
    activity_key: str = "dapi_activity",
    split_key: str = "split",
    source_indices: Sequence[int] | None = None,
    num_bins: int = 4,
    seed: int = 2026,
    num_samples: int | None = None,
) -> ActivitySamplingPlan:
    """Create a train-only sampler from precomputed manifest scalar values."""

    row_values = tuple(rows)
    resolved_indices = _source_indices(row_values, source_indices)
    if not enabled:
        return ActivitySamplingPlan(source_indices=resolved_indices, sampler=None)
    if not str(activity_key).strip():
        raise ValueError("activity_key cannot be empty when activity sampling is enabled")
    sampler = ActivityStratifiedSampler.from_manifest(
        row_values,
        activity_key=str(activity_key),
        split_key=str(split_key),
        sample_indices=resolved_indices,
        num_bins=num_bins,
        seed=seed,
        num_samples=num_samples,
    )
    return ActivitySamplingPlan(source_indices=resolved_indices, sampler=sampler)
