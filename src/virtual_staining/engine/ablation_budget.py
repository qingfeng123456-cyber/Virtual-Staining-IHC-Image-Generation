"""Deterministic ablation-budget and independent-evidence planning.

This module only resolves experiment intent.  It does not start training,
inspect images, or infer whether a fold is safe.  Callers remain responsible
for constructing authoritative grouped folds and for using the returned
validation mode exactly as declared.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SUPPORTED_BUDGETS = ("smoke", "screen", "confirm", "full")
INNER_FOLD_VALIDATION = "grouped_inner_fold"
OFFICIAL_OUTER_VALIDATION = "official_outer_validation"


@dataclass(frozen=True, slots=True)
class EvidenceRun:
    """One independently identified fold/seed experiment."""

    fold: int | None
    seed: int
    suffix: str
    evidence_id: str
    validation_mode: str

    @property
    def record_fold(self) -> int | str:
        """Fold value used in validation records and promotion provenance."""

        return self.fold if self.fold is not None else "official_outer"

    def make_run_id(self, base_run_id: str) -> str:
        """Append the stable evidence suffix without consulting filesystem state."""

        normalized = str(base_run_id).strip()
        if not normalized:
            raise ValueError("base_run_id cannot be empty")
        return f"{normalized}{self.suffix}"

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "fold": self.fold,
            "seed": self.seed,
            "suffix": self.suffix,
            "evidence_id": self.evidence_id,
            "validation_mode": self.validation_mode,
            "record_fold": self.record_fold,
        }


@dataclass(frozen=True, slots=True)
class AblationBudgetPlan:
    """Resolved epoch budget and evidence grid for one ablation budget."""

    budget: str
    epochs: int
    validation_mode: str
    evidence_runs: tuple[EvidenceRun, ...]

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_runs)

    @property
    def uses_official_outer_validation(self) -> bool:
        return self.validation_mode == OFFICIAL_OUTER_VALIDATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "epochs": self.epochs,
            "validation_mode": self.validation_mode,
            "evidence_count": self.evidence_count,
            "evidence_runs": [run.to_dict() for run in self.evidence_runs],
        }


def _integer(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{field} must be {qualifier}")
    return parsed


def _integer_sequence(
    value: Any,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence of integers")
    parsed = tuple(_integer(item, field=field, minimum=0) for item in value)
    if not parsed and not allow_empty:
        raise ValueError(f"{field} cannot be empty")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{field} must contain unique values")
    return parsed


def _validation_mode(
    budget: str,
    entry: Mapping[str, Any],
    *,
    folds: tuple[int, ...],
) -> str:
    requested = str(entry.get("validation_mode", "")).strip().casefold()
    if requested and requested not in {INNER_FOLD_VALIDATION, OFFICIAL_OUTER_VALIDATION}:
        raise ValueError(
            "validation_mode must be grouped_inner_fold or official_outer_validation"
        )
    inferred = OFFICIAL_OUTER_VALIDATION if not folds else INNER_FOLD_VALIDATION
    mode = requested or inferred
    if mode == OFFICIAL_OUTER_VALIDATION:
        if budget != "full":
            raise ValueError("official_outer_validation is permitted only for the full budget")
        if folds:
            raise ValueError("official_outer_validation requires an empty folds list")
    elif not folds:
        raise ValueError("grouped_inner_fold validation requires at least one fold")
    return mode


def _evidence_run(fold: int | None, seed: int, validation_mode: str) -> EvidenceRun:
    if fold is None:
        suffix = f"_outer_seed{seed}"
        evidence_id = f"official_outer_seed{seed}"
    else:
        suffix = f"_fold{fold}_seed{seed}"
        evidence_id = f"fold{fold}_seed{seed}"
    return EvidenceRun(
        fold=fold,
        seed=seed,
        suffix=suffix,
        evidence_id=evidence_id,
        validation_mode=validation_mode,
    )


def resolve_ablation_budget(
    budget_mapping: Mapping[str, Any],
    budget: str,
    *,
    default_seed: int = 2026,
    default_fold: int = 0,
) -> AblationBudgetPlan:
    """Resolve a named budget into a deterministic fold-by-seed evidence grid.

    Missing fold/seed declarations produce one default evidence run for smoke
    and screen.  Confirm uses the same defaulting rule and then fails unless at
    least two independent fold/seed combinations were explicitly obtainable.
    An empty full-budget fold list denotes official outer validation.
    """

    normalized_budget = str(budget).strip().casefold()
    if normalized_budget not in SUPPORTED_BUDGETS:
        raise ValueError(f"Unsupported ablation budget: {budget}")
    if not isinstance(budget_mapping, Mapping):
        raise TypeError("budget_mapping must be a mapping")
    raw_entry = budget_mapping.get(normalized_budget)
    if not isinstance(raw_entry, Mapping):
        raise ValueError(f"Missing budget mapping for {normalized_budget}")
    entry = dict(raw_entry)
    epochs = _integer(entry.get("epochs"), field=f"{normalized_budget}.epochs", minimum=1)
    fallback_seed = _integer(default_seed, field="default_seed", minimum=0)
    fallback_fold = _integer(default_fold, field="default_fold", minimum=0)

    seeds = _integer_sequence(
        entry.get("seeds", [fallback_seed]),
        field=f"{normalized_budget}.seeds",
        allow_empty=False,
    )
    if "folds" in entry:
        folds = _integer_sequence(
            entry["folds"],
            field=f"{normalized_budget}.folds",
            allow_empty=normalized_budget == "full",
        )
    elif normalized_budget == "full":
        folds = ()
    else:
        folds = (fallback_fold,)
    mode = _validation_mode(normalized_budget, entry, folds=folds)

    if mode == OFFICIAL_OUTER_VALIDATION:
        evidence_runs = tuple(_evidence_run(None, seed, mode) for seed in seeds)
    else:
        evidence_runs = tuple(
            _evidence_run(fold, seed, mode) for fold in folds for seed in seeds
        )
    if not evidence_runs:
        raise ValueError(f"{normalized_budget} must resolve to at least one evidence run")
    evidence_ids = [run.evidence_id for run in evidence_runs]
    suffixes = [run.suffix for run in evidence_runs]
    if len(set(evidence_ids)) != len(evidence_ids) or len(set(suffixes)) != len(suffixes):
        raise ValueError(f"{normalized_budget} resolved duplicate evidence identities")
    if normalized_budget == "confirm" and len(evidence_runs) < 2:
        raise ValueError("confirm requires at least two independent fold/seed evidence runs")

    return AblationBudgetPlan(
        budget=normalized_budget,
        epochs=epochs,
        validation_mode=mode,
        evidence_runs=evidence_runs,
    )


def _validate_evidence_runs(evidence_runs: Iterable[EvidenceRun]) -> tuple[EvidenceRun, ...]:
    runs = tuple(evidence_runs)
    if not runs:
        raise ValueError("evidence_runs cannot be empty")
    identities = [run.evidence_id for run in runs]
    if any(not identity.strip() for identity in identities):
        raise ValueError("evidence_id cannot be empty")
    if len(set(identities)) != len(identities):
        raise ValueError("evidence_runs must contain unique evidence_id values")
    return runs


def bind_evidence_records(
    records: Iterable[Mapping[str, Any]],
    evidence_run: EvidenceRun,
) -> list[dict[str, Any]]:
    """Copy per-image records and bind them to one fold/seed evidence run.

    Existing nonempty evidence fields must already agree.  This prevents a
    validation record set from being relabeled as a different independent run.
    """

    source = [dict(record) for record in records]
    if not source:
        raise ValueError("validation records cannot be empty")
    expected: dict[str, int | str] = {
        "fold": evidence_run.record_fold,
        "seed": evidence_run.seed,
        "evidence_id": evidence_run.evidence_id,
    }
    bound: list[dict[str, Any]] = []
    for index, record in enumerate(source):
        for field, value in expected.items():
            existing = record.get(field)
            if existing is not None and str(existing).strip() and str(existing) != str(value):
                raise ValueError(f"record {index} has conflicting {field}")
            record[field] = value
        bound.append(record)
    return bound


def build_promotion_provenance(
    evidence_runs: Iterable[EvidenceRun],
    *,
    parent_validation_hashes: Mapping[str, str],
    candidate_validation_hashes: Mapping[str, str],
) -> list[dict[str, int | str]]:
    """Build strictly paired manifest provenance for promotion evaluation.

    Hash mappings are keyed by ``EvidenceRun.evidence_id``.  Parent and
    candidate must declare exactly the planned identities and must use the same
    nonempty validation-manifest hash for each identity.
    """

    runs = _validate_evidence_runs(evidence_runs)
    expected_ids = {run.evidence_id for run in runs}
    parent_ids = {str(key) for key in parent_validation_hashes}
    candidate_ids = {str(key) for key in candidate_validation_hashes}
    if parent_ids != expected_ids:
        raise ValueError("parent validation hashes do not match planned evidence ids")
    if candidate_ids != expected_ids:
        raise ValueError("candidate validation hashes do not match planned evidence ids")

    provenance: list[dict[str, int | str]] = []
    for run in runs:
        parent_hash = str(parent_validation_hashes[run.evidence_id]).strip()
        candidate_hash = str(candidate_validation_hashes[run.evidence_id]).strip()
        if not parent_hash or not candidate_hash:
            raise ValueError(f"validation hash cannot be empty for {run.evidence_id}")
        if parent_hash != candidate_hash:
            raise ValueError(f"validation hash mismatch for {run.evidence_id}")
        provenance.append(
            {
                "evidence_id": run.evidence_id,
                "fold": run.record_fold,
                "seed": run.seed,
                "validation_mode": run.validation_mode,
                "parent_manifest_sha256": parent_hash,
                "candidate_manifest_sha256": candidate_hash,
            }
        )
    return provenance
