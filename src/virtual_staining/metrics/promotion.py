"""Strict ROI-grouped final-JPEG promotion decisions for Performance V2.

The gate is intentionally independent of training and inference.  It consumes
immutable per-image validation records plus explicit audit/provenance evidence,
and never infers missing ROI safety facts.  A report can therefore be produced
for a blocked experiment without accidentally making it eligible for the final
default configuration.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .domain_metrics import roi_bootstrap_difference

JPG_METRICS = ("jpg_ssim", "jpg_psnr")
DEFAULT_STRATA = ("organ", "border_class", "activity_bin")
EvidenceIdentity = tuple[str, ...]


def _unique_reasons(reasons: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(reason) for reason in reasons if str(reason)))


def _audit_mapping(audit: Any) -> Mapping[str, Any]:
    if isinstance(audit, Mapping):
        return audit
    to_dict = getattr(audit, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError("roi_audit must be a mapping or expose to_dict()")


def _audit_reasons(roi_audit: Any) -> list[str]:
    audit = _audit_mapping(roi_audit)
    reasons: list[str] = []
    total_rows = int(audit.get("total_rows", 0))
    parsed_rows = int(audit.get("parsed_rows", 0))
    if total_rows <= 0:
        reasons.append("empty_roi_audit")
    if parsed_rows != total_rows:
        reasons.append("incomplete_filename_coordinates")
    if audit.get("filename_grid_verified") is not True:
        reasons.append("unverified_filename_grid")

    leakage_fields = {
        "duplicate_coordinates": "duplicate_coordinates",
        "train_val_shared_rois": "train_val_roi_leakage",
        "cross_split_adjacent_pairs": "train_val_neighborhood_leakage",
    }
    for field, failure in leakage_fields.items():
        if field not in audit:
            reasons.append(f"missing_audit_evidence:{field}")
        elif audit[field]:
            reasons.append(failure)

    boundary = audit.get("boundary")
    if not isinstance(boundary, Mapping):
        reasons.append("missing_audit_evidence:boundary")
    else:
        if boundary.get("direction_verified") is not True:
            reasons.append("coordinate_direction_not_verified")
        if boundary.get("continuity_verified") is not True:
            reasons.append("boundary_continuity_not_verified")

    gate_reasons = audit.get("context_gate_reasons")
    if gate_reasons is None:
        reasons.append("missing_audit_evidence:context_gate_reasons")
    elif isinstance(gate_reasons, Sequence) and not isinstance(gate_reasons, (str, bytes)):
        reasons.extend(f"roi_audit:{reason}" for reason in gate_reasons if str(reason))
    else:
        reasons.append("invalid_audit_evidence:context_gate_reasons")
    if audit.get("context_enabled") is not True:
        reasons.append("roi_grid_gate_disabled")
    return _unique_reasons(reasons)


def _normalize_provenance(
    provenance: Iterable[Mapping[str, Any]] | Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[EvidenceIdentity, ...], list[str]]:
    if isinstance(provenance, Mapping):
        rows = [dict(provenance)]
    else:
        rows = [dict(item) for item in provenance]
    reasons: list[str] = []
    identities: list[EvidenceIdentity] = []
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        fold = row.get("fold")
        seed = row.get("seed")
        evidence_id = str(row.get("evidence_id", "")).strip()
        fold_available = fold is not None and str(fold).strip() != ""
        seed_available = seed is not None and str(seed).strip() != ""
        identity: EvidenceIdentity | None
        if evidence_id:
            identity = ("evidence_id", evidence_id)
        elif fold_available and seed_available:
            identity = ("fold_seed", str(fold), str(seed))
        else:
            identity = None
            reasons.append(f"missing_provenance_evidence_identity:{index}")
        if fold_available != seed_available:
            reasons.append(f"incomplete_provenance_fold_seed:{index}")
        if identity is not None:
            if identity in identities:
                reasons.append(f"duplicate_provenance_identity:{_format_evidence(identity)}")
            identities.append(identity)

        parent_manifest = row.get("parent_manifest_sha256")
        candidate_manifest = row.get("candidate_manifest_sha256")
        evidence_label = _format_evidence(identity) if identity is not None else str(index)
        if parent_manifest is None or candidate_manifest is None:
            reasons.append(f"missing_manifest_provenance:{evidence_label}")
        elif str(parent_manifest) != str(candidate_manifest):
            reasons.append(f"validation_manifest_mismatch:{evidence_label}")
        normalized.append(
            {
                "evidence_identity": evidence_label,
                "evidence_id": evidence_id or None,
                "fold": fold if fold_available else None,
                "seed": seed if seed_available else None,
                "parent_manifest_sha256": (
                    str(parent_manifest) if parent_manifest is not None else None
                ),
                "candidate_manifest_sha256": (
                    str(candidate_manifest) if candidate_manifest is not None else None
                ),
            }
        )

    unique_identities = tuple(sorted(set(identities)))
    if len(unique_identities) < 2:
        reasons.append("insufficient_independent_fold_seed_evidence")
    by_identity = {
        str(row["evidence_identity"]): row
        for row in normalized
        if row["evidence_identity"] != ""
    }
    summary = [by_identity[_format_evidence(identity)] for identity in unique_identities]
    return summary, unique_identities, _unique_reasons(reasons)


def _format_evidence(identity: EvidenceIdentity) -> str:
    if identity[0] == "evidence_id":
        return f"evidence_id={identity[1]}"
    return f"fold={identity[1]},seed={identity[2]}"


def _record_evidence_identity(
    row: Mapping[str, Any],
    *,
    expected_evidence: tuple[EvidenceIdentity, ...],
    require_explicit: bool,
) -> EvidenceIdentity:
    evidence_id = str(row.get("evidence_id", "")).strip()
    fold = row.get("fold")
    seed = row.get("seed")
    fold_available = fold is not None and str(fold).strip() != ""
    seed_available = seed is not None and str(seed).strip() != ""
    if evidence_id:
        return ("evidence_id", evidence_id)
    if fold_available and seed_available:
        return ("fold_seed", str(fold), str(seed))
    if fold_available != seed_available:
        raise ValueError("incomplete_record_fold_seed")
    if not require_explicit and len(expected_evidence) == 1:
        return expected_evidence[0]
    raise ValueError("missing_record_evidence_binding")


def _record_identity(
    row: Mapping[str, Any], evidence: EvidenceIdentity
) -> tuple[EvidenceIdentity, str, str]:
    canonical_key = str(row.get("canonical_key", "")).strip()
    if not canonical_key:
        raise ValueError("missing_canonical_key")
    return evidence, str(row.get("target", "output")), canonical_key


def _index_records(
    records: Iterable[Mapping[str, Any]],
    label: str,
    *,
    expected_evidence: tuple[EvidenceIdentity, ...],
    require_explicit: bool,
) -> tuple[
    dict[tuple[EvidenceIdentity, str, str], dict[str, Any]],
    set[EvidenceIdentity],
    list[str],
]:
    indexed: dict[tuple[EvidenceIdentity, str, str], dict[str, Any]] = {}
    observed_evidence: set[EvidenceIdentity] = set()
    reasons: list[str] = []
    for index, source in enumerate(records):
        row = dict(source)
        try:
            evidence = _record_evidence_identity(
                row,
                expected_evidence=expected_evidence,
                require_explicit=require_explicit,
            )
            identity = _record_identity(row, evidence)
        except ValueError as error:
            reasons.append(f"{label}_{error.args[0]}:{index}")
            continue
        observed_evidence.add(evidence)
        if identity in indexed:
            reasons.append(
                f"duplicate_{label}_record:{_format_evidence(identity[0])}:"
                f"{identity[1]}:{identity[2]}"
            )
            continue
        roi_id = str(row.get("roi_id", "")).strip()
        if not roi_id:
            reasons.append(
                f"missing_{label}_roi_id:{_format_evidence(identity[0])}:"
                f"{identity[1]}:{identity[2]}"
            )
        for metric in JPG_METRICS:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                reasons.append(
                    f"invalid_{label}_{metric}:{_format_evidence(identity[0])}:"
                    f"{identity[1]}:{identity[2]}"
                )
                continue
            if not np.isfinite(value):
                reasons.append(
                    f"nonfinite_{label}_{metric}:{_format_evidence(identity[0])}:"
                    f"{identity[1]}:{identity[2]}"
                )
        indexed[identity] = row
    if not indexed:
        reasons.append(f"empty_{label}_records")
    expected = set(expected_evidence)
    for evidence in sorted(expected - observed_evidence):
        reasons.append(f"{label}_records_missing_evidence:{_format_evidence(evidence)}")
    for evidence in sorted(observed_evidence - expected):
        reasons.append(f"{label}_records_undeclared_evidence:{_format_evidence(evidence)}")
    return indexed, observed_evidence, _unique_reasons(reasons)


def _paired_records(
    parent_records: Iterable[Mapping[str, Any]],
    candidate_records: Iterable[Mapping[str, Any]],
    strata_keys: Sequence[str],
    expected_evidence: tuple[EvidenceIdentity, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    require_explicit = len(expected_evidence) >= 2
    parent, _, parent_reasons = _index_records(
        parent_records,
        "parent",
        expected_evidence=expected_evidence,
        require_explicit=require_explicit,
    )
    candidate, _, candidate_reasons = _index_records(
        candidate_records,
        "candidate",
        expected_evidence=expected_evidence,
        require_explicit=require_explicit,
    )
    reasons = [*parent_reasons, *candidate_reasons]
    parent_keys = set(parent)
    candidate_keys = set(candidate)
    if parent_keys != candidate_keys:
        missing_candidate = sorted(parent_keys - candidate_keys)
        missing_parent = sorted(candidate_keys - parent_keys)
        if missing_candidate:
            reasons.append(f"unpaired_records_missing_candidate:{len(missing_candidate)}")
        if missing_parent:
            reasons.append(f"unpaired_records_missing_parent:{len(missing_parent)}")
    keys = sorted(parent_keys & candidate_keys)
    paired_parent: list[dict[str, Any]] = []
    paired_candidate: list[dict[str, Any]] = []
    for identity in keys:
        parent_row = dict(parent[identity])
        candidate_row = dict(candidate[identity])
        parent_roi = str(parent_row.get("roi_id", "")).strip()
        candidate_roi = str(candidate_row.get("roi_id", "")).strip()
        if parent_roi != candidate_roi:
            reasons.append(
                f"paired_roi_mismatch:{_format_evidence(identity[0])}:"
                f"{identity[1]}:{identity[2]}"
            )
        for stratum in strata_keys:
            if stratum not in parent_row or stratum not in candidate_row:
                reasons.append(f"missing_stratum:{stratum}")
                continue
            parent_label = str(parent_row[stratum]).strip()
            candidate_label = str(candidate_row[stratum]).strip()
            if not parent_label or parent_label.casefold() in {"unknown", "unavailable"}:
                reasons.append(f"unresolved_parent_stratum:{stratum}")
            if not candidate_label or candidate_label.casefold() in {"unknown", "unavailable"}:
                reasons.append(f"unresolved_candidate_stratum:{stratum}")
            if parent_label != candidate_label:
                reasons.append(
                    f"stratum_label_mismatch:{stratum}:{_format_evidence(identity[0])}:"
                    f"{identity[2]}"
                )
        evidence_label = _format_evidence(identity[0])
        parent_row["_promotion_evidence"] = evidence_label
        candidate_row["_promotion_evidence"] = evidence_label
        parent_row["_promotion_roi_group"] = f"{evidence_label}|roi={parent_roi}"
        candidate_row["_promotion_roi_group"] = f"{evidence_label}|roi={candidate_roi}"
        paired_parent.append(parent_row)
        paired_candidate.append(candidate_row)
    if not paired_parent:
        reasons.append("no_paired_records")
    return paired_parent, paired_candidate, _unique_reasons(reasons)


def _bootstrap_metric(
    parent: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    result = roi_bootstrap_difference(
        candidate,
        parent,
        metric=metric,
        group_key="_promotion_roi_group",
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
        tie_tolerance=tie_tolerance,
    )
    result["significant_improvement"] = bool(float(result["ci_low"]) > 0.0)
    result["significant_decline"] = bool(float(result["ci_high"]) < 0.0)
    result["no_significant_decline"] = not result["significant_decline"]
    return result


def _stratified_differences(
    parent: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    strata_keys: Sequence[str],
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    tie_tolerance: float,
) -> tuple[dict[str, Any], list[str]]:
    result: dict[str, Any] = {}
    reasons: list[str] = []
    for stratum_index, stratum in enumerate(strata_keys):
        parent_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        candidate_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for parent_row, candidate_row in zip(parent, candidate, strict=True):
            if stratum not in parent_row or stratum not in candidate_row:
                continue
            parent_groups[str(parent_row[stratum])].append(parent_row)
            candidate_groups[str(candidate_row[stratum])].append(candidate_row)
        stratum_result: dict[str, Any] = {}
        for label in sorted(set(parent_groups) & set(candidate_groups)):
            metric_result: dict[str, Any] = {}
            for metric_index, metric in enumerate(JPG_METRICS):
                summary = _bootstrap_metric(
                    parent_groups[label],
                    candidate_groups[label],
                    metric,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    seed=seed + 1009 * (stratum_index + 1) + 37 * metric_index,
                    tie_tolerance=tie_tolerance,
                )
                metric_result[metric] = summary
                if summary["significant_decline"]:
                    reasons.append(f"stratum_significant_decline:{stratum}:{label}:{metric}")
            stratum_result[label] = {
                "paired_count": len(parent_groups[label]),
                "metrics": metric_result,
            }
        result[stratum] = stratum_result
    return result, _unique_reasons(reasons)


def evaluate_roi_jpg_promotion(
    parent_records: Iterable[Mapping[str, Any]],
    candidate_records: Iterable[Mapping[str, Any]],
    *,
    roi_audit: Any,
    fold_seed_provenance: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    strata_keys: Sequence[str] = DEFAULT_STRATA,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 2026,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Evaluate the non-negotiable final-default Performance V2 promotion gate.

    Eligibility requires a verified filename coordinate grid, explicit proof of
    no ROI/adjacent-patch leakage, exact paired records, at least two distinct
    ``(fold, seed)`` evidence points, one significantly improved final-JPEG
    metric, no significantly declining final-JPEG metric, and no significant
    decline in any declared validation stratum.  With multiple evidence points,
    every record must carry the declared ``fold``/``seed`` pair or an explicit
    ``evidence_id``; provenance declarations alone never multiply evidence.
    """

    if int(bootstrap_samples) < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if float(tie_tolerance) < 0.0:
        raise ValueError("tie_tolerance must be nonnegative")
    normalized_strata = tuple(dict.fromkeys(str(value) for value in strata_keys))
    if not normalized_strata:
        raise ValueError("At least one stratum is required for promotion")

    audit_reasons = _audit_reasons(roi_audit)
    provenance_summary, evidence_identities, provenance_reasons = _normalize_provenance(
        fold_seed_provenance
    )
    parent, candidate, record_reasons = _paired_records(
        parent_records,
        candidate_records,
        normalized_strata,
        evidence_identities,
    )
    reasons = [*audit_reasons, *provenance_reasons, *record_reasons]
    metric_results: dict[str, Any] = {}
    stratified: dict[str, Any] = {}
    metric_reasons: list[str] = []
    stratified_reasons: list[str] = []

    record_evidence_valid = not record_reasons and bool(parent)
    if record_evidence_valid:
        try:
            for index, metric in enumerate(JPG_METRICS):
                metric_results[metric] = _bootstrap_metric(
                    parent,
                    candidate,
                    metric,
                    bootstrap_samples=int(bootstrap_samples),
                    confidence=float(confidence),
                    seed=int(seed) + index,
                    tie_tolerance=float(tie_tolerance),
                )
            stratified, stratified_reasons = _stratified_differences(
                parent,
                candidate,
                normalized_strata,
                bootstrap_samples=int(bootstrap_samples),
                confidence=float(confidence),
                seed=int(seed),
                tie_tolerance=float(tie_tolerance),
            )
        except ValueError as error:
            metric_reasons.append(f"bootstrap_failed:{error}")

    if metric_results:
        improvements = [
            metric for metric, summary in metric_results.items() if summary["significant_improvement"]
        ]
        declines = [
            metric for metric, summary in metric_results.items() if summary["significant_decline"]
        ]
        if not improvements:
            metric_reasons.append("no_significant_final_jpg_improvement")
        metric_reasons.extend(f"significant_final_jpg_decline:{metric}" for metric in declines)
    else:
        improvements = []
        declines = []
        if not metric_reasons:
            metric_reasons.append("jpg_metric_evidence_unavailable")

    reasons = _unique_reasons([*reasons, *metric_reasons, *stratified_reasons])
    roi_ids = {
        str(row.get("_promotion_roi_group", ""))
        for row in parent
        if row.get("_promotion_roi_group")
    }
    promotable = not reasons
    return {
        "schema_version": 1,
        "decision": "promote" if promotable else "blocked",
        "promotable": promotable,
        "final_default_eligible": promotable,
        "primary_domain": "jpg",
        "direction": "candidate_minus_parent",
        "paired_record_count": len(parent) if record_evidence_valid else 0,
        "paired_roi_count": len(roi_ids) if record_evidence_valid else 0,
        "independent_evidence_count": len(provenance_summary),
        "independent_fold_seed_count": len(provenance_summary),
        "fold_seed_provenance": provenance_summary,
        "criteria": {
            "one_metric_ci_low_above_zero": bool(improvements),
            "no_metric_ci_high_below_zero": not declines and bool(metric_results),
            "no_significant_stratum_decline": not stratified_reasons and bool(stratified),
            "minimum_independent_fold_seed_count": 2,
            "confidence": float(confidence),
            "bootstrap_samples": int(bootstrap_samples),
        },
        "jpg_metrics": metric_results,
        "stratified": stratified,
        "reasons": reasons,
    }


def write_promotion_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically persist a promotion report as UTF-8 JSON."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    payload = json.dumps(dict(report), indent=2, ensure_ascii=False, allow_nan=False)
    temporary.write_text(f"{payload}\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
