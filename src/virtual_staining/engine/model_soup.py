"""Architecture-safe checkpoint averaging and validation-gated greedy soups.

Model soup operates in weight space.  It is deliberately separate from prediction
ensembling and accepts a candidate only after the supplied validation callback has
evaluated the complete tentative state dictionary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from virtual_staining.constants import normalize_marker
from virtual_staining.data.roi_index import (
    coordinate_value_from_row,
    parse_patch_coordinate,
)


class SoupCompatibilityError(ValueError):
    """Raised when state dictionaries cannot be averaged without ambiguity."""


class SoupValidationContractError(SoupCompatibilityError):
    """Raised when model selection is not backed by authoritative JPG evidence."""


_PROVENANCE_VERSION = 1
_VALIDATION_CONTRACT_VERSION = 1
_MODEL_SOUP_PROVENANCE_VERSION = 2
_AUTHORITATIVE_ROI_SOURCES = frozenset({"filename_regex", "filename_coordinate"})
_VALIDATION_SPLITS = frozenset(
    {"val", "valid", "validation", "official_val", "final_val"}
)
_JPG_METRIC_DOMAIN = "jpg_roundtrip"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unique_reasons(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


@dataclass(frozen=True)
class SoupValidationContract:
    """Immutable evidence binding for strict model-soup selection.

    The contract binds the complete validation manifest, ROI audit, and target-level
    per-image JPG records by exact SHA256.  An engineering override never converts
    this object into authoritative evidence: its unsafe fields are part of the
    contract hash and must be propagated to every derived soup checkpoint.
    """

    target_marker: str
    metric_domain: str
    primary_metric: str
    validation_manifest_path: str
    validation_manifest_sha256: str
    validation_sample_count: int
    validation_keys_sha256: str
    roi_audit_path: str
    roi_audit_sha256: str
    roi_audit_total_rows: int
    audited_manifest_row_count: int
    per_image_evidence_path: str
    per_image_evidence_sha256: str
    per_image_record_count: int
    audited_manifests: tuple[tuple[str, str], ...]
    evaluated_sample_count: int
    max_val_samples: int | None
    full_validation: bool
    filename_grid_verified: bool
    authoritative: bool
    unsafe_validation_override_used: bool
    unsafe_engineering_override_used: bool
    unsafe_reasons: tuple[str, ...]
    version: int = _VALIDATION_CONTRACT_VERSION

    def _payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "safety_mode": (
                "strict" if self.authoritative else "UNSAFE_ENGINEERING_VALIDATION"
            ),
            "target_marker": self.target_marker,
            "metric_domain": self.metric_domain,
            "primary_metric": self.primary_metric,
            "validation_manifest_path": self.validation_manifest_path,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            "validation_sample_count": self.validation_sample_count,
            "validation_keys_sha256": self.validation_keys_sha256,
            "roi_audit_path": self.roi_audit_path,
            "roi_audit_sha256": self.roi_audit_sha256,
            "roi_audit_total_rows": self.roi_audit_total_rows,
            "audited_manifest_row_count": self.audited_manifest_row_count,
            "per_image_evidence_path": self.per_image_evidence_path,
            "per_image_evidence_sha256": self.per_image_evidence_sha256,
            "per_image_record_count": self.per_image_record_count,
            "audited_manifests": [
                {"path": path, "sha256": digest}
                for path, digest in self.audited_manifests
            ],
            "evaluated_sample_count": self.evaluated_sample_count,
            "max_val_samples": self.max_val_samples,
            "full_validation": self.full_validation,
            "filename_grid_verified": self.filename_grid_verified,
            "authoritative": self.authoritative,
            "unsafe_validation_override_used": self.unsafe_validation_override_used,
            "unsafe_engineering_override_used": self.unsafe_engineering_override_used,
            "unsafe_reasons": list(self.unsafe_reasons),
        }

    @property
    def contract_sha256(self) -> str:
        return _json_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["contract_sha256"] = self.contract_sha256
        return payload


def _read_csv_rows(path: Path, *, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SoupValidationContractError(f"{label} has no CSV header: {path}")
        return [dict(row) for row in reader]


def _keys_sha256(keys: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(keys),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _strict_manifest_reasons(rows: Sequence[Mapping[str, str]]) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    keys = [str(row.get("canonical_key", "")).strip() for row in rows]
    if not rows:
        reasons.append("empty_validation_manifest")
    if any(not key for key in keys):
        reasons.append("missing_canonical_key")
    if len(set(keys)) != len(keys):
        reasons.append("duplicate_canonical_key")
    for index, row in enumerate(rows):
        split = str(row.get("split", "")).strip().casefold()
        source_split = str(row.get("source_split", "")).strip().casefold()
        if "test" in split or "test" in source_split:
            raise SoupValidationContractError(
                f"Validation manifest row {index} belongs to test data"
            )
        if split not in _VALIDATION_SPLITS:
            reasons.append(f"non_validation_split:{index}")
        if source_split not in _VALIDATION_SPLITS:
            reasons.append(f"non_validation_source_split:{index}")
        if str(row.get("is_smoke", "")).strip().casefold() in {
            "1",
            "true",
            "yes",
            "y",
        }:
            reasons.append(f"smoke_validation_row:{index}")
        scope = str(
            row.get("manifest_scope")
            or row.get("data_scope")
            or row.get("subset_reason")
            or ""
        ).strip().casefold()
        if scope in {"smoke", "tiny", "subset", "truncated", "debug"}:
            reasons.append(f"non_full_validation_scope:{index}")
        if str(row.get("is_truncated", "")).strip().casefold() in {
            "1",
            "true",
            "yes",
            "y",
        }:
            reasons.append(f"truncated_validation_row:{index}")
        source = str(row.get("roi_id_source", "")).strip().casefold()
        if source not in _AUTHORITATIVE_ROI_SOURCES:
            reasons.append(f"non_authoritative_roi_source:{index}")
        coordinate = parse_patch_coordinate(coordinate_value_from_row(row))
        if coordinate is None:
            reasons.append(f"unverified_filename_coordinate:{index}")
        elif str(row.get("roi_id", "")).strip().upper() != coordinate.roi_id:
            reasons.append(f"roi_id_coordinate_mismatch:{index}")
    return reasons, keys


def _strict_roi_audit_reasons(
    audit: Mapping[str, Any],
    *,
    validation_sample_count: int,
) -> list[str]:
    reasons: list[str] = []
    total_rows = int(audit.get("total_rows", 0))
    parsed_rows = int(audit.get("parsed_rows", 0))
    if total_rows < validation_sample_count:
        reasons.append("roi_audit_does_not_cover_validation_manifest")
    if total_rows <= 0:
        reasons.append("empty_roi_audit")
    if parsed_rows != total_rows:
        reasons.append("incomplete_roi_audit_coordinates")
    if audit.get("filename_grid_verified") is not True:
        reasons.append("unverified_filename_roi_grid")
    for field, reason in (
        ("duplicate_coordinates", "duplicate_roi_coordinates"),
        ("train_val_shared_rois", "train_val_roi_leakage"),
        ("cross_split_adjacent_pairs", "train_val_neighborhood_leakage"),
    ):
        if field not in audit:
            reasons.append(f"missing_roi_audit_field:{field}")
        elif audit[field]:
            reasons.append(reason)
    boundary = audit.get("boundary")
    if not isinstance(boundary, Mapping):
        reasons.append("missing_roi_boundary_audit")
    else:
        if boundary.get("direction_verified") is not True:
            reasons.append("roi_direction_not_verified")
        if boundary.get("continuity_verified") is not True:
            reasons.append("roi_boundary_continuity_not_verified")
    gate_reasons = audit.get("context_gate_reasons")
    if gate_reasons is None:
        reasons.append("missing_roi_context_gate_reasons")
    elif isinstance(gate_reasons, Sequence) and not isinstance(gate_reasons, (str, bytes)):
        reasons.extend(f"roi_audit:{reason}" for reason in gate_reasons if str(reason))
    else:
        reasons.append("invalid_roi_context_gate_reasons")
    if audit.get("context_enabled") is not True:
        reasons.append("roi_context_gate_disabled")
    return reasons


def _strict_per_image_reasons(
    rows: Sequence[Mapping[str, str]],
    *,
    target_marker: str,
    manifest_keys: Sequence[str],
) -> tuple[list[str], list[dict[str, str]]]:
    reasons: list[str] = []
    selected: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        raw_target = str(row.get("target", "")).strip()
        if not raw_target:
            reasons.append(f"missing_per_image_target:{index}")
            continue
        try:
            normalized = normalize_marker(raw_target)
        except ValueError:
            reasons.append(f"invalid_per_image_target:{index}")
            continue
        if normalized == target_marker:
            selected.append(dict(row))
    evidence_keys = [str(row.get("canonical_key", "")).strip() for row in selected]
    if not selected:
        reasons.append("empty_target_per_image_evidence")
    if any(not key for key in evidence_keys):
        reasons.append("missing_per_image_canonical_key")
    if len(set(evidence_keys)) != len(evidence_keys):
        reasons.append("duplicate_target_per_image_record")
    if set(evidence_keys) != set(manifest_keys) or len(evidence_keys) != len(manifest_keys):
        reasons.append("per_image_manifest_key_mismatch")
    for index, row in enumerate(selected):
        for field in ("jpg_ssim", "jpg_psnr"):
            try:
                value = float(row.get(field, ""))
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value):
                reasons.append(f"invalid_{field}:{index}")
    return reasons, selected


def build_soup_validation_contract(
    validation_manifest_path: str | Path,
    roi_audit_path: str | Path,
    per_image_evidence_path: str | Path,
    *,
    target_marker: str,
    evaluated_sample_count: int,
    metric_domain: str = _JPG_METRIC_DOMAIN,
    primary_metric: str = "local_proxy_score",
    max_val_samples: int | None = None,
    audited_manifest_paths: Sequence[str | Path] | None = None,
    allow_unsafe_engineering_override: bool = False,
) -> SoupValidationContract:
    """Build a strict, externally anchored soup-validation evidence contract.

    Strict mode requires a complete authoritative validation manifest, a verified
    filename-coordinate audit with no ROI/neighborhood leakage, one finite JPG metric
    record per target image, and no configured validation limit.  The explicit unsafe
    mode is intended only for engineering smoke runs and is permanently encoded.
    """

    manifest_path = Path(validation_manifest_path).expanduser().resolve()
    audit_path = Path(roi_audit_path).expanduser().resolve()
    evidence_path = Path(per_image_evidence_path).expanduser().resolve()
    manifest_rows = _read_csv_rows(manifest_path, label="Validation manifest")
    evidence_rows = _read_csv_rows(evidence_path, label="Per-image evidence")
    if not audit_path.is_file():
        raise FileNotFoundError(f"ROI audit does not exist: {audit_path}")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SoupValidationContractError(f"ROI audit is not valid JSON: {audit_path}") from error
    if not isinstance(audit, Mapping):
        raise SoupValidationContractError("ROI audit JSON must contain an object")
    try:
        marker = normalize_marker(target_marker)
    except ValueError as error:
        raise SoupValidationContractError(str(error)) from error
    if marker == "DAPI":
        raise SoupValidationContractError("DAPI cannot be a virtual-staining soup target")
    if isinstance(evaluated_sample_count, bool) or int(evaluated_sample_count) < 0:
        raise ValueError("evaluated_sample_count must be a nonnegative integer")
    evaluated_count = int(evaluated_sample_count)
    if max_val_samples is not None and (
        isinstance(max_val_samples, bool) or int(max_val_samples) < 1
    ):
        raise ValueError("max_val_samples must be None or a positive integer")
    resolved_limit = int(max_val_samples) if max_val_samples is not None else None

    reasons, manifest_keys = _strict_manifest_reasons(manifest_rows)
    reasons.extend(
        _strict_roi_audit_reasons(
            audit,
            validation_sample_count=len(manifest_rows),
        )
    )
    evidence_reasons, target_evidence = _strict_per_image_reasons(
        evidence_rows,
        target_marker=marker,
        manifest_keys=manifest_keys,
    )
    reasons.extend(evidence_reasons)
    normalized_domain = str(metric_domain).strip().casefold()
    if normalized_domain in {"jpg", "jpeg", _JPG_METRIC_DOMAIN}:
        normalized_domain = _JPG_METRIC_DOMAIN
    else:
        reasons.append("primary_metric_domain_is_not_jpg_roundtrip")
    if not str(primary_metric).strip():
        reasons.append("empty_primary_metric")
    if resolved_limit is not None:
        reasons.append("validation_manifest_was_configured_with_a_sample_limit")
    if evaluated_count != len(manifest_rows):
        reasons.append("evaluated_sample_count_mismatch")
    if len(target_evidence) != evaluated_count:
        reasons.append("per_image_record_count_mismatch")

    audit_manifest_values = tuple(audited_manifest_paths or (manifest_path,))
    audited: list[tuple[str, str]] = []
    audited_row_count = 0
    audited_keys: list[str] = []
    seen_audited_paths: set[Path] = set()
    for value in audit_manifest_values:
        resolved = Path(value).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Audited manifest does not exist: {resolved}")
        if resolved in seen_audited_paths:
            reasons.append("duplicate_audited_manifest_path")
            continue
        seen_audited_paths.add(resolved)
        audited_rows = _read_csv_rows(resolved, label="Audited manifest")
        audited_row_count += len(audited_rows)
        audited_keys.extend(
            str(row.get("canonical_key", "")).strip() for row in audited_rows
        )
        audited.append((str(resolved), _file_sha256(resolved)))
    manifest_digest = _file_sha256(manifest_path)
    if not any(
        Path(path) == manifest_path and digest == manifest_digest
        for path, digest in audited
    ):
        reasons.append("validation_manifest_not_bound_to_roi_audit_inputs")
    if any(not key for key in audited_keys):
        reasons.append("audited_manifest_missing_canonical_key")
    if len(set(audited_keys)) != len(audited_keys):
        reasons.append("audited_manifest_canonical_key_overlap")
    if int(audit.get("total_rows", 0)) != audited_row_count:
        reasons.append("roi_audit_manifest_row_count_mismatch")

    failures = _unique_reasons(reasons)
    if failures and not allow_unsafe_engineering_override:
        raise SoupValidationContractError(
            "Strict model-soup validation contract failed: " + ", ".join(failures)
        )
    if allow_unsafe_engineering_override and not failures:
        failures = ("explicit_unsafe_engineering_override",)
    full_validation = bool(
        resolved_limit is None
        and evaluated_count == len(manifest_rows)
        and len(target_evidence) == len(manifest_rows)
        and set(row.get("canonical_key", "").strip() for row in target_evidence)
        == set(manifest_keys)
    )
    authoritative = not failures and not allow_unsafe_engineering_override
    contract = SoupValidationContract(
        target_marker=marker,
        metric_domain=normalized_domain,
        primary_metric=str(primary_metric).strip(),
        validation_manifest_path=str(manifest_path),
        validation_manifest_sha256=manifest_digest,
        validation_sample_count=len(manifest_rows),
        validation_keys_sha256=_keys_sha256(manifest_keys),
        roi_audit_path=str(audit_path),
        roi_audit_sha256=_file_sha256(audit_path),
        roi_audit_total_rows=int(audit.get("total_rows", 0)),
        audited_manifest_row_count=audited_row_count,
        per_image_evidence_path=str(evidence_path),
        per_image_evidence_sha256=_file_sha256(evidence_path),
        per_image_record_count=len(target_evidence),
        audited_manifests=tuple(audited),
        evaluated_sample_count=evaluated_count,
        max_val_samples=resolved_limit,
        full_validation=full_validation,
        filename_grid_verified=audit.get("filename_grid_verified") is True,
        authoritative=authoritative,
        unsafe_validation_override_used=not authoritative,
        unsafe_engineering_override_used=bool(allow_unsafe_engineering_override),
        unsafe_reasons=failures,
    )
    validate_soup_validation_contract(
        contract,
        require_strict=not allow_unsafe_engineering_override,
        verify_files=True,
    )
    return contract


def validate_soup_validation_contract(
    contract: SoupValidationContract | Mapping[str, Any],
    *,
    require_strict: bool = True,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Validate a persisted contract and optionally re-hash its evidence files."""

    payload = contract.to_dict() if isinstance(contract, SoupValidationContract) else dict(contract)
    expected_contract_hash = _sha256_or_none(
        payload.pop("contract_sha256", None),
        field="validation_contract.contract_sha256",
        required=True,
    )
    if int(payload.get("version", -1)) != _VALIDATION_CONTRACT_VERSION:
        raise SoupValidationContractError("Unsupported soup validation contract version")
    if _json_sha256(payload) != expected_contract_hash:
        raise SoupValidationContractError("Soup validation contract hash is invalid")
    for field in (
        "validation_manifest_sha256",
        "validation_keys_sha256",
        "roi_audit_sha256",
        "per_image_evidence_sha256",
    ):
        _sha256_or_none(payload.get(field), field=f"validation_contract.{field}", required=True)
    sample_count = int(payload.get("validation_sample_count", 0))
    evidence_count = int(payload.get("per_image_record_count", -1))
    evaluated_count = int(payload.get("evaluated_sample_count", -1))
    if sample_count <= 0:
        raise SoupValidationContractError("Validation contract sample count must be positive")
    if require_strict:
        strict_failures: list[str] = []
        if str(payload.get("safety_mode", "")).casefold() != "strict":
            strict_failures.append("unsafe_safety_mode")
        if payload.get("authoritative") is not True:
            strict_failures.append("non_authoritative_validation")
        if payload.get("unsafe_validation_override_used") is True:
            strict_failures.append("unsafe_validation_override_used")
        if payload.get("unsafe_engineering_override_used") is True:
            strict_failures.append("unsafe_engineering_override_used")
        if payload.get("unsafe_reasons"):
            strict_failures.append("unsafe_validation_reasons_present")
        if payload.get("metric_domain") != _JPG_METRIC_DOMAIN:
            strict_failures.append("primary_metric_domain_is_not_jpg_roundtrip")
        if payload.get("full_validation") is not True or payload.get("max_val_samples") is not None:
            strict_failures.append("validation_is_truncated")
        if payload.get("filename_grid_verified") is not True:
            strict_failures.append("unverified_filename_roi_grid")
        if evidence_count != sample_count or evaluated_count != sample_count:
            strict_failures.append("validation_count_mismatch")
        if int(payload.get("roi_audit_total_rows", 0)) != int(
            payload.get("audited_manifest_row_count", -1)
        ):
            strict_failures.append("roi_audit_manifest_row_count_mismatch")
        audited = payload.get("audited_manifests")
        if not isinstance(audited, list) or not any(
            isinstance(item, Mapping)
            and item.get("path") == payload.get("validation_manifest_path")
            and item.get("sha256") == payload.get("validation_manifest_sha256")
            for item in audited
        ):
            strict_failures.append("validation_manifest_not_bound_to_roi_audit_inputs")
        if strict_failures:
            raise SoupValidationContractError(
                "Strict soup cannot use this validation contract: "
                + ", ".join(_unique_reasons(strict_failures))
            )
    if verify_files:
        for path_field, hash_field in (
            ("validation_manifest_path", "validation_manifest_sha256"),
            ("roi_audit_path", "roi_audit_sha256"),
            ("per_image_evidence_path", "per_image_evidence_sha256"),
        ):
            path = Path(str(payload.get(path_field, ""))).expanduser().resolve()
            if not path.is_file() or _file_sha256(path) != payload.get(hash_field):
                raise SoupValidationContractError(f"Evidence file hash mismatch: {path_field}")
        audited = payload.get("audited_manifests", [])
        if not isinstance(audited, list):
            raise SoupValidationContractError("audited_manifests must be a list")
        for item in audited:
            if not isinstance(item, Mapping):
                raise SoupValidationContractError("Invalid audited manifest record")
            path = Path(str(item.get("path", ""))).expanduser().resolve()
            if not path.is_file() or _file_sha256(path) != item.get("sha256"):
                raise SoupValidationContractError("Audited manifest file hash mismatch")
    payload["contract_sha256"] = expected_contract_hash
    return payload


def _unsafe_soup_provenance(provenance: Mapping[str, Any]) -> bool:
    unsafe_flags = (
        "unsafe_lineage_override_used",
        "unsafe_validation_override_used",
        "unsafe_engineering_override_used",
    )
    if any(bool(provenance.get(field, False)) for field in unsafe_flags):
        return True
    safety_mode = str(provenance.get("safety_mode", "strict")).strip().casefold()
    if safety_mode != "strict":
        return True
    contract = provenance.get("validation_contract")
    return isinstance(contract, Mapping) and (
        any(bool(contract.get(field, False)) for field in unsafe_flags)
        or str(contract.get("safety_mode", "strict")).strip().casefold() != "strict"
    )


def bind_soup_validation_contract(
    soup_provenance: Mapping[str, Any],
    contract: SoupValidationContract | Mapping[str, Any],
) -> dict[str, Any]:
    """Bind validation evidence while preserving every inherited unsafe marker."""

    normalized_contract = validate_soup_validation_contract(
        contract,
        require_strict=False,
        verify_files=False,
    )
    payload = dict(soup_provenance)
    inherited_unsafe = _unsafe_soup_provenance(payload)
    unsafe_validation = bool(
        payload.get("unsafe_validation_override_used", False)
        or normalized_contract.get("unsafe_validation_override_used", False)
    )
    unsafe_engineering = bool(
        payload.get("unsafe_engineering_override_used", False)
        or normalized_contract.get("unsafe_engineering_override_used", False)
    )
    unsafe_lineage = bool(payload.get("unsafe_lineage_override_used", False))
    any_unsafe = inherited_unsafe or unsafe_validation or unsafe_engineering or unsafe_lineage
    payload.update(
        {
            "version": _MODEL_SOUP_PROVENANCE_VERSION,
            "safety_mode": "UNSAFE_ENGINEERING_OVERRIDE" if any_unsafe else "strict",
            "unsafe_lineage_override_used": unsafe_lineage,
            "unsafe_validation_override_used": unsafe_validation,
            "unsafe_engineering_override_used": unsafe_engineering,
            "validation_contract": normalized_contract,
        }
    )
    return payload


@dataclass(frozen=True)
class SoupCheckpointProvenance:
    """Verified identity and initialization lineage for one checkpoint."""

    checkpoint_path: str
    checkpoint_sha256: str
    architecture_sha256: str
    initialization_lineage: str | None
    initial_state_sha256: str | None
    parent_checkpoint_sha256: str | None
    pretrain_checkpoint_sha256: str | None
    provenance_source: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "architecture_sha256": self.architecture_sha256,
            "initialization_lineage": self.initialization_lineage,
            "initial_state_sha256": self.initial_state_sha256,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "pretrain_checkpoint_sha256": self.pretrain_checkpoint_sha256,
            "provenance_source": self.provenance_source,
        }


@dataclass(frozen=True)
class SoupMember:
    """One architecture-compatible model-soup candidate.

    ``validation_score`` is used only to establish the deterministic trial order.
    Every accepted score is recomputed by ``greedy_model_soup`` through its callback.
    """

    name: str
    state_dict: Mapping[str, Tensor]
    validation_score: float
    weight_source: str = "raw"
    lineage: str | None = None
    checkpoint_sha256: str | None = None
    parent_checkpoint_sha256: str | None = None
    pretrain_checkpoint_sha256: str | None = None
    checkpoint_path: str | None = None
    architecture_sha256: str | None = None


@dataclass(frozen=True)
class SoupTrial:
    """Audit record for one greedy candidate evaluation."""

    member_name: str
    validation_score: float
    accepted: bool
    member_count: int


@dataclass(frozen=True)
class GreedySoupResult:
    """Final averaged state and the full validation decision trail."""

    state_dict: dict[str, Tensor]
    member_names: tuple[str, ...]
    validation_score: float
    trials: tuple[SoupTrial, ...]
    weight_source: str
    architecture_sha256: str
    common_initialization_lineage: str | None
    common_parent_checkpoint_sha256: str | None
    common_pretrain_checkpoint_sha256: str | None
    unsafe_lineage_override_used: bool


ValidationCallback = Callable[[Mapping[str, Tensor]], float]


def checkpoint_file_sha256(path: str | Path) -> str:
    """Hash the exact checkpoint bytes used as a soup input."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(value: Tensor) -> bytes:
    contiguous = value.detach().cpu().contiguous().reshape(-1)
    return contiguous.view(torch.uint8).numpy().tobytes()


def state_dict_architecture_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Fingerprint tensor names, shapes, dtypes, and layouts without their values."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise TypeError(f"Invalid state-dict entry for architecture hash: {key!r}")
        record = (
            key,
            tuple(int(dimension) for dimension in value.shape),
            str(value.dtype),
            str(value.layout),
        )
        digest.update(repr(record).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def state_dict_value_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Fingerprint an exact model state independently of tensor device."""

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise TypeError(f"Invalid state-dict entry for value hash: {key!r}")
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(repr(tuple(int(item) for item in value.shape)).encode())
        digest.update(b"\0")
        digest.update(_tensor_bytes(value))
        digest.update(b"\0")
    return digest.hexdigest()


def build_initialization_provenance(
    state_dict: Mapping[str, Tensor],
    *,
    parent_checkpoint_sha256: str | None = None,
    pretrain_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Create the checkpoint-native contract for a run's exact starting state."""

    if parent_checkpoint_sha256 is not None and pretrain_checkpoint_sha256 is not None:
        raise ValueError("Initialization cannot use parent and pretrain checkpoints together")
    initial_state_sha256 = state_dict_value_sha256(state_dict)
    return {
        "version": _PROVENANCE_VERSION,
        "initialization_lineage": initial_state_sha256,
        "initial_state_sha256": initial_state_sha256,
        "architecture_sha256": state_dict_architecture_sha256(state_dict),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "pretrain_checkpoint_sha256": pretrain_checkpoint_sha256,
    }


def _sha256_or_none(value: Any, *, field: str, required: bool) -> str | None:
    if value is None or not str(value).strip():
        if required:
            raise SoupCompatibilityError(f"Checkpoint provenance is missing {field}")
        return None
    normalized = str(value).strip().casefold()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise SoupCompatibilityError(f"Checkpoint provenance has invalid {field}")
    return normalized


def extract_checkpoint_provenance(
    checkpoint: Mapping[str, Any],
    path: str | Path,
    *,
    weight_source: str = "raw",
    require_complete: bool = True,
) -> SoupCheckpointProvenance:
    """Read and verify soup provenance, rejecting legacy/malformed checkpoints by default."""

    checkpoint_path = Path(path).expanduser().resolve()
    state = extract_checkpoint_state(checkpoint, weight_source=weight_source)
    architecture_sha256 = state_dict_architecture_sha256(state)
    extra = checkpoint.get("extra", {})
    soup_provenance = (
        extra.get("model_soup_provenance", {})
        if isinstance(extra, Mapping)
        else {}
    )
    config = checkpoint.get("config", {})
    project = config.get("project", {}) if isinstance(config, Mapping) else {}
    initialization = (
        project.get("initialization_provenance", {})
        if isinstance(project, Mapping)
        else {}
    )
    provenance_source = "training_initialization"
    if isinstance(soup_provenance, Mapping) and soup_provenance:
        if _unsafe_soup_provenance(soup_provenance) and require_complete:
            raise SoupCompatibilityError(
                "A soup produced with an unsafe lineage, validation, or engineering "
                "override cannot enter a strict soup"
            )
        if require_complete:
            if int(soup_provenance.get("version", -1)) != _MODEL_SOUP_PROVENANCE_VERSION:
                raise SoupCompatibilityError(
                    "Legacy model-soup provenance has no complete authoritative "
                    "validation contract; use only through an explicit unsafe override"
                )
            validation_contract = soup_provenance.get("validation_contract")
            if not isinstance(validation_contract, Mapping):
                raise SoupCompatibilityError(
                    "Strict model-soup provenance is missing validation_contract"
                )
            validate_soup_validation_contract(
                validation_contract,
                require_strict=True,
                verify_files=False,
            )
        provenance_source = "model_soup"
        lineage = _sha256_or_none(
            soup_provenance.get("common_initialization_lineage"),
            field="common_initialization_lineage",
            required=require_complete,
        )
        initial_state_sha256 = None
        parent_sha256 = _sha256_or_none(
            soup_provenance.get("common_parent_checkpoint_sha256"),
            field="common_parent_checkpoint_sha256",
            required=False,
        )
        pretrain_sha256 = _sha256_or_none(
            soup_provenance.get("common_pretrain_checkpoint_sha256"),
            field="common_pretrain_checkpoint_sha256",
            required=False,
        )
        declared_architecture = _sha256_or_none(
            soup_provenance.get("architecture_sha256"),
            field="architecture_sha256",
            required=require_complete,
        )
        if declared_architecture is not None and declared_architecture != architecture_sha256:
            raise SoupCompatibilityError(
                "Soup architecture does not match its recorded provenance"
            )
    elif isinstance(initialization, Mapping) and initialization:
        if int(initialization.get("version", -1)) != _PROVENANCE_VERSION:
            raise SoupCompatibilityError("Unsupported initialization provenance version")
        declared_architecture = _sha256_or_none(
            initialization.get("architecture_sha256"),
            field="architecture_sha256",
            required=require_complete,
        )
        if declared_architecture is not None and declared_architecture != architecture_sha256:
            raise SoupCompatibilityError(
                "Checkpoint architecture does not match its initialization provenance"
            )
        initial_state_sha256 = _sha256_or_none(
            initialization.get("initial_state_sha256"),
            field="initial_state_sha256",
            required=require_complete,
        )
        lineage = _sha256_or_none(
            initialization.get("initialization_lineage"),
            field="initialization_lineage",
            required=require_complete,
        )
        if lineage is not None and initial_state_sha256 is not None and lineage != initial_state_sha256:
            raise SoupCompatibilityError(
                "Initialization lineage must identify the exact recorded starting state"
            )
        parent_sha256 = _sha256_or_none(
            initialization.get("parent_checkpoint_sha256"),
            field="parent_checkpoint_sha256",
            required=False,
        )
        pretrain_sha256 = _sha256_or_none(
            initialization.get("pretrain_checkpoint_sha256"),
            field="pretrain_checkpoint_sha256",
            required=False,
        )
    else:
        if require_complete:
            raise SoupCompatibilityError(
                "Checkpoint has no initialization_provenance; legacy checkpoints "
                "require the explicit unsafe lineage override"
            )
        lineage = None
        initial_state_sha256 = None
        parent_sha256 = None
        pretrain_sha256 = None
        provenance_source = "missing"
    if parent_sha256 is not None and pretrain_sha256 is not None:
        raise SoupCompatibilityError(
            "Checkpoint provenance cannot declare both parent and pretrain starting points"
        )
    return SoupCheckpointProvenance(
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_file_sha256(checkpoint_path),
        architecture_sha256=architecture_sha256,
        initialization_lineage=lineage,
        initial_state_sha256=initial_state_sha256,
        parent_checkpoint_sha256=parent_sha256,
        pretrain_checkpoint_sha256=pretrain_sha256,
        provenance_source=provenance_source,
    )


def validate_state_dict_compatibility(
    reference: Mapping[str, Tensor],
    candidate: Mapping[str, Tensor],
    *,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> None:
    """Require identical tensor keys, shapes, and dtypes.

    Devices may differ because checkpoints are commonly loaded on CPU before a model
    is evaluated on a GPU.  Non-floating buffers are also required to have identical
    values because averaging counters, masks, or categorical state is undefined.
    """

    reference_keys = set(reference)
    candidate_keys = set(candidate)
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        unexpected = sorted(candidate_keys - reference_keys)
        raise SoupCompatibilityError(
            f"State-dict keys differ for {reference_name!r} and {candidate_name!r}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key in reference:
        reference_value = reference[key]
        candidate_value = candidate[key]
        if not isinstance(reference_value, Tensor) or not isinstance(candidate_value, Tensor):
            raise SoupCompatibilityError(f"State entry {key!r} must be a tensor in every member")
        if reference_value.shape != candidate_value.shape:
            raise SoupCompatibilityError(
                f"Shape mismatch at {key!r}: {tuple(reference_value.shape)} != "
                f"{tuple(candidate_value.shape)}"
            )
        if reference_value.dtype != candidate_value.dtype:
            raise SoupCompatibilityError(
                f"Dtype mismatch at {key!r}: {reference_value.dtype} != "
                f"{candidate_value.dtype}"
            )
        if reference_value.layout != torch.strided or candidate_value.layout != torch.strided:
            raise SoupCompatibilityError(f"Only strided tensors can be averaged; found {key!r}")
        if reference_value.is_quantized or candidate_value.is_quantized:
            raise SoupCompatibilityError(f"Quantized state cannot be averaged safely at {key!r}")
        if not (reference_value.is_floating_point() or reference_value.is_complex()):
            comparable = candidate_value.to(device=reference_value.device)
            if not torch.equal(reference_value, comparable):
                raise SoupCompatibilityError(
                    f"Non-floating state differs at {key!r}; it cannot be averaged"
                )


def _normalized_weights(weights: Sequence[float] | None, count: int) -> tuple[float, ...]:
    if count < 1:
        raise ValueError("At least one state dictionary is required")
    if weights is None:
        return (1.0 / count,) * count
    if len(weights) != count:
        raise ValueError(f"Expected {count} soup weights, received {len(weights)}")
    values = tuple(float(value) for value in weights)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Soup weights must be finite and nonnegative")
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError("At least one soup weight must be positive")
    return tuple(value / total for value in values)


def average_state_dicts(
    state_dicts: Sequence[Mapping[str, Tensor]],
    weights: Sequence[float] | None = None,
) -> dict[str, Tensor]:
    """Average compatible floating state while preserving every original dtype.

    Integer and Boolean buffers are copied only after compatibility validation has
    established that every member stores exactly the same value.
    """

    if not state_dicts:
        raise ValueError("At least one state dictionary is required")
    normalized = _normalized_weights(weights, len(state_dicts))
    reference = state_dicts[0]
    for index, candidate in enumerate(state_dicts[1:], start=1):
        validate_state_dict_compatibility(
            reference,
            candidate,
            reference_name="member[0]",
            candidate_name=f"member[{index}]",
        )

    averaged: dict[str, Tensor] = {}
    for key, reference_value in reference.items():
        if not (reference_value.is_floating_point() or reference_value.is_complex()):
            averaged[key] = reference_value.detach().clone()
            continue
        accumulator_dtype = torch.complex128 if reference_value.is_complex() else torch.float64
        accumulator = torch.zeros(
            reference_value.shape,
            dtype=accumulator_dtype,
            device=reference_value.device,
        )
        for weight, state in zip(normalized, state_dicts, strict=True):
            value = state[key].detach().to(
                device=reference_value.device,
                dtype=accumulator_dtype,
            )
            accumulator.add_(value, alpha=weight)
        averaged[key] = accumulator.to(dtype=reference_value.dtype)
    return averaged


def _validated_score(callback: ValidationCallback, state: Mapping[str, Tensor]) -> float:
    score = float(callback(state))
    if not math.isfinite(score):
        raise ValueError(f"Validation callback returned a non-finite score: {score}")
    return score


def greedy_model_soup(
    members: Sequence[SoupMember],
    validation_callback: ValidationCallback,
    *,
    min_improvement: float = 0.0,
    acceptance_tolerance: float = 0.0,
    require_matching_lineage: bool = True,
    allow_unsafe_lineage_mismatch: bool = False,
) -> GreedySoupResult:
    """Build a deterministic greedy soup gated by complete validation.

    Candidates are ordered by their recorded validation score (then by name), but the
    callback re-evaluates both the initial member and each tentative equal-weight soup.
    A candidate is retained only when its fresh validation score does not regress after
    accounting for ``min_improvement`` and ``acceptance_tolerance``.
    """

    if not members:
        raise ValueError("At least one soup member is required")
    if not math.isfinite(min_improvement) or min_improvement < 0.0:
        raise ValueError("min_improvement must be finite and nonnegative")
    if not math.isfinite(acceptance_tolerance) or acceptance_tolerance < 0.0:
        raise ValueError("acceptance_tolerance must be finite and nonnegative")
    names = [member.name for member in members]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Soup member names must be non-empty and unique")
    if any(not math.isfinite(float(member.validation_score)) for member in members):
        raise ValueError("Recorded validation scores must be finite")
    if not require_matching_lineage and not allow_unsafe_lineage_mismatch:
        raise ValueError(
            "Disabling lineage checks requires allow_unsafe_lineage_mismatch=true"
        )

    ordered = sorted(members, key=lambda member: (-member.validation_score, member.name))
    reference = ordered[0]
    if any(member.weight_source != reference.weight_source for member in ordered[1:]):
        raise SoupCompatibilityError("Raw, EMA, and SWA states cannot be mixed in one soup")
    if require_matching_lineage and not allow_unsafe_lineage_mismatch:
        if reference.lineage is None or any(
            member.lineage != reference.lineage for member in ordered[1:]
        ):
            raise SoupCompatibilityError(
                "All soup members must declare the same initialization lineage"
            )
    architecture_hashes: list[str] = []
    for candidate in ordered[1:]:
        validate_state_dict_compatibility(
            reference.state_dict,
            candidate.state_dict,
            reference_name=reference.name,
            candidate_name=candidate.name,
        )
    for member in ordered:
        computed_architecture = state_dict_architecture_sha256(member.state_dict)
        if (
            member.architecture_sha256 is not None
            and member.architecture_sha256 != computed_architecture
        ):
            raise SoupCompatibilityError(
                f"Declared architecture hash is invalid for {member.name!r}"
            )
        architecture_hashes.append(computed_architecture)
    if len(set(architecture_hashes)) != 1:
        raise SoupCompatibilityError("All soup members must have the same architecture hash")

    accepted = [reference]
    current_state = average_state_dicts([reference.state_dict])
    current_score = _validated_score(validation_callback, current_state)
    trials = [SoupTrial(reference.name, current_score, True, 1)]
    for candidate in ordered[1:]:
        tentative_members = [member.state_dict for member in accepted] + [candidate.state_dict]
        tentative_state = average_state_dicts(tentative_members)
        tentative_score = _validated_score(validation_callback, tentative_state)
        keep = tentative_score + acceptance_tolerance >= current_score + min_improvement
        trials.append(
            SoupTrial(
                member_name=candidate.name,
                validation_score=tentative_score,
                accepted=keep,
                member_count=len(accepted) + int(keep),
            )
        )
        if keep:
            accepted.append(candidate)
            current_state = tentative_state
            current_score = tentative_score

    def common_value(values: Sequence[str | None]) -> str | None:
        reference_value = values[0]
        return reference_value if reference_value is not None and all(
            value == reference_value for value in values[1:]
        ) else None

    return GreedySoupResult(
        state_dict=current_state,
        member_names=tuple(member.name for member in accepted),
        validation_score=current_score,
        trials=tuple(trials),
        weight_source=reference.weight_source,
        architecture_sha256=architecture_hashes[0],
        common_initialization_lineage=common_value(
            [member.lineage for member in ordered]
        ),
        common_parent_checkpoint_sha256=common_value(
            [member.parent_checkpoint_sha256 for member in ordered]
        ),
        common_pretrain_checkpoint_sha256=common_value(
            [member.pretrain_checkpoint_sha256 for member in ordered]
        ),
        unsafe_lineage_override_used=bool(allow_unsafe_lineage_mismatch),
    )


def extract_checkpoint_state(
    checkpoint: Mapping[str, Any],
    *,
    weight_source: str = "raw",
) -> dict[str, Tensor]:
    """Extract raw, EMA, or SWA tensors from an already loaded checkpoint."""

    source = weight_source.strip().lower()
    if source == "raw":
        state: Any = checkpoint.get("model", checkpoint.get("state_dict"))
    elif source == "ema":
        state = checkpoint.get("ema")
        if isinstance(state, Mapping) and "shadow" in state:
            state = state["shadow"]
    elif source == "swa":
        state = checkpoint.get("swa")
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
    else:
        raise ValueError("weight_source must be one of: raw, ema, swa")
    if not isinstance(state, Mapping) or not state:
        raise KeyError(f"Checkpoint does not contain {source!r} model state")
    extracted: dict[str, Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise TypeError(f"Invalid checkpoint state entry for {key!r}")
        normalized_key = key[7:] if key.startswith("module.") else key
        if normalized_key in extracted:
            raise ValueError(f"Duplicate normalized checkpoint key: {normalized_key}")
        extracted[normalized_key] = value.detach().clone()
    return extracted


def load_checkpoint_state(
    path: str | Path,
    *,
    weight_source: str = "raw",
    map_location: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    """Load one checkpoint state without constructing or mutating a model."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint payload must be a mapping")
    return extract_checkpoint_state(payload, weight_source=weight_source)
