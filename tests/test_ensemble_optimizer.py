from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from virtual_staining.cli import command_optimize_ensemble
from virtual_staining.engine import ensemble_optimizer as optimizer_module
from virtual_staining.engine.ensemble_optimizer import (
    blend_predictions,
    build_ensemble_manifest_anchor,
    coordinate_search_weights,
    default_sidecar_path,
    load_verified_ensemble_inputs,
    optimize_ensemble_weights,
    slsqp_weights,
    validate_prediction_source,
    write_ensemble_array_sidecar,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keys_sha256(keys: list[str]) -> str:
    payload = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(
    root: Path,
    *,
    source: str,
    count: int = 4,
    authoritative: bool = True,
    scope: str = "full",
    filename: str = "manifest.csv",
) -> tuple[Path, list[dict[str, str]]]:
    split = "val" if source == "validation" else "train"
    rows: list[dict[str, str]] = []
    for index in range(count):
        roi_number = index // 2
        coordinate_stem = f"ROI{roi_number:03d}_00_{index % 2:02d}"
        stem = coordinate_stem if authoritative else f"{index:05d}"
        rows.append(
            {
                "organ": "colon",
                "split": split,
                "source_split": split,
                "stem": stem,
                "patch_id": stem,
                "canonical_key": f"colon/{stem}",
                "roi_id": f"ROI{roi_number:03d}" if authoritative else "surrogate_0000",
                "roi_id_source": (
                    "filename_regex" if authoritative else "surrogate_numeric_block"
                ),
                "group_id": (
                    f"colon/ROI{roi_number:03d}"
                    if authoritative
                    else "colon/surrogate_0000"
                ),
                "cd68_path": f"colon/CD68/{stem}.jpg",
                "is_paired": "true",
                "missing_targets": "",
                "manifest_scope": scope,
            }
        )
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def _audit_kwargs(
    root: Path,
    manifest: Path,
    *,
    source: str,
) -> dict[str, Any]:
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        target_rows = [dict(row) for row in csv.DictReader(handle)]
    if not target_rows:
        raise ValueError("Test ensemble manifest must not be empty")
    partner_role = "train" if source == "validation" else "val"
    partner_rows: list[dict[str, str]] = []
    for index, row in enumerate(target_rows):
        roi = f"ROI{800 + index // 2:03d}"
        stem = f"{roi}_00_{index % 2:02d}"
        partner = dict(row)
        partner.update(
            {
                "organ": "colon",
                "split": partner_role,
                "source_split": partner_role,
                "stem": stem,
                "patch_id": stem,
                "canonical_key": f"colon/{stem}",
                "roi_id": roi,
                "roi_id_source": "filename_regex",
                "group_id": f"colon/{roi}",
                "cd68_path": f"colon/CD68/{stem}.jpg",
                "is_paired": "true",
                "missing_targets": "",
                "manifest_scope": "full",
            }
        )
        partner_rows.append(partner)
    partner_path = root / f"{manifest.stem}_{partner_role}_audit.csv"
    with partner_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(partner_rows[0]))
        writer.writeheader()
        writer.writerows(partner_rows)
    train_manifest, val_manifest = (
        (partner_path, manifest)
        if source == "validation"
        else (manifest, partner_path)
    )
    total_rows = len(target_rows) + len(partner_rows)
    audit = root / f"{manifest.stem}_{source}_roi_audit.json"
    audit.write_text(
        json.dumps(
            {
                "total_rows": total_rows,
                "parsed_rows": total_rows,
                "filename_grid_verified": True,
                "duplicate_coordinates": [],
                "train_val_shared_rois": [],
                "cross_split_adjacent_pairs": [],
                "boundary": {
                    "direction_verified": True,
                    "continuity_verified": True,
                },
                "context_gate_reasons": [],
                "context_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    return {
        "roi_audit_path": audit,
        "audited_manifest_paths": (train_manifest, val_manifest),
    }


def _write_fold_assignment(
    root: Path,
    rows: list[dict[str, str]],
    *,
    filename: str = "folds.csv",
) -> Path:
    assignment_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        parts = row["stem"].split("_")
        assignment_rows.append(
            {
                "source_index": index,
                "canonical_key": row["canonical_key"],
                "organ": row["organ"],
                "roi_id": parts[0],
                "row": int(parts[1]),
                "col": int(parts[2]),
                "fold": index // 2,
            }
        )
    path = root / filename
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignment_rows[0]))
        writer.writeheader()
        writer.writerows(assignment_rows)
    return path


def _write_array(
    root: Path,
    name: str,
    values: np.ndarray,
    *,
    role: str,
    anchor: Any,
    artifact_id: str | None = None,
) -> Path:
    path = root / f"{name}.npy"
    np.save(path, values)
    write_ensemble_array_sidecar(
        path,
        role=role,
        anchor=anchor,
        artifact_id=artifact_id or name,
    )
    return path


def _validation_contract(
    tmp_path: Path,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, Any, list[Path], Path, Path]:
    manifest, _ = _write_manifest(tmp_path, source="validation")
    audit_kwargs = _audit_kwargs(tmp_path, manifest, source="validation")
    anchor = build_ensemble_manifest_anchor(
        manifest,
        source="validation",
        target_marker="CD68",
        **audit_kwargs,
    )
    target = np.array([0.0, 1.0, 0.5, 0.25], dtype=np.float32)
    paths = [
        _write_array(
            tmp_path,
            "model_a",
            target.copy(),
            role="prediction",
            anchor=anchor,
        ),
        _write_array(
            tmp_path,
            "model_b",
            np.zeros_like(target),
            role="prediction",
            anchor=anchor,
        ),
    ]
    target_path = _write_array(
        tmp_path,
        "target",
        target,
        role="target",
        anchor=anchor,
    )
    predictions, loaded_target, provenance = load_verified_ensemble_inputs(
        paths,
        target_path,
        expected_source="val",
        manifest_path=manifest,
        target_marker="CD68",
        **audit_kwargs,
    )
    return predictions, loaded_target, provenance, paths, target_path, manifest


@pytest.mark.parametrize("source", ["test", "train", "official_test", ""])
def test_optimizer_rejects_non_validation_prediction_sources(source: str) -> None:
    with pytest.raises(ValueError, match="only from explicit validation or OOF"):
        coordinate_search_weights([np.zeros(2)], np.zeros(2), source=source)


def test_source_aliases_are_normalized() -> None:
    assert validate_prediction_source("VAL") == "validation"
    assert validate_prediction_source("out-of-fold") == "oof"


def test_coordinate_search_is_deterministic_nonnegative_and_improves_uniform() -> None:
    target = np.array([0.0, 0.5, 1.0, 0.25])
    predictions = [target.copy(), np.ones_like(target), np.zeros_like(target)]
    first = coordinate_search_weights(predictions, target, source="oof")
    second = coordinate_search_weights(predictions, target, source="oof")
    assert first == second
    assert all(weight >= 0.0 for weight in first.weights)
    assert sum(first.weights) == pytest.approx(1.0, abs=1e-12)
    assert first.score > first.uniform_score
    assert first.weights[0] > 0.99


def test_slsqp_is_deterministic_nonnegative_unit_sum_and_improves_uniform() -> None:
    target = np.array([0.0, 0.5, 1.0, 0.25], dtype=np.float64)
    predictions = [target.copy(), np.ones_like(target), np.zeros_like(target)]
    first = slsqp_weights(predictions, target, source="oof")
    second = slsqp_weights(predictions, target, source="oof")
    assert first.weights == pytest.approx(second.weights, abs=1e-12)
    assert first.score == pytest.approx(second.score, abs=1e-15)
    assert first.optimizer == "slsqp"
    assert first.fallback_reason is None
    assert all(weight >= 0.0 for weight in first.weights)
    assert sum(first.weights) == pytest.approx(1.0, abs=1e-12)
    assert first.weights[0] > 0.99


def test_slsqp_failure_policy_is_auditable_uniform_or_explicit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_solver(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            success=False,
            message="synthetic failure",
            nfev=7,
            x=np.array([0.5, 0.5]),
        )

    monkeypatch.setattr(optimizer_module, "minimize", fail_solver)
    target = np.array([0.0, 1.0])
    predictions = [np.zeros_like(target), np.ones_like(target)]
    fallback = slsqp_weights(
        predictions, target, source="validation", failure_policy="uniform"
    )
    assert fallback.weights == pytest.approx((0.5, 0.5), abs=1e-12)
    assert fallback.fallback_reason == "solver:synthetic failure"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        slsqp_weights(
            predictions, target, source="validation", failure_policy="error"
        )


def test_public_optimizer_requires_verified_sidecars() -> None:
    with pytest.raises(ValueError, match="source string alone is insufficient"):
        optimize_ensemble_weights([np.zeros(2)], np.zeros(2), source="validation")


def test_verified_validation_optimizer_and_external_anchor(tmp_path: Path) -> None:
    predictions, target, provenance, _, _, manifest = _validation_contract(tmp_path)
    result = optimize_ensemble_weights(
        predictions,
        target,
        source="validation",
        provenance=provenance,
        cross_validate_weights=False,
    )
    assert all(weight >= 0.0 for weight in result.weights)
    assert sum(result.weights) == pytest.approx(1.0, abs=1e-12)
    assert result.weights[0] > 0.99
    summary = provenance.summary()
    assert summary["manifest_sha256"] == _sha256(manifest)
    assert summary["external_manifest_anchor"]["metric_domain"] == "jpg_roundtrip"
    assert summary["external_manifest_anchor"]["roi_authority"] == (
        "authoritative_filename_grid"
    )
    assert not summary["unsafe_engineering_override_used"]


def test_verified_contract_rejects_in_memory_array_mutation(tmp_path: Path) -> None:
    predictions, target, provenance, _, _, _ = _validation_contract(tmp_path)
    predictions[0][0] = 0.75
    with pytest.raises(ValueError, match="content changed"):
        optimize_ensemble_weights(
            predictions,
            target,
            source="validation",
            provenance=provenance,
            cross_validate_weights=False,
        )


def test_loader_requires_external_source_manifest_and_marker(tmp_path: Path) -> None:
    values = np.zeros(2, dtype=np.float32)
    prediction = tmp_path / "prediction.npy"
    target = tmp_path / "target.npy"
    np.save(prediction, values)
    np.save(target, values)
    with pytest.raises(ValueError, match="expected_source is required"):
        load_verified_ensemble_inputs([prediction], target)
    with pytest.raises(ValueError, match="manifest_path is required"):
        load_verified_ensemble_inputs(
            [prediction], target, expected_source="validation"
        )
    manifest, _ = _write_manifest(tmp_path, source="validation")
    with pytest.raises(ValueError, match="target_marker is required"):
        load_verified_ensemble_inputs(
            [prediction],
            target,
            expected_source="validation",
            manifest_path=manifest,
        )


def test_sidecar_writer_requires_complete_manifest_coverage(tmp_path: Path) -> None:
    manifest, _ = _write_manifest(tmp_path, source="validation")
    anchor = build_ensemble_manifest_anchor(
        manifest,
        source="validation",
        target_marker="CD68",
        **_audit_kwargs(tmp_path, manifest, source="validation"),
    )
    array = tmp_path / "partial.npy"
    np.save(array, np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="fully cover"):
        write_ensemble_array_sidecar(
            array, role="prediction", anchor=anchor, artifact_id="partial"
        )


def test_schema_v1_self_signed_sidecar_is_rejected(tmp_path: Path) -> None:
    manifest, _ = _write_manifest(tmp_path, source="validation")
    values = np.zeros(4, dtype=np.float32)
    prediction = tmp_path / "prediction.npy"
    target = tmp_path / "target.npy"
    np.save(prediction, values)
    np.save(target, values)
    for path, role in ((prediction, "prediction"), (target, "target")):
        default_sidecar_path(path).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "role": role,
                    "split": "validation",
                    "array_filename": path.name,
                    "array_sha256": _sha256(path),
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="schema_version must be 2"):
        load_verified_ensemble_inputs(
            [prediction],
            target,
            expected_source="validation",
            manifest_path=manifest,
            target_marker="CD68",
            **_audit_kwargs(tmp_path, manifest, source="validation"),
        )


def test_forged_sidecar_manifest_hash_is_rejected_against_actual_file(
    tmp_path: Path,
) -> None:
    _, _, _, paths, target, manifest = _validation_contract(tmp_path)
    for array in [*paths, target]:
        sidecar = default_sidecar_path(array)
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata["manifest_sha256"] = "b" * 64
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="actual manifest"):
        load_verified_ensemble_inputs(
            paths,
            target,
            expected_source="validation",
            manifest_path=manifest,
            target_marker="CD68",
            **_audit_kwargs(tmp_path, manifest, source="validation"),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("array_sha256", "0" * 64, "array SHA256"),
        ("metric_domain", "float", "JPG round-trip"),
        ("target_marker", "Vimentin", "target_marker"),
        ("sample_organs", ["liver"] * 4, "sample_organs_sha256"),
    ],
)
def test_sidecar_tampering_is_rejected(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    _, _, _, paths, target, manifest = _validation_contract(tmp_path)
    sidecar = default_sidecar_path(paths[0])
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata[field] = value
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_verified_ensemble_inputs(
            paths,
            target,
            expected_source="validation",
            manifest_path=manifest,
            target_marker="CD68",
            **_audit_kwargs(tmp_path, manifest, source="validation"),
        )


def test_sidecar_sample_order_cannot_override_manifest_order(tmp_path: Path) -> None:
    _, _, _, paths, target, manifest = _validation_contract(tmp_path)
    sidecar = default_sidecar_path(paths[0])
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["sample_keys"] = list(reversed(metadata["sample_keys"]))
    metadata["sample_keys_sha256"] = _keys_sha256(metadata["sample_keys"])
    metadata["manifest_sample_keys_sha256"] = metadata["sample_keys_sha256"]
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="actual manifest"):
        load_verified_ensemble_inputs(
            paths,
            target,
            expected_source="validation",
            manifest_path=manifest,
            target_marker="CD68",
            **_audit_kwargs(tmp_path, manifest, source="validation"),
        )


def test_actual_manifest_mutation_invalidates_existing_sidecars(tmp_path: Path) -> None:
    _, _, _, paths, target, manifest = _validation_contract(tmp_path)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="actual manifest"):
        load_verified_ensemble_inputs(
            paths,
            target,
            expected_source="validation",
            manifest_path=manifest,
            target_marker="CD68",
            **_audit_kwargs(tmp_path, manifest, source="validation"),
        )


def test_metric_domain_must_be_final_jpg_roundtrip(tmp_path: Path) -> None:
    manifest, _ = _write_manifest(tmp_path, source="validation")
    with pytest.raises(ValueError, match="JPG round-trip"):
        build_ensemble_manifest_anchor(
            manifest,
            source="validation",
            target_marker="CD68",
            metric_domain="float",
        )


@pytest.mark.parametrize("scope", ["tiny", "smoke", "truncated", "debug"])
def test_strict_anchor_rejects_tiny_or_debug_manifest(
    tmp_path: Path, scope: str
) -> None:
    manifest, _ = _write_manifest(
        tmp_path, source="validation", scope=scope
    )
    with pytest.raises(ValueError, match="strict ensemble fitting is blocked"):
        build_ensemble_manifest_anchor(
            manifest,
            source="validation",
            target_marker="CD68",
            **_audit_kwargs(tmp_path, manifest, source="validation"),
        )


def test_surrogate_manifest_requires_explicit_permanently_marked_unsafe_path(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_manifest(
        tmp_path,
        source="validation",
        authoritative=False,
    )
    with pytest.raises(ValueError, match="strict ensemble fitting is blocked"):
        build_ensemble_manifest_anchor(
            manifest,
            source="validation",
            target_marker="CD68",
            **_audit_kwargs(tmp_path, manifest, source="validation"),
        )
    audit_kwargs = _audit_kwargs(tmp_path, manifest, source="validation")
    anchor = build_ensemble_manifest_anchor(
        manifest,
        source="validation",
        target_marker="CD68",
        allow_unsafe_engineering_manifest=True,
        **audit_kwargs,
    )
    assert anchor.unsafe_engineering_override_used
    assert anchor.roi_authority == "unsafe_engineering_manifest"
    values = np.zeros(4, dtype=np.float32)
    prediction = _write_array(
        tmp_path, "unsafe_prediction", values, role="prediction", anchor=anchor
    )
    target = _write_array(
        tmp_path, "unsafe_target", values, role="target", anchor=anchor
    )
    with pytest.raises(ValueError, match="strict ensemble fitting is blocked"):
        load_verified_ensemble_inputs(
            [prediction],
            target,
            expected_source="validation",
            manifest_path=manifest,
            target_marker="CD68",
            **audit_kwargs,
        )
    _, _, provenance = load_verified_ensemble_inputs(
        [prediction],
        target,
        expected_source="validation",
        manifest_path=manifest,
        target_marker="CD68",
        allow_unsafe_engineering_manifest=True,
        **audit_kwargs,
    )
    assert provenance.summary()["unsafe_engineering_override_used"]
    sidecar = json.loads(
        default_sidecar_path(prediction).read_text(encoding="utf-8")
    )
    assert sidecar["unsafe_engineering_override_used"] is True
    assert sidecar["unsafe_reasons"]


def test_test_manifest_is_rejected_even_with_unsafe_override(tmp_path: Path) -> None:
    manifest, rows = _write_manifest(tmp_path, source="validation")
    for row in rows:
        row["split"] = "test"
        row["source_split"] = "official_test"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="test data"):
        build_ensemble_manifest_anchor(
            manifest,
            source="validation",
            target_marker="CD68",
            allow_unsafe_engineering_manifest=True,
        )


def test_validation_requires_at_least_two_roi_groups_in_strict_mode(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_manifest(tmp_path, source="validation", count=2)
    with pytest.raises(ValueError, match="fewer_than_two_roi_groups"):
        build_ensemble_manifest_anchor(
            manifest,
            source="validation",
            target_marker="CD68",
            **_audit_kwargs(tmp_path, manifest, source="validation"),
        )


def test_strict_validation_anchor_requires_bound_train_val_roi_audit(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_manifest(tmp_path, source="validation")

    with pytest.raises(ValueError, match="missing_train_val_audited_manifest_pair"):
        build_ensemble_manifest_anchor(
            manifest,
            source="validation",
            target_marker="CD68",
        )


def test_strict_anchor_derives_train_val_roi_overlap_from_actual_manifests(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_manifest(tmp_path, source="validation")
    audit_kwargs = _audit_kwargs(tmp_path, manifest, source="validation")
    train_manifest = Path(audit_kwargs["audited_manifest_paths"][0])
    with train_manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows):
        roi = f"ROI{index // 2:03d}"
        stem = f"{roi}_10_{index % 2:02d}"
        row["stem"] = stem
        row["patch_id"] = stem
        row["roi_id"] = roi
        row["canonical_key"] = f"colon/train_{stem}"
    with train_manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="audited_train_val_roi_overlap"):
        build_ensemble_manifest_anchor(
            manifest,
            source="validation",
            target_marker="CD68",
            **audit_kwargs,
        )


def test_roi_direction_audit_and_sidecar_hash_are_fail_closed(
    tmp_path: Path,
) -> None:
    _, _, _, paths, target, manifest = _validation_contract(tmp_path)
    audit_kwargs = {
        "roi_audit_path": tmp_path / "manifest_validation_roi_audit.json",
        "audited_manifest_paths": (
            tmp_path / "manifest_train_audit.csv",
            manifest,
        ),
    }
    audit = Path(audit_kwargs["roi_audit_path"])
    audit.write_text(audit.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="roi_audit_sha256"):
        load_verified_ensemble_inputs(
            paths,
            target,
            expected_source="validation",
            manifest_path=manifest,
            target_marker="CD68",
            **audit_kwargs,
        )

    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["boundary"]["direction_verified"] = False
    audit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="roi_audit_direction_not_verified"):
        build_ensemble_manifest_anchor(
            manifest,
            source="validation",
            target_marker="CD68",
            **audit_kwargs,
        )


def test_unsafe_roi_audit_cannot_be_reloaded_as_strict(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_manifest(tmp_path, source="validation")
    audit_kwargs = _audit_kwargs(tmp_path, manifest, source="validation")
    audit = Path(audit_kwargs["roi_audit_path"])
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["boundary"]["continuity_verified"] = False
    payload["context_enabled"] = False
    payload["context_gate_reasons"] = ["boundary_continuity_not_verified"]
    audit.write_text(json.dumps(payload), encoding="utf-8")
    anchor = build_ensemble_manifest_anchor(
        manifest,
        source="validation",
        target_marker="CD68",
        allow_unsafe_engineering_manifest=True,
        **audit_kwargs,
    )
    values = np.zeros(4, dtype=np.float32)
    prediction = _write_array(
        tmp_path, "unsafe_audit_prediction", values, role="prediction", anchor=anchor
    )
    target = _write_array(
        tmp_path, "unsafe_audit_target", values, role="target", anchor=anchor
    )
    with pytest.raises(ValueError, match="strict ensemble fitting is blocked"):
        load_verified_ensemble_inputs(
            [prediction],
            target,
            expected_source="validation",
            manifest_path=manifest,
            target_marker="CD68",
            **audit_kwargs,
        )
    _, _, provenance = load_verified_ensemble_inputs(
        [prediction],
        target,
        expected_source="validation",
        manifest_path=manifest,
        target_marker="CD68",
        allow_unsafe_engineering_manifest=True,
        **audit_kwargs,
    )
    assert provenance.anchor.unsafe_engineering_override_used is True
    assert "roi_audit_continuity_not_verified" in provenance.anchor.unsafe_reasons


def test_oof_external_assignment_is_hashed_and_cross_validated(
    tmp_path: Path,
) -> None:
    manifest, rows = _write_manifest(tmp_path, source="oof", count=8)
    assignment = _write_fold_assignment(tmp_path, rows)
    audit_kwargs = _audit_kwargs(tmp_path, manifest, source="oof")
    anchor = build_ensemble_manifest_anchor(
        manifest,
        source="oof",
        target_marker="CD68",
        fold_assignment_path=assignment,
        **audit_kwargs,
    )
    target_values = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    prediction_paths = [
        _write_array(
            tmp_path,
            "good_model",
            target_values.copy(),
            role="prediction",
            anchor=anchor,
        ),
        _write_array(
            tmp_path,
            "weak_model",
            np.zeros_like(target_values),
            role="prediction",
            anchor=anchor,
        ),
    ]
    target_path = _write_array(
        tmp_path, "oof_target", target_values, role="target", anchor=anchor
    )
    predictions, target, provenance = load_verified_ensemble_inputs(
        prediction_paths,
        target_path,
        expected_source="oof",
        manifest_path=manifest,
        target_marker="CD68",
        fold_assignment_path=assignment,
        **audit_kwargs,
    )
    assert provenance.anchor.fold_assignment_sha256 == _sha256(assignment)
    for optimizer in ("coordinate", "slsqp"):
        result = optimize_ensemble_weights(
            predictions,
            target,
            source="oof",
            provenance=provenance,
            cross_validate_weights=True,
            optimizer=optimizer,
        )
        assert result.cross_validated
        assert len(result.fold_results) == 4
        assert sum(fold.held_out_samples for fold in result.fold_results) == 8
        assert all(fold.held_out_groups == 1 for fold in result.fold_results)
        assert all(fold.optimizer == optimizer for fold in result.fold_results)
        assert result.cross_validated_score is not None
        assert result.cross_validated_uniform_score is not None
        assert result.cross_validated_score > result.cross_validated_uniform_score
        assert all(weight >= 0.0 for weight in result.weights)
        assert sum(result.weights) == pytest.approx(1.0, abs=1e-12)


def test_oof_assignment_mutation_invalidates_existing_sidecars(tmp_path: Path) -> None:
    manifest, rows = _write_manifest(tmp_path, source="oof", count=4)
    assignment = _write_fold_assignment(tmp_path, rows)
    audit_kwargs = _audit_kwargs(tmp_path, manifest, source="oof")
    anchor = build_ensemble_manifest_anchor(
        manifest,
        source="oof",
        target_marker="CD68",
        fold_assignment_path=assignment,
        **audit_kwargs,
    )
    values = np.zeros(4, dtype=np.float32)
    prediction = _write_array(
        tmp_path, "prediction", values, role="prediction", anchor=anchor
    )
    target = _write_array(tmp_path, "target", values, role="target", anchor=anchor)
    with assignment.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="fold_assignment_sha256"):
        load_verified_ensemble_inputs(
            [prediction],
            target,
            expected_source="oof",
            manifest_path=manifest,
            target_marker="CD68",
            fold_assignment_path=assignment,
            **audit_kwargs,
        )


def test_oof_assignment_rejects_partial_coverage_and_group_crossing(
    tmp_path: Path,
) -> None:
    manifest, rows = _write_manifest(tmp_path, source="oof", count=4)
    assignment = _write_fold_assignment(tmp_path, rows)
    with assignment.open(newline="", encoding="utf-8-sig") as handle:
        assignment_rows = list(csv.DictReader(handle))
    with assignment.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignment_rows[0]))
        writer.writeheader()
        writer.writerows(assignment_rows[:-1])
    with pytest.raises(ValueError, match="cover every manifest row"):
        build_ensemble_manifest_anchor(
            manifest,
            source="oof",
            target_marker="CD68",
            fold_assignment_path=assignment,
        )

    assignment = _write_fold_assignment(tmp_path, rows)
    with assignment.open(newline="", encoding="utf-8-sig") as handle:
        assignment_rows = list(csv.DictReader(handle))
    assignment_rows[1]["fold"] = "1"
    with assignment.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignment_rows[0]))
        writer.writeheader()
        writer.writerows(assignment_rows)
    with pytest.raises(ValueError, match="crosses OOF fold"):
        build_ensemble_manifest_anchor(
            manifest,
            source="oof",
            target_marker="CD68",
            fold_assignment_path=assignment,
        )


def test_oof_assignment_must_match_manifest_coordinate_and_order(tmp_path: Path) -> None:
    manifest, rows = _write_manifest(tmp_path, source="oof", count=4)
    assignment = _write_fold_assignment(tmp_path, rows)
    with assignment.open(newline="", encoding="utf-8-sig") as handle:
        assignment_rows = list(csv.DictReader(handle))
    assignment_rows[0]["canonical_key"] = "colon/forged"
    with assignment.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignment_rows[0]))
        writer.writeheader()
        writer.writerows(assignment_rows)
    with pytest.raises(ValueError, match="canonical_key/organ order"):
        build_ensemble_manifest_anchor(
            manifest,
            source="oof",
            target_marker="CD68",
            fold_assignment_path=assignment,
        )


def test_validation_contract_cannot_enable_grouped_oof_cross_validation(
    tmp_path: Path,
) -> None:
    predictions, target, provenance, _, _, _ = _validation_contract(tmp_path)
    with pytest.raises(ValueError, match="requires grouped OOF"):
        optimize_ensemble_weights(
            predictions,
            target,
            source="validation",
            provenance=provenance,
            cross_validate_weights=True,
        )


def test_cli_uses_external_roi_audit_anchor_and_writes_provenance(
    tmp_path: Path,
) -> None:
    _, _, _, prediction_paths, target_path, manifest = _validation_contract(
        tmp_path
    )
    audit_kwargs = {
        "roi_audit": str(tmp_path / "manifest_validation_roi_audit.json"),
        "audited_manifests": [
            str(tmp_path / "manifest_train_audit.csv"),
            str(manifest),
        ],
    }
    config_path = tmp_path / "ensemble.yaml"
    config_path.write_text(
        "ensemble:\n"
        "  cross_validate_weights: false\n"
        "  validation_only: true\n"
        "  optimizer: coordinate\n"
        "  optimizer_failure_policy: error\n",
        encoding="utf-8",
    )
    output = tmp_path / "weights.json"
    payload = command_optimize_ensemble(
        SimpleNamespace(
            config=str(config_path),
            set=[],
            predictions=[str(path) for path in prediction_paths],
            prediction_sidecars=None,
            target_array=str(target_path),
            target_sidecar=None,
            source="validation",
            manifest=str(manifest),
            target_marker="CD68",
            metric_domain="jpg_roundtrip",
            fold_assignment=None,
            allow_unsafe_engineering_manifest=False,
            cross_validate_weights=None,
            min_gain_over_uniform=0.0,
            uniform_shrinkage=0.0,
            output=str(output),
            **audit_kwargs,
        )
    )
    anchor = payload["provenance"]["external_manifest_anchor"]
    assert payload["output"] == str(output.resolve())
    assert anchor["roi_audit_sha256"] == _sha256(
        Path(audit_kwargs["roi_audit"])
    )
    assert anchor["audited_manifest_row_count"] == 8
    assert anchor["unsafe_engineering_override_used"] is False
    assert output.is_file()


def test_optimizer_can_keep_uniform_as_overfitting_guard() -> None:
    target = np.array([0.0, 1.0])
    predictions = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    result = coordinate_search_weights(
        predictions,
        target,
        source="validation",
        min_gain_over_uniform=0.01,
    )
    assert result.weights == pytest.approx((0.5, 0.5))
    assert not result.used_learned_weights


def test_custom_score_and_global_blending() -> None:
    target = np.array([0.0, 1.0, 0.5])
    predictions = [np.zeros(3), target]

    def negative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -float(np.mean(np.abs(y_true - y_pred)))

    result = coordinate_search_weights(
        predictions,
        target,
        source="val",
        score_function=negative_absolute_error,
        uniform_shrinkage=0.1,
    )
    blended = blend_predictions(predictions, result.weights)
    assert result.weights[1] > result.weights[0]
    assert np.allclose(blended, result.weights[1] * target)


def test_optimizer_rejects_shape_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        coordinate_search_weights(
            [np.zeros(2), np.zeros(3)], np.zeros(2), source="val"
        )
    with pytest.raises(ValueError, match="finite"):
        coordinate_search_weights([np.array([np.nan])], np.zeros(1), source="oof")
