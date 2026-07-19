from __future__ import annotations

import json
from pathlib import Path

from virtual_staining.metrics.promotion import (
    evaluate_roi_jpg_promotion,
    write_promotion_report,
)


def _verified_audit() -> dict[str, object]:
    return {
        "total_rows": 24,
        "parsed_rows": 24,
        "filename_grid_verified": True,
        "duplicate_coordinates": [],
        "train_val_shared_rois": [],
        "cross_split_adjacent_pairs": [],
        "boundary": {
            "direction_verified": True,
            "continuity_verified": True,
        },
        "context_enabled": True,
        "context_gate_reasons": [],
    }


def _records(
    ssim_deltas: list[float],
    psnr_deltas: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    parent: list[dict[str, object]] = []
    candidate: list[dict[str, object]] = []
    for index, (ssim_delta, psnr_delta) in enumerate(
        zip(ssim_deltas, psnr_deltas, strict=True)
    ):
        label = "border" if index == len(ssim_deltas) - 1 else "interior"
        base = {
            "target": "CD68",
            "canonical_key": f"colon/ROI{index:03d}_00_00",
            "roi_id": f"ROI{index:03d}",
            "organ": "colon",
            "border_class": label,
            "activity_bin": "high" if index % 2 else "low",
        }
        parent.append({**base, "jpg_ssim": 0.70, "jpg_psnr": 24.0})
        candidate.append(
            {
                **base,
                "jpg_ssim": 0.70 + ssim_delta,
                "jpg_psnr": 24.0 + psnr_delta,
            }
        )
    return parent, candidate


def _provenance() -> list[dict[str, object]]:
    return [
        {
            "fold": 0,
            "seed": 2026,
            "parent_manifest_sha256": "same-fold-0",
            "candidate_manifest_sha256": "same-fold-0",
        },
        {
            "fold": 1,
            "seed": 2026,
            "parent_manifest_sha256": "same-fold-1",
            "candidate_manifest_sha256": "same-fold-1",
        },
    ]


def _bound_records(
    ssim_deltas: list[float],
    psnr_deltas: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    parent, candidate = _records(ssim_deltas, psnr_deltas)
    bound_parent: list[dict[str, object]] = []
    bound_candidate: list[dict[str, object]] = []
    for fold in (0, 1):
        bound_parent.extend({**row, "fold": fold, "seed": 2026} for row in parent)
        bound_candidate.extend({**row, "fold": fold, "seed": 2026} for row in candidate)
    return bound_parent, bound_candidate


def test_promotion_requires_significant_jpg_gain_and_writes_atomically(tmp_path: Path) -> None:
    parent, candidate = _bound_records([0.02] * 4, [0.0] * 4)
    report = evaluate_roi_jpg_promotion(
        parent,
        candidate,
        roi_audit=_verified_audit(),
        fold_seed_provenance=_provenance(),
        bootstrap_samples=300,
    )

    assert report["promotable"] is True
    assert report["final_default_eligible"] is True
    assert report["jpg_metrics"]["jpg_ssim"]["ci_low"] > 0.0
    assert report["jpg_metrics"]["jpg_ssim"]["win_tie_loss"] == {
        "win": 8,
        "tie": 0,
        "loss": 0,
    }
    assert report["jpg_metrics"]["jpg_psnr"]["ci_low"] == 0.0
    assert report["reasons"] == []

    destination = write_promotion_report(report, tmp_path / "目录" / "promotion.json")
    assert json.loads(destination.read_text(encoding="utf-8"))["decision"] == "promote"
    assert not list(destination.parent.glob("*.tmp"))


def test_surrogate_grid_is_blocked_even_when_metrics_improve() -> None:
    parent, candidate = _bound_records([0.03] * 4, [0.5] * 4)
    surrogate_audit = {
        "total_rows": 1346,
        "parsed_rows": 0,
        "filename_grid_verified": False,
        "duplicate_coordinates": [],
        "train_val_shared_rois": [],
        "cross_split_adjacent_pairs": [],
        "boundary": {
            "direction_verified": False,
            "continuity_verified": False,
        },
        "context_enabled": False,
        "context_gate_reasons": [
            "unverified_filename_coordinates",
            "coordinate_direction_not_verified",
            "boundary_continuity_not_verified",
        ],
    }

    report = evaluate_roi_jpg_promotion(
        parent,
        candidate,
        roi_audit=surrogate_audit,
        fold_seed_provenance=_provenance(),
        bootstrap_samples=200,
    )

    assert report["promotable"] is False
    assert report["jpg_metrics"]["jpg_ssim"]["ci_low"] > 0.0
    assert "incomplete_filename_coordinates" in report["reasons"]
    assert "unverified_filename_grid" in report["reasons"]
    assert "roi_grid_gate_disabled" in report["reasons"]


def test_promotion_blocks_leakage_unpaired_records_and_one_evidence_run() -> None:
    parent, candidate = _records([0.02] * 4, [0.2] * 4)
    candidate.pop()
    audit = _verified_audit()
    audit["train_val_shared_rois"] = ["ROI000"]
    audit["cross_split_adjacent_pairs"] = [{"left": "a", "right": "b"}]

    report = evaluate_roi_jpg_promotion(
        parent,
        candidate,
        roi_audit=audit,
        fold_seed_provenance=[{"fold": 0, "seed": 2026}],
        bootstrap_samples=100,
    )

    assert report["promotable"] is False
    assert report["jpg_metrics"] == {}
    assert "train_val_roi_leakage" in report["reasons"]
    assert "train_val_neighborhood_leakage" in report["reasons"]
    assert "unpaired_records_missing_candidate:1" in report["reasons"]
    assert "insufficient_independent_fold_seed_evidence" in report["reasons"]


def test_other_jpg_metric_must_not_significantly_decline() -> None:
    parent, candidate = _bound_records([0.02] * 4, [-0.5] * 4)
    report = evaluate_roi_jpg_promotion(
        parent,
        candidate,
        roi_audit=_verified_audit(),
        fold_seed_provenance=_provenance(),
        bootstrap_samples=200,
    )

    assert report["promotable"] is False
    assert report["jpg_metrics"]["jpg_ssim"]["significant_improvement"] is True
    assert report["jpg_metrics"]["jpg_psnr"]["significant_decline"] is True
    assert "significant_final_jpg_decline:jpg_psnr" in report["reasons"]


def test_concentrated_stratum_regression_blocks_overall_gain() -> None:
    parent, candidate = _bound_records([0.10, 0.10, 0.10, -0.01], [0.4] * 4)
    report = evaluate_roi_jpg_promotion(
        parent,
        candidate,
        roi_audit=_verified_audit(),
        fold_seed_provenance=_provenance(),
        bootstrap_samples=4000,
    )

    assert report["jpg_metrics"]["jpg_ssim"]["ci_low"] > 0.0
    assert report["promotable"] is False
    assert (
        "stratum_significant_decline:border_class:border:jpg_ssim" in report["reasons"]
    )


def test_manifest_provenance_must_pair_parent_and_candidate() -> None:
    parent, candidate = _bound_records([0.02] * 4, [0.1] * 4)
    provenance = _provenance()
    provenance[1]["candidate_manifest_sha256"] = "different"
    report = evaluate_roi_jpg_promotion(
        parent,
        candidate,
        roi_audit=_verified_audit(),
        fold_seed_provenance=provenance,
        bootstrap_samples=100,
    )

    assert report["promotable"] is False
    assert "validation_manifest_mismatch:fold=1,seed=2026" in report["reasons"]


def test_two_declared_evidence_points_cannot_reuse_one_unbound_record_set() -> None:
    parent, candidate = _records([0.03] * 4, [0.2] * 4)
    report = evaluate_roi_jpg_promotion(
        parent,
        candidate,
        roi_audit=_verified_audit(),
        fold_seed_provenance=_provenance(),
        bootstrap_samples=100,
    )

    assert report["promotable"] is False
    assert report["jpg_metrics"] == {}
    assert "parent_missing_record_evidence_binding:0" in report["reasons"]
    assert "candidate_missing_record_evidence_binding:0" in report["reasons"]
    assert "parent_records_missing_evidence:fold=0,seed=2026" in report["reasons"]
    assert "candidate_records_missing_evidence:fold=1,seed=2026" in report["reasons"]


def test_single_evidence_can_be_scored_but_cannot_enter_final_default() -> None:
    parent, candidate = _records([0.03] * 4, [0.2] * 4)
    report = evaluate_roi_jpg_promotion(
        parent,
        candidate,
        roi_audit=_verified_audit(),
        fold_seed_provenance=[
            {
                "fold": 0,
                "seed": 2026,
                "parent_manifest_sha256": "single",
                "candidate_manifest_sha256": "single",
            }
        ],
        bootstrap_samples=100,
    )

    assert report["jpg_metrics"]["jpg_ssim"]["ci_low"] > 0.0
    assert report["promotable"] is False
    assert report["independent_evidence_count"] == 1
    assert "insufficient_independent_fold_seed_evidence" in report["reasons"]


def test_explicit_evidence_ids_bind_records_without_fold_seed_fields() -> None:
    parent, candidate = _records([0.02] * 4, [0.0] * 4)
    bound_parent: list[dict[str, object]] = []
    bound_candidate: list[dict[str, object]] = []
    for evidence_id in ("fold-zero", "fold-one"):
        bound_parent.extend({**row, "evidence_id": evidence_id} for row in parent)
        bound_candidate.extend({**row, "evidence_id": evidence_id} for row in candidate)
    provenance = [
        {
            "evidence_id": evidence_id,
            "parent_manifest_sha256": f"manifest-{evidence_id}",
            "candidate_manifest_sha256": f"manifest-{evidence_id}",
        }
        for evidence_id in ("fold-zero", "fold-one")
    ]

    report = evaluate_roi_jpg_promotion(
        bound_parent,
        bound_candidate,
        roi_audit=_verified_audit(),
        fold_seed_provenance=provenance,
        bootstrap_samples=100,
    )

    assert report["promotable"] is True
    assert report["independent_fold_seed_count"] == 2
    assert {row["evidence_id"] for row in report["fold_seed_provenance"]} == {
        "fold-zero",
        "fold-one",
    }
