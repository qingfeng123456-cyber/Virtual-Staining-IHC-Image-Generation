"""Deterministic ablation budget and evidence provenance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.engine.ablation_budget import (
    INNER_FOLD_VALIDATION,
    OFFICIAL_OUTER_VALIDATION,
    bind_evidence_records,
    build_promotion_provenance,
    resolve_ablation_budget,
)


def _budgets() -> dict[str, object]:
    return {
        "smoke": {"epochs": 2},
        "screen": {"epochs": 20},
        "confirm": {"epochs": 80, "folds": [0, 1], "seeds": [2026]},
        "full": {"epochs": 120, "folds": [], "seeds": [2026]},
    }


def test_screen_defaults_to_one_stable_evidence_run_independent_of_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = resolve_ablation_budget(_budgets(), "screen", default_fold=3, default_seed=17)
    nested = tmp_path / "中文 工作区" / "另一个目录"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    second = resolve_ablation_budget(_budgets(), "screen", default_fold=3, default_seed=17)

    assert first == second
    assert first.epochs == 20
    assert first.validation_mode == INNER_FOLD_VALIDATION
    assert first.evidence_count == 1
    evidence = first.evidence_runs[0]
    assert evidence.fold == 3
    assert evidence.seed == 17
    assert evidence.suffix == "_fold3_seed17"
    assert evidence.evidence_id == "fold3_seed17"
    assert evidence.make_run_id("套件_A1_screen") == "套件_A1_screen_fold3_seed17"


def test_confirm_builds_fold_by_seed_product_in_declared_order() -> None:
    budgets = _budgets()
    budgets["confirm"] = {
        "epochs": 80,
        "folds": [2, 0],
        "seeds": [2026, 9],
    }

    plan = resolve_ablation_budget(budgets, "confirm")

    assert [(run.fold, run.seed) for run in plan.evidence_runs] == [
        (2, 2026),
        (2, 9),
        (0, 2026),
        (0, 9),
    ]
    assert len({run.evidence_id for run in plan.evidence_runs}) == 4
    assert len({run.suffix for run in plan.evidence_runs}) == 4


def test_confirm_with_only_one_independent_evidence_hard_fails() -> None:
    budgets = _budgets()
    budgets["confirm"] = {"epochs": 80, "folds": [0], "seeds": [2026]}

    with pytest.raises(ValueError, match="at least two independent"):
        resolve_ablation_budget(budgets, "confirm")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("folds", [0, 0], "unique"),
        ("seeds", [2026, 2026], "unique"),
        ("folds", [-1], "nonnegative"),
        ("seeds", [-3], "nonnegative"),
        ("seeds", [], "cannot be empty"),
    ],
)
def test_duplicate_empty_and_negative_evidence_axes_are_rejected(
    field: str,
    value: list[int],
    message: str,
) -> None:
    budgets = _budgets()
    budgets["screen"] = {"epochs": 20, "folds": [0], "seeds": [2026]}
    budgets["screen"][field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=message):
        resolve_ablation_budget(budgets, "screen")


def test_full_empty_folds_explicitly_means_official_outer_validation() -> None:
    plan = resolve_ablation_budget(_budgets(), "full")

    assert plan.validation_mode == OFFICIAL_OUTER_VALIDATION
    assert plan.uses_official_outer_validation is True
    assert plan.evidence_runs[0].fold is None
    assert plan.evidence_runs[0].record_fold == "official_outer"
    assert plan.evidence_runs[0].make_run_id("full") == "full_outer_seed2026"

    budgets = _budgets()
    budgets["screen"] = {
        "epochs": 20,
        "folds": [],
        "validation_mode": OFFICIAL_OUTER_VALIDATION,
    }
    with pytest.raises(ValueError, match="cannot be empty"):
        resolve_ablation_budget(budgets, "screen")


def test_bind_evidence_records_copies_and_rejects_relabeling() -> None:
    evidence = resolve_ablation_budget(_budgets(), "screen").evidence_runs[0]
    source = [{"canonical_key": "colon/ROI000_00_00", "jpg_ssim": 0.8}]

    bound = bind_evidence_records(source, evidence)

    assert bound[0]["fold"] == 0
    assert bound[0]["seed"] == 2026
    assert bound[0]["evidence_id"] == "fold0_seed2026"
    assert "fold" not in source[0]
    with pytest.raises(ValueError, match="conflicting fold"):
        bind_evidence_records([{**source[0], "fold": 1}], evidence)


def test_promotion_provenance_requires_exact_paired_validation_hashes() -> None:
    runs = resolve_ablation_budget(_budgets(), "confirm").evidence_runs
    hashes = {run.evidence_id: f"manifest-{run.fold}" for run in runs}

    provenance = build_promotion_provenance(
        runs,
        parent_validation_hashes=hashes,
        candidate_validation_hashes=hashes,
    )

    assert [row["evidence_id"] for row in provenance] == [
        "fold0_seed2026",
        "fold1_seed2026",
    ]
    assert all(row["parent_manifest_sha256"] == row["candidate_manifest_sha256"] for row in provenance)

    mismatched = dict(hashes)
    mismatched["fold1_seed2026"] = "different"
    with pytest.raises(ValueError, match="hash mismatch"):
        build_promotion_provenance(
            runs,
            parent_validation_hashes=hashes,
            candidate_validation_hashes=mismatched,
        )
    with pytest.raises(ValueError, match="planned evidence ids"):
        build_promotion_provenance(
            runs,
            parent_validation_hashes={"fold0_seed2026": "manifest-0"},
            candidate_validation_hashes=hashes,
        )


def test_invalid_epoch_budget_and_run_id_fail_early() -> None:
    budgets = _budgets()
    budgets["smoke"] = {"epochs": 0}
    with pytest.raises(ValueError, match="positive"):
        resolve_ablation_budget(budgets, "smoke")

    evidence = resolve_ablation_budget(_budgets(), "screen").evidence_runs[0]
    with pytest.raises(ValueError, match="cannot be empty"):
        evidence.make_run_id("  ")
