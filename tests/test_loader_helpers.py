from __future__ import annotations

from pathlib import Path

import pytest

from virtual_staining.data.loader_helpers import (
    prepare_activity_sampling,
    prepare_authoritative_fold_split,
    require_verified_filename_grid,
)


def _official_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for organ in ("colon", "liver"):
        for roi_number in range(3):
            for col in range(2):
                stem = f"ROI{roi_number:03d}_00_{col:02d}"
                rows.append(
                    {
                        "organ": organ,
                        "split": "official_train",
                        "source_split": "official_train",
                        "stem": stem,
                        "canonical_key": f"{organ}/{stem}",
                        "roi_id_source": "filename_coordinate",
                    }
                )
    return rows


def _verified_audit(row_count: int) -> dict[str, object]:
    return {
        "total_rows": row_count,
        "parsed_rows": row_count,
        "filename_grid_verified": True,
        "duplicate_coordinates": [],
        "train_val_shared_rois": [],
        "cross_split_adjacent_pairs": [],
    }


def test_prepare_authoritative_fold_split_writes_deterministic_contract(
    tmp_path: Path,
) -> None:
    rows = _official_rows()
    source_indices = [10 + 2 * index for index in range(len(rows))]
    output = tmp_path / "含 空格" / "inner_folds.csv"

    split = prepare_authoritative_fold_split(
        rows,
        fold=1,
        fold_count=3,
        seed=2026,
        source_indices=source_indices,
        roi_audit=_verified_audit(len(rows)),
        output_csv=output,
    )
    repeated = prepare_authoritative_fold_split(
        rows,
        fold=1,
        fold_count=3,
        seed=2026,
        source_indices=source_indices,
        roi_audit=_verified_audit(len(rows)),
    )

    assert set(split.train_indices).isdisjoint(split.validation_indices)
    assert set(split.train_indices) | set(split.validation_indices) == set(source_indices)
    assert len(split.train_rows) + len(split.validation_rows) == len(rows)
    assert split.assignment_sha256 == repeated.assignment_sha256
    assert split.assignment_csv == output.resolve()
    assert split.assignment_hash_file is not None
    assert split.assignment_hash_file.read_text(encoding="utf-8").startswith(
        split.assignment_sha256
    )
    assert split.to_dict()["assignment_sha256"] == split.assignment_sha256


@pytest.mark.parametrize(
    ("audit_update", "expected"),
    (
        ({"filename_grid_verified": False}, "unverified_filename_coordinates"),
        ({"parsed_rows": 1}, "incomplete_filename_coordinates"),
        ({"duplicate_coordinates": [{"row": 0}]}, "duplicate_coordinates"),
        ({"train_val_shared_rois": ["colon/ROI000"]}, "train_val_roi_overlap"),
        (
            {"cross_split_adjacent_pairs": [{"first": "a", "second": "b"}]},
            "cross_split_adjacent_patches",
        ),
    ),
)
def test_verified_grid_helper_rejects_unverified_or_leaking_audits(
    audit_update: dict[str, object],
    expected: str,
) -> None:
    audit = _verified_audit(4)
    audit.update(audit_update)

    with pytest.raises(ValueError, match=expected):
        require_verified_filename_grid(audit, minimum_rows=4)


def test_activity_plan_default_off_does_not_require_activity_values() -> None:
    rows = [{"split": "val"}, {"split": "official_test"}]

    plan = prepare_activity_sampling(rows)

    assert plan.enabled is False
    assert plan.dataloader_kwargs() == {"shuffle": True}
    assert plan.state_dict()["sampler"] is None


def test_activity_plan_uses_precomputed_train_values_and_source_indices() -> None:
    rows = [
        {"split": "train", "dapi_activity": value, "dapi_path": "missing.jpg"}
        for value in (0.1, 0.2, 0.8, 0.9)
    ]
    source_indices = [2, 5, 9, 12]
    plan = prepare_activity_sampling(
        rows,
        enabled=True,
        source_indices=source_indices,
        num_bins=2,
        seed=8,
    )

    assert plan.enabled is True
    assert plan.dataloader_kwargs()["shuffle"] is False
    assert set(plan.sampler or ()) == set(source_indices)


def test_activity_plan_resume_restores_exact_remaining_indices() -> None:
    rows = [
        {"split": "official_train", "dapi_activity": value}
        for value in (0.1, 0.3, 0.7, 0.9)
    ]
    first = prepare_activity_sampling(
        rows,
        enabled=True,
        num_bins=2,
        seed=11,
        num_samples=9,
    )
    first.set_epoch(3)
    assert first.sampler is not None
    iterator = iter(first.sampler)
    consumed = [next(iterator), next(iterator)]
    state = first.state_dict()
    expected_remaining = list(iterator)

    resumed = prepare_activity_sampling(
        rows,
        enabled=True,
        num_bins=2,
        seed=11,
        num_samples=9,
    )
    resumed.load_state_dict(state)
    assert resumed.sampler is not None
    assert list(resumed.sampler) == expected_remaining
    assert len(consumed + expected_remaining) == 9


@pytest.mark.parametrize("forbidden_split", ("val", "test", "official_test"))
def test_activity_plan_rejects_validation_and_test_rows(forbidden_split: str) -> None:
    rows = [
        {"split": "train", "dapi_activity": 0.1},
        {"split": forbidden_split, "dapi_activity": 0.9},
    ]

    with pytest.raises(ValueError, match="training rows only"):
        prepare_activity_sampling(rows, enabled=True)
