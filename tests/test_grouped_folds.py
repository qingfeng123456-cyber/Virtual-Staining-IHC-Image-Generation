from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

from virtual_staining.data.grouped_folds import build_authoritative_roi_folds


def _official_rows(specification: dict[str, list[int]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for organ, roi_sizes in specification.items():
        for roi_number, patch_count in enumerate(roi_sizes):
            for patch in range(patch_count):
                row, col = divmod(patch, 3)
                stem = f"ROI{roi_number:03d}_{row:02d}_{col:02d}"
                rows.append(
                    {
                        "organ": organ,
                        "split": "train",
                        "source_split": "train",
                        "stem": stem,
                        "canonical_key": f"{organ}/{stem}",
                        "roi_id_source": "filename_regex",
                    }
                )
    return rows


def test_grouped_folds_never_split_roi_and_write_stable_assignments(tmp_path: Path) -> None:
    rows = _official_rows({"colon": [4, 3, 2], "liver": [3, 2, 1]})
    result = build_authoritative_roi_folds(rows, fold_count=3, seed=2026)
    repeated = build_authoritative_roi_folds(rows, fold_count=3, seed=2026)

    folds_by_roi: dict[tuple[str, str], set[int]] = defaultdict(set)
    for assignment in result.assignments:
        folds_by_roi[(assignment.organ, assignment.roi_id)].add(assignment.fold)
    assert all(len(folds) == 1 for folds in folds_by_roi.values())
    assert result.assignments == repeated.assignments
    assert result.assignment_sha256 == repeated.assignment_sha256
    assert len(result.assignment_sha256) == 64
    assert sum(result.fold_patch_counts) == len(rows)
    for fold in range(3):
        train = set(result.training_indices(fold))
        validation = set(result.validation_indices(fold))
        assert train.isdisjoint(validation)
        assert train | validation == set(range(len(rows)))

    output = result.write_csv(tmp_path / "含 空格" / "folds.csv")
    with output.open(newline="", encoding="utf-8-sig") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(rows)
    assert [int(row["source_index"]) for row in written] == list(range(len(rows)))


def test_grouped_folds_balance_patch_counts_within_each_organ() -> None:
    rows = _official_rows({"colon": [2] * 6, "liver": [2] * 3, "stomach": [2] * 3})

    result = build_authoritative_roi_folds(rows, fold_count=3, seed=17)

    assert result.fold_patch_counts == (8, 8, 8)
    for organ in ("colon", "liver", "stomach"):
        per_fold = [counts.get(organ, 0) for counts in result.fold_organ_patch_counts]
        assert max(per_fold) == min(per_fold)


@pytest.mark.parametrize("invalid_split", ("val", "test", "official_test"))
def test_grouped_folds_reject_non_train_rows(invalid_split: str) -> None:
    rows = _official_rows({"colon": [1, 1]})
    rows[0]["split"] = invalid_split
    rows[0]["source_split"] = invalid_split

    with pytest.raises(ValueError, match="not a training row"):
        build_authoritative_roi_folds(rows, fold_count=2)


def test_grouped_folds_reject_unparsed_surrogate_and_nonofficial_source() -> None:
    rows = _official_rows({"colon": [1, 1]})
    rows[0]["stem"] = "00000"
    rows[0]["roi_id_source"] = "surrogate_numeric_block"
    with pytest.raises(ValueError, match="no authoritative filename"):
        build_authoritative_roi_folds(rows, fold_count=2)

    rows = _official_rows({"colon": [1, 1]})
    rows[0]["source_split"] = "generated"
    with pytest.raises(ValueError, match="official train source"):
        build_authoritative_roi_folds(rows, fold_count=2)


def test_grouped_folds_reject_duplicate_coordinates() -> None:
    rows = _official_rows({"colon": [1, 1]})
    duplicate = dict(rows[0])
    duplicate["canonical_key"] = "colon/duplicate-name"
    rows.append(duplicate)

    with pytest.raises(ValueError, match="Duplicate authoritative ROI coordinate"):
        build_authoritative_roi_folds(rows, fold_count=2)


def test_grouped_folds_preserve_noncontiguous_manifest_indices() -> None:
    rows = _official_rows({"colon": [2, 2], "liver": [2, 2]})
    source_indices = [100 + 3 * index for index in range(len(rows))]

    result = build_authoritative_roi_folds(
        rows,
        fold_count=2,
        seed=5,
        source_indices=source_indices,
    )

    train = set(result.training_indices(0))
    validation = set(result.validation_indices(0))
    assert train.isdisjoint(validation)
    assert train | validation == set(source_indices)


@pytest.mark.parametrize(
    "source_indices",
    ([0], [0, 0], [0, -1], [0, True]),
)
def test_grouped_folds_reject_invalid_source_indices(
    source_indices: list[int],
) -> None:
    rows = _official_rows({"colon": [1, 1]})

    with pytest.raises(ValueError, match="source_indices"):
        build_authoritative_roi_folds(
            rows,
            fold_count=2,
            source_indices=source_indices,
        )
