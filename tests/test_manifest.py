from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from virtual_staining.data.manifest import (
    MANIFEST_FIELDS,
    build_manifests,
    parse_roi_id,
    read_manifest,
)

MARKERS = ("DAPI", "HLA-DR", "CD45RO", "Vimentin", "CD68")


def _write_sample(root: Path, index: int, organ: str = "colon") -> None:
    for marker_index, marker in enumerate(MARKERS):
        directory = root / organ / marker
        directory.mkdir(parents=True, exist_ok=True)
        y, x = np.indices((12, 12))
        array = np.stack(
            (
                (x + index + marker_index) % 256,
                (y * 2 + index + marker_index) % 256,
                (x + y + index + marker_index) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(directory / f"{index:05d}.jpg", quality=95)


def test_roi_regex_and_numeric_surrogate_groups() -> None:
    assert parse_roi_id("ROI025_00_00") == ("ROI025", "filename_regex")
    assert parse_roi_id("00031") == ("surrogate_00000", "surrogate_numeric_block")
    assert parse_roi_id("00032") == ("surrogate_00001", "surrogate_numeric_block")


def test_build_manifest_pairs_by_stem_and_prevents_hash_leakage(tmp_path: Path) -> None:
    root = tmp_path / "数据 根" / "dataset_sample"
    for index in range(96):
        _write_sample(root, index)
    shutil.copyfile(root / "colon" / "DAPI" / "00000.jpg", root / "colon" / "DAPI" / "00033.jpg")

    result = build_manifests(root, workspace=tmp_path, output_dir="artifacts/manifests", smoke_count=5)
    train = read_manifest(result.train_manifest)
    val = read_manifest(result.val_manifest)
    test = read_manifest(result.test_manifest)
    smoke = read_manifest(result.smoke_test_manifest)

    assert len(train) + len(val) == 96
    assert train and val
    assert test == []
    assert len(smoke) == min(5, len(val))
    assert set(train[0]) == set(MANIFEST_FIELDS)
    assert all(row["canonical_key"] == f"colon/{int(row['stem']):05d}" for row in train + val)
    assert all(row["dapi_path"].startswith("colon/DAPI/") for row in train + val)
    assert all(row["cd68_path"].startswith("colon/CD68/") for row in train + val)
    split_by_stem = {row["stem"]: row["split"] for row in train + val}
    assert split_by_stem["00000"] == split_by_stem["00033"]
    train_groups = {row["group_id"] for row in train}
    val_groups = {row["group_id"] for row in val}
    assert train_groups.isdisjoint(val_groups)
    train_hashes = {row["dapi_sha1"] for row in train}
    val_hashes = {row["dapi_sha1"] for row in val}
    assert train_hashes.isdisjoint(val_hashes)
    assert all(not row["cd68_path"] and row["split"] == "smoke_test" for row in smoke)
    assert all(row["cd68_channels"] == "3" and row["cd68_mode"] == "RGB" for row in smoke)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["official_test_missing"] is True
    assert metadata["official_test_status"] == "missing"
    assert metadata["smoke_test_is_official"] is False
    leakage = json.loads(result.leakage_report_path.read_text(encoding="utf-8"))
    assert leakage["leakage_checks_passed"] is True
    assert leakage["roi_leakage_verifiable"] is False


def test_missing_target_is_excluded_and_recorded(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for index in range(64):
        _write_sample(root, index)
    (root / "colon" / "CD68" / "00010.jpg").unlink()

    result = build_manifests(root, workspace=tmp_path, output_dir="artifacts/manifests")
    rows = read_manifest(result.train_manifest) + read_manifest(result.val_manifest)

    assert len(rows) == 63
    assert "colon/00010" not in {row["canonical_key"] for row in rows}
    with result.bad_samples_path.open("r", encoding="utf-8-sig", newline="") as handle:
        bad = list(csv.DictReader(handle))
    assert any(row["canonical_key"] == "colon/00010" and row["issue"] == "missing_target" for row in bad)


def test_manifest_can_select_one_organ(tmp_path: Path) -> None:
    root = tmp_path / "multi_organ"
    for organ in ("colon", "liver"):
        for index in range(64):
            _write_sample(root, index, organ)
    result = build_manifests(
        root,
        workspace=tmp_path,
        output_dir="selected/manifests",
        organ="LIVER",
    )
    rows = read_manifest(result.train_manifest) + read_manifest(result.val_manifest)
    assert len(rows) == 64
    assert {row["organ"] for row in rows} == {"liver"}


def test_preserves_official_train_val_and_test_is_input_only(tmp_path: Path) -> None:
    root = tmp_path / "official"
    for split, stem in (("train", "ROI001_00_00"), ("val", "ROI002_00_00")):
        for marker_index, marker in enumerate(MARKERS):
            path = root / split / "colon" / marker / f"{stem}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((8, 8), 10 + marker_index, np.uint8), mode="L").save(path)
    test_dapi = root / "test" / "colon" / "DAPI" / "ROI003_00_00.png"
    test_dapi.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), 20, np.uint8), mode="L").save(test_dapi)

    result = build_manifests(root, workspace=tmp_path, output_dir="manifests")

    assert [row["split"] for row in read_manifest(result.train_manifest)] == ["train"]
    assert [row["split"] for row in read_manifest(result.val_manifest)] == ["val"]
    test_rows = read_manifest(result.test_manifest)
    assert len(test_rows) == 1
    assert test_rows[0]["dapi_path"].endswith("ROI003_00_00.png")
    assert not test_rows[0]["cd68_path"]
    assert result.official_test_missing is False


def test_split_marker_layout_builds_roi_grouped_validation_and_official_test(
    tmp_path: Path,
) -> None:
    root = tmp_path / "official"
    for roi_index in range(5):
        stem = f"ROI{roi_index:03d}_00_00"
        for marker_index, marker in enumerate(MARKERS):
            path = root / "train" / marker / f"{stem}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                np.full(
                    (8, 8, 3),
                    10 + roi_index * len(MARKERS) + marker_index,
                    np.uint8,
                ),
                mode="RGB",
            ).save(path)
    test_path = root / "test" / "DAPI" / "ROI005_00_00.jpg"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 20, np.uint8), mode="RGB").save(test_path)

    result = build_manifests(
        root,
        workspace=tmp_path,
        output_dir="manifests",
        val_fraction=0.2,
        seed=2026,
    )

    train_rows = read_manifest(result.train_manifest)
    val_rows = read_manifest(result.val_manifest)
    test_rows = read_manifest(result.test_manifest)
    assert len(train_rows) == 4
    assert len(val_rows) == 1
    assert len(test_rows) == 1
    assert {row["organ"] for row in train_rows + val_rows + test_rows} == {"unknown"}
    assert {row["roi_id"] for row in train_rows}.isdisjoint(
        {row["roi_id"] for row in val_rows}
    )
    assert all(row["roi_id_source"] == "filename_regex" for row in train_rows + val_rows)
    assert test_rows[0]["dapi_path"] == "test/DAPI/ROI005_00_00.jpg"
    assert not test_rows[0]["cd68_path"]
    assert result.official_test_missing is False
