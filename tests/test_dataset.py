from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from virtual_staining.data.dataset import InferenceDataset, VirtualStainingDataset
from virtual_staining.data.manifest import MANIFEST_FIELDS


def _write_image(path: Path, *, rgb: bool, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((10, 12))
    if rgb:
        array = np.stack(((x + offset) % 256, (y + offset) % 256, (x + y + offset) % 256), axis=-1)
        Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)
    else:
        Image.fromarray(((x + y + offset) % 256).astype(np.uint8), mode="L").save(path)


def _manifest(root: Path, tmp_path: Path) -> Path:
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        {
            "organ": "colon",
            "split": "train",
            "roi_id": "ROI001",
            "group_id": "colon/ROI001",
            "patch_id": "ROI001_00_00",
            "stem": "ROI001_00_00",
            "canonical_key": "colon/roi001_00_00",
            "dapi_path": "colon/DAPI/ROI001_00_00.png",
            "hla_dr_path": "colon/HLA-DR/ROI001_00_00.png",
            "cd45ro_path": "colon/CD45RO/ROI001_00_00.png",
            "vimentin_path": "colon/Vimentin/ROI001_00_00.png",
            "cd68_path": "colon/CD68/ROI001_00_00.png",
        }
    )
    _write_image(root / row["dapi_path"], rgb=False, offset=1)
    for index, column in enumerate(("hla_dr_path", "cd45ro_path", "vimentin_path", "cd68_path")):
        _write_image(root / row[column], rgb=True, offset=10 + index)
    manifest = tmp_path / "含中文 manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return manifest


def test_single_task_dataset_preserves_gray_and_rgb_channels(tmp_path: Path) -> None:
    root = tmp_path / "含 空格数据"
    manifest = _manifest(root, tmp_path)
    dataset = VirtualStainingDataset(manifest, root, targets="CD68")

    item = dataset[0]

    assert item["input"].shape == (1, 10, 12)
    assert item["target"].shape == (3, 10, 12)
    assert item["targets"]["CD68"].data_ptr() == item["target"].data_ptr()
    assert item["input"].dtype == torch.float32
    assert 0.0 <= float(item["input"].min()) <= float(item["input"].max()) <= 1.0
    assert item["canonical_key"] == "colon/roi001_00_00"
    assert Path(item["dapi_path"]).is_file()


def test_multi_task_dataset_and_channel_conversion(tmp_path: Path) -> None:
    root = tmp_path / "数据"
    manifest = _manifest(root, tmp_path)
    dataset = VirtualStainingDataset(
        manifest,
        root,
        targets=("hla_dr", "cd45ro", "vimentin", "cd68"),
        input_channels=3,
        target_channels=1,
    )

    item = dataset[0]

    assert item["input"].shape == (3, 10, 12)
    assert set(item["targets"]) == {"HLA-DR", "CD45RO", "Vimentin", "CD68"}
    assert all(tensor.shape == (1, 10, 12) for tensor in item["targets"].values())
    assert "target" not in item


def test_inference_dataset_requires_only_dapi(tmp_path: Path) -> None:
    root = tmp_path / "data"
    manifest = _manifest(root, tmp_path)
    inference = InferenceDataset(manifest, root)

    item = inference[0]

    assert item["targets"] == {}
    assert "target" not in item


def test_missing_and_corrupt_images_raise_clear_errors(tmp_path: Path) -> None:
    root = tmp_path / "data"
    manifest = _manifest(root, tmp_path)
    missing = root / "colon" / "CD68" / "ROI001_00_00.png"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="missing CD68 image"):
        VirtualStainingDataset(manifest, root, targets="CD68")

    missing.write_bytes(b"not an image")
    dataset = VirtualStainingDataset(manifest, root, targets="CD68", strict=False)
    with pytest.raises(ValueError, match="Failed to load CD68"):
        dataset[0]


def test_empty_manifest_has_official_test_guidance(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    manifest = tmp_path / "empty.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS).writeheader()
    with pytest.raises(ValueError, match="official test data is missing"):
        InferenceDataset(manifest, root)
