from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import virtual_staining.data.neighborhood as neighborhood_module
from virtual_staining.data.neighborhood import (
    NeighborhoodDataset,
    NeighborhoodInferenceDataset,
    context_offsets,
)


def _write_gray(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), value, dtype=np.uint8), mode="L").save(path)


def _neighborhood_rows(root: Path) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    target_path = root / "colon" / "CD68" / "center.png"
    _write_gray(target_path, 231)
    center_index = -1
    for row in range(3):
        for col in range(3):
            split = "val" if (row, col) == (0, 0) else "train"
            stem = f"ROI010_{row:02d}_{col:02d}"
            dapi_path = root / "colon" / "DAPI" / f"{stem}.png"
            _write_gray(dapi_path, 10 + row * 20 + col)
            rows.append(
                {
                    "organ": "colon",
                    "split": split,
                    "stem": stem,
                    "patch_id": stem,
                    "roi_id": "ROI010",
                    "canonical_key": f"colon/{stem}",
                    "dapi_path": dapi_path.relative_to(root).as_posix(),
                    "cd68_path": (
                        target_path.relative_to(root).as_posix()
                        if (row, col) == (1, 1)
                        else "target_neighbor_must_not_be_opened.png"
                    ),
                }
            )
            if (row, col) == (1, 1):
                center_index = len(rows) - 1
    return rows, center_index


def test_context_offsets_are_canonical_row_major() -> None:
    offsets = context_offsets(3)
    assert offsets.dtype == torch.int64
    assert offsets.tolist() == [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 0],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ]
    with pytest.raises(ValueError, match="positive odd"):
        context_offsets(2)


def test_neighborhood_is_dapi_only_and_never_crosses_split(tmp_path: Path) -> None:
    rows, center_index = _neighborhood_rows(tmp_path)
    dataset = NeighborhoodDataset(
        rows,
        tmp_path,
        targets="CD68",
        input_channels=1,
        target_channels=1,
        strict=False,
    )

    sample = dataset[center_index]

    assert sample["context_tiles"].shape == (9, 1, 8, 8)
    assert sample["context_valid_mask"].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    center = sample["input"]
    assert torch.equal(sample["context_tiles"][0], center)
    assert torch.equal(sample["context_tiles"][4], center)
    assert float(sample["target"].mean()) == pytest.approx(231 / 255)
    assert sample["metadata"]["coordinate"].roi_id == "ROI010"
    assert sample["organ_id"] == "colon"
    assert sample["task_id"] == "CD68"


def test_neighborhood_never_crosses_roi_or_organ(tmp_path: Path) -> None:
    rows, center_index = _neighborhood_rows(tmp_path)
    for organ, roi, value in (("colon", "ROI999", 180), ("liver", "ROI010", 190)):
        stem = f"{roi}_00_00"
        path = tmp_path / organ / "DAPI" / f"{stem}.png"
        _write_gray(path, value)
        rows.append(
            {
                "organ": organ,
                "split": "train",
                "stem": stem,
                "canonical_key": f"{organ}/{stem}",
                "dapi_path": path.relative_to(tmp_path).as_posix(),
                "cd68_path": "unused.png",
            }
        )
    dataset = NeighborhoodDataset(rows, tmp_path, strict=False, input_channels=1)
    sample = dataset[center_index]

    assert sample["context_valid_mask"][0].item() == 0.0
    assert torch.equal(sample["context_tiles"][0], sample["input"])


def test_include_center_false_masks_center_and_copies_pixels(tmp_path: Path) -> None:
    rows, center_index = _neighborhood_rows(tmp_path)
    dataset = NeighborhoodDataset(
        rows,
        tmp_path,
        strict=False,
        input_channels=1,
        include_center=False,
    )
    sample = dataset[center_index]

    assert sample["context_valid_mask"][4].item() == 0.0
    assert torch.equal(sample["context_tiles"][4], sample["input"])


def test_numeric_stems_cannot_enable_verified_neighborhood(tmp_path: Path) -> None:
    dapi = tmp_path / "colon" / "DAPI" / "00000.png"
    target = tmp_path / "colon" / "CD68" / "00000.png"
    _write_gray(dapi, 20)
    _write_gray(target, 30)
    rows = [
        {
            "organ": "colon",
            "split": "train",
            "stem": "00000",
            "roi_id": "surrogate_00000",
            "roi_id_source": "surrogate_numeric_block",
            "canonical_key": "colon/00000",
            "dapi_path": dapi.relative_to(tmp_path).as_posix(),
            "cd68_path": target.relative_to(tmp_path).as_posix(),
        }
    ]

    with pytest.raises(ValueError, match="verified ROI_row_col"):
        NeighborhoodDataset(rows, tmp_path)


def test_unverified_opt_out_is_center_only_not_inferred(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for index in range(2):
        dapi = tmp_path / "colon" / "DAPI" / f"{index:05d}.png"
        target = tmp_path / "colon" / "CD68" / f"{index:05d}.png"
        _write_gray(dapi, 20 + index * 100)
        _write_gray(target, 30)
        rows.append(
            {
                "organ": "colon",
                "split": "train",
                "stem": f"{index:05d}",
                "canonical_key": f"colon/{index:05d}",
                "dapi_path": dapi.relative_to(tmp_path).as_posix(),
                "cd68_path": target.relative_to(tmp_path).as_posix(),
            }
        )
    dataset = NeighborhoodDataset(rows, tmp_path, require_verified_grid=False, input_channels=1)
    sample = dataset[0]

    assert sample["context_valid_mask"].sum().item() == 1.0
    assert sample["context_valid_mask"][4].item() == 1.0
    assert all(torch.equal(tile, sample["input"]) for tile in sample["context_tiles"])


@pytest.mark.parametrize(
    "dataset_class",
    (NeighborhoodDataset, NeighborhoodInferenceDataset),
)
def test_dapi_lru_evicts_least_recently_used_entry_and_clears_on_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_class: type[NeighborhoodDataset] | type[NeighborhoodInferenceDataset],
) -> None:
    rows, _ = _neighborhood_rows(tmp_path)
    original_load = neighborhood_module.load_image_tensor
    calls: dict[str, int] = {}

    def counting_load(path: str | Path, logical_channels: int | None = None) -> torch.Tensor:
        name = Path(path).name
        calls[name] = calls.get(name, 0) + 1
        return original_load(path, logical_channels=logical_channels)

    monkeypatch.setattr(neighborhood_module, "load_image_tensor", counting_load)
    dataset = dataset_class(
        rows,
        tmp_path,
        strict=False,
        input_channels=1,
        cache_size=2,
    )
    paths = [tmp_path / str(rows[index]["dapi_path"]) for index in range(3)]

    dataset._load_dapi_tensor(paths[0])
    dataset._load_dapi_tensor(paths[1])
    dataset._load_dapi_tensor(paths[0])
    dataset._load_dapi_tensor(paths[2])
    dataset._load_dapi_tensor(paths[1])

    assert calls == {
        paths[0].name: 1,
        paths[1].name: 2,
        paths[2].name: 1,
    }
    assert len(dataset._dapi_cache) == 2
    assert not dataset.__getstate__()["_dapi_cache"]


@pytest.mark.parametrize(
    "dataset_class",
    (NeighborhoodDataset, NeighborhoodInferenceDataset),
)
def test_dapi_cache_returns_clones_that_cannot_be_polluted_by_callers(
    tmp_path: Path,
    dataset_class: type[NeighborhoodDataset] | type[NeighborhoodInferenceDataset],
) -> None:
    rows, center_index = _neighborhood_rows(tmp_path)
    dataset = dataset_class(
        rows,
        tmp_path,
        strict=False,
        input_channels=1,
        cache_size=32,
    )
    first = dataset[center_index]
    expected_center = first["input"].clone()
    expected_context = first["context_tiles"].clone()

    first["input"].zero_()
    first["context_tiles"].fill_(1.0)
    second = dataset[center_index]

    assert torch.equal(second["input"], expected_center)
    assert torch.equal(second["context_tiles"], expected_context)


@pytest.mark.parametrize(
    "dataset_class",
    (NeighborhoodDataset, NeighborhoodInferenceDataset),
)
def test_cache_size_zero_is_disabled_and_reloads_dapi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_class: type[NeighborhoodDataset] | type[NeighborhoodInferenceDataset],
) -> None:
    rows, center_index = _neighborhood_rows(tmp_path)
    original_load = neighborhood_module.load_image_tensor
    call_count = 0

    def counting_load(path: str | Path, logical_channels: int | None = None) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return original_load(path, logical_channels=logical_channels)

    monkeypatch.setattr(neighborhood_module, "load_image_tensor", counting_load)
    dataset = dataset_class(rows, tmp_path, strict=False, input_channels=1)

    dataset[center_index]
    first_pass_calls = call_count
    dataset[center_index]

    assert first_pass_calls > 0
    assert call_count == first_pass_calls * 2
    assert dataset.cache_size == 0
    assert not dataset._dapi_cache
