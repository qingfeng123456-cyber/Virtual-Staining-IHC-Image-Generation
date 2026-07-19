from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from virtual_staining.data.roi_index import (
    PatchCoordinate,
    ROIIndex,
    audit_roi_grid,
    parse_patch_coordinate,
    parse_patch_coordinate_status,
)


def _grid_rows(
    root: Path,
    *,
    transposed_names: bool = False,
    split_by_coordinate: dict[tuple[int, int], str] | None = None,
) -> list[dict[str, str]]:
    directory = root / "colon" / "DAPI"
    directory.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((16, 16))
    mosaic = (20 + y * 2 + x * 3).astype(np.uint8)
    rows: list[dict[str, str]] = []
    for physical_row in range(2):
        for physical_col in range(2):
            named_row, named_col = (
                (physical_col, physical_row)
                if transposed_names
                else (physical_row, physical_col)
            )
            stem = f"ROI007_{named_row:02d}_{named_col:02d}"
            path = directory / f"{stem}.PNG"
            patch = mosaic[
                physical_row * 8 : (physical_row + 1) * 8,
                physical_col * 8 : (physical_col + 1) * 8,
            ]
            Image.fromarray(patch, mode="L").save(path)
            coordinate = (named_row, named_col)
            rows.append(
                {
                    "organ": "colon",
                    "split": (split_by_coordinate or {}).get(coordinate, "train"),
                    "stem": stem,
                    "patch_id": stem,
                    "canonical_key": f"colon/{stem}",
                    "dapi_path": path.relative_to(root).as_posix(),
                }
            )
    return rows


def test_strict_patch_coordinate_parser_accepts_only_authoritative_pattern() -> None:
    expected = PatchCoordinate("ROI000", 0, 123)
    assert parse_patch_coordinate("ROI000_00_123.jpg") == expected
    assert parse_patch_coordinate("roi000_0_0123.JPEG") == expected
    assert parse_patch_coordinate(Path("目录") / "RoI000_000_123.PnG") == expected
    for invalid in (
        "00000.jpg",
        "prefix_ROI000_00_00.jpg",
        "ROI000_00_00_extra.jpg",
        "ROI000_-1_00.jpg",
        "ROI000_00.jpg",
        "ROIABC_00_00.jpg",
        "",
    ):
        result = parse_patch_coordinate_status(invalid)
        assert result.coordinate is None
        assert result.source == "unparsed"
        assert result.reason


def test_roi_index_never_crosses_organ_split_or_roi() -> None:
    rows = [
        {
            "organ": organ,
            "split": split,
            "stem": f"{roi}_00_{col:02d}",
            "canonical_key": f"{organ}/{split}/{roi}/{col}",
        }
        for organ, split, roi, col in (
            ("colon", "train", "ROI001", 0),
            ("colon", "train", "ROI001", 1),
            ("colon", "val", "ROI001", 1),
            ("colon", "train", "ROI002", 1),
            ("liver", "train", "ROI001", 1),
        )
    ]
    index = ROIIndex(rows, require_verified=True)
    assert index.neighbor_index(0, 0, 1) == 1
    assert index.neighbor_index(0, 0, -1) is None
    neighborhood = dict(index.neighborhood_indices(0))
    assert neighborhood[(0, 0)] == 0
    assert neighborhood[(0, 1)] == 1
    assert sum(value is not None for value in neighborhood.values()) == 2


def test_roi_grid_audit_verifies_normal_direction_and_continuity(tmp_path: Path) -> None:
    rows = _grid_rows(tmp_path)
    audit = audit_roi_grid(rows, tmp_path, border_width=1)

    assert audit.parsed_rows == 4
    assert audit.filename_grid_verified is True
    assert audit.boundary.horizontal_pair_count == 2
    assert audit.boundary.vertical_pair_count == 2
    assert audit.boundary.best_orientation == "row_down_col_right"
    assert audit.boundary.direction_verified is True
    assert audit.boundary.continuity_verified is True
    assert audit.context_enabled is True
    assert audit.grids[0].missing_coordinates == ()


def test_roi_grid_audit_detects_coordinate_transpose(tmp_path: Path) -> None:
    rows = _grid_rows(tmp_path, transposed_names=True)
    audit = audit_roi_grid(rows, tmp_path, border_width=1)

    assert audit.boundary.best_orientation == "transposed_row_down_col_right"
    assert audit.boundary.direction_verified is False
    assert audit.context_enabled is False
    assert "coordinate_direction_not_verified" in audit.context_gate_reasons


def test_roi_grid_audit_blocks_cross_split_roi_and_adjacent_context(tmp_path: Path) -> None:
    rows = _grid_rows(tmp_path, split_by_coordinate={(0, 1): "val", (1, 1): "val"})
    audit = audit_roi_grid(rows, tmp_path, border_width=1)

    assert audit.train_val_shared_rois == ("colon/ROI007",)
    assert len(audit.cross_split_adjacent_pairs) == 2
    assert audit.context_enabled is False
    assert "train_val_roi_overlap" in audit.context_gate_reasons
    assert "cross_split_adjacent_patches" in audit.context_gate_reasons


def test_numeric_stems_are_audit_evidence_not_verified_coordinates(tmp_path: Path) -> None:
    rows = [
        {
            "organ": "colon",
            "split": "train",
            "stem": f"{index:05d}",
            "canonical_key": f"colon/{index:05d}",
            "dapi_path": f"colon/DAPI/{index:05d}.jpg",
            "roi_id": "surrogate_00000",
            "roi_id_source": "surrogate_numeric_block",
        }
        for index in range(4)
    ]
    audit = audit_roi_grid(rows, tmp_path)

    assert audit.parsed_rows == 0
    assert audit.filename_grid_verified is False
    assert audit.context_enabled is False
    assert "unverified_filename_coordinates" in audit.context_gate_reasons

