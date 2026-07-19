from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from virtual_staining.constants import normalize_marker
from virtual_staining.data.discovery import (
    discover_data_root,
    find_marker_directories,
    normalize_stem,
)


def _save_image(path: Path, value: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8), mode="RGB").save(path)


def test_marker_and_stem_normalization() -> None:
    assert normalize_marker("hla_dr") == "HLA-DR"
    assert normalize_marker("HLA-DR") == "HLA-DR"
    assert normalize_marker("cd45_ro") == "CD45RO"
    assert normalize_marker("VIMENTIN") == "Vimentin"
    assert normalize_stem(" ROI000_00_00 ") == "roi000_00_00"


def test_discovers_nested_dataset_sample_and_writes_audit(tmp_path: Path) -> None:
    workspace = tmp_path / "含 空格的工作区"
    data_root = workspace / "dataset" / "dataset_sample"
    for index, marker in enumerate(("dapi", "hla_dr", "CD45RO", "vimentin", "cd68")):
        _save_image(data_root / "colon" / marker / "ROI000_00_00.jpg", value=30 + index)
        _save_image(
            workspace / "outputs" / "old_run" / "predictions" / marker / "fake.jpg",
            value=90 + index,
        )
    output = workspace / "artifacts" / "data_discovery.json"

    result = discover_data_root("AUTO", workspace=workspace, output_path=output)

    assert result.selected_root == data_root.resolve()
    assert {item.marker for item in result.marker_directories} == {
        "DAPI",
        "HLA-DR",
        "CD45RO",
        "Vimentin",
        "CD68",
    }
    assert all(item.organ == "colon" for item in result.marker_directories)
    assert all(
        "outputs" not in item.path.parts
        for candidate in result.candidates
        for item in candidate.marker_directories
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert Path(payload["selected_root"]) == data_root.resolve()
    assert payload["candidate_count"] >= 2


def test_recognizes_split_organ_marker_layout(tmp_path: Path) -> None:
    root = tmp_path / "比赛数据"
    _save_image(root / "train" / "colon" / "DAPI" / "00000.png")
    _save_image(root / "train" / "colon" / "CD68" / "00000.png")
    _save_image(root / "test" / "colon" / "DAPI" / "00001.png")

    found = find_marker_directories(root)

    contexts = {(item.marker, item.organ, item.split) for item in found}
    assert ("DAPI", "colon", "train") in contexts
    assert ("CD68", "colon", "train") in contexts
    assert ("DAPI", "colon", "test") in contexts


def test_config_relative_root_is_resolved_from_project(tmp_path: Path) -> None:
    root = tmp_path / "dataset" / "custom_name"
    _save_image(root / "colon" / "DAPI" / "00000.png")
    _save_image(root / "colon" / "CD68" / "00000.png")
    config = tmp_path / "configs" / "local.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("data:\n  root: dataset/custom_name\n", encoding="utf-8")
    result = discover_data_root(
        "AUTO",
        workspace=tmp_path,
        config_path=config,
        output_path=tmp_path / "audit.json",
    )
    assert result.selected_root == root.resolve()


def test_explicit_root_wins_over_other_valid_workspace_candidates(tmp_path: Path) -> None:
    workspace_root = tmp_path / "dataset" / "workspace_data"
    explicit_root = tmp_path / "incoming" / "official_data"
    for root, value in ((workspace_root, 32), (explicit_root, 96)):
        _save_image(root / "train" / "colon" / "DAPI" / "ROI000_00_00.jpg", value)
        _save_image(root / "train" / "colon" / "CD68" / "ROI000_00_00.jpg", value)

    result = discover_data_root(
        explicit_root,
        workspace=tmp_path,
        output_path=tmp_path / "audit.json",
    )

    assert result.selected_root == explicit_root.resolve()
    assert result.candidates[0].source.startswith("explicit")


def test_missing_data_root_reports_bounded_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    workspace.mkdir()
    try:
        discover_data_root("AUTO", workspace=workspace, output_path=workspace / "audit.json")
    except FileNotFoundError as exc:
        assert "Bounded candidates" in str(exc)
    else:
        raise AssertionError("Discovery must fail clearly when no image data exists")
