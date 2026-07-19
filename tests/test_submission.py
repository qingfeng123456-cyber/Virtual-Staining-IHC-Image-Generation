from __future__ import annotations

import csv
import zipfile

import numpy as np
import pytest
from PIL import Image

from virtual_staining.submission.validator import SubmissionValidationError, validate_submission
from virtual_staining.submission.writer import build_submission, submission_filename

FIELDS = [
    "organ",
    "split",
    "roi_id",
    "patch_id",
    "canonical_key",
    "dapi_path",
    "width",
    "height",
    "cd68_channels",
]


def _write_manifest(path, stems: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for stem in stems:
            writer.writerow(
                {
                    "organ": "colon",
                    "split": "smoke_test",
                    "roi_id": "surrogate_000",
                    "patch_id": stem,
                    "canonical_key": f"colon/{stem}",
                    "dapi_path": f"colon/DAPI/{stem}.jpg",
                    "width": 16,
                    "height": 16,
                    "cd68_channels": 3,
                }
            )


def _write_prediction(path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(1, 254, size=(16, 16, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=100, subsampling=0)


def test_submission_build_validate_and_zip_root(tmp_path) -> None:
    manifest = tmp_path / "manifests" / "smoke_test_manifest.csv"
    stems = ["00000", "00001"]
    _write_manifest(manifest, stems)
    predictions = tmp_path / "predictions" / "CD68"
    for index, stem in enumerate(stems):
        _write_prediction(predictions / f"{stem}.jpg", index)
    build = build_submission(
        tmp_path / "predictions",
        manifest,
        "cd68",
        tmp_path / "smoke_submission",
        smoke=True,
    )
    assert build["count"] == 2
    assert submission_filename("00000_fake") == "00000_fake.jpg"
    report = validate_submission(
        tmp_path / "smoke_submission" / "results",
        manifest,
        "CD68",
        zip_path=build["zip_path"],
        artifact_dir=tmp_path / "artifacts",
    )
    assert report["valid"]
    with zipfile.ZipFile(build["zip_path"]) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
    assert members
    assert all(name.startswith("results/") for name in members)
    assert not any(name.startswith("smoke_submission/") for name in members)


def test_empty_manifest_and_missing_output_fail(tmp_path) -> None:
    empty = tmp_path / "test_manifest.csv"
    _write_manifest(empty, [])
    (tmp_path / "predictions").mkdir()
    with pytest.raises(ValueError, match="empty"):
        build_submission(tmp_path / "predictions", empty, "CD68", tmp_path / "results")

    manifest = tmp_path / "smoke_test_manifest.csv"
    _write_manifest(manifest, ["missing"])
    results = tmp_path / "package" / "results" / "test" / "CD68"
    results.mkdir(parents=True)
    with pytest.raises(SubmissionValidationError, match="Missing output"):
        validate_submission(tmp_path / "package", manifest, "CD68", artifact_dir=tmp_path / "audit")


def test_submission_rebuild_removes_stale_files_and_rejects_wrong_mode(tmp_path) -> None:
    manifest = tmp_path / "renamed.csv"
    _write_manifest(manifest, ["00000"])
    prediction = tmp_path / "predictions" / "CD68" / "00000.jpg"
    _write_prediction(prediction, 7)
    package = tmp_path / "package"
    stale = package / "results" / "test" / "CD68" / "stale_fake.jpg"
    _write_prediction(stale, 8)
    build_submission(tmp_path / "predictions", manifest, "CD68", package, smoke=True)
    assert not stale.exists()

    gray = np.full((16, 16), 100, dtype=np.uint8)
    output = package / "results" / "test" / "CD68" / "00000_fake.jpg"
    Image.fromarray(gray, mode="L").save(output, format="JPEG")
    with pytest.raises(SubmissionValidationError, match="channel count"):
        validate_submission(package, manifest, "CD68", artifact_dir=tmp_path / "audit")
