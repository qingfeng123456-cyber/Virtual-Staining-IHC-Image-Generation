"""Strict, report-producing validation for result directories and ZIP files."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .writer import _manifest_rows, _row_stem, canonical_target, submission_filename


class SubmissionValidationError(ValueError):
    """Raised when a submission violates one or more hard requirements."""


def _results_root(submission_dir: str | Path, root_name: str) -> Path:
    base = Path(submission_dir).expanduser().resolve()
    if base.name.casefold() == root_name.casefold():
        return base
    nested = base / root_name
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"Could not find {root_name}/ below {base}")


def _target_channel_column(target: str) -> str:
    return {
        "HLA-DR": "hla_dr_channels",
        "CD45RO": "cd45ro_channels",
        "Vimentin": "vimentin_channels",
        "CD68": "cd68_channels",
    }[target]


def _target_mode_column(target: str) -> str:
    return {
        "HLA-DR": "hla_dr_mode",
        "CD45RO": "cd45ro_mode",
        "Vimentin": "vimentin_mode",
        "CD68": "cd68_mode",
    }[target]


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _validate_zip(
    zip_path: Path,
    expected_members: set[str],
    root_name: str,
) -> list[str]:
    errors: list[str] = []
    if not zip_path.is_file():
        return [f"ZIP file does not exist: {zip_path}"]
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            files = {name for name in archive.namelist() if not name.endswith("/")}
            invalid_roots = sorted(
                name for name in files if not name.startswith(f"{root_name}/")
            )
            if invalid_roots:
                errors.append(f"ZIP members outside {root_name}/: {invalid_roots}")
            missing = sorted(expected_members - files)
            extra = sorted(files - expected_members)
            if missing:
                errors.append(f"ZIP is missing files: {missing}")
            if extra:
                errors.append(f"ZIP has extra files: {extra}")
            bad_member = archive.testzip()
            if bad_member is not None:
                errors.append(f"ZIP CRC check failed for {bad_member}")
    except zipfile.BadZipFile as error:
        errors.append(f"Invalid ZIP file: {error}")
    return errors


def validate_submission(
    submission_dir: str | Path,
    test_manifest: str | Path,
    target: str,
    *,
    zip_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    root_name: str = "results",
    split_name: str = "test",
    fake_suffix: str = "_fake",
    extension: str = ".jpg",
    expected_size: int = 256,
    expected_mode: str | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Validate hierarchy, names, image properties, counts, and ZIP structure."""
    rows = _manifest_rows(test_manifest)
    normalized_target = canonical_target(target)
    results_root = _results_root(submission_dir, root_name)
    target_dir = results_root / split_name / normalized_target
    errors: list[str] = []
    warnings: list[str] = []
    if not target_dir.is_dir():
        errors.append(f"Required target directory does not exist: {target_dir}")
        actual_files: dict[str, Path] = {}
    else:
        actual_files = {
            path.name: path
            for path in target_dir.iterdir()
            if path.is_file()
        }
    row_by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        filename = submission_filename(
            _row_stem(row), fake_suffix=fake_suffix, extension=extension
        )
        key = filename.casefold()
        if key in {name.casefold() for name in row_by_name}:
            errors.append(f"Duplicate expected output filename: {filename}")
        row_by_name[filename] = row
    expected_names = set(row_by_name)
    missing = sorted(expected_names - set(actual_files))
    extra = sorted(set(actual_files) - expected_names)
    if missing:
        errors.append(f"Missing output files: {missing}")
    if extra:
        errors.append(f"Extra output files: {extra}")

    split_dir = results_root / split_name
    if split_dir.is_dir():
        other_targets = sorted(
            path.name for path in split_dir.iterdir() if path.is_dir() and path.name != normalized_target
        )
        if other_targets:
            errors.append(f"Unexpected target directories: {other_targets}")

    channel_column = _target_channel_column(normalized_target)
    mode_column = _target_mode_column(normalized_target)
    file_reports: list[dict[str, Any]] = []
    black_count = 0
    white_count = 0
    for filename in sorted(expected_names & set(actual_files), key=str.casefold):
        path = actual_files[filename]
        row = row_by_name[filename]
        item_errors: list[str] = []
        record: dict[str, Any] = {"filename": filename, "path": str(path)}
        if path.suffix.casefold() != extension.casefold():
            item_errors.append(f"extension is {path.suffix}, expected {extension}")
        if not path.stem.casefold().endswith(fake_suffix.casefold()):
            item_errors.append(f"filename does not end in {fake_suffix}")
        try:
            with Image.open(path) as image:
                image.load()
                array = np.asarray(image)
                record.update(
                    {
                        "format": image.format,
                        "mode": image.mode,
                        "width": image.width,
                        "height": image.height,
                        "dtype": str(array.dtype),
                        "min": int(array.min()),
                        "max": int(array.max()),
                        "mean": float(array.mean()),
                    }
                )
                expected_width = _integer(row.get("width"), expected_size)
                expected_height = _integer(row.get("height"), expected_size)
                if image.size != (expected_width, expected_height):
                    item_errors.append(
                        f"size is {image.size}, expected {(expected_width, expected_height)}"
                    )
                channels = 1 if array.ndim == 2 else int(array.shape[-1])
                row_mode = str(row.get(mode_column) or "").upper() or None
                resolved_mode = expected_mode or row_mode
                expected_channels = _integer(row.get(channel_column), 0)
                if not expected_channels and resolved_mode in {"L", "RGB"}:
                    expected_channels = 1 if resolved_mode == "L" else 3
                if expected_channels and channels != expected_channels:
                    item_errors.append(
                        f"channel count is {channels}, expected {expected_channels}"
                    )
                if resolved_mode is not None and image.mode != resolved_mode:
                    item_errors.append(f"mode is {image.mode}, expected {resolved_mode}")
                if image.format != "JPEG":
                    item_errors.append(f"encoded format is {image.format}, expected JPEG")
                if array.dtype != np.uint8:
                    item_errors.append(f"dtype is {array.dtype}, expected uint8")
                if not np.isfinite(array).all():
                    item_errors.append("pixels contain NaN or Inf")
                if int(array.min()) < 0 or int(array.max()) > 255:
                    item_errors.append("pixel range is outside [0, 255]")
                if np.all(array == 0):
                    black_count += 1
                    item_errors.append("image is completely black")
                if np.all(array == 255):
                    white_count += 1
                    item_errors.append("image is completely white")
                if float(array.std()) < 0.25:
                    warnings.append(f"Near-constant output image: {filename}")
        except (UnidentifiedImageError, OSError, ValueError) as error:
            item_errors.append(f"cannot decode image: {error}")
        record["valid"] = not item_errors
        record["errors"] = " | ".join(item_errors)
        file_reports.append(record)
        errors.extend(f"{filename}: {message}" for message in item_errors)

    expected_members = {
        (Path(root_name) / split_name / normalized_target / name).as_posix()
        for name in expected_names
    }
    zip_errors: list[str] = []
    resolved_zip: Path | None = None
    if zip_path is not None:
        resolved_zip = Path(zip_path).expanduser().resolve()
        zip_errors = _validate_zip(resolved_zip, expected_members, root_name)
        errors.extend(zip_errors)

    report: dict[str, Any] = {
        "valid": not errors,
        "target": normalized_target,
        "results_root": str(results_root),
        "expected_count": len(expected_names),
        "actual_count": len(actual_files),
        "validated_count": len(file_reports),
        "missing": missing,
        "extra": extra,
        "all_black_count": black_count,
        "all_white_count": white_count,
        "errors": errors,
        "warnings": warnings,
        "zip_path": str(resolved_zip) if resolved_zip is not None else None,
        "zip_errors": zip_errors,
    }
    artifacts = (
        Path(artifact_dir).expanduser().resolve()
        if artifact_dir is not None
        else results_root.parent / "artifacts"
    )
    artifacts.mkdir(parents=True, exist_ok=True)
    report_path = artifacts / "submission_report.json"
    files_path = artifacts / "submission_files.csv"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    fieldnames = [
        "filename",
        "path",
        "format",
        "mode",
        "width",
        "height",
        "dtype",
        "min",
        "max",
        "mean",
        "valid",
        "errors",
    ]
    with files_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(file_reports)
    report["report_path"] = str(report_path)
    report["files_report_path"] = str(files_path)
    if errors and raise_on_error:
        raise SubmissionValidationError("Submission validation failed:\n- " + "\n- ".join(errors))
    return report
