"""Build an exact competition directory and ZIP without double JPEG encoding."""

from __future__ import annotations

import csv
import shutil
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _portable_stem(value: str) -> str:
    return Path(str(value).replace("\\", "/").rsplit("/", 1)[-1]).stem


def canonical_target(target: str) -> str:
    """Normalize competition marker spellings while retaining canonical case."""
    key = "".join(character for character in str(target).casefold() if character.isalnum())
    aliases = {
        "dapi": "DAPI",
        "hladr": "HLA-DR",
        "cd45ro": "CD45RO",
        "vimentin": "Vimentin",
        "cd68": "CD68",
    }
    if key not in aliases:
        raise ValueError(f"Unknown competition target: {target!r}")
    normalized = aliases[key]
    if normalized == "DAPI":
        raise ValueError("DAPI is the input marker and cannot be a submission target")
    return normalized


def _manifest_rows(path: str | Path) -> list[dict[str, str]]:
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Test manifest does not exist: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(
            "Test manifest is empty. Official submission generation cannot treat zero inputs as success."
        )
    return rows


def _row_stem(row: dict[str, str]) -> str:
    explicit = row.get("stem")
    if explicit:
        return _portable_stem(str(explicit))
    dapi_path = row.get("dapi_path")
    if dapi_path:
        return _portable_stem(str(dapi_path))
    patch_id = row.get("patch_id")
    if patch_id:
        return _portable_stem(str(patch_id))
    raise ValueError("Every test manifest row must contain dapi_path, stem, or patch_id")


def submission_filename(stem: str, *, fake_suffix: str = "_fake", extension: str = ".jpg") -> str:
    """Create one output filename while preventing an accidental fake_fake suffix."""
    normalized = _portable_stem(str(stem))
    while normalized.casefold().endswith(fake_suffix.casefold()):
        normalized = normalized[: -len(fake_suffix)]
    if not normalized:
        raise ValueError("Submission stem becomes empty after suffix normalization")
    suffix = extension if extension.startswith(".") else f".{extension}"
    return f"{normalized}{fake_suffix}{suffix.lower()}"


def _prediction_candidates(prediction_root: Path, target: str, stem: str) -> list[Path]:
    names: set[str] = set()
    for extension in IMAGE_EXTENSIONS:
        names.update((f"{stem}{extension}".casefold(), f"{stem}_fake{extension}".casefold()))
    directories = (
        prediction_root / target,
        prediction_root / target.casefold(),
        prediction_root,
    )
    found: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        for candidate in directory.iterdir():
            if candidate.is_file() and candidate.name.casefold() in names:
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(resolved)
    return found


def _write_as_jpeg(source: Path, destination: Path, quality: int, subsampling: int) -> None:
    with Image.open(source) as image:
        image.load()
        source_is_jpeg = (image.format or "").upper() == "JPEG"
    if source_is_jpeg:
        shutil.copy2(source, destination)
        return
    with Image.open(source) as image:
        mode = "L" if image.mode in {"1", "L", "I", "I;16", "F"} else "RGB"
        converted = image.convert(mode)
        converted.save(
            destination,
            format="JPEG",
            quality=int(quality),
            subsampling=int(subsampling),
            optimize=False,
        )


def build_submission(
    pred_dir: str | Path,
    test_manifest: str | Path,
    target: str,
    output_dir: str | Path,
    *,
    root_name: str = "results",
    split_name: str = "test",
    fake_suffix: str = "_fake",
    extension: str = ".jpg",
    create_zip: bool = True,
    jpeg_quality: int = 100,
    jpeg_subsampling: int = 0,
    smoke: bool | None = None,
) -> dict[str, Any]:
    """Copy predictions into the required hierarchy and create a compliant ZIP."""
    prediction_root = Path(pred_dir).expanduser().resolve()
    if not prediction_root.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {prediction_root}")
    manifest_path = Path(test_manifest).expanduser().resolve()
    rows = _manifest_rows(manifest_path)
    normalized_target = canonical_target(target)
    base = Path(output_dir).expanduser().resolve()
    if base.name.casefold() == root_name.casefold():
        results_root = base
        package_root = base.parent
    else:
        results_root = base / root_name
        package_root = base
    is_smoke = "smoke" in manifest_path.stem.casefold() if smoke is None else bool(smoke)
    canonical_project_results = (Path.cwd().resolve() / root_name).resolve()
    if is_smoke and results_root == canonical_project_results:
        raise ValueError("Smoke submissions must not be written to the official project results directory")
    expected_names: set[str] = set()
    source_records: list[dict[str, str]] = []
    planned_sources: list[tuple[str, str, Path]] = []
    for row in rows:
        stem = _row_stem(row)
        filename = submission_filename(stem, fake_suffix=fake_suffix, extension=extension)
        if filename.casefold() in {name.casefold() for name in expected_names}:
            raise ValueError(f"Duplicate manifest stem after normalization: {stem}")
        expected_names.add(filename)
        candidates = _prediction_candidates(prediction_root, normalized_target, stem)
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one prediction for {stem!r}, found {len(candidates)}: {candidates}"
            )
        planned_sources.append((stem, filename, candidates[0]))

    target_dir = results_root / split_name / normalized_target
    target_dir.mkdir(parents=True, exist_ok=True)
    for existing in target_dir.iterdir():
        if not existing.is_file():
            raise ValueError(f"Submission target directory contains an unexpected directory: {existing}")
        existing.unlink()
    for stem, filename, source in planned_sources:
        destination = target_dir / filename
        _write_as_jpeg(source, destination, jpeg_quality, jpeg_subsampling)
        source_records.append(
            {"stem": stem, "source": str(source), "destination": str(destination)}
        )

    zip_path: Path | None = None
    if create_zip:
        package_root.mkdir(parents=True, exist_ok=True)
        zip_path = package_root / f"submission_{normalized_target}.zip"
        temporary = zip_path.with_name(f".{zip_path.name}.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in sorted(target_dir.iterdir(), key=lambda path: path.name.casefold()):
                if file_path.is_file() and file_path.name in expected_names:
                    archive.write(
                        file_path,
                        arcname=(Path(root_name) / file_path.relative_to(results_root)).as_posix(),
                    )
        temporary.replace(zip_path)
    return {
        "results_root": str(results_root),
        "target_dir": str(target_dir),
        "target": normalized_target,
        "count": len(source_records),
        "files": source_records,
        "zip_path": str(zip_path) if zip_path is not None else None,
        "smoke": is_smoke,
    }
