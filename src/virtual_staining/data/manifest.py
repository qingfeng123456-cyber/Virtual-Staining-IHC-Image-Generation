"""Pair marker images and build leakage-aware CSV manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from virtual_staining.constants import TARGET_MARKERS, normalize_marker

from .discovery import (
    DiscoveryResult,
    MarkerDirectory,
    discover_data_root,
    find_marker_directories,
    image_files,
    normalize_stem,
)

MARKER_PATH_COLUMNS = {
    "HLA-DR": "hla_dr_path",
    "CD45RO": "cd45ro_path",
    "Vimentin": "vimentin_path",
    "CD68": "cd68_path",
}
MARKER_PREFIXES = {
    "HLA-DR": "hla_dr",
    "CD45RO": "cd45ro",
    "Vimentin": "vimentin",
    "CD68": "cd68",
}
REAL_ROI_PATTERNS = (
    re.compile(r"^(ROI\d+)(?:[_-]|$)", re.IGNORECASE),
    re.compile(r"^([A-Za-z]+ROI[_-]?\d+)(?:[_-]|$)", re.IGNORECASE),
)

MANIFEST_FIELDS = [
    "organ",
    "split",
    "source_split",
    "roi_id",
    "raw_roi_id",
    "roi_id_source",
    "group_id",
    "patch_id",
    "stem",
    "canonical_key",
    "dapi_path",
    "hla_dr_path",
    "cd45ro_path",
    "vimentin_path",
    "cd68_path",
    "width",
    "height",
    "dapi_channels",
    "hla_dr_channels",
    "cd45ro_channels",
    "vimentin_channels",
    "cd68_channels",
    "dapi_mode",
    "hla_dr_mode",
    "cd45ro_mode",
    "vimentin_mode",
    "cd68_mode",
    "dapi_format",
    "hla_dr_format",
    "cd45ro_format",
    "vimentin_format",
    "cd68_format",
    "dapi_file_size",
    "hla_dr_file_size",
    "cd45ro_file_size",
    "vimentin_file_size",
    "cd68_file_size",
    "dapi_sha1",
    "hla_dr_sha1",
    "cd45ro_sha1",
    "vimentin_sha1",
    "cd68_sha1",
    "sample_sha1",
    "is_paired",
    "missing_targets",
]

BAD_SAMPLE_FIELDS = ["organ", "source_split", "canonical_key", "path", "issue", "details"]


@dataclass(frozen=True)
class ImageMetadata:
    width: int
    height: int
    channels: int
    mode: str
    format: str
    file_size: int
    sha1: str


@dataclass(frozen=True)
class ManifestBuildResult:
    data_root: Path
    train_manifest: Path
    val_manifest: Path
    test_manifest: Path
    smoke_test_manifest: Path
    metadata_path: Path
    leakage_report_path: Path
    bad_samples_path: Path
    train_count: int
    val_count: int
    test_count: int
    smoke_test_count: int
    official_test_missing: bool

    @property
    def manifest_dir(self) -> Path:
        return self.train_manifest.parent

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        winner, loser = sorted((left_root, right_root), key=str.casefold)
        self.parent[loser] = winner


def marker_path_column(marker: str) -> str:
    canonical = normalize_marker(marker)
    if canonical not in MARKER_PATH_COLUMNS:
        raise ValueError(f"Unsupported target marker: {marker}")
    return MARKER_PATH_COLUMNS[canonical]


def quick_sha1(path: str | Path, length: int = 16) -> str:
    """Return a deterministic SHA1 prefix after streaming the complete file."""

    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def inspect_image(path: str | Path) -> ImageMetadata:
    """Decode an image and collect storage metadata, raising a clear error if invalid."""

    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            channels = len(image.getbands())
            image_format = (image.format or image_path.suffix.lstrip(".")).upper()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValueError(f"Failed to decode image {image_path}: {exc}") from exc
    return ImageMetadata(
        width=width,
        height=height,
        channels=channels,
        mode=mode,
        format=image_format,
        file_size=image_path.stat().st_size,
        sha1=quick_sha1(image_path),
    )


def parse_roi_id(stem: str, surrogate_block_size: int = 32) -> tuple[str, str]:
    """Parse a real ROI ID or create a conservative, explicitly labelled surrogate."""

    for pattern in REAL_ROI_PATTERNS:
        match = pattern.match(stem)
        if match is not None:
            return match.group(1).upper(), "filename_regex"
    if stem.isdecimal():
        block = int(stem) // surrogate_block_size
        return f"surrogate_{block:05d}", "surrogate_numeric_block"
    normalized = normalize_stem(stem)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"surrogate_stem_{digest}", "surrogate_unique_stem"


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Image path lies outside selected data root: {path} (root={root})") from exc


def _index_marker_directory(directory: MarkerDirectory) -> tuple[dict[str, Path], list[dict[str, str]]]:
    indexed: dict[str, Path] = {}
    bad: list[dict[str, str]] = []
    for path in image_files(directory.path):
        stem = normalize_stem(path.stem)
        if stem in indexed:
            bad.append(
                {
                    "organ": directory.organ,
                    "source_split": directory.split,
                    "canonical_key": f"{directory.organ}/{stem}",
                    "path": str(path),
                    "issue": "duplicate_normalized_stem",
                    "details": f"conflicts with {indexed[stem]}",
                }
            )
            continue
        indexed[stem] = path
    return indexed, bad


def _group_marker_directories(
    marker_directories: Sequence[MarkerDirectory],
) -> dict[tuple[str, str], dict[str, list[MarkerDirectory]]]:
    grouped: dict[tuple[str, str], dict[str, list[MarkerDirectory]]] = defaultdict(lambda: defaultdict(list))
    for directory in marker_directories:
        grouped[(directory.organ, directory.split)][directory.marker].append(directory)
    return grouped


def _combined_marker_index(
    directories: Sequence[MarkerDirectory],
) -> tuple[dict[str, Path], list[dict[str, str]]]:
    combined: dict[str, Path] = {}
    bad: list[dict[str, str]] = []
    for directory in directories:
        indexed, directory_bad = _index_marker_directory(directory)
        bad.extend(directory_bad)
        for stem, path in indexed.items():
            if stem in combined:
                bad.append(
                    {
                        "organ": directory.organ,
                        "source_split": directory.split,
                        "canonical_key": f"{directory.organ}/{stem}",
                        "path": str(path),
                        "issue": "duplicate_normalized_stem",
                        "details": f"conflicts with {combined[stem]}",
                    }
                )
            else:
                combined[stem] = path
    return combined, bad


def _empty_row() -> dict[str, Any]:
    return {field: "" for field in MANIFEST_FIELDS}


def _build_context_rows(
    root: Path,
    organ: str,
    source_split: str,
    marker_groups: Mapping[str, Sequence[MarkerDirectory]],
    surrogate_block_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, str]] = []
    dapi_index, dapi_bad = _combined_marker_index(marker_groups.get("DAPI", []))
    bad.extend(dapi_bad)
    target_indexes: dict[str, dict[str, Path]] = {}
    for marker in TARGET_MARKERS:
        target_index, target_bad = _combined_marker_index(marker_groups.get(marker, []))
        target_indexes[marker] = target_index
        bad.extend(target_bad)
    for stem in sorted(dapi_index, key=str.casefold):
        dapi_path = dapi_index[stem]
        canonical_key = f"{organ}/{stem}"
        try:
            dapi_meta = inspect_image(dapi_path)
        except ValueError as exc:
            bad.append(
                {
                    "organ": organ,
                    "source_split": source_split,
                    "canonical_key": canonical_key,
                    "path": str(dapi_path),
                    "issue": "corrupt_dapi",
                    "details": str(exc),
                }
            )
            continue
        row = _empty_row()
        raw_roi, roi_source = parse_roi_id(dapi_path.stem, surrogate_block_size)
        row.update(
            {
                "organ": organ,
                "split": source_split,
                "source_split": source_split,
                "roi_id": raw_roi,
                "raw_roi_id": raw_roi,
                "roi_id_source": roi_source,
                "group_id": f"{organ}/{raw_roi}",
                "patch_id": dapi_path.stem,
                "stem": dapi_path.stem,
                "canonical_key": canonical_key,
                "dapi_path": _relative_path(dapi_path, root),
                "width": dapi_meta.width,
                "height": dapi_meta.height,
                "dapi_channels": dapi_meta.channels,
                "dapi_mode": dapi_meta.mode,
                "dapi_format": dapi_meta.format,
                "dapi_file_size": dapi_meta.file_size,
                "dapi_sha1": dapi_meta.sha1,
            }
        )
        missing: list[str] = []
        sample_digests = [dapi_meta.sha1]
        include_targets = source_split != "test"
        for marker in TARGET_MARKERS:
            prefix = MARKER_PREFIXES[marker]
            target_path = target_indexes[marker].get(stem) if include_targets else None
            if target_path is None:
                if include_targets:
                    missing.append(marker)
                continue
            try:
                target_meta = inspect_image(target_path)
            except ValueError as exc:
                missing.append(marker)
                bad.append(
                    {
                        "organ": organ,
                        "source_split": source_split,
                        "canonical_key": canonical_key,
                        "path": str(target_path),
                        "issue": "corrupt_target",
                        "details": f"{marker}: {exc}",
                    }
                )
                continue
            if (target_meta.width, target_meta.height) != (dapi_meta.width, dapi_meta.height):
                missing.append(marker)
                bad.append(
                    {
                        "organ": organ,
                        "source_split": source_split,
                        "canonical_key": canonical_key,
                        "path": str(target_path),
                        "issue": "shape_mismatch",
                        "details": (
                            f"{marker} is {target_meta.width}x{target_meta.height}; "
                            f"DAPI is {dapi_meta.width}x{dapi_meta.height}"
                        ),
                    }
                )
                continue
            row[f"{prefix}_path"] = _relative_path(target_path, root)
            row[f"{prefix}_channels"] = target_meta.channels
            row[f"{prefix}_mode"] = target_meta.mode
            row[f"{prefix}_format"] = target_meta.format
            row[f"{prefix}_file_size"] = target_meta.file_size
            row[f"{prefix}_sha1"] = target_meta.sha1
            sample_digests.append(target_meta.sha1)
        row["missing_targets"] = "|".join(missing)
        row["is_paired"] = source_split != "test" and not missing
        row["sample_sha1"] = hashlib.sha1("|".join(sample_digests).encode("ascii")).hexdigest()[:16]
        if missing and source_split != "test":
            bad.append(
                {
                    "organ": organ,
                    "source_split": source_split,
                    "canonical_key": canonical_key,
                    "path": str(dapi_path),
                    "issue": "missing_target",
                    "details": "|".join(missing),
                }
            )
            continue
        rows.append(row)
    return rows, bad


def _merge_duplicate_dapi_groups(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups = sorted({str(row["group_id"]) for row in rows}, key=str.casefold)
    union_find = _UnionFind(groups)
    groups_by_hash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups_by_hash[str(row["dapi_sha1"])].append(str(row["group_id"]))
    duplicate_hash_groups: dict[str, list[str]] = {}
    for digest, digest_groups in groups_by_hash.items():
        unique_groups = sorted(set(digest_groups), key=str.casefold)
        if len(digest_groups) > 1:
            duplicate_hash_groups[digest] = unique_groups
        for group in unique_groups[1:]:
            union_find.union(unique_groups[0], group)
    components: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        components[union_find.find(group)].append(group)
    component_name: dict[str, str] = {}
    for members in components.values():
        stable_members = sorted(members, key=str.casefold)
        name = stable_members[0]
        for member in stable_members:
            component_name[member] = name
    for row in rows:
        original = str(row["group_id"])
        row["group_id"] = component_name[original]
        row["roi_id"] = component_name[original].split("/", 1)[-1]
    return duplicate_hash_groups


def _stable_group_split(
    rows: list[dict[str, Any]], seed: int, val_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must lie strictly between 0 and 1, got {val_fraction}")
    groups = sorted({str(row["group_id"]) for row in rows}, key=str.casefold)
    if len(groups) < 2:
        return rows, []
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_group_count = min(len(groups) - 1, max(1, round(len(groups) * val_fraction)))
    val_groups = set(groups[:val_group_count])
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row["group_id"]) in val_groups:
            row["split"] = "val"
            val_rows.append(row)
        else:
            row["split"] = "train"
            train_rows.append(row)
    return train_rows, val_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_sha1(path: str | Path) -> str:
    return quick_sha1(path, length=40)


def _leakage_report(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    duplicate_hash_groups: Mapping[str, Sequence[str]],
    bad_samples: Sequence[Mapping[str, str]],
    *,
    official_split: bool,
) -> dict[str, Any]:
    train_keys = {str(row["canonical_key"]) for row in train_rows}
    val_keys = {str(row["canonical_key"]) for row in val_rows}
    train_hashes = {str(row["dapi_sha1"]) for row in train_rows}
    val_hashes = {str(row["dapi_sha1"]) for row in val_rows}
    train_groups = {str(row["group_id"]) for row in train_rows}
    val_groups = {str(row["group_id"]) for row in val_rows}
    sources = {str(row["roi_id_source"]) for row in [*train_rows, *val_rows]}
    return {
        "official_split_preserved": official_split,
        "roi_id_sources": sorted(sources),
        "roi_leakage_verifiable": sources.issubset({"filename_regex"}),
        "train_val_duplicate_canonical_keys": sorted(train_keys.intersection(val_keys)),
        "train_val_duplicate_dapi_hashes": sorted(train_hashes.intersection(val_hashes)),
        "train_val_group_overlap": sorted(train_groups.intersection(val_groups)),
        "duplicate_dapi_hash_groups_before_merge": duplicate_hash_groups,
        "missing_or_corrupt_count": len(bad_samples),
        "missing_or_corrupt_examples": list(bad_samples[:50]),
        "leakage_checks_passed": not (
            train_keys.intersection(val_keys)
            or train_hashes.intersection(val_hashes)
            or train_groups.intersection(val_groups)
        ),
    }


def _resolve_root_and_markers(
    data_root: str | Path | DiscoveryResult,
    workspace: Path,
) -> tuple[Path, tuple[MarkerDirectory, ...], DiscoveryResult | None]:
    if isinstance(data_root, DiscoveryResult):
        return data_root.selected_root, data_root.marker_directories, data_root
    if str(data_root).strip().casefold() == "auto":
        result = discover_data_root("AUTO", workspace=workspace)
        return result.selected_root, result.marker_directories, result
    root = Path(data_root).expanduser()
    if not root.is_absolute():
        root = workspace / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    marker_directories = find_marker_directories(root)
    if not marker_directories:
        result = discover_data_root(root, workspace=workspace)
        return result.selected_root, result.marker_directories, result
    return root, marker_directories, None


def build_manifests(
    data_root: str | Path | DiscoveryResult = "AUTO",
    *,
    output_dir: str | Path = "artifacts/manifests",
    workspace: str | Path | None = None,
    seed: int = 2026,
    val_fraction: float = 0.2,
    surrogate_block_size: int = 32,
    smoke_count: int = 8,
    organ: str | None = "auto",
) -> ManifestBuildResult:
    """Build paired train/val manifests and isolated official/smoke test manifests."""

    workspace_path = Path(workspace or Path.cwd()).expanduser().resolve()
    root, marker_directories, _ = _resolve_root_and_markers(data_root, workspace_path)
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = workspace_path / destination
    destination = destination.resolve()
    contexts = _group_marker_directories(marker_directories)
    requested_organ = str(organ or "auto").strip()
    if requested_organ.casefold() != "auto":
        contexts = {
            key: value
            for key, value in contexts.items()
            if key[0].casefold() == requested_organ.casefold()
        }
        if not contexts:
            available = sorted({item.organ for item in marker_directories}, key=str.casefold)
            raise ValueError(
                f"Requested organ {requested_organ!r} was not found; available organs: {available}"
            )
    context_rows: list[dict[str, Any]] = []
    bad_samples: list[dict[str, str]] = []
    for (organ, source_split), marker_groups in sorted(
        contexts.items(), key=lambda item: (item[0][0].casefold(), item[0][1])
    ):
        rows, bad = _build_context_rows(
            root,
            organ,
            source_split,
            marker_groups,
            surrogate_block_size,
        )
        context_rows.extend(rows)
        bad_samples.extend(bad)

    official_test_rows = [row for row in context_rows if row["source_split"] == "test"]
    supervised_rows = [row for row in context_rows if row["source_split"] != "test"]
    duplicate_hash_groups = _merge_duplicate_dapi_groups(supervised_rows)
    has_official_val = any(row["source_split"] == "val" for row in supervised_rows)
    if has_official_val:
        train_rows = [row for row in supervised_rows if row["source_split"] in {"train", "unspecified"}]
        val_rows = [row for row in supervised_rows if row["source_split"] == "val"]
        for row in train_rows:
            row["split"] = "train"
        for row in val_rows:
            row["split"] = "val"
    else:
        train_rows, val_rows = _stable_group_split(supervised_rows, seed, val_fraction)

    train_rows.sort(key=lambda row: (str(row["organ"]).casefold(), str(row["canonical_key"])))
    val_rows.sort(key=lambda row: (str(row["organ"]).casefold(), str(row["canonical_key"])))
    official_test_rows.sort(key=lambda row: (str(row["organ"]).casefold(), str(row["canonical_key"])))
    smoke_rows: list[dict[str, Any]] = []
    for source in val_rows[: max(0, smoke_count)]:
        row = dict(source)
        row["split"] = "smoke_test"
        row["source_split"] = "held_out_val_for_smoke"
        row["is_paired"] = False
        row["missing_targets"] = ""
        for marker in TARGET_MARKERS:
            prefix = MARKER_PREFIXES[marker]
            # Remove label-bearing paths and content fingerprints while retaining
            # non-label storage specifications.  The latter let the smoke
            # validator enforce the same RGB/L contract as official inference.
            for suffix in ("path", "file_size", "sha1"):
                row[f"{prefix}_{suffix}"] = ""
        smoke_rows.append(row)

    train_path = destination / "train_manifest.csv"
    val_path = destination / "val_manifest.csv"
    test_path = destination / "test_manifest.csv"
    smoke_path = destination / "smoke_test_manifest.csv"
    _write_csv(train_path, train_rows, MANIFEST_FIELDS)
    _write_csv(val_path, val_rows, MANIFEST_FIELDS)
    _write_csv(test_path, official_test_rows, MANIFEST_FIELDS)
    _write_csv(smoke_path, smoke_rows, MANIFEST_FIELDS)
    bad_path = destination.parent / "bad_samples.csv"
    _write_csv(bad_path, bad_samples, BAD_SAMPLE_FIELDS)

    leakage = _leakage_report(
        train_rows,
        val_rows,
        duplicate_hash_groups,
        bad_samples,
        official_split=has_official_val,
    )
    leakage_path = destination.parent / "leakage_report.json"
    leakage_path.parent.mkdir(parents=True, exist_ok=True)
    leakage_path.write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    official_test_missing = len(official_test_rows) == 0
    metadata = {
        "data_root": str(root),
        "seed": seed,
        "val_fraction": val_fraction,
        "surrogate_block_size": surrogate_block_size,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "official_test_count": len(official_test_rows),
        "smoke_test_count": len(smoke_rows),
        "official_test_missing": official_test_missing,
        "official_test_status": "missing" if official_test_missing else "available",
        "smoke_test_is_official": False,
        "smoke_test_source": "held-out validation inputs; labels intentionally removed",
        "roi_id_source": sorted({str(row["roi_id_source"]) for row in supervised_rows}),
        "roi_leakage_verifiable": leakage["roi_leakage_verifiable"],
        "manifest_sha1": {
            "train": manifest_sha1(train_path),
            "val": manifest_sha1(val_path),
            "test": manifest_sha1(test_path),
            "smoke_test": manifest_sha1(smoke_path),
        },
        "missing_or_corrupt_count": len(bad_samples),
        "missing_fraction": len(bad_samples) / max(1, len(supervised_rows) + len(bad_samples)),
    }
    metadata_path = destination / "manifest_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return ManifestBuildResult(
        data_root=root,
        train_manifest=train_path,
        val_manifest=val_path,
        test_manifest=test_path,
        smoke_test_manifest=smoke_path,
        metadata_path=metadata_path,
        leakage_report_path=leakage_path,
        bad_samples_path=bad_path,
        train_count=len(train_rows),
        val_count=len(val_rows),
        test_count=len(official_test_rows),
        smoke_test_count=len(smoke_rows),
        official_test_missing=official_test_missing,
    )


def build_manifest(*args: Any, **kwargs: Any) -> ManifestBuildResult:
    """Compatibility alias for callers using the singular command name."""

    return build_manifests(*args, **kwargs)
