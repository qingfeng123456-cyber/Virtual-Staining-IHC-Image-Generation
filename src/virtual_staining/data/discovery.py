"""Bounded, explainable discovery of competition image data."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from virtual_staining.constants import CANONICAL_MARKERS, TARGET_MARKERS, normalize_marker

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})
DATA_NAME_HINTS = ("dataset", "datasets", "data", "比赛数据", "训练数据")
GENERATED_DIRECTORY_NAMES = frozenset(
    {".git", ".venv", "__pycache__", "artifacts", "outputs", "results"}
)
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
ROI_PATTERN = re.compile(r"^roi\d+(?:[_-]\d+){1,}", re.IGNORECASE)


def _canonical_marker(name: str) -> str | None:
    """Return a known canonical marker or ``None`` for an unrelated directory."""

    try:
        marker = normalize_marker(name)
    except (KeyError, ValueError):
        return None
    return marker if marker in CANONICAL_MARKERS else None


def normalize_stem(stem: str) -> str:
    """Normalize a file stem for cross-marker matching without losing coordinates."""

    normalized = stem.strip().casefold()
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def image_files(directory: Path) -> list[Path]:
    """Return supported image files directly below *directory* in stable order."""

    try:
        files = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ]
    except OSError:
        return []
    return sorted(files, key=lambda path: (path.stem.casefold(), path.suffix.casefold(), path.name))


@dataclass(frozen=True)
class MarkerDirectory:
    """A marker directory and its inferred organ/split context."""

    path: Path
    marker: str
    organ: str
    split: str
    image_count: int
    relative_depth: int

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


@dataclass(frozen=True)
class DiscoveryCandidate:
    """Scored possible data root."""

    path: Path
    source: str
    source_priority: int
    score: float
    reasons: tuple[str, ...]
    marker_directories: tuple[MarkerDirectory, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "source": self.source,
            "source_priority": self.source_priority,
            "score": self.score,
            "reasons": list(self.reasons),
            "marker_directories": [item.to_dict() for item in self.marker_directories],
        }


@dataclass(frozen=True)
class DiscoveryResult:
    """Selected data root plus the complete bounded candidate audit trail."""

    selected_root: Path
    selected_score: float
    candidates: tuple[DiscoveryCandidate, ...]
    marker_directories: tuple[MarkerDirectory, ...]
    output_path: Path | None = None

    @property
    def data_root(self) -> Path:
        return self.selected_root

    def __fspath__(self) -> str:
        return str(self.selected_root)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_root": str(self.selected_root),
            "selected_score": self.selected_score,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "marker_directories": [item.to_dict() for item in self.marker_directories],
        }


def _walk_directories(root: Path, max_depth: int) -> Iterable[tuple[Path, int]]:
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        yield directory, depth
        if depth >= max_depth:
            continue
        try:
            children = sorted(
                (
                    child
                    for child in directory.iterdir()
                    if child.is_dir()
                    and child.name.casefold() not in GENERATED_DIRECTORY_NAMES
                    and not child.name.casefold().endswith(".egg-info")
                ),
                key=lambda child: child.name.casefold(),
                reverse=True,
            )
        except OSError:
            children = []
        stack.extend((child, depth + 1) for child in children)


def _infer_context(data_root: Path, marker_dir: Path) -> tuple[str, str]:
    relative = marker_dir.relative_to(data_root)
    parents = list(relative.parts[:-1])
    lowered = [part.casefold() for part in parents]
    split_index = next((i for i, part in enumerate(lowered) if part in SPLIT_ALIASES), None)
    if split_index is None:
        split = "unspecified"
        organ = parents[-1] if parents else "unknown"
    else:
        split = SPLIT_ALIASES[lowered[split_index]]
        if split_index + 1 < len(parents):
            organ = parents[split_index + 1]
        elif split_index > 0:
            organ = parents[split_index - 1]
        else:
            organ = "unknown"
    return organ, split


def find_marker_directories(data_root: str | Path, max_depth: int = 4) -> tuple[MarkerDirectory, ...]:
    """Find marker folders without assuming a single split/organ nesting order."""

    root = Path(data_root).expanduser().resolve()
    found: list[MarkerDirectory] = []
    if not root.is_dir():
        return tuple()
    for directory, depth in _walk_directories(root, max_depth=max_depth):
        marker = _canonical_marker(directory.name)
        if marker is None:
            continue
        files = image_files(directory)
        if not files:
            continue
        organ, split = _infer_context(root, directory)
        found.append(
            MarkerDirectory(
                path=directory,
                marker=marker,
                organ=organ,
                split=split,
                image_count=len(files),
                relative_depth=depth,
            )
        )
    return tuple(sorted(found, key=lambda item: (item.organ.casefold(), item.split, item.marker, str(item.path))))


def _score_candidate(path: Path, source: str, source_priority: int) -> DiscoveryCandidate:
    marker_dirs = find_marker_directories(path)
    reasons: list[str] = []
    markers = {item.marker for item in marker_dirs}
    image_count = sum(item.image_count for item in marker_dirs)
    score = 0.0
    if "DAPI" in markers:
        score += 25.0
        reasons.append("contains DAPI")
    targets = sorted(markers.intersection(TARGET_MARKERS))
    if targets:
        score += 8.0 * len(targets)
        reasons.append(f"contains {len(targets)} target marker(s): {', '.join(targets)}")
    if image_count:
        image_score = min(30.0, 6.0 * math.log10(image_count + 1.0))
        score += image_score
        reasons.append(f"contains {image_count} supported image(s)")
    splits = {item.split for item in marker_dirs if item.split != "unspecified"}
    if splits:
        score += min(9.0, 3.0 * len(splits))
        reasons.append(f"recognized split(s): {', '.join(sorted(splits))}")
    depths = [item.relative_depth for item in marker_dirs]
    if depths:
        common_depth = max(set(depths), key=lambda depth: (depths.count(depth), -depth))
        if common_depth == 2:
            score += 12.0
            reasons.append("marker folders match organ/marker or split/marker depth")
        elif common_depth == 3 and splits:
            score += 10.0
            reasons.append("marker folders match split/organ/marker depth")
        elif common_depth == 1:
            score -= 5.0
            reasons.append("marker folders are direct children; likely an organ subdirectory")
        else:
            score -= float(max(0, common_depth - 3) * 3)
    roi_matches = 0
    for marker_dir in marker_dirs:
        for path_item in image_files(marker_dir.path)[:256]:
            roi_matches += int(ROI_PATTERN.match(path_item.stem) is not None)
    if roi_matches:
        score += min(8.0, 2.0 + math.log2(roi_matches + 1.0))
        reasons.append(f"found {roi_matches} ROI-style filename sample(s)")
    return DiscoveryCandidate(
        path=path,
        source=source,
        source_priority=source_priority,
        score=round(score, 4),
        reasons=tuple(reasons),
        marker_directories=marker_dirs,
    )


def _config_data_roots(config_path: Path | None) -> list[Path]:
    if config_path is None or not config_path.is_file():
        return []
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, ValueError, ImportError):
        return []
    roots: list[Path] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                visit(nested_value, str(nested_key).casefold())
        elif key in {"data_root", "dataset_root", "root"} and isinstance(value, str):
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                base = (
                    config_path.parent.parent
                    if config_path.parent.name.casefold() == "configs"
                    else config_path.parent
                )
                candidate = base / candidate
            roots.append(candidate.resolve())

    visit(payload)
    return roots


def _seed_candidates(
    workspace: Path,
    explicit_root: Path | None,
    config_path: Path | None,
) -> list[tuple[Path, str, int]]:
    seeds: list[tuple[Path, str, int]] = []
    if explicit_root is not None:
        # An explicit CLI path is an instruction, not merely another discovery
        # hint.  A negative priority keeps it ahead of config, environment, and
        # workspace candidates while still allowing a nested structural root.
        seeds.append((explicit_root.resolve(), "explicit", -1))
    for root in _config_data_roots(config_path):
        seeds.append((root, "config", 0))
    env_value = os.environ.get("AIC_DATA_ROOT", "").strip()
    if env_value:
        seeds.append((Path(env_value).expanduser().resolve(), "environment:AIC_DATA_ROOT", 1))
    seeds.append((workspace, "workspace", 2))
    try:
        workspace_children = list(workspace.iterdir())
    except OSError:
        workspace_children = []
    for child in workspace_children:
        lowered = child.name.casefold()
        if child.is_dir() and any(hint in lowered for hint in DATA_NAME_HINTS):
            seeds.append((child.resolve(), "workspace-name-match", 2))
    parent = workspace.parent
    try:
        parent_children = list(parent.iterdir())
    except OSError:
        parent_children = []
    for child in parent_children:
        lowered = child.name.casefold()
        if child.is_dir() and any(hint in lowered for hint in DATA_NAME_HINTS):
            seeds.append((child.resolve(), "parent-name-match", 3))
    return seeds


def _expand_candidate_roots(seeds: list[tuple[Path, str, int]]) -> list[tuple[Path, str, int]]:
    expanded: dict[Path, tuple[str, int]] = {}
    for seed, source, priority in seeds:
        if not seed.is_dir():
            continue
        existing = expanded.get(seed)
        if existing is None or priority < existing[1]:
            expanded[seed] = (source, priority)
        for directory, depth in _walk_directories(seed, max_depth=2):
            if depth == 0 or _canonical_marker(directory.name) is not None:
                continue
            existing = expanded.get(directory)
            derived_source = f"{source}:nested-{depth}"
            if existing is None or priority < existing[1]:
                expanded[directory] = (derived_source, priority)
    return [(path, source, priority) for path, (source, priority) in expanded.items()]


def discover_data_root(
    data_root: str | Path | None = None,
    *,
    workspace: str | Path | None = None,
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> DiscoveryResult:
    """Discover and score likely data roots within the explicitly bounded search area.

    ``data_root='AUTO'`` is equivalent to leaving it unset. An explicitly supplied
    directory is searched for an immediately nested structural root rather than
    requiring an exact expected directory name.
    """

    workspace_path = Path(workspace or Path.cwd()).expanduser().resolve()
    explicit = None
    if data_root is not None and str(data_root).strip().casefold() != "auto":
        explicit = Path(data_root).expanduser()
        if not explicit.is_absolute():
            explicit = workspace_path / explicit
        explicit = explicit.resolve()
        if not explicit.is_dir():
            raise FileNotFoundError(f"Configured data root is not a directory: {explicit}")
    config = Path(config_path).expanduser().resolve() if config_path is not None else None
    candidates = [
        _score_candidate(path, source, priority)
        for path, source, priority in _expand_candidate_roots(
            _seed_candidates(workspace_path, explicit, config)
        )
    ]
    candidates = [candidate for candidate in candidates if candidate.marker_directories]
    candidates.sort(
        key=lambda candidate: (
            candidate.source_priority,
            -candidate.score,
            len(candidate.path.parts),
            str(candidate.path).casefold(),
        )
    )
    if not candidates:
        searched = [str(path) for path, _, _ in _seed_candidates(workspace_path, explicit, config)]
        raise FileNotFoundError(
            "No competition data directory containing DAPI/target images was found. "
            f"Bounded candidates: {searched}"
        )
    selected = candidates[0]
    destination = Path(output_path) if output_path is not None else workspace_path / "artifacts/data_discovery.json"
    result = DiscoveryResult(
        selected_root=selected.path,
        selected_score=selected.score,
        candidates=tuple(candidates),
        marker_directories=selected.marker_directories,
        output_path=destination,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def discover_data(*args: Any, **kwargs: Any) -> DiscoveryResult:
    """Compatibility alias used by CLI integrations."""

    return discover_data_root(*args, **kwargs)
