"""PyTorch datasets backed by auditable CSV manifests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from virtual_staining.constants import TARGET_MARKERS, normalize_marker
from virtual_staining.utils.image_io import load_image_tensor

from .manifest import MARKER_PATH_COLUMNS, read_manifest

Transform = Callable[..., Any]


def marker_path_column(marker: str) -> str:
    """Return the manifest path column for a canonical or normalized marker name."""

    canonical = normalize_marker(marker)
    if canonical not in MARKER_PATH_COLUMNS:
        raise ValueError(f"DAPI is an input, not a target marker: {marker}")
    return MARKER_PATH_COLUMNS[canonical]


def _coerce_targets(targets: str | Sequence[str] | None) -> tuple[str, ...]:
    if targets is None:
        return ("CD68",)
    values = (targets,) if isinstance(targets, str) else tuple(targets)
    canonical = tuple(normalize_marker(value) for value in values)
    if not canonical:
        return tuple()
    invalid = [marker for marker in canonical if marker not in TARGET_MARKERS]
    if invalid:
        raise ValueError(f"Only target markers are allowed in targets: {invalid}")
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"Duplicate target marker requested: {canonical}")
    return canonical


def _resolve_image_path(value: str, data_root: Path) -> Path:
    if not value.strip():
        raise ValueError("Manifest contains an empty required image path")
    path = Path(value)
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


class VirtualStainingDataset(Dataset[dict[str, Any]]):
    """Load DAPI input and one or more paired target markers.

    Images are returned as CHW ``float32`` tensors in ``[0, 1]``. Relative
    manifest paths are resolved against ``data_root`` using :class:`Path`, so
    spaces and non-ASCII Windows paths remain supported.
    """

    def __init__(
        self,
        manifest: str | Path | Sequence[Mapping[str, Any]],
        data_root: str | Path,
        *,
        targets: str | Sequence[str] | None = "CD68",
        split: str | None = None,
        transform: Transform | None = None,
        input_channels: int | None = None,
        target_channels: int | Mapping[str, int] | None = None,
        strict: bool = True,
        allow_empty: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest).resolve() if isinstance(manifest, (str, Path)) else None
        if self.manifest_path is not None:
            rows: list[dict[str, Any]] = [dict(row) for row in read_manifest(self.manifest_path)]
        else:
            rows = [dict(row) for row in manifest]
        if split is not None:
            rows = [row for row in rows if str(row.get("split", "")) == split]
        self.rows = rows
        self.data_root = Path(data_root).expanduser().resolve()
        self.targets = _coerce_targets(targets)
        self.transform = transform
        self.input_channels = input_channels
        self.target_channels = target_channels
        self.strict = strict
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.data_root}")
        if not self.rows and not allow_empty:
            source = self.manifest_path if self.manifest_path is not None else "in-memory manifest"
            raise ValueError(
                f"Manifest contains no usable samples: {source}. "
                "An empty official test manifest means official test data is missing."
            )
        self._validate_rows()

    def _channels_for_target(self, marker: str) -> int | None:
        if isinstance(self.target_channels, Mapping):
            for key, value in self.target_channels.items():
                if normalize_marker(key) == marker:
                    return int(value)
            return None
        return int(self.target_channels) if self.target_channels is not None else None

    def _validate_rows(self) -> None:
        required_columns = {"dapi_path", "canonical_key"}
        required_columns.update(marker_path_column(marker) for marker in self.targets)
        for index, row in enumerate(self.rows):
            missing_columns = sorted(column for column in required_columns if column not in row)
            if missing_columns:
                raise ValueError(f"Manifest row {index} lacks required column(s): {missing_columns}")
            if not self.strict:
                continue
            paths = [("DAPI", str(row["dapi_path"]))]
            paths.extend((marker, str(row[marker_path_column(marker)])) for marker in self.targets)
            for marker, value in paths:
                try:
                    path = _resolve_image_path(value, self.data_root)
                except ValueError as exc:
                    raise ValueError(f"Manifest row {index} ({marker}): {exc}") from exc
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Manifest row {index} ({row.get('canonical_key', '?')}) "
                        f"references missing {marker} image: {path}"
                    )

    def __len__(self) -> int:
        return len(self.rows)

    def _load_dapi_tensor(self, path: Path) -> torch.Tensor:
        """Load one raw DAPI tensor; context datasets may override this I/O hook."""

        return load_image_tensor(path, logical_channels=self.input_channels)

    def _apply_transform(
        self,
        image: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.transform is None:
            return image, targets
        transformed = self.transform(image, targets)
        if isinstance(transformed, tuple) and len(transformed) == 2:
            transformed_image, transformed_targets = transformed
        elif isinstance(transformed, Mapping):
            transformed_image = transformed.get("input", transformed.get("image"))
            transformed_targets = transformed.get("targets")
        else:
            raise TypeError("Paired transform must return (input, targets) or a mapping")
        if not isinstance(transformed_image, torch.Tensor):
            raise TypeError("Paired transform returned a non-tensor input")
        if not isinstance(transformed_targets, Mapping):
            raise TypeError("Paired transform returned non-mapping targets")
        typed_targets = {
            normalize_marker(str(marker)): tensor
            for marker, tensor in transformed_targets.items()
            if isinstance(tensor, torch.Tensor)
        }
        if set(typed_targets) != set(targets):
            raise ValueError(
                f"Paired transform changed target keys: before={sorted(targets)}, "
                f"after={sorted(typed_targets)}"
            )
        return transformed_image, typed_targets

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        dapi_path = _resolve_image_path(str(row["dapi_path"]), self.data_root)
        try:
            image = self._load_dapi_tensor(dapi_path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Failed to load DAPI image {dapi_path}: {exc}") from exc
        targets: dict[str, torch.Tensor] = {}
        target_paths: dict[str, str] = {}
        for marker in self.targets:
            path = _resolve_image_path(str(row[marker_path_column(marker)]), self.data_root)
            try:
                tensor = load_image_tensor(path, logical_channels=self._channels_for_target(marker))
            except (OSError, ValueError) as exc:
                raise ValueError(f"Failed to load {marker} image {path}: {exc}") from exc
            targets[marker] = tensor
            target_paths[marker] = str(path)
        for marker, target in targets.items():
            if target.shape[-2:] != image.shape[-2:]:
                raise ValueError(
                    f"Spatial mismatch for {row['canonical_key']}: DAPI={tuple(image.shape)}, "
                    f"{marker}={tuple(target.shape)}"
                )
        image, targets = self._apply_transform(image, targets)
        image = image.to(dtype=torch.float32).clamp_(0.0, 1.0)
        targets = {
            marker: target.to(dtype=torch.float32).clamp_(0.0, 1.0)
            for marker, target in targets.items()
        }
        item: dict[str, Any] = {
            "input": image,
            "image": image,
            "targets": targets,
            "canonical_key": str(row["canonical_key"]),
            "stem": str(row.get("stem") or row.get("patch_id") or Path(str(row["dapi_path"])).stem),
            "patch_id": str(row.get("patch_id", "")),
            "roi_id": str(row.get("roi_id", "")),
            "group_id": str(row.get("group_id", row.get("roi_id", ""))),
            "organ": str(row.get("organ", "unknown")),
            "split": str(row.get("split", "")),
            "dapi_path": str(dapi_path),
            "target_paths": target_paths,
            "index": index,
        }
        if len(self.targets) == 1:
            item["target"] = targets[self.targets[0]]
            item["target_marker"] = self.targets[0]
        return item


class InferenceDataset(VirtualStainingDataset):
    """Input-only manifest dataset for official or smoke inference."""

    def __init__(self, manifest: str | Path | Sequence[Mapping[str, Any]], data_root: str | Path, **kwargs: Any) -> None:
        kwargs.pop("targets", None)
        super().__init__(manifest, data_root, targets=(), **kwargs)


VirtualStainDataset = VirtualStainingDataset
