"""Leakage-safe DAPI-only ROI neighborhood dataset."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from virtual_staining.utils.image_io import load_image_tensor

from .dataset import InferenceDataset, VirtualStainingDataset
from .roi_index import ROIGridAudit, ROIIndex

ContextTransform = Callable[..., Mapping[str, Any]]


def context_offsets(grid_size: int = 3) -> torch.Tensor:
    """Return canonical row-major ``(row, col)`` offsets for an odd grid."""

    if grid_size < 1 or grid_size % 2 == 0:
        raise ValueError(f"grid_size must be a positive odd integer, got {grid_size}")
    radius = grid_size // 2
    return torch.tensor(
        [
            (row_offset, col_offset)
            for row_offset in range(-radius, radius + 1)
            for col_offset in range(-radius, radius + 1)
        ],
        dtype=torch.int64,
    )


def _resolve_path(value: str, root: Path) -> Path:
    if not value.strip():
        raise ValueError("Manifest contains an empty DAPI path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


class _RawDAPITensorLRU:
    """Per-dataset, per-process cache of unaugmented decoded DAPI tensors."""

    input_channels: int | None

    def _initialize_dapi_cache(self, cache_size: int) -> None:
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        self.cache_size = cache_size
        self._dapi_cache: OrderedDict[Path, torch.Tensor] = OrderedDict()

    def _load_dapi_tensor(self, path: Path) -> torch.Tensor:
        resolved = path.resolve()
        if self.cache_size > 0 and resolved in self._dapi_cache:
            cached = self._dapi_cache.pop(resolved)
            self._dapi_cache[resolved] = cached
            return cached.clone()
        tensor = load_image_tensor(resolved, logical_channels=self.input_channels)
        if self.cache_size > 0:
            self._dapi_cache[resolved] = tensor.detach().clone()
            while len(self._dapi_cache) > self.cache_size:
                self._dapi_cache.popitem(last=False)
            return tensor.clone()
        return tensor

    def _discard_cached_dapi(self, path: Path) -> None:
        self._dapi_cache.pop(path.resolve(), None)

    def __getstate__(self) -> dict[str, Any]:
        """Never transfer populated cache entries into a spawned worker process."""

        state = self.__dict__.copy()
        state["_dapi_cache"] = OrderedDict()
        return state


class NeighborhoodDataset(_RawDAPITensorLRU, VirtualStainingDataset):
    """Return a center sample plus a leakage-safe DAPI-only neighborhood.

    Neighbors are indexed by the five-part key ``organ/split/ROI/row/col``.
    Missing tiles copy the center pixels but carry mask value zero.  No neighbor
    target path is ever opened; supervision remains the center target only.
    """

    def __init__(
        self,
        manifest: str | Path | Sequence[Mapping[str, Any]],
        data_root: str | Path,
        *,
        targets: str | Sequence[str] | None = "CD68",
        split: str | None = None,
        transform: ContextTransform | None = None,
        input_channels: int | None = None,
        target_channels: int | Mapping[str, int] | None = None,
        strict: bool = True,
        allow_empty: bool = False,
        grid_size: int = 3,
        missing_policy: str = "center",
        include_center: bool = True,
        require_verified_grid: bool = True,
        grid_audit: ROIGridAudit | None = None,
        cache_size: int = 0,
    ) -> None:
        if grid_size < 1 or grid_size % 2 == 0:
            raise ValueError(f"grid_size must be a positive odd integer, got {grid_size}")
        if missing_policy != "center":
            raise ValueError("Only the audited missing_policy='center' is supported")
        if grid_audit is not None and require_verified_grid and not grid_audit.context_enabled:
            reasons = ", ".join(grid_audit.context_gate_reasons) or "unknown_reason"
            raise ValueError(f"ROI context gate is disabled: {reasons}")
        self._initialize_dapi_cache(cache_size)
        super().__init__(
            manifest,
            data_root,
            targets=targets,
            split=split,
            transform=None,
            input_channels=input_channels,
            target_channels=target_channels,
            strict=strict,
            allow_empty=allow_empty,
        )
        self.grid_size = grid_size
        self.missing_policy = missing_policy
        self.include_center = include_center
        self.context_transform = transform
        self.roi_index = ROIIndex(self.rows, require_verified=require_verified_grid)
        self._offsets = context_offsets(grid_size)

    def _load_neighbor_dapi(self, row_index: int, center: torch.Tensor) -> tuple[torch.Tensor, str]:
        row = self.rows[row_index]
        path = _resolve_path(str(row.get("dapi_path", "")), self.data_root)
        try:
            tensor = self._load_dapi_tensor(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Failed to load context DAPI image {path}: {exc}") from exc
        if tensor.shape != center.shape:
            self._discard_cached_dapi(path)
            raise ValueError(
                f"Context DAPI shape mismatch for {path}: "
                f"center={tuple(center.shape)}, neighbor={tuple(tensor.shape)}"
            )
        return tensor.to(dtype=torch.float32).clamp_(0.0, 1.0), str(path)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        center = item["input"]
        targets = item["targets"]
        if not isinstance(center, torch.Tensor) or not isinstance(targets, Mapping):
            raise TypeError("Base dataset returned an invalid sample")
        tiles: list[torch.Tensor] = []
        mask: list[float] = []
        source_paths: list[str] = []
        for offset, neighbor_index in self.roi_index.neighborhood_indices(
            index, grid_size=self.grid_size
        ):
            is_center = offset == (0, 0)
            if is_center:
                neighbor_index = index if self.include_center else None
            if neighbor_index is None:
                tiles.append(center.clone())
                mask.append(0.0)
                source_paths.append(str(item["dapi_path"]))
            elif neighbor_index == index:
                tiles.append(center.clone())
                mask.append(1.0)
                source_paths.append(str(item["dapi_path"]))
            else:
                neighbor, path = self._load_neighbor_dapi(neighbor_index, center)
                tiles.append(neighbor)
                mask.append(1.0)
                source_paths.append(path)
        context_tiles = torch.stack(tiles, dim=0)
        valid_mask = torch.tensor(mask, dtype=torch.float32)
        offsets = self._offsets.clone()
        if self.context_transform is not None:
            transformed = self.context_transform(
                center,
                context_tiles,
                valid_mask,
                offsets,
                targets,
            )
            required = {
                "input",
                "targets",
                "context_tiles",
                "context_valid_mask",
                "context_offsets",
            }
            missing = required.difference(transformed)
            if missing:
                raise ValueError(f"Context transform omitted keys: {sorted(missing)}")
            center = transformed["input"]
            targets = transformed["targets"]
            context_tiles = transformed["context_tiles"]
            valid_mask = transformed["context_valid_mask"]
            offsets = transformed["context_offsets"]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (center, context_tiles, valid_mask, offsets)
            ) or not isinstance(targets, Mapping):
                raise TypeError("Context transform returned invalid tensor or target values")
        typed_targets = {
            str(marker): tensor.to(dtype=torch.float32).clamp_(0.0, 1.0)
            for marker, tensor in targets.items()
            if isinstance(tensor, torch.Tensor)
        }
        item.update(
            {
                "input": center.to(dtype=torch.float32).clamp_(0.0, 1.0),
                "image": center.to(dtype=torch.float32).clamp_(0.0, 1.0),
                "targets": typed_targets,
                "context_tiles": context_tiles.to(dtype=torch.float32).clamp_(0.0, 1.0),
                "context_valid_mask": valid_mask.to(dtype=torch.float32),
                "context_offsets": offsets.to(dtype=torch.int64),
                "context_source_paths": source_paths,
                "organ_id": str(item["organ"]),
                "task_id": str(item.get("target_marker", "multi")),
                "metadata": {
                    "canonical_key": str(item["canonical_key"]),
                    "organ": str(item["organ"]),
                    "split": str(item["split"]),
                    "coordinate": self.roi_index.coordinate_for(index),
                },
            }
        )
        if len(self.targets) == 1:
            item["target"] = typed_targets[self.targets[0]]
        return item


class NeighborhoodInferenceDataset(_RawDAPITensorLRU, InferenceDataset):
    """Input-only counterpart that applies the same strict ROI context gate."""

    def __init__(
        self,
        manifest: str | Path | Sequence[Mapping[str, Any]],
        data_root: str | Path,
        *,
        input_channels: int | None = None,
        strict: bool = True,
        allow_empty: bool = False,
        grid_size: int = 3,
        missing_policy: str = "center",
        include_center: bool = True,
        require_verified_grid: bool = True,
        grid_audit: ROIGridAudit | None = None,
        cache_size: int = 0,
    ) -> None:
        if grid_size < 1 or grid_size % 2 == 0:
            raise ValueError(f"grid_size must be a positive odd integer, got {grid_size}")
        if missing_policy != "center":
            raise ValueError("Only the audited missing_policy='center' is supported")
        if grid_audit is not None and require_verified_grid and not grid_audit.context_enabled:
            reasons = ", ".join(grid_audit.context_gate_reasons) or "unknown_reason"
            raise ValueError(f"ROI context gate is disabled: {reasons}")
        self._initialize_dapi_cache(cache_size)
        super().__init__(
            manifest,
            data_root,
            input_channels=input_channels,
            strict=strict,
            allow_empty=allow_empty,
        )
        self.grid_size = int(grid_size)
        self.include_center = bool(include_center)
        self.roi_index = ROIIndex(self.rows, require_verified=require_verified_grid)
        self._offsets = context_offsets(grid_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        center = item["input"]
        if not isinstance(center, torch.Tensor):
            raise TypeError("Base inference dataset returned a non-tensor input")
        tiles: list[torch.Tensor] = []
        mask: list[float] = []
        source_paths: list[str] = []
        for offset, neighbor_index in self.roi_index.neighborhood_indices(
            index, grid_size=self.grid_size
        ):
            if offset == (0, 0):
                neighbor_index = index if self.include_center else None
            if neighbor_index is None:
                tile = center.clone()
                valid = 0.0
                source = str(item["dapi_path"])
            elif neighbor_index == index:
                tile = center.clone()
                valid = 1.0
                source = str(item["dapi_path"])
            else:
                row = self.rows[neighbor_index]
                path = _resolve_path(str(row.get("dapi_path", "")), self.data_root)
                try:
                    tile = self._load_dapi_tensor(path)
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Failed to load context DAPI image {path}: {exc}") from exc
                if tile.shape != center.shape:
                    self._discard_cached_dapi(path)
                    raise ValueError(
                        f"Context DAPI shape mismatch for {path}: "
                        f"center={tuple(center.shape)}, neighbor={tuple(tile.shape)}"
                    )
                tile = tile.to(dtype=torch.float32).clamp_(0.0, 1.0)
                valid = 1.0
                source = str(path)
            tiles.append(tile)
            mask.append(valid)
            source_paths.append(source)
        item.update(
            {
                "context_tiles": torch.stack(tiles),
                "context_valid_mask": torch.tensor(mask, dtype=torch.float32),
                "context_offsets": self._offsets.clone(),
                "context_source_paths": source_paths,
                "organ_id": str(item["organ"]),
            }
        )
        return item
