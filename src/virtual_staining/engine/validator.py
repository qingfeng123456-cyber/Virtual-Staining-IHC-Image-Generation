"""Per-image reference validation with task, ROI, and macro aggregation."""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn

from virtual_staining.data.roi_index import parse_patch_coordinate
from virtual_staining.data.transforms import apply_context_d4
from virtual_staining.metrics.aggregation import (
    aggregate_stratified_metrics,
    aggregate_target_metrics,
    weighted_organ_metrics,
)
from virtual_staining.metrics.domain_metrics import (
    attach_rank_proxy,
    calculate_domain_metric_record,
)
from virtual_staining.metrics.image_metrics import calculate_image_metrics
from virtual_staining.utils.image_io import ImageSpec

from .common import (
    call_model,
    config_get,
    extract_predictions,
    metadata_item,
    model_kwargs_from_metadata,
    move_to_device,
    resolve_target_pairs,
    unpack_batch,
)
from .ema import ExponentialMovingAverage
from .inferencer import D4_TRANSFORMS, apply_d4, d4_context_transform_id, invert_d4
from .prototype_diagnostics import PrototypeDiagnosticsAggregator


def _reference_statistics(reference: torch.Tensor) -> dict[str, float]:
    image = reference.detach().float()
    horizontal = (
        torch.mean(torch.abs(image[..., :, 1:] - image[..., :, :-1]))
        if image.shape[-1] > 1
        else image.new_zeros(())
    )
    vertical = (
        torch.mean(torch.abs(image[..., 1:, :] - image[..., :-1, :]))
        if image.shape[-2] > 1
        else image.new_zeros(())
    )
    return {
        "image_mean": float(image.mean().cpu()),
        "image_std": float(image.std(unbiased=False).cpu()),
        "activity_proxy": float((horizontal + vertical).cpu()),
    }


def _context_availability(metadata: Mapping[str, Any], index: int) -> tuple[str, float]:
    mask = metadata.get("context_valid_mask")
    if not isinstance(mask, torch.Tensor):
        return "local_only", 0.0
    item = mask[index] if mask.ndim > 1 else mask
    fraction = float(item.detach().float().mean().cpu()) if item.numel() else 0.0
    if fraction >= 1.0 - 1e-12:
        return "full", fraction
    if fraction <= 1e-12:
        return "none", fraction
    return "partial", fraction


def _assign_quantile_bins(
    records: list[dict[str, Any]], value_key: str, output_key: str
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("target", "output"))].append(record)
    for rows in grouped.values():
        ordered = sorted(float(row[value_key]) for row in rows)
        low_index = min(len(ordered) - 1, int((len(ordered) - 1) / 3))
        high_index = min(len(ordered) - 1, int(2 * (len(ordered) - 1) / 3))
        low = ordered[low_index]
        high = ordered[high_index]
        for row in rows:
            value = float(row[value_key])
            row[output_key] = "low" if value < low else "high" if value > high else "mid"


def _assign_border_classes(records: list[dict[str, Any]]) -> None:
    extents: dict[tuple[str, str], list[int]] = {}
    for record in records:
        row = record.get("grid_row")
        col = record.get("grid_col")
        if row is None or col is None:
            continue
        key = (str(record.get("organ", "unknown")), str(record.get("roi_id", "")))
        values = extents.setdefault(key, [int(row), int(row), int(col), int(col)])
        values[0] = min(values[0], int(row))
        values[1] = max(values[1], int(row))
        values[2] = min(values[2], int(col))
        values[3] = max(values[3], int(col))
    for record in records:
        row = record.get("grid_row")
        col = record.get("grid_col")
        key = (str(record.get("organ", "unknown")), str(record.get("roi_id", "")))
        bounds = extents.get(key)
        if row is None or col is None or bounds is None:
            record["border_class"] = "unknown"
        elif int(row) in bounds[:2] or int(col) in bounds[2:]:
            record["border_class"] = "border"
        else:
            record["border_class"] = "interior"


class Validator:
    """Evaluate a restoration model without altering images before scoring."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: Any,
        *,
        device: str | torch.device | None = None,
        config: Any = None,
        ema: ExponentialMovingAverage | None = None,
        use_ema: bool | None = None,
        task_name: str | None = None,
        tta: str | bool | None = None,
        image_spec: Any = None,
        prototype_diagnostics_dir: str | Path | None = None,
    ) -> None:
        configured_device = config_get(
            config, "validation.device", config_get(config, "train.device", "auto")
        )
        requested_device = configured_device if device is None else device
        if str(requested_device).lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA validation was requested but CUDA is unavailable")
        self.model = model.to(self.device)
        # channels_last must match the trainer's setting: if the model was
        # compiled/converted with channels_last during training, validation inputs
        # must also be channels_last for the conv kernels to use the NHWC path.
        # The model is shared with the trainer instance, so its parameters are
        # already channels_last; we only need to convert the input tensors here.
        self.channels_last_enabled = bool(
            config_get(config, "train.channels_last", False)
        ) and self.device.type == "cuda"
        self.dataloader = dataloader
        self.config = config
        self.ema = ema
        self.use_ema = bool(
            config_get(config, "inference.use_ema", True) if use_ema is None else use_ema
        )
        self.task_name = task_name or config_get(config, "validation.task_name", None)
        tta_setting = config_get(config, "validation.tta", False) if tta is None else tta
        self.tta = str(tta_setting).casefold() in {"d4", "true", "1"} or tta_setting is True
        self.amp = bool(config_get(config, "train.amp", True))
        requested_dtype = str(config_get(config, "train.amp_dtype", "auto")).lower()
        self.amp_dtype = (
            torch.bfloat16
            if requested_dtype in {"auto", "bfloat16", "bf16"}
            and (self.device.type == "cpu" or torch.cuda.is_bf16_supported())
            else torch.float16
        )
        self.psnr_cap = config_get(config, "validation.psnr_cap", None)
        self.psnr_norm_min = float(config_get(config, "validation.psnr_norm_min", 15.0))
        self.psnr_norm_max = float(config_get(config, "validation.psnr_norm_max", 40.0))
        self.image_spec = image_spec
        self.primary_domain = str(
            config_get(config, "validation.primary_domain", "float")
        ).casefold()
        if self.primary_domain not in {"float", "uint8", "jpg"}:
            raise ValueError(f"Unsupported validation.primary_domain: {self.primary_domain}")
        self.jpeg_quality = int(config_get(config, "inference.jpeg_quality", 100))
        self.jpeg_subsampling = int(config_get(config, "inference.jpeg_subsampling", 0))
        self.prototype_diagnostics_enabled = bool(
            config_get(config, "train.prototype_monitor.enabled", False)
        )
        self.prototype_dead_threshold = float(
            config_get(
                config,
                "train.prototype_monitor.dead_threshold",
                config_get(config, "model.prototypes.dead_threshold", 1e-4),
            )
        )
        self.prototype_attention_visuals_enabled = bool(
            config_get(
                config,
                "train.prototype_monitor.attention_visuals_enabled",
                False,
            )
        )
        self.prototype_attention_visual_count = int(
            config_get(
                config,
                "train.prototype_monitor.attention_visual_count",
                4,
            )
        )
        self.prototype_attention_visual_seed = int(
            config_get(
                config,
                "train.prototype_monitor.attention_visual_seed",
                config_get(config, "project.seed", 2026),
            )
        )
        self.prototype_attention_visual_size = int(
            config_get(
                config,
                "train.prototype_monitor.attention_visual_size",
                256,
            )
        )
        self.prototype_diagnostics_dir = (
            Path(prototype_diagnostics_dir).expanduser().resolve()
            if prototype_diagnostics_dir is not None
            else None
        )
        self.prototype_diagnostics_epoch: int | None = None
        self.prototype_diagnostics_weight_source: str | None = None

    def _target_image_spec(
        self, target: str, reference: torch.Tensor
    ) -> ImageSpec | None:
        value = self.image_spec
        if isinstance(value, Mapping):
            candidate: Any = value.get(target, value.get(target.casefold(), value))
            if isinstance(candidate, Mapping):
                mode_counts = candidate.get("mode_counts")
                mode = candidate.get("storage_mode") or candidate.get("mode")
                if mode is None and isinstance(mode_counts, Mapping) and len(mode_counts) == 1:
                    mode = next(iter(mode_counts))
                channels = candidate.get("storage_channels")
                if channels is None:
                    channels = 1 if str(mode).upper() == "L" else reference.shape[0]
                logical = candidate.get("logical_channels", reference.shape[0])
                if mode is not None:
                    return ImageSpec(
                        width=int(reference.shape[-1]),
                        height=int(reference.shape[-2]),
                        storage_channels=int(channels),
                        logical_channels=int(logical),
                        mode=str(mode).upper(),
                    )
        return value if isinstance(value, ImageSpec) else None

    def _autocast(self) -> Any:
        return torch.autocast(
            self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp and self.device.type in {"cpu", "cuda"},
        )

    @staticmethod
    def _transform_context_kwargs(
        model_kwargs: dict[str, Any], transform: int
    ) -> dict[str, Any]:
        transformed = dict(model_kwargs)
        tiles = transformed.get("context_tiles")
        mask = transformed.get("context_valid_mask")
        offsets = transformed.get("context_offsets")
        if not all(isinstance(value, torch.Tensor) for value in (tiles, mask, offsets)):
            return transformed
        if tiles.ndim != 5 or mask.ndim != 2 or offsets.ndim != 3:
            raise ValueError("Batched context must be [B,N,C,H,W], [B,N], and [B,N,2]")
        batches = [
            apply_context_d4(
                tiles[index],
                mask[index],
                offsets[index],
                d4_context_transform_id(transform),
            )
            for index in range(tiles.shape[0])
        ]
        transformed["context_tiles"] = torch.stack([item[0] for item in batches])
        transformed["context_valid_mask"] = torch.stack([item[1] for item in batches])
        transformed["context_offsets"] = torch.stack([item[2] for item in batches])
        return transformed

    def _predict(
        self,
        inputs: torch.Tensor,
        task_name: str | None,
        model_kwargs: dict[str, Any] | None = None,
        prototype_diagnostics: PrototypeDiagnosticsAggregator | None = None,
        prototype_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        kwargs = model_kwargs or {}
        if not self.tta:
            with self._autocast():
                output = call_model(self.model, inputs, task_name, model_kwargs=kwargs)
            if prototype_diagnostics is not None:
                prototype_diagnostics.observe(output, metadata=prototype_metadata)
            return extract_predictions(output)
        accumulated: dict[str, torch.Tensor] = {}
        for transform in D4_TRANSFORMS:
            transformed = apply_d4(inputs, transform)
            transformed_kwargs = self._transform_context_kwargs(kwargs, transform)
            with self._autocast():
                output = call_model(
                    self.model,
                    transformed,
                    task_name,
                    model_kwargs=transformed_kwargs,
                )
            if prototype_diagnostics is not None:
                is_identity = transform == D4_TRANSFORMS[0]
                prototype_diagnostics.observe(
                    output,
                    metadata=prototype_metadata if is_identity else None,
                    capture_visuals=is_identity,
                )
            predictions = extract_predictions(output)
            for target, prediction in predictions.items():
                restored = invert_d4(prediction, transform).float()
                accumulated[target] = accumulated.get(
                    target, torch.zeros_like(restored)
                ) + restored
        return {target: value / len(D4_TRANSFORMS) for target, value in accumulated.items()}

    @torch.inference_mode()
    def evaluate(self, *, records_path: str | Path | None = None) -> dict[str, Any]:
        """Return all per-image records and aggregate validation metrics."""
        was_training = self.model.training
        self.model.eval()
        records: list[dict[str, Any]] = []
        prototype_diagnostics = (
            PrototypeDiagnosticsAggregator(
                dead_threshold=self.prototype_dead_threshold,
                attention_visuals_enabled=self.prototype_attention_visuals_enabled,
                attention_visual_count=self.prototype_attention_visual_count,
                attention_visual_seed=self.prototype_attention_visual_seed,
                attention_visual_size=self.prototype_attention_visual_size,
            )
            if self.prototype_diagnostics_enabled
            else None
        )
        started = time.perf_counter()
        ema_context = (
            self.ema.average_parameters(self.model)
            if self.use_ema and self.ema is not None
            else nullcontext()
        )
        try:
            with ema_context:
                for batch in self.dataloader:
                    inputs, targets, metadata = unpack_batch(batch)
                    if not targets:
                        raise ValueError("Validation batch contains no target tensors")
                    inputs = move_to_device(inputs, self.device)
                    if (
                        self.channels_last_enabled
                        and isinstance(inputs, torch.Tensor)
                        and inputs.ndim == 4
                        and torch.is_floating_point(inputs)
                    ):
                        inputs = inputs.to(memory_format=torch.channels_last)
                    targets = move_to_device(targets, self.device)
                    metadata = move_to_device(metadata, self.device)
                    effective_task = self.task_name
                    if effective_task is None and len(targets) == 1:
                        effective_task = next(iter(targets))
                    predictions = self._predict(
                        inputs,
                        effective_task,
                        model_kwargs_from_metadata(metadata),
                        prototype_diagnostics,
                        metadata,
                    )
                    for target_name, prediction, reference in resolve_target_pairs(
                        predictions, targets
                    ):
                        prediction = prediction.float().clamp(0.0, 1.0)
                        reference = reference.float()
                        if prediction.shape != reference.shape:
                            raise ValueError(
                                f"Validation prediction shape {tuple(prediction.shape)} does not "
                                f"match target {tuple(reference.shape)} for {target_name}"
                            )
                        for index in range(prediction.shape[0]):
                            stem = str(metadata_item(metadata, "stem", index, index))
                            coordinate = parse_patch_coordinate(stem)
                            context_label, context_fraction = _context_availability(
                                metadata, index
                            )
                            float_metrics = calculate_image_metrics(
                                prediction[index],
                                reference[index],
                                data_range=1.0,
                                psnr_cap=self.psnr_cap,
                            )
                            domain_metrics = calculate_domain_metric_record(
                                prediction[index],
                                reference[index],
                                image_spec=self._target_image_spec(
                                    target_name, reference[index]
                                ),
                                psnr_cap=self.psnr_cap,
                                jpeg_quality=self.jpeg_quality,
                                jpeg_subsampling=self.jpeg_subsampling,
                            )
                            primary_metrics = {
                                "ssim": domain_metrics[f"{self.primary_domain}_ssim"],
                                "psnr": domain_metrics[f"{self.primary_domain}_psnr"],
                            }
                            record = {
                                    "target": target_name,
                                    "canonical_key": str(
                                        metadata_item(metadata, "canonical_key", index, index)
                                    ),
                                    "stem": stem,
                                    "roi_id": str(
                                        metadata_item(metadata, "roi_id", index, f"sample-{index}")
                                    ),
                                    "organ": str(
                                        metadata_item(metadata, "organ", index, "unknown")
                                    ),
                                    "grid_row": coordinate.row if coordinate is not None else None,
                                    "grid_col": coordinate.col if coordinate is not None else None,
                                    "coordinate_source": (
                                        "filename" if coordinate is not None else "unavailable"
                                    ),
                                    "context_availability": context_label,
                                    "context_valid_fraction": context_fraction,
                                    **_reference_statistics(reference[index]),
                                    **primary_metrics,
                                    **domain_metrics,
                                }
                            if self.primary_domain == "float":
                                record["ssim"] = float_metrics["ssim"]
                                record["psnr"] = float_metrics["psnr"]
                            records.append(record)
        finally:
            self.model.train(was_training)
        if not records:
            raise ValueError("Validation dataloader produced no scored images")
        records = attach_rank_proxy(records)
        _assign_quantile_bins(records, "activity_proxy", "activity_bin")
        _assign_quantile_bins(records, "image_mean", "image_mean_bin")
        _assign_border_classes(records)
        domains: dict[str, Any] = {}
        for domain in ("float", "uint8", "jpg"):
            domain_rows = [
                {
                    **record,
                    "ssim": record[f"{domain}_ssim"],
                    "psnr": record[f"{domain}_psnr"],
                }
                for record in records
            ]
            domain_aggregate = aggregate_target_metrics(
                domain_rows,
                psnr_norm_min=self.psnr_norm_min,
                psnr_norm_max=self.psnr_norm_max,
            )
            domain_aggregate["stratified"] = aggregate_stratified_metrics(
                domain_rows,
                psnr_norm_min=self.psnr_norm_min,
                psnr_norm_max=self.psnr_norm_max,
            )
            configured_organ_weights = config_get(
                self.config,
                "validation.organ_weights",
                {"colon": 0.1, "liver": 0.2, "stomach": 0.7},
            )
            if isinstance(configured_organ_weights, Mapping):
                weighted = weighted_organ_metrics(
                    domain_rows,
                    weights={
                        str(key): float(value)
                        for key, value in configured_organ_weights.items()
                    },
                    psnr_norm_min=self.psnr_norm_min,
                    psnr_norm_max=self.psnr_norm_max,
                )
                if weighted is not None:
                    domain_aggregate["weighted_organs"] = weighted
            domain_aggregate["raw_ssim"] = domain_aggregate["macro"]["mean_ssim"]
            domain_aggregate["raw_psnr"] = domain_aggregate["macro"]["mean_psnr"]
            domain_aggregate["rank_proxy"] = float(
                sum(float(row["rank_proxy"]) for row in domain_rows) / len(domain_rows)
            )
            domain_aggregate["configurable_official_proxy"] = domain_aggregate["macro"][
                "local_proxy_score"
            ]
            domains[domain] = domain_aggregate
        aggregate = dict(domains[self.primary_domain])
        aggregate["domains"] = domains
        aggregate["primary_domain"] = self.primary_domain
        aggregate["records"] = records
        aggregate["duration_seconds"] = float(time.perf_counter() - started)
        aggregate["tta"] = "d4" if self.tta else "none"
        if prototype_diagnostics is not None:
            diagnostics_dir = self.prototype_diagnostics_dir
            if diagnostics_dir is None and records_path is not None:
                record_destination = Path(records_path).expanduser().resolve()
                diagnostics_dir = record_destination.parent / (
                    f"{record_destination.stem}_prototype_diagnostics"
                )
            if diagnostics_dir is not None:
                if self.prototype_diagnostics_epoch is not None:
                    diagnostics_dir = diagnostics_dir / (
                        f"epoch_{self.prototype_diagnostics_epoch:04d}"
                    )
                if self.prototype_diagnostics_weight_source is not None:
                    diagnostics_dir = diagnostics_dir / (
                        f"weight_{self.prototype_diagnostics_weight_source}"
                    )
                aggregate["prototype_diagnostics"] = prototype_diagnostics.write(
                    diagnostics_dir,
                    allow_empty=True,
                )
        if records_path is not None:
            destination = Path(records_path).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        return aggregate

    validate = evaluate
