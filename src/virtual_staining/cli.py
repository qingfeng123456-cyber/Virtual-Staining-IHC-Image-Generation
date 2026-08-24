"""Command-line entry point for the complete virtual-staining workflow."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import logging
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from virtual_staining.config import config_hash, load_config, save_effective_config
from virtual_staining.constants import TARGET_MARKERS, normalize_marker
from virtual_staining.data.activity import compute_training_activity
from virtual_staining.data.audit import audit_data
from virtual_staining.data.dataset import InferenceDataset, VirtualStainingDataset
from virtual_staining.data.discovery import DiscoveryResult, discover_data_root
from virtual_staining.data.loader_helpers import (
    ActivitySamplingPlan,
    prepare_activity_sampling,
    prepare_authoritative_fold_split,
)
from virtual_staining.data.manifest import (
    ManifestBuildResult,
    build_manifests,
    manifest_sha1,
    read_manifest,
)
from virtual_staining.data.neighborhood import (
    NeighborhoodDataset,
    NeighborhoodInferenceDataset,
    context_offsets,
)
from virtual_staining.data.roi_audit import write_roi_grid_audit
from virtual_staining.data.transforms import ContextPairedTransform, PairedTransform
from virtual_staining.engine import (
    AblationBudgetPlan,
    EvidenceRun,
    Inferencer,
    Trainer,
    Validator,
    bind_evidence_records,
    build_promotion_provenance,
    load_checkpoint,
    resolve_ablation_budget,
    resume_from_checkpoint,
)
from virtual_staining.engine.common import call_model, extract_predictions
from virtual_staining.engine.ensemble import validation_score_weights
from virtual_staining.engine.ensemble_optimizer import (
    load_verified_ensemble_inputs,
    optimize_ensemble_weights,
)
from virtual_staining.engine.experiment_registry import (
    ExperimentRegistry,
    compare_experiments,
)
from virtual_staining.engine.model_soup import (
    SoupMember,
    bind_soup_validation_contract,
    build_initialization_provenance,
    build_soup_validation_contract,
    checkpoint_file_sha256,
    extract_checkpoint_provenance,
    greedy_model_soup,
    load_checkpoint_state,
)
from virtual_staining.engine.multistage import MultiStageController
from virtual_staining.engine.multitask_optimizer import build_task_balancer
from virtual_staining.engine.pretrainer import (
    DAPIPretrainer,
    transfer_local_encoder_from_checkpoint,
)
from virtual_staining.engine.trainer import collect_optimizer_parameters
from virtual_staining.losses import (
    CompositeRestorationLoss,
    ScheduledCompositeLoss,
    TwoPhaseLossSchedule,
)
from virtual_staining.metrics.promotion import (
    evaluate_roi_jpg_promotion,
    write_promotion_report,
)
from virtual_staining.models import (
    CAMPVSv2,
    MultiMarkerRestorer,
    ResidualUNet,
    RestorationOutput,
    count_parameters,
    model_statistics,
)
from virtual_staining.models.dapi_mae import DAPIMaskedAutoencoder
from virtual_staining.submission import build_submission, validate_submission
from virtual_staining.utils.device import environment_report, hardware_profile
from virtual_staining.utils.logging import (
    ExperimentLogSession,
    activate_experiment_log,
    active_experiment_log,
    configure_logging,
)
from virtual_staining.utils.seed import seed_worker, set_seed


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else "nan"
    return value


def _write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return destination


def _write_rows_csv(
    path: str | Path, rows: Sequence[Mapping[str, Any]]
) -> Path:
    if not rows:
        raise ValueError("Cannot write an empty resolved manifest")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination


def _print_json(payload: Any) -> None:
    # ``conda run`` captures stdout using the active Windows code page.  ASCII
    # escaping keeps machine-readable output reliable even when the workspace
    # path contains Chinese characters; artifact files remain UTF-8.
    print(json.dumps(_json_safe(payload), ensure_ascii=True, indent=2, allow_nan=False))


def _load_effective(
    args: argparse.Namespace, *, apply_target_to_model: bool = False
) -> dict[str, Any]:
    config = load_config(args.config, getattr(args, "set", None))
    target = getattr(args, "target", None)
    if target and apply_target_to_model:
        canonical = normalize_marker(target)
        if canonical not in TARGET_MARKERS:
            raise ValueError("DAPI cannot be used as a prediction target")
        config["data"]["targets"] = [canonical]
    max_epochs = getattr(args, "max_epochs", None)
    if max_epochs is not None:
        config["train"]["epochs"] = int(max_epochs)
    return config


def _resolve_discovery(
    data_root: str | Path | None,
    *,
    config: Mapping[str, Any] | None = None,
    output_path: str | Path = "artifacts/data_discovery.json",
) -> DiscoveryResult:
    requested = data_root or (config or {}).get("data", {}).get("root", "AUTO")
    return discover_data_root(
        requested,
        workspace=Path.cwd(),
        output_path=output_path,
    )


def _ensure_manifests(
    root: str | Path | DiscoveryResult,
    config: Mapping[str, Any],
) -> ManifestBuildResult:
    data = config["data"]
    return build_manifests(
        root,
        workspace=Path.cwd(),
        seed=int(
            data.get("split_seed", config["project"].get("seed", 2026))
        ),
        val_fraction=float(data.get("val_ratio", 0.2)),
        surrogate_block_size=int(data.get("surrogate_group_size", 32)),
        smoke_count=int(data.get("smoke_count", 8)),
        organ=str(data.get("organ", "auto")),
    )


def _apply_hardware_defaults(config: dict[str, Any]) -> None:
    profile = hardware_profile()
    train = config["train"]
    model = config["model"]
    if train.get("hardware_profile", "auto") == "auto":
        train["resolved_hardware_profile"] = profile["name"]
        if train.get("batch_size") == "auto":
            train["batch_size"] = profile["batch_size"]
        if train.get("gradient_accumulation") == "auto":
            train["gradient_accumulation"] = profile["gradient_accumulation"]
        # Only override model width when it is still at the config default, so
        # an explicit per-config base_channels is never clobbered.
        if int(model.get("base_channels", 48)) == 48:
            model["base_channels"] = profile["base_channels"]
    if train.get("batch_size") == "auto":
        train["batch_size"] = 1
    if train.get("gradient_accumulation") == "auto":
        train["gradient_accumulation"] = 1


def _selected_targets(config: Mapping[str, Any]) -> tuple[str, ...]:
    targets = tuple(normalize_marker(value) for value in config["data"].get("targets", ["CD68"]))
    if not targets or any(target not in TARGET_MARKERS for target in targets):
        raise ValueError(f"Invalid target list: {targets}")
    return targets


def _submission_targets(config: Mapping[str, Any]) -> tuple[str, ...]:
    targets = tuple(
        normalize_marker(value) for value in config["data"].get("submit_targets", [])
    )
    if not targets or any(target not in TARGET_MARKERS for target in targets):
        raise ValueError(f"Invalid submission target list: {targets}")
    if not set(targets).issubset(_selected_targets(config)):
        raise ValueError("Submission targets must be present in the model target list")
    return targets


def _manifest_smoke_status(path: str | Path) -> bool:
    rows = read_manifest(path)
    if not rows:
        return False
    smoke_flags = [
        str(row.get("split", "")).casefold() == "smoke_test"
        and str(row.get("source_split", "")).casefold() == "held_out_val_for_smoke"
        for row in rows
    ]
    if any(smoke_flags) and not all(smoke_flags):
        raise ValueError("Manifest mixes smoke and non-smoke rows")
    return all(smoke_flags)


def _channel_mapping(config: Mapping[str, Any], targets: Sequence[str]) -> tuple[int, dict[str, int]]:
    data = config["data"]
    input_channels = int(data.get("input_channels", 3))
    configured = data.get("target_channels", {})
    if isinstance(configured, Mapping):
        output_channels = {target: int(configured.get(target, configured.get(target.lower(), 3))) for target in targets}
    elif configured == "auto":
        output_channels = {target: 3 for target in targets}
    else:
        output_channels = {target: int(configured) for target in targets}
    return input_channels, output_channels


def _filter_kwargs(callable_object: Callable[..., Any], values: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_object)
    return {key: value for key, value in values.items() if key in signature.parameters}


def _build_model(config: Mapping[str, Any]) -> nn.Module:
    targets = _selected_targets(config)
    input_channels, output_channels = _channel_mapping(config, targets)
    options = dict(config["model"])
    name = str(options.pop("name", "multi_marker_restorer")).lower()
    if name in {"residual_unet", "baseline_unet", "unet"}:
        if len(targets) != 1:
            raise ValueError("Residual U-Net supports exactly one target per model")
        options.update(
            {
                "in_channels": input_channels,
                "out_channels": output_channels[targets[0]],
                "target_name": targets[0],
            }
        )
        return ResidualUNet(**_filter_kwargs(ResidualUNet, options))
    if name in {"camp_vs_v2", "campvsv2", "camp"}:
        options.update(
            {
                "in_channels": input_channels,
                "out_channels": output_channels,
                "target_names": targets,
            }
        )
        return CAMPVSv2(**_filter_kwargs(CAMPVSv2, options))
    if name not in {"multi_marker_restorer", "multimarkerrestorer", "main"}:
        raise ValueError(
            "Unknown model name "
            f"{name!r}; expected residual_unet, camp_vs_v2, or multi_marker_restorer"
        )
    options.update(
        {
            "in_channels": input_channels,
            "out_channels": output_channels,
            "target_names": targets,
        }
    )
    model = MultiMarkerRestorer(**_filter_kwargs(MultiMarkerRestorer, options))
    configured_flags = {
        key: bool(value.get("enabled", False))
        for key, value in config["model"].items()
        if key
        in {
            "context",
            "spatial_frequency",
            "lightweight_unet",
            "heavy_unet",
            "multi_route_fusion",
        }
        and isinstance(value, Mapping)
    }
    actual_flags = getattr(model, "feature_flags", {})
    mismatches = {
        key: {"configured": enabled, "constructed": actual_flags.get(key)}
        for key, enabled in configured_flags.items()
        if actual_flags.get(key) is not enabled
    }
    if mismatches:
        raise RuntimeError(
            "Enabled model features were not constructed; refusing a silent fallback: "
            f"{mismatches}"
        )
    return model


def _build_loss(config: Mapping[str, Any]) -> nn.Module:
    loss_config = config["loss"]
    if str(loss_config.get("schedule", "none")).casefold() == "two_phase":
        schedule = TwoPhaseLossSchedule(
            phase_a=loss_config.get("phase_a", {}),
            phase_b=loss_config.get("phase_b", {}),
            phase_a_ratio=float(loss_config.get("phase_a_ratio", 0.7)),
            interpolation=str(loss_config.get("interpolation", "cosine")),
        )
        pyramid = loss_config.get("pyramid", {})
        statistics = loss_config.get("statistics", {})
        prototype = loss_config.get("prototype", {})
        prototype = prototype if isinstance(prototype, Mapping) else {}
        shift_tolerant = loss_config.get("shift_tolerant", {})
        shift_tolerant = (
            shift_tolerant if isinstance(shift_tolerant, Mapping) else {}
        )
        foreground = loss_config.get("fluorescence_foreground", {})
        foreground = foreground if isinstance(foreground, Mapping) else {}
        shift_enabled = bool(shift_tolerant.get("enabled", False))
        foreground_enabled = bool(foreground.get("enabled", False))
        multitask_mode = str(config.get("multitask", {}).get("optimizer", "equal"))
        task_balancer = build_task_balancer(multitask_mode, _selected_targets(config))
        return ScheduledCompositeLoss(
            schedule,
            pyramid_levels=int(
                pyramid.get("levels", 4) if isinstance(pyramid, Mapping) else 4
            ),
            statistics_pooled_weight=float(
                statistics.get("pooled_weight", 0.0)
                if isinstance(statistics, Mapping)
                else 0.0
            ),
            task_balancer=task_balancer,
            deep_supervision_weights=tuple(
                float(value)
                for value in loss_config.get(
                    "deep_supervision_weights", (1.0, 0.5, 0.25)
                )
            ),
            deep_supervision_final_weights=(
                tuple(
                    float(value)
                    for value in loss_config["deep_supervision_final_weights"]
                )
                if loss_config.get("deep_supervision_final_weights") is not None
                else None
            ),
            frequency=float(loss_config.get("frequency", 0.0)),
            correlation=float(loss_config.get("correlation", 0.0)),
            prototype_activation=float(
                prototype.get(
                    "activation", loss_config.get("prototype_activation", 0.0)
                )
            ),
            prototype_diversity=float(
                prototype.get(
                    "diversity", loss_config.get("prototype_diversity", 0.0)
                )
            ),
            prototype_usage_entropy=float(
                prototype.get(
                    "usage_entropy",
                    loss_config.get("prototype_usage_entropy", 0.0),
                )
            ),
            shift_tolerant_enabled=shift_enabled,
            shift_tolerant_weight=float(
                shift_tolerant.get("weight", 1.0) if shift_enabled else 0.0
            ),
            shift_tolerant_max_shift=int(shift_tolerant.get("max_shift", 1)),
            shift_tolerant_mode=str(shift_tolerant.get("mode", "hard")),
            shift_tolerant_temperature=float(
                shift_tolerant.get("temperature", 0.05)
            ),
            foreground_enabled=foreground_enabled,
            foreground_weight=float(foreground.get("weight", 0.0)),
            foreground_threshold_std_scale=float(
                foreground.get("threshold_std_scale", 0.25)
            ),
            foreground_temperature=float(foreground.get("temperature", 0.05)),
            foreground_mse_weight=float(foreground.get("mse_weight", 1.0)),
            foreground_dice_weight=float(foreground.get("dice_weight", 0.10)),
            foreground_intensity_weight=float(
                foreground.get("intensity_weight", 0.25)
            ),
            foreground_min_activity_std=float(
                foreground.get("min_activity_std", 0.01)
            ),
            competition_proxy_enabled=bool(
                isinstance(loss_config.get("competition_proxy", {}), Mapping)
                and loss_config.get("competition_proxy", {}).get("enabled", False)
            ),
            competition_proxy_weight=float(
                loss_config.get("competition_proxy", {}).get("weight", 0.0)
            ),
            competition_proxy_start_ratio=float(
                loss_config.get("competition_proxy", {}).get("start_ratio", 0.55)
            ),
            competition_proxy_ssim_weight=float(
                loss_config.get("competition_proxy", {}).get("ssim_weight", 0.7)
            ),
            competition_proxy_window_size=int(
                loss_config.get("competition_proxy", {}).get("window_size", 7)
            ),
            competition_proxy_psnr_cap=float(
                loss_config.get("competition_proxy", {}).get("psnr_cap", 60.0)
            ),
            data_range=float(loss_config.get("data_range", 1.0)),
        )
    multitask_mode = str(config.get("multitask", {}).get("optimizer", "equal"))
    if multitask_mode.casefold() != "equal":
        raise ValueError("FAMO/uncertainty require loss.schedule=two_phase")
    options = _filter_kwargs(CompositeRestorationLoss, loss_config)
    return CompositeRestorationLoss(**options)


def _limited_rows(
    source: str | Path | Sequence[Mapping[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    rows = (
        read_manifest(source)
        if isinstance(source, (str, Path))
        else [dict(row) for row in source]
    )
    return rows if limit is None else rows[: max(0, int(limit))]


def _make_loader(
    manifest: str | Path | Sequence[Mapping[str, Any]],
    data_root: str | Path,
    config: Mapping[str, Any],
    *,
    train: bool,
    inference: bool = False,
    activity_plan: ActivitySamplingPlan | None = None,
) -> DataLoader[Any]:
    data = config["data"]
    targets = _selected_targets(config)
    input_channels, target_channels = _channel_mapping(config, targets)
    limit_key = "max_train_samples" if train else "max_val_samples"
    limit = data.get(limit_key)
    rows = _limited_rows(manifest, int(limit) if limit is not None else None)
    if inference and not rows:
        raise ValueError(
            "Inference manifest is empty. Missing official test inputs cannot be treated as success."
        )
    context_options = config.get("model", {}).get("context", {})
    context_enabled = bool(
        isinstance(context_options, Mapping) and context_options.get("enabled", False)
    )
    if context_enabled and inference:
        dataset = NeighborhoodInferenceDataset(
            rows,
            data_root,
            input_channels=input_channels,
            strict=True,
            grid_size=int(context_options.get("grid_size", 3)),
            missing_policy=str(context_options.get("missing_policy", "center")),
            include_center=bool(context_options.get("include_center", True)),
            require_verified_grid=bool(context_options.get("require_verified_grid", True)),
            cache_size=int(context_options.get("cache_size", 0)),
        )
    elif inference:
        dataset = InferenceDataset(rows, data_root, input_channels=input_channels, strict=True)
    else:
        transform = None
        if train:
            augmentation = config.get("augmentation", {})
            shared_transform_options = {
                "horizontal_flip_probability": float(
                    augmentation.get("horizontal_flip_probability", 0.5)
                ),
                "vertical_flip_probability": float(
                    augmentation.get("vertical_flip_probability", 0.5)
                ),
                "rotate_probability": float(
                    augmentation.get("rotate_probability", 1.0)
                ),
                "gamma_range": tuple(
                    float(value)
                    for value in augmentation.get("gamma_range", (0.9, 1.1))
                ),
                "gamma_probability": float(
                    augmentation.get("gamma_probability", 0.5)
                ),
                "brightness_delta": float(
                    augmentation.get("brightness_delta", 0.05)
                ),
                "brightness_probability": float(
                    augmentation.get("brightness_probability", 0.5)
                ),
                "contrast_range": tuple(
                    float(value)
                    for value in augmentation.get("contrast_range", (0.95, 1.05))
                ),
                "contrast_probability": float(
                    augmentation.get("contrast_probability", 0.5)
                ),
                "noise_std": float(augmentation.get("noise_std", 0.005)),
                "noise_probability": float(
                    augmentation.get("noise_probability", 0.25)
                ),
            }
            if context_enabled:
                transform = ContextPairedTransform(**shared_transform_options)
            else:
                transform = PairedTransform(
                    **shared_transform_options,
                    max_translation=int(augmentation.get("max_translation", 0)),
                )
        if context_enabled:
            dataset = NeighborhoodDataset(
                rows,
                data_root,
                targets=targets,
                transform=transform,
                input_channels=input_channels,
                target_channels=target_channels,
                strict=True,
                grid_size=int(context_options.get("grid_size", 3)),
                missing_policy=str(context_options.get("missing_policy", "center")),
                include_center=bool(context_options.get("include_center", True)),
                require_verified_grid=bool(context_options.get("require_verified_grid", True)),
                cache_size=int(context_options.get("cache_size", 0)),
            )
        else:
            dataset = VirtualStainingDataset(
                rows,
                data_root,
                targets=targets,
                transform=transform,
                input_channels=input_channels,
                target_channels=target_channels,
                strict=True,
            )
    train_config = config["train"]
    inference_config = config["inference"]
    configured_batch = inference_config.get("batch_size", 1) if inference else train_config.get("batch_size", 1)
    if configured_batch == "auto":
        configured_batch = train_config.get("batch_size", 1)
    if configured_batch == "auto":
        configured_batch = hardware_profile()["batch_size"]
    batch_size = int(configured_batch)
    if batch_size < 1:
        batch_size = 1
    generator = set_seed(
        int(config["project"].get("seed", 2026)),
        deterministic=bool(train_config.get("deterministic", True)),
        benchmark=train_config.get("cudnn_benchmark", None),
    )
    workers = int(data.get("num_workers", 0))
    persistent = bool(data.get("persistent_workers", False)) and workers > 0
    pin = bool(data.get("pin_memory", False)) and torch.cuda.is_available()
    prefetch_factor = int(data.get("prefetch_factor", 2)) if workers > 0 else None
    sampler_options = (
        activity_plan.dataloader_kwargs()
        if train and activity_plan is not None
        else {"shuffle": train}
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch_factor,
        worker_init_fn=seed_worker if workers else None,
        generator=generator,
        **sampler_options,
    )


def _audit_image_specs() -> dict[str, Any] | None:
    path = Path("artifacts/data_audit.json")
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("markers")


def _expected_target_mode(target: str, image_specs: Any = None) -> str | None:
    canonical = normalize_marker(target)
    specs = _audit_image_specs() if image_specs is None else image_specs
    if not isinstance(specs, Mapping):
        return None
    candidate = specs.get(canonical, specs.get(canonical.casefold()))
    if not isinstance(candidate, Mapping):
        return None
    explicit = candidate.get("storage_mode") or candidate.get("mode")
    if explicit:
        return str(explicit).upper()
    mode_counts = candidate.get("mode_counts")
    if isinstance(mode_counts, Mapping) and len(mode_counts) == 1:
        return str(next(iter(mode_counts))).upper()
    channels = candidate.get("storage_channels")
    if channels is not None:
        return "L" if int(channels) == 1 else "RGB"
    return None


def _scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any]) -> Any:
    scheduler_name = str(config["train"].get("scheduler", "cosine")).casefold()
    if scheduler_name in {"none", "constant"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    epochs = int(config["train"].get("epochs", 1))
    warmup = int(config["train"].get("warmup_epochs", 0))

    def factor(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return max(1e-8, float(epoch + 1) / float(warmup))
        progress = (epoch - warmup) / max(1, epochs - warmup - 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def _prepare_training(
    config: dict[str, Any],
    root: Path,
    manifests: ManifestBuildResult,
    run_dir: Path,
) -> tuple[Trainer, Validator, nn.Module, DataLoader[Any], DataLoader[Any]]:
    train_rows: list[dict[str, Any]] = [dict(row) for row in read_manifest(manifests.train_manifest)]
    val_rows: list[dict[str, Any]] = [dict(row) for row in read_manifest(manifests.val_manifest)]
    fold_provenance: dict[str, Any] | None = None
    grouped_options = config["data"].get("grouped_inner_folds", {})
    if isinstance(grouped_options, Mapping) and bool(grouped_options.get("enabled", False)):
        roi_audit = write_roi_grid_audit(
            [*train_rows, *val_rows],
            root,
            run_dir / "artifacts" / "roi_grid",
        )
        fold = int(config.get("validation", {}).get("fold", 0))
        assignment_csv = run_dir / "artifacts" / "inner_fold_assignments.csv"
        split = prepare_authoritative_fold_split(
            train_rows,
            fold=fold,
            roi_audit=roi_audit,
            fold_count=int(grouped_options.get("fold_count", 5)),
            seed=int(grouped_options.get("seed", config["project"].get("seed", 2026))),
            output_csv=assignment_csv,
        )
        train_rows = [dict(row) for row in split.train_rows]
        val_rows = [dict(row) for row in split.validation_rows]
        fold_provenance = split.to_dict()
        grouped_options["output_csv"] = str(assignment_csv)
        config["project"]["fold"] = fold
        _write_json(run_dir / "artifacts" / "fold_provenance.json", fold_provenance)
    train_limit = config["data"].get("max_train_samples")
    val_limit = config["data"].get("max_val_samples")
    train_rows = _limited_rows(
        train_rows, int(train_limit) if train_limit is not None else None
    )
    val_rows = _limited_rows(val_rows, int(val_limit) if val_limit is not None else None)
    if not train_rows or not val_rows:
        raise ValueError("Resolved training and validation rows must both be non-empty")
    dapi_pretrain_manifest = _write_rows_csv(
        run_dir / "manifests" / "dapi_pretrain_manifest.csv", train_rows
    )
    activity_options = config["data"].get("activity_sampler", {})
    activity_enabled = bool(
        isinstance(activity_options, Mapping) and activity_options.get("enabled", False)
    )
    activity_report: dict[str, Any] | None = None
    if activity_enabled:
        activity_key = str(activity_options.get("activity_key", "target_activity"))
        train_rows, activity_report = compute_training_activity(
            train_rows,
            root,
            target=_selected_targets(config)[0],
            activity_key=activity_key,
            output_csv=run_dir / "artifacts" / "train_activity.csv",
        )
        _write_json(run_dir / "artifacts" / "train_activity.json", activity_report)
    else:
        activity_key = str(
            activity_options.get("activity_key", "target_activity")
            if isinstance(activity_options, Mapping)
            else "target_activity"
        )
    activity_plan = prepare_activity_sampling(
        train_rows,
        enabled=activity_enabled,
        activity_key=activity_key,
        source_indices=range(len(train_rows)),
        num_bins=int(activity_options.get("num_bins", 4))
        if isinstance(activity_options, Mapping)
        else 4,
        seed=int(activity_options.get("seed", config["project"].get("seed", 2026)))
        if isinstance(activity_options, Mapping)
        else int(config["project"].get("seed", 2026)),
        num_samples=activity_options.get("num_samples")
        if isinstance(activity_options, Mapping)
        else None,
        strategy=str(activity_options.get("strategy", "balanced"))
        if isinstance(activity_options, Mapping)
        else "balanced",
        start_epoch=int(
            math.ceil(
                float(activity_options.get("start_ratio", 0.0))
                * int(config["train"].get("epochs", 1))
            )
        )
        if isinstance(activity_options, Mapping)
        else 0,
        hard_fraction=float(activity_options.get("hard_fraction", 0.0))
        if isinstance(activity_options, Mapping)
        else 0.0,
    )
    resolved_train_manifest = _write_rows_csv(
        run_dir / "manifests" / "train_manifest.csv", train_rows
    )
    resolved_val_manifest = _write_rows_csv(
        run_dir / "manifests" / "val_manifest.csv", val_rows
    )
    train_loader = _make_loader(
        train_rows,
        root,
        config,
        train=True,
        activity_plan=activity_plan,
    )
    val_loader = _make_loader(val_rows, root, config, train=False)
    model = _build_model(config)
    stage_controller = None
    stage = config["train"].get("stage")
    if stage:
        stage_controller = MultiStageController()
        stage_controller.transition(str(stage))
        stage_controller.configure_trainable_parameters(model)
    loss_fn = _build_loss(config)
    optimizer_options = config["train"].get("optimizer_options", {})
    optimized_adamw = bool(
        isinstance(optimizer_options, Mapping)
        and optimizer_options.get("enabled", False)
    )
    optimizer = None
    scheduler = None
    if not optimized_adamw:
        optimizer = torch.optim.AdamW(
            collect_optimizer_parameters(model, loss_fn),
            lr=float(config["train"].get("lr", 2e-4)),
            weight_decay=float(config["train"].get("weight_decay", 1e-4)),
        )
        scheduler = _scheduler(optimizer, config)
    trainer = Trainer(
        model,
        train_loader,
        loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        stage_controller=stage_controller,
    )
    trainer.activity_sampling_plan = activity_plan
    trainer.fold_provenance = fold_provenance
    trainer.activity_report = activity_report
    trainer.training_manifest_path = resolved_train_manifest
    trainer.validation_manifest_path = resolved_val_manifest
    trainer.training_manifest_hash = manifest_sha1(resolved_train_manifest)
    trainer.validation_manifest_hash = manifest_sha1(resolved_val_manifest)
    trainer.dapi_pretrain_manifest_path = dapi_pretrain_manifest
    trainer.dapi_pretrain_manifest_hash = manifest_sha1(dapi_pretrain_manifest)
    validator = Validator(
        model,
        val_loader,
        device=trainer.device,
        config=config,
        ema=trainer.ema,
        task_name=config["train"].get("task_name"),
        image_spec=_audit_image_specs(),
        prototype_diagnostics_dir=run_dir
        / "artifacts"
        / "prototype_validation",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return trainer, validator, model, train_loader, val_loader


def _append_experiment(config: Mapping[str, Any], run_dir: Path, stats: Mapping[str, Any], history: Sequence[Any]) -> None:
    path = Path(config["project"].get("artifact_root", "artifacts")) / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    macro: Mapping[str, Any] = {}
    if history and isinstance(history[-1], Mapping):
        macro = history[-1].get("validation", {}).get("macro", {})
    row = {
        "run_id": run_dir.name,
        "git_commit": "",
        "config_hash": config_hash(dict(config)),
        "seed": config["project"].get("seed"),
        "fold": 0,
        "target": "+".join(_selected_targets(config)),
        "model": config["model"].get("name"),
        "params": stats.get("parameters"),
        "flops": int(stats.get("approximate_macs", 0)) * 2,
        "train_time": history[-1].get("train", {}).get("duration_seconds") if history else None,
        "peak_vram": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "val_ssim": macro.get("mean_ssim"),
        "val_psnr": macro.get("mean_psnr"),
        "roi_ssim": macro.get("roi_ssim"),
        "roi_psnr": macro.get("roi_psnr"),
        "tta": config["inference"].get("tta"),
        "ensemble": False,
        "checkpoint": str(run_dir / "checkpoints" / "best_ssim.ckpt"),
    }
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _v2_registry_record(
    config: Mapping[str, Any],
    run_result: Mapping[str, Any],
    *,
    parent_run: str = "",
    status: str = "completed",
    failure_reason: str = "",
) -> dict[str, Any]:
    run_dir = Path(str(run_result.get("run_dir", "")))
    history_path = run_dir / "metrics.json"
    pretrain_history_path = run_dir / "pretrain_metrics.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    elif pretrain_history_path.is_file():
        history = json.loads(pretrain_history_path.read_text(encoding="utf-8"))
    else:
        history = []
    validation = history[-1].get("validation", {}) if history else {}
    domains = validation.get("domains", {}) if isinstance(validation, Mapping) else {}

    def macro(domain: str, metric: str) -> Any:
        values = domains.get(domain, {}) if isinstance(domains, Mapping) else {}
        return values.get("macro", {}).get(metric) if isinstance(values, Mapping) else None

    stats_path = run_dir / "model_stats.json"
    stats = (
        json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_path.is_file()
        else {}
    )
    train_time = sum(
        float(
            item.get("train", {}).get(
                "duration_seconds", item.get("duration_seconds", 0.0)
            )
        )
        for item in history
    )
    model_config = config.get("model", {})
    context = model_config.get("context", {}) if isinstance(model_config, Mapping) else {}
    prototypes = model_config.get("prototypes", {}) if isinstance(model_config, Mapping) else {}
    targets = "+".join(_selected_targets(config))
    return {
        "run_id": run_dir.name or str(run_result.get("run_id", "unknown")),
        "parent_run": parent_run,
        "git_commit": "not_a_git_repo",
        "config_hash": config_hash(dict(config)),
        "manifest_hash": run_result.get("manifest_hash", ""),
        "model": model_config.get("name") if isinstance(model_config, Mapping) else "",
        "target": targets,
        "organ": config.get("data", {}).get("organ", "auto"),
        "fold": config.get("project", {}).get("fold", 0),
        "seed": config.get("project", {}).get("seed", 2026),
        "context": bool(context.get("enabled", False)) if isinstance(context, Mapping) else False,
        "pretrain": bool(config.get("pretrain", {}).get("enabled", False)),
        "prototype": (
            bool(prototypes.get("enabled", model_config.get("use_prototypes", False)))
            if isinstance(prototypes, Mapping)
            else bool(model_config.get("use_prototypes", False))
        ),
        "task_optimizer": config.get("multitask", {}).get("optimizer", "equal"),
        "loss_schedule": config.get("loss", {}).get("schedule", "none"),
        "params": stats.get("parameters"),
        "flops": int(stats.get("approximate_macs", 0)) * 2,
        "peak_vram": max(
            (
                int(item.get("train", {}).get("peak_vram_bytes", 0))
                for item in history
            ),
            default=0,
        ),
        "train_time": train_time,
        "float_ssim": macro("float", "mean_ssim"),
        "float_psnr": macro("float", "mean_psnr"),
        "uint8_ssim": macro("uint8", "mean_ssim"),
        "uint8_psnr": macro("uint8", "mean_psnr"),
        "jpg_ssim": macro("jpg", "mean_ssim"),
        "jpg_psnr": macro("jpg", "mean_psnr"),
        "weighted_score_proxy": macro("jpg", "local_proxy_score"),
        "checkpoint": run_result.get("best_ssim_checkpoint", ""),
        "status": status,
        "failure_reason": failure_reason,
    }


def _run_train(
    config: dict[str, Any],
    data_root: str | Path | None,
    run_id: str,
    *,
    initial_checkpoint: str | Path | None = None,
    pretrain_checkpoint: str | Path | None = None,
    initial_strict: bool = True,
    stage: str | None = None,
) -> dict[str, Any]:
    logger = logging.getLogger("virtual_staining")
    if initial_checkpoint is not None and pretrain_checkpoint is not None:
        raise ValueError(
            "An initial restoration checkpoint and a DAPI pretraining checkpoint "
            "cannot be applied in the same run"
        )
    _apply_hardware_defaults(config)
    if stage is not None:
        config["train"]["stage"] = stage
    targets = _selected_targets(config)
    config["train"]["task_name"] = targets[0] if len(targets) == 1 else None
    logger.info("Preparing training data and model for target(s): %s", ", ".join(targets))
    discovery = _resolve_discovery(data_root, config=config)
    root = discovery.selected_root
    config["data"]["root"] = str(root)
    logger.info("Using data root: %s", root)
    manifests = _ensure_manifests(root, config)
    context_options = config.get("model", {}).get("context", {})
    if isinstance(context_options, Mapping) and bool(context_options.get("enabled", False)):
        combined_rows = [
            *read_manifest(manifests.train_manifest),
            *read_manifest(manifests.val_manifest),
        ]
        roi_report = write_roi_grid_audit(
            combined_rows,
            root,
            config["project"].get("artifact_root", "artifacts/performance_v2"),
        )
        if bool(context_options.get("require_verified_grid", True)) and not bool(
            roi_report.get("context_enabled", False)
        ):
            reasons = ", ".join(roi_report.get("context_gate_reasons", []))
            raise ValueError(f"ROI context gate is disabled: {reasons or 'unverified_grid'}")
    output_root = Path(config["project"].get("output_root", "outputs"))
    run_dir = (output_root / run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already contains artifacts and cannot be overwritten: {run_dir}. "
            "Use a new run id or the resume command."
        )
    config["project"]["run_id"] = run_id
    save_effective_config(config, run_dir / "effective_config.yaml")
    logger.info("Building datasets, loaders, model, optimizer, and validator")
    trainer, validator, model, _, _ = _prepare_training(config, root, manifests, run_dir)
    save_effective_config(config, run_dir / "effective_config.yaml")
    initialization: dict[str, Any] | None = None
    pretrain_transfer: dict[str, Any] | None = None
    if initial_checkpoint is not None:
        checkpoint_path = Path(initial_checkpoint).expanduser().resolve()
        initialization = load_checkpoint(checkpoint_path, map_location="cpu")
        weight_source = str(config["train"].get("initial_weight_source", "raw"))
        initial_state = load_checkpoint_state(
            checkpoint_path,
            weight_source=weight_source,
        )
        if initial_strict:
            model.load_state_dict(initial_state, strict=True)
            initialization["compatible_tensor_count"] = len(initial_state)
            initialization["skipped_tensor_count"] = 0
        else:
            destination_state = model.state_dict()
            compatible = {
                key: value
                for key, value in initial_state.items()
                if key in destination_state
                and destination_state[key].shape == value.shape
                and destination_state[key].dtype == value.dtype
            }
            if not compatible:
                raise ValueError("Initial checkpoint has no compatible tensors")
            model.load_state_dict(compatible, strict=False)
            initialization["compatible_tensor_count"] = len(compatible)
            initialization["skipped_tensor_count"] = len(initial_state) - len(compatible)
        if trainer.stage_controller is not None:
            source_extra = initialization.get("extra", {})
            source_stage = (
                source_extra.get("multistage")
                if isinstance(source_extra, Mapping)
                else None
            )
            if isinstance(source_stage, Mapping):
                trainer.stage_controller.load_state_dict(dict(source_stage))
            trainer.stage_controller.transition(
                str(config["train"]["stage"]),
                source_checkpoint=str(checkpoint_path),
            )
            trainer.stage_controller.configure_trainable_parameters(model)
        initialization["ema_synchronized"] = trainer.sync_ema_from_model()
    if pretrain_checkpoint is not None:
        pretrain_transfer = transfer_local_encoder_from_checkpoint(
            pretrain_checkpoint,
            model,
            expected_manifest_hash=trainer.dapi_pretrain_manifest_hash,
        )
        pretrain_transfer["ema_synchronized"] = trainer.sync_ema_from_model()
        trainer.pretrain_transfer_report = pretrain_transfer
        _write_json(run_dir / "artifacts" / "pretrain_transfer.json", pretrain_transfer)
    parent_checkpoint_sha256 = (
        checkpoint_file_sha256(initial_checkpoint)
        if initial_checkpoint is not None
        else None
    )
    pretrain_checkpoint_sha256 = (
        str(pretrain_transfer["source_checkpoint_sha256"])
        if pretrain_transfer is not None
        else None
    )
    config["project"]["initialization_provenance"] = build_initialization_provenance(
        model.state_dict(),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        pretrain_checkpoint_sha256=pretrain_checkpoint_sha256,
    )
    save_effective_config(config, run_dir / "effective_config.yaml")
    input_channels, _ = _channel_mapping(config, targets)
    image_size = int(config["data"].get("image_size", 256))
    statistics_kwargs: dict[str, Any] = {}
    if isinstance(context_options, Mapping) and bool(context_options.get("enabled", False)):
        grid_size = int(context_options.get("grid_size", 3))
        tile_count = grid_size * grid_size
        statistics_kwargs = {
            "context_tiles": torch.zeros(
                1,
                tile_count,
                input_channels,
                image_size,
                image_size,
                device=trainer.device,
            ),
            "context_valid_mask": torch.ones(
                1, tile_count, dtype=torch.bool, device=trainer.device
            ),
            "context_offsets": context_offsets(grid_size)
            .unsqueeze(0)
            .to(device=trainer.device),
        }
    logger.info("Measuring model size and compute; batch progress starts immediately after this step")
    stats = model_statistics(
        model,
        (1, input_channels, image_size, image_size),
        device=trainer.device,
        forward_kwargs=statistics_kwargs,
    )
    _write_json(run_dir / "model_stats.json", stats)
    log_session = active_experiment_log()
    if log_session is not None:
        log_session.bind_training_run(
            run_dir,
            config=config,
            model=model,
            model_stats=stats,
        )
        trainer.add_epoch_callback(log_session.record_epoch)
    train_manifest_hash = trainer.training_manifest_hash
    logger.info("Preparation complete. Starting batch-by-batch training progress now")
    history = trainer.fit(
        epochs=int(config["train"].get("epochs", 1)),
        validator=validator,
        checkpoint_dir=run_dir / "checkpoints",
        manifest_hash=train_manifest_hash,
        image_spec=_audit_image_specs(),
        targets=list(targets),
    )
    _write_json(run_dir / "metrics.json", history)
    _append_experiment(config, run_dir, stats, history)
    return {
        "run_dir": str(run_dir),
        "targets": list(targets),
        "model_parameters": count_parameters(model),
        "manifest_hash": train_manifest_hash,
        "validation_manifest_hash": trainer.validation_manifest_hash,
        "dapi_pretrain_manifest_hash": trainer.dapi_pretrain_manifest_hash,
        "fold_provenance": trainer.fold_provenance,
        "activity_report": trainer.activity_report,
        "epochs_completed": len(history),
        "stop_reason": history[-1].get("stop_reason") if history else None,
        "last_checkpoint": str(run_dir / "checkpoints" / "last.ckpt"),
        "best_ssim_checkpoint": str(run_dir / "checkpoints" / "best_ssim.ckpt"),
        "initial_checkpoint": str(Path(initial_checkpoint).resolve())
        if initial_checkpoint is not None
        else None,
        "initial_checkpoint_version": initialization.get("checkpoint_version")
        if initialization is not None
        else None,
        "initial_compatible_tensors": initialization.get("compatible_tensor_count")
        if initialization is not None
        else None,
        "initial_skipped_tensors": initialization.get("skipped_tensor_count")
        if initialization is not None
        else None,
        "initial_ema_synchronized": initialization.get("ema_synchronized")
        if initialization is not None
        else None,
        "pretrain_checkpoint": str(Path(pretrain_checkpoint).resolve())
        if pretrain_checkpoint is not None
        else None,
        "pretrain_transfer": pretrain_transfer,
        "stage": config["train"].get("stage"),
    }


def command_env(args: argparse.Namespace) -> dict[str, Any]:
    report = environment_report()
    report["hardware_profile"] = hardware_profile()
    _write_json(args.output, report)
    return report


def command_discover(args: argparse.Namespace) -> dict[str, Any]:
    result = discover_data_root(
        args.data_root,
        workspace=Path.cwd(),
        config_path=args.config,
        output_path=args.output,
    )
    return result.to_dict()


def command_manifest(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, getattr(args, "set", None), include_resolved=False)
    discovery = _resolve_discovery(args.data_root, config=config)
    return _ensure_manifests(discovery, config).to_dict()


def command_audit(args: argparse.Namespace) -> dict[str, Any]:
    result = audit_data(
        args.data_root,
        workspace=Path.cwd(),
        correlation_sample_count=args.correlation_samples,
        alignment_sample_count=args.alignment_samples,
        figure_count=args.figure_count,
    )
    return result.to_dict()


def command_train(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_effective(args, apply_target_to_model=True)
    return _run_train(config, args.data_root, args.run_id)


def command_resume(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_effective(args, apply_target_to_model=True)
    _apply_hardware_defaults(config)
    targets = _selected_targets(config)
    config["train"]["task_name"] = targets[0] if len(targets) == 1 else None
    discovery = _resolve_discovery(args.data_root, config=config)
    root = discovery.selected_root
    manifests = _ensure_manifests(root, config)
    checkpoint_path = Path(args.checkpoint).resolve()
    run_dir = checkpoint_path.parent.parent
    trainer, validator, model, _, _ = _prepare_training(config, root, manifests, run_dir)
    resumed = resume_from_checkpoint(
        checkpoint_path,
        model,
        ema=trainer.ema,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        scaler=trainer.scaler,
        loss_fn=trainer.loss_fn,
        dataloader_generator=getattr(trainer.dataloader, "generator", None),
        map_location=trainer.device,
    )
    expected_hash = trainer.training_manifest_hash
    legacy_manifest_hash = manifest_sha1(manifests.train_manifest)
    if resumed.get("manifest_hash") not in {None, expected_hash, legacy_manifest_hash}:
        raise ValueError("Checkpoint manifest hash does not match the current train manifest")
    saved_targets = tuple(resumed.get("targets") or [])
    if saved_targets and saved_targets != targets:
        raise ValueError(f"Checkpoint targets {saved_targets} do not match requested {targets}")
    trainer.global_step = int(resumed["global_step"])
    trainer.metric_history = list(resumed.get("metric_history") or [])
    extra = resumed.get("checkpoint", {}).get("extra", {})
    if trainer.stage_controller is not None and isinstance(extra, Mapping):
        stage_state = extra.get("multistage")
        if isinstance(stage_state, Mapping):
            trainer.stage_controller.load_state_dict(dict(stage_state))
    if isinstance(extra, Mapping):
        prototype_state = extra.get("prototype_monitor")
        if prototype_state is not None:
            if trainer.prototype_monitor is None or not isinstance(
                prototype_state, Mapping
            ):
                raise ValueError(
                    "Checkpoint prototype monitor state is incompatible with the "
                    "current configuration"
                )
            trainer.prototype_monitor.load_state_dict(prototype_state)
        elif trainer.prototype_reset_enabled:
            raise ValueError(
                "A checkpoint resumed with prototype reset enabled must contain "
                "prototype monitor state"
            )
        saved_reset_contract = extra.get("prototype_reset")
        if saved_reset_contract is not None:
            if not isinstance(saved_reset_contract, Mapping) or dict(
                saved_reset_contract
            ) != trainer.prototype_reset_contract():
                raise ValueError(
                    "Checkpoint prototype reset contract does not match the "
                    "current configuration"
                )
        elif trainer.prototype_reset_enabled:
            raise ValueError(
                "Checkpoint does not contain the enabled prototype reset contract"
            )
        resolved_manifests = extra.get("resolved_manifests")
        if isinstance(resolved_manifests, Mapping):
            saved_dapi_hash = str(
                resolved_manifests.get("dapi_pretrain_hash", "")
            )
            if (
                saved_dapi_hash
                and saved_dapi_hash != trainer.dapi_pretrain_manifest_hash
            ):
                raise ValueError(
                    "Checkpoint DAPI-pretraining manifest does not match the "
                    "current training fold"
                )
        pretrain_transfer = extra.get("pretrain_transfer")
        if isinstance(pretrain_transfer, Mapping):
            trainer.pretrain_transfer_report = dict(pretrain_transfer)
        activity_state = extra.get("activity_sampler")
        activity_plan = getattr(trainer, "activity_sampling_plan", None)
        if activity_state is not None:
            if activity_plan is None or not isinstance(activity_state, Mapping):
                raise ValueError("Checkpoint activity sampler state is incompatible")
            activity_plan.load_state_dict(activity_state)
        saved_fold = extra.get("fold_provenance")
        current_fold = getattr(trainer, "fold_provenance", None)
        if isinstance(saved_fold, Mapping) or isinstance(current_fold, Mapping):
            saved_hash = (
                str(saved_fold.get("assignment_sha256", ""))
                if isinstance(saved_fold, Mapping)
                else ""
            )
            current_hash = (
                str(current_fold.get("assignment_sha256", ""))
                if isinstance(current_fold, Mapping)
                else ""
            )
            if not saved_hash or saved_hash != current_hash:
                raise ValueError("Checkpoint grouped-fold assignment does not match current fold")
    start_epoch = int(resumed["start_epoch"])
    log_session = active_experiment_log()
    if log_session is not None:
        stats_path = run_dir / "model_stats.json"
        model_stats = (
            json.loads(stats_path.read_text(encoding="utf-8"))
            if stats_path.is_file()
            else None
        )
        log_session.bind_training_run(
            run_dir,
            config=config,
            model=model,
            model_stats=model_stats,
        )
        trainer.add_epoch_callback(log_session.record_epoch)
    history = trainer.fit(
        epochs=int(config["train"].get("epochs", start_epoch + 1)),
        start_epoch=start_epoch,
        validator=validator,
        checkpoint_dir=run_dir / "checkpoints",
        manifest_hash=expected_hash,
        image_spec=resumed.get("image_spec"),
        targets=list(targets),
    )
    _write_json(run_dir / "metrics.json", history)
    return {"run_dir": str(run_dir), "start_epoch": start_epoch, "epochs_completed": len(history)}


def _checkpoint_runtime_config(
    config: Mapping[str, Any], checkpoint_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind loader/model settings to immutable checkpoint architecture metadata."""

    effective = copy.deepcopy(dict(config))
    saved_config = checkpoint_payload.get("config")
    if not isinstance(saved_config, Mapping):
        return effective
    saved_model = saved_config.get("model")
    saved_data = saved_config.get("data")
    if isinstance(saved_model, Mapping):
        effective["model"] = copy.deepcopy(dict(saved_model))
    if isinstance(saved_data, Mapping):
        for key in ("targets", "input_channels", "target_channels"):
            if key in saved_data:
                effective["data"][key] = copy.deepcopy(saved_data[key])
    saved_context = effective.get("model", {}).get("context", {})
    checkpoint_requires_context = bool(
        isinstance(saved_context, Mapping) and saved_context.get("enabled", False)
    )
    requested_context = bool(effective.get("inference", {}).get("context", False))
    if requested_context and not checkpoint_requires_context:
        raise ValueError(
            "inference.context=true is incompatible with a checkpoint whose model context is disabled"
        )
    effective["inference"]["context"] = checkpoint_requires_context
    return effective


def _evaluation_weight_source(
    config: Mapping[str, Any], checkpoint_payload: Mapping[str, Any]
) -> str:
    requested = str(config.get("inference", {}).get("weight_source", "auto")).casefold()
    explicit = requested in {"raw", "ema"}
    if requested in {"auto", "best_jpg"}:
        extra = checkpoint_payload.get("extra", {})
        selected = extra.get("selected_weight_source") if isinstance(extra, Mapping) else None
        requested = str(
            selected
            or ("ema" if config.get("inference", {}).get("use_ema", True) else "raw")
        ).casefold()
    if requested not in {"raw", "ema"}:
        raise ValueError("inference.weight_source must be raw, ema, auto, or best_jpg")
    if requested == "ema" and checkpoint_payload.get("ema") is None:
        if explicit:
            raise ValueError("EMA weights were requested but the checkpoint has no EMA state")
        return "raw"
    return requested


def _load_model_for_evaluation(config: dict[str, Any], checkpoint: str | Path) -> tuple[nn.Module, dict[str, Any]]:
    metadata = load_checkpoint(checkpoint, map_location="cpu")
    model_config = _checkpoint_runtime_config(config, metadata)
    model = _build_model(model_config)
    weight_source = _evaluation_weight_source(config, metadata)
    payload = load_checkpoint(
        checkpoint,
        model,
        map_location="cpu",
        use_ema_as_model=weight_source == "ema",
    )
    payload["loaded_weight_source"] = weight_source
    saved_targets = tuple(payload.get("targets") or [])
    expected = _selected_targets(model_config)
    if saved_targets and saved_targets != expected:
        raise ValueError(f"Checkpoint targets {saved_targets} do not match config targets {expected}")
    return model, payload


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_effective(args)
    _apply_hardware_defaults(config)
    discovery = _resolve_discovery(args.data_root, config=config)
    manifests = _ensure_manifests(discovery, config)
    model, checkpoint_payload = _load_model_for_evaluation(config, args.checkpoint)
    runtime_config = _checkpoint_runtime_config(config, checkpoint_payload)
    loader = _make_loader(
        manifests.val_manifest, discovery.selected_root, runtime_config, train=False
    )
    selected_target = normalize_marker(args.target) if args.target else None
    validator = Validator(
        model,
        loader,
        config=runtime_config,
        task_name=selected_target or runtime_config["train"].get("task_name"),
        image_spec=checkpoint_payload.get("image_spec"),
        progress_label=(
            "final validation, no TTA, "
            f"weights={checkpoint_payload['loaded_weight_source']}"
        ),
    )
    run_dir = Path(args.checkpoint).resolve().parent.parent
    result = validator.evaluate(records_path=run_dir / "validation" / "per_image.csv")
    if str(runtime_config["inference"].get("tta", "none")).casefold() == "d4":
        tta_validator = Validator(
            model,
            loader,
            config=runtime_config,
            task_name=selected_target or runtime_config["train"].get("task_name"),
            tta="d4",
            image_spec=checkpoint_payload.get("image_spec"),
            progress_label=(
                "final validation, D4 x8, "
                f"weights={checkpoint_payload['loaded_weight_source']}"
            ),
        )
        tta_result = tta_validator.evaluate(
            records_path=run_dir / "validation" / "per_image_tta_d4.csv"
        )
        decisions: dict[str, Any] = {}
        roi_audit = _context_audit_for_config(runtime_config, discovery.selected_root)
        validation_hash = manifest_sha1(manifests.val_manifest)
        provenance = {
            "fold": runtime_config.get("project", {}).get("fold", 0),
            "seed": runtime_config.get("project", {}).get("seed", 2026),
            "parent_manifest_sha256": validation_hash,
            "candidate_manifest_sha256": validation_hash,
        }
        for target in tta_result["per_target"]:
            baseline_records = [
                record
                for record in result["records"]
                if str(record.get("target")) == target
            ]
            candidate_records = [
                record
                for record in tta_result["records"]
                if str(record.get("target")) == target
            ]
            promotion = evaluate_roi_jpg_promotion(
                baseline_records,
                candidate_records,
                roi_audit=roi_audit,
                fold_seed_provenance=provenance,
                bootstrap_samples=int(
                    runtime_config.get("validation", {}).get(
                        "bootstrap_samples", 1000
                    )
                ),
                seed=int(runtime_config.get("project", {}).get("seed", 2026)),
            )
            single_evidence_reason = "insufficient_independent_fold_seed_evidence"
            screen_blockers = [
                reason
                for reason in promotion["reasons"]
                if reason != single_evidence_reason
            ]
            decisions[target] = {
                "enabled": not screen_blockers,
                "screen_roi_jpg_eligible": not screen_blockers,
                "final_default_eligible": bool(
                    promotion["final_default_eligible"]
                ),
                "reasons": screen_blockers,
                "confirmation_requirement": (
                    single_evidence_reason
                    if single_evidence_reason in promotion["reasons"]
                    else None
                ),
                "promotion_report": promotion,
            }
        result["tta_comparison"] = {
            "d4": tta_result,
            "decisions": decisions,
            "selection_metric": "roi_grouped_final_jpg_bootstrap",
        }
        _write_json(run_dir / "validation" / "tta_decisions.json", decisions)
    result["checkpoint_image_spec"] = checkpoint_payload.get("image_spec")
    result["loaded_weight_source"] = checkpoint_payload["loaded_weight_source"]
    _write_json(run_dir / "validation" / "metrics.json", result)
    return result


def _resolve_prediction_tta(
    configured_tta: Any,
    policy: str,
    decisions: Mapping[str, Any],
    target: str | None,
) -> str:
    """Resolve D4 without allowing an inconclusive audit to override config."""

    normalized_tta = str(configured_tta).strip().casefold()
    configured = "d4" if normalized_tta in {"d4", "true", "1"} else "none"
    normalized_policy = str(policy).strip().casefold()
    if normalized_policy == "configured":
        return configured
    if normalized_policy != "validation_decision":
        raise ValueError(
            "inference.tta_policy must be configured or validation_decision"
        )
    decision = decisions.get(target) if target is not None else None
    if isinstance(decision, Mapping):
        return "d4" if bool(decision.get("enabled")) else "none"
    return configured


def command_predict(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_effective(args)
    _apply_hardware_defaults(config)
    discovery = _resolve_discovery(args.data_root, config=config)
    manifests = _ensure_manifests(discovery, config)
    manifest = Path(args.manifest) if args.manifest else manifests.test_manifest
    if not manifest.is_absolute():
        manifest = Path.cwd() / manifest
    model, checkpoint_payload = _load_model_for_evaluation(config, args.checkpoint)
    runtime_config = _checkpoint_runtime_config(config, checkpoint_payload)
    loader = _make_loader(
        manifest,
        discovery.selected_root,
        runtime_config,
        train=False,
        inference=True,
    )
    run_dir = Path(args.checkpoint).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "predictions"
    selected_target = normalize_marker(args.target) if args.target else None
    configured_tta = runtime_config["inference"].get("tta", "none")
    tta_policy = str(
        runtime_config["inference"].get("tta_policy", "configured")
    ).strip().casefold()
    decision_path = run_dir / "validation" / "tta_decisions.json"
    decisions = (
        json.loads(decision_path.read_text(encoding="utf-8"))
        if decision_path.is_file()
        else {}
    )
    use_validation_decisions = tta_policy == "validation_decision"
    if decisions and not use_validation_decisions:
        logging.getLogger("virtual_staining").info(
            "Ignoring validation TTA decisions because inference.tta_policy=%s; "
            "configured inference.tta=%s is authoritative.",
            tta_policy,
            configured_tta,
        )

    def predict_one(target: str | None) -> dict[str, Any]:
        target_tta = _resolve_prediction_tta(
            configured_tta,
            tta_policy,
            decisions,
            target,
        )
        inferencer = Inferencer(
            model,
            config=runtime_config,
            tta=target_tta,
            image_spec=checkpoint_payload.get("image_spec"),
        )
        return inferencer.predict_loader(loader, output_dir=output_dir, target=target)

    if selected_target is None and decisions and use_validation_decisions:
        per_target = {
            target: predict_one(target) for target in _selected_targets(runtime_config)
        }
        result = {
            "count": next(iter(per_target.values()))["count"],
            "files": [path for report in per_target.values() for path in report["files"]],
            "per_target": per_target,
            "tta_decisions_applied": decisions,
        }
    else:
        result = predict_one(selected_target)
        result["tta_decisions_applied"] = (
            decisions if use_validation_decisions else {}
        )
    result["tta_decisions_available"] = decisions
    result["tta_policy"] = tta_policy
    result["configured_tta"] = str(configured_tta).strip().casefold()
    result["loaded_weight_source"] = checkpoint_payload["loaded_weight_source"]
    result["context_enabled"] = bool(runtime_config["inference"].get("context", False))
    result["output_dir"] = str(output_dir)
    _write_json(run_dir / "inference_report.json", result)
    return result


class _AveragedRestorer(nn.Module):
    def __init__(self, models: Sequence[nn.Module], weights: Sequence[float]) -> None:
        super().__init__()
        if not models or len(models) != len(weights):
            raise ValueError("Ensemble models and weights must be non-empty and aligned")
        total = sum(max(0.0, float(value)) for value in weights)
        if total <= 0:
            raise ValueError("Ensemble weights must contain a positive value")
        self.models = nn.ModuleList(models)
        self.weights = [max(0.0, float(value)) / total for value in weights]

    def forward(
        self,
        inputs: torch.Tensor,
        task_name: str | None = None,
        **model_kwargs: Any,
    ) -> RestorationOutput:
        combined: dict[str, torch.Tensor] = {}
        for model, weight in zip(self.models, self.weights, strict=True):
            output = call_model(
                model,
                inputs,
                task_name,
                model_kwargs=model_kwargs,
            )
            for target, prediction in extract_predictions(output).items():
                combined[target] = combined.get(target, torch.zeros_like(prediction)) + prediction * weight
        return RestorationOutput(predictions=combined)


def command_ensemble(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_effective(args)
    _apply_hardware_defaults(config)
    loaded = [_load_model_for_evaluation(config, path) for path in args.checkpoints]
    models = [model for model, _ in loaded]
    runtime_configs = [
        _checkpoint_runtime_config(config, payload) for _, payload in loaded
    ]
    context_flags = [
        bool(value.get("model", {}).get("context", {}).get("enabled", False))
        for value in runtime_configs
    ]
    if len(set(context_flags)) != 1:
        raise ValueError("Ensemble checkpoints disagree on context requirements")
    runtime_config = runtime_configs[0]
    image_specs = [payload.get("image_spec") for _, payload in loaded]
    if any(spec != image_specs[0] for spec in image_specs[1:]):
        raise ValueError("Ensemble checkpoints have incompatible ImageSpec metadata")
    if args.weights is not None and args.validation_scores is not None:
        raise ValueError("Use either explicit weights or validation SSIM scores, not both")
    if args.validation_scores is not None:
        if len(args.validation_scores) != len(models):
            raise ValueError("Validation score count must match checkpoint count")
        weights = validation_score_weights(
            args.validation_scores, temperature=float(args.weight_temperature)
        )
        weight_source = "validation_ssim_softmax"
    else:
        weights = args.weights or [1.0] * len(models)
        weight_source = "explicit" if args.weights is not None else "arithmetic_mean"
    averaged = _AveragedRestorer(models, weights)
    discovery = _resolve_discovery(args.data_root, config=config)
    manifests = _ensure_manifests(discovery, config)
    manifest = Path(args.manifest) if args.manifest else manifests.test_manifest
    loader = _make_loader(
        manifest,
        discovery.selected_root,
        runtime_config,
        train=False,
        inference=True,
    )
    output_dir = Path(args.output_dir or "outputs/ensemble/predictions").resolve()
    selected_target = normalize_marker(args.target) if args.target else None
    result = Inferencer(
        averaged, config=runtime_config, image_spec=image_specs[0]
    ).predict_loader(loader, output_dir=output_dir, target=selected_target)
    result["weights"] = list(averaged.weights)
    result["weight_source"] = weight_source
    result["member_weight_sources"] = [
        payload["loaded_weight_source"] for _, payload in loaded
    ]
    result["context_enabled"] = context_flags[0]
    result["output_dir"] = str(output_dir)
    return result


def command_make_submission(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_effective(args)
    target = normalize_marker(args.target)
    if target not in _submission_targets(config):
        raise ValueError(
            f"Target {target} is not enabled by data.submit_targets={list(_submission_targets(config))}"
        )
    is_smoke = _manifest_smoke_status(args.test_manifest)
    if is_smoke and not bool(config["submission"].get("allow_smoke_manifest", False)):
        raise ValueError("This configuration does not allow a smoke manifest")
    if not is_smoke and not bool(config["submission"].get("official", True)):
        raise ValueError("A smoke-only configuration cannot build an official submission")
    return build_submission(
        args.pred_dir,
        args.test_manifest,
        target,
        args.output_dir,
        root_name=config["submission"].get("root_name", "results"),
        split_name=config["submission"].get("split_name", "test"),
        fake_suffix=config["submission"].get("fake_suffix", "_fake"),
        extension=config["submission"].get("extension", ".jpg"),
        create_zip=bool(config["submission"].get("create_zip", True)),
        jpeg_quality=int(config["inference"].get("jpeg_quality", 100)),
        jpeg_subsampling=int(config["inference"].get("jpeg_subsampling", 0)),
        smoke=is_smoke,
    )


def command_validate_submission(args: argparse.Namespace) -> dict[str, Any]:
    requested_mode = getattr(args, "expected_mode", "auto")
    expected_mode = (
        _expected_target_mode(args.target)
        if str(requested_mode).casefold() == "auto"
        else str(requested_mode).upper()
    )
    return validate_submission(
        args.submission_dir,
        args.test_manifest,
        args.target,
        zip_path=args.zip_path,
        artifact_dir=args.artifact_dir,
        expected_mode=expected_mode,
    )


def command_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {"official_submission": False}
    report["environment"] = command_env(
        argparse.Namespace(output="artifacts/environment.json")
    )
    discovery = discover_data_root(
        args.data_root,
        workspace=Path.cwd(),
        config_path=args.config,
        output_path="artifacts/data_discovery.json",
    )
    report["discovery"] = discovery.to_dict()
    base_config = load_config(args.config, args.set, include_resolved=False)
    pipeline_targets = _selected_targets(base_config)
    submit_targets = _submission_targets(base_config)
    if len(pipeline_targets) != 1 or len(submit_targets) != 1:
        raise ValueError("The smoke pipeline requires exactly one configured prediction target")
    pipeline_target = submit_targets[0]
    manifests = _ensure_manifests(discovery, base_config)
    report["manifests"] = manifests.to_dict()
    report["audit"] = audit_data(discovery, workspace=Path.cwd()).to_dict()
    config = load_config(args.config, args.set, include_resolved=True)
    config["train"]["epochs"] = 1
    report["train"] = _run_train(config, discovery.selected_root, args.run_id)
    last = Path(report["train"]["last_checkpoint"])
    resume_args = argparse.Namespace(
        config=args.config,
        set=[*(args.set or []), "train.epochs=2"],
        target=pipeline_target,
        max_epochs=2,
        data_root=str(discovery.selected_root),
        checkpoint=str(last),
    )
    report["resume"] = command_resume(resume_args)
    best = last.parent / "best_ssim.ckpt"
    common = {
        "config": args.config,
        "set": args.set,
        "target": pipeline_target,
        "max_epochs": None,
        "data_root": str(discovery.selected_root),
    }
    report["validation"] = command_validate(
        argparse.Namespace(**common, checkpoint=str(best))
    )
    prediction_dir = Path(report["train"]["run_dir"]) / "predictions"
    report["prediction"] = command_predict(
        argparse.Namespace(
            **common,
            checkpoint=str(best),
            manifest=str(manifests.smoke_test_manifest),
            output_dir=str(prediction_dir),
        )
    )
    submission_root = Path(report["train"]["run_dir"]) / "smoke_submission"
    report["submission"] = command_make_submission(
        argparse.Namespace(
            config=args.config,
            set=args.set,
            target=pipeline_target,
            max_epochs=None,
            pred_dir=str(prediction_dir),
            test_manifest=str(manifests.smoke_test_manifest),
            output_dir=str(submission_root),
            submit_target=True,
        )
    )
    report["submission_validation"] = validate_submission(
        submission_root / "results",
        manifests.smoke_test_manifest,
        pipeline_target,
        zip_path=report["submission"]["zip_path"],
        artifact_dir=Path(report["train"]["run_dir"]) / "artifacts",
        expected_mode=_expected_target_mode(pipeline_target),
    )
    report_path = Path(report["train"]["run_dir"]) / "pipeline_report.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def _autodl_overrides(overrides: Sequence[str] | None) -> list[str]:
    """Use parallel JPEG loading on Linux unless the user chose a worker count."""

    resolved = list(overrides or [])
    worker_is_explicit = any(
        item.split("=", 1)[0].strip() == "data.num_workers"
        for item in resolved
        if "=" in item
    )
    if os.name != "nt" and not worker_is_explicit:
        worker_count = max(1, min(8, (os.cpu_count() or 2) // 2))
        resolved.extend(
            [
                f"data.num_workers={worker_count}",
                "data.persistent_workers=true",
            ]
        )
        logging.getLogger("virtual_staining").info(
            "AutoDL Linux data loading enabled with %d workers. Override with "
            "--set data.num_workers=N if needed.",
            worker_count,
        )
    return resolved


def command_autodl_run(args: argparse.Namespace) -> dict[str, Any]:
    """One-command training + validation + log for AutoDL CUDA GPUs.

    Chains env -> discover-data -> build-manifest -> audit-data -> train(full
    config epochs) -> validate(best selected weights in JPG) -> pipeline_report.json.
    The user only runs this single command; all internal steps and logging are
    handled automatically. Logs and artifacts land under log/<run-id>/ and
    outputs/<...>/<run-id>/.
    """
    report: dict[str, Any] = {"command": "autodl-run", "run_id": args.run_id}
    effective_overrides = _autodl_overrides(args.set)
    # 1. Environment report (CUDA / GPU / PyTorch version sanity).
    report["environment"] = command_env(
        argparse.Namespace(output="artifacts/environment.json")
    )
    # 2. Discover the official data root and build leakage-aware manifests.
    discovery = discover_data_root(
        args.data_root,
        workspace=Path.cwd(),
        config_path=args.config,
        output_path="artifacts/data_discovery.json",
    )
    report["discovery"] = discovery.to_dict()
    base_config = load_config(args.config, effective_overrides, include_resolved=False)
    pipeline_targets = _selected_targets(base_config)
    submit_targets = _submission_targets(base_config)
    if len(pipeline_targets) != 1 or len(submit_targets) != 1:
        raise ValueError(
            "autodl-run requires exactly one configured prediction target "
            f"(got targets={pipeline_targets}, submit_targets={submit_targets})"
        )
    pipeline_target = submit_targets[0]
    manifests = _ensure_manifests(discovery, base_config)
    report["manifests"] = manifests.to_dict()
    report["audit"] = audit_data(discovery, workspace=Path.cwd()).to_dict()
    # 3. Train with the requested config. --max-epochs overrides train.epochs
    #    for a quick sanity run if needed.
    config = load_config(args.config, effective_overrides, include_resolved=True)
    if args.max_epochs is not None:
        config["train"]["epochs"] = int(args.max_epochs)
    report["train"] = _run_train(config, discovery.selected_root, args.run_id)
    run_dir = Path(report["train"]["run_dir"])
    best = Path(report["train"]["best_ssim_checkpoint"])
    # 4. Validate the best checkpoint on the val split with raw+ema + jpg domain.
    report["validation"] = command_validate(
        argparse.Namespace(
            config=args.config,
            set=effective_overrides,
            target=pipeline_target,
            max_epochs=None,
            data_root=str(discovery.selected_root),
            checkpoint=str(best),
        )
    )
    # 5. Write a single consolidated report so the user can inspect everything
    #    from one file under log/<run-id>/ without hunting for sub-logs.
    report_path = run_dir / "pipeline_report.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    report["best_checkpoint"] = str(best)
    report["validation_metrics"] = str(run_dir / "validation" / "metrics.json")
    logger = logging.getLogger("virtual_staining")
    logger.info(
        "autodl-run complete | run_dir=%s | best=%s | validation=%s",
        run_dir,
        best,
        report["validation_metrics"],
    )
    return report


def command_autodl_submit(args: argparse.Namespace) -> dict[str, Any]:
    """One-command predict + make-submission + validate-submission -> ZIP.

    Resolves the best checkpoint (auto or --checkpoint), runs deterministic
    inference on the official test manifest, builds the competition results
    hierarchy and ZIP, then strictly validates the ZIP. The user only runs this
    single command after autodl-run has produced a checkpoint.
    """
    report: dict[str, Any] = {"command": "autodl-submit"}
    effective_overrides = _autodl_overrides(args.set)
    # 1. Resolve the checkpoint: explicit --checkpoint wins, else auto-detect
    #    the most recent best_ssim.ckpt under outputs/.
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
    else:
        checkpoint = _auto_checkpoint()
    report["checkpoint"] = str(checkpoint)
    # 2. Discover data + manifests to get the official test manifest path.
    discovery = discover_data_root(
        args.data_root,
        workspace=Path.cwd(),
        config_path=args.config,
        output_path="artifacts/data_discovery.json",
    )
    base_config = load_config(args.config, effective_overrides, include_resolved=False)
    submit_targets = _submission_targets(base_config)
    if len(submit_targets) != 1:
        raise ValueError(
            "autodl-submit requires exactly one submit target "
            f"(got {submit_targets})"
        )
    pipeline_target = submit_targets[0]
    manifests = _ensure_manifests(discovery, base_config)
    test_manifest = manifests.test_manifest
    # 3. Predict the official test set (DAPI-only, no labels).
    run_dir = checkpoint.parent.parent
    prediction_dir = run_dir / "predictions"
    report["prediction"] = command_predict(
        argparse.Namespace(
            config=args.config,
            set=effective_overrides,
            target=pipeline_target,
            max_epochs=None,
            data_root=str(discovery.selected_root),
            checkpoint=str(checkpoint),
            manifest=str(test_manifest),
            output_dir=str(prediction_dir),
        )
    )
    # 4. Build the competition results/ hierarchy and ZIP.
    submission_root = run_dir / "submission"
    report["submission"] = command_make_submission(
        argparse.Namespace(
            config=args.config,
            set=effective_overrides,
            target=pipeline_target,
            max_epochs=None,
            pred_dir=str(prediction_dir),
            test_manifest=str(test_manifest),
            output_dir=str(submission_root),
            submit_target=True,
        )
    )
    # 5. Strictly validate the ZIP (file names, count, size, mode).
    report["submission_validation"] = validate_submission(
        submission_root / "results",
        test_manifest,
        pipeline_target,
        zip_path=report["submission"]["zip_path"],
        artifact_dir=run_dir / "artifacts",
        expected_mode=_expected_target_mode(pipeline_target),
    )
    report["zip_path"] = report["submission"]["zip_path"]
    logger = logging.getLogger("virtual_staining")
    logger.info(
        "autodl-submit complete | zip=%s | checkpoint=%s",
        report["zip_path"],
        checkpoint,
    )
    return report


def command_autodl_ensemble_submit(args: argparse.Namespace) -> dict[str, Any]:
    """Average independent checkpoints with D4 and build one validated ZIP."""

    if len(args.checkpoints) < 2:
        raise ValueError("autodl-ensemble-submit requires at least two checkpoints")
    effective_overrides = _autodl_overrides(args.set)
    discovery = discover_data_root(
        args.data_root,
        workspace=Path.cwd(),
        config_path=args.config,
        output_path="artifacts/data_discovery.json",
    )
    base_config = load_config(args.config, effective_overrides, include_resolved=False)
    submit_targets = _submission_targets(base_config)
    if len(submit_targets) != 1:
        raise ValueError(
            "autodl-ensemble-submit requires exactly one submit target "
            f"(got {submit_targets})"
        )
    target = normalize_marker(args.target) if args.target else submit_targets[0]
    if target not in submit_targets:
        raise ValueError(
            f"Requested target {target!r} is not in submit_targets {submit_targets}"
        )
    manifests = _ensure_manifests(discovery, base_config)
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Ensemble output directory is not empty: {output_root}. "
            "Choose a new --output-dir to prevent stale predictions entering the ZIP."
        )
    prediction_dir = output_root / "predictions"
    ensemble_result = command_ensemble(
        argparse.Namespace(
            config=args.config,
            set=effective_overrides,
            target=target,
            max_epochs=None,
            data_root=str(discovery.selected_root),
            checkpoints=args.checkpoints,
            weights=args.weights,
            validation_scores=args.validation_scores,
            weight_temperature=args.weight_temperature,
            manifest=str(manifests.test_manifest),
            output_dir=str(prediction_dir),
        )
    )
    submission_root = output_root / "submission"
    submission = command_make_submission(
        argparse.Namespace(
            config=args.config,
            set=effective_overrides,
            target=target,
            max_epochs=None,
            pred_dir=str(prediction_dir),
            test_manifest=str(manifests.test_manifest),
            output_dir=str(submission_root),
            submit_target=True,
        )
    )
    validation = validate_submission(
        submission_root / "results",
        manifests.test_manifest,
        target,
        zip_path=submission["zip_path"],
        artifact_dir=output_root / "artifacts",
        expected_mode=_expected_target_mode(target),
    )
    report = {
        "command": "autodl-ensemble-submit",
        "checkpoints": [str(Path(path).expanduser().resolve()) for path in args.checkpoints],
        "ensemble": ensemble_result,
        "submission": submission,
        "submission_validation": validation,
        "zip_path": submission["zip_path"],
    }
    _write_json(output_root / "ensemble_submission_report.json", report)
    return report


def _auto_checkpoint() -> Path:
    candidates = sorted(
        Path("outputs").glob("**/best_ssim.ckpt"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No best_ssim.ckpt was found under outputs")
    return candidates[0].resolve()


def command_benchmark_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """Re-evaluate one immutable checkpoint across weights, TTA, and save domains."""

    base_config = _load_effective(args)
    _apply_hardware_defaults(base_config)
    if args.max_samples is None:
        base_config["data"].pop("max_val_samples", None)
    else:
        base_config["data"]["max_val_samples"] = int(args.max_samples)
    base_config["validation"]["primary_domain"] = "jpg"
    base_config["validation"]["domains"] = ["float", "uint8", "jpg"]
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else _auto_checkpoint()
    discovery = _resolve_discovery(args.data_root, config=base_config)
    manifests = _ensure_manifests(discovery, base_config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combinations: dict[str, Any] = {}
    for weight_source in args.evaluate_weights:
        normalized_weight = str(weight_source).casefold()
        if normalized_weight not in {"raw", "ema"}:
            raise ValueError("--evaluate-weights accepts only raw and ema")
        for tta in args.tta:
            normalized_tta = str(tta).casefold()
            if normalized_tta not in {"none", "d4"}:
                raise ValueError("--tta accepts only none and d4")
            config = copy.deepcopy(base_config)
            config["inference"]["use_ema"] = normalized_weight == "ema"
            config["inference"]["weight_source"] = normalized_weight
            model, payload = _load_model_for_evaluation(config, checkpoint)
            runtime_config = _checkpoint_runtime_config(config, payload)
            loader = _make_loader(
                manifests.val_manifest,
                discovery.selected_root,
                runtime_config,
                train=False,
            )
            name = f"{normalized_weight}_{normalized_tta}"
            validator = Validator(
                model,
                loader,
                config=runtime_config,
                use_ema=False,
                tta=normalized_tta,
                task_name=getattr(args, "target", None),
                image_spec=payload.get("image_spec") or _audit_image_specs(),
            )
            combinations[name] = validator.evaluate(
                records_path=output_dir / f"baseline_{name}_per_image.csv"
            )
    result = {
        "checkpoint": str(checkpoint),
        "data_root": str(discovery.selected_root),
        "validation_manifest": str(manifests.val_manifest),
        "validation_count": len(read_manifest(manifests.val_manifest))
        if args.max_samples is None
        else min(int(args.max_samples), len(read_manifest(manifests.val_manifest))),
        "combinations": combinations,
        "selection_warning": "Short-run EMA is reported independently and is not assumed better.",
    }
    _write_json(output_dir / "baseline_benchmark.json", result)
    return {
        **{key: value for key, value in result.items() if key != "combinations"},
        "combinations": {
            name: {
                "primary_domain": report["primary_domain"],
                "domains": {
                    domain: values["macro"]
                    for domain, values in report["domains"].items()
                },
                "duration_seconds": report["duration_seconds"],
                "tta": report["tta"],
                "records_path": str(output_dir / f"baseline_{name}_per_image.csv"),
            }
            for name, report in combinations.items()
        },
        "full_report": str(output_dir / "baseline_benchmark.json"),
    }


def command_audit_roi_grid(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, getattr(args, "set", None), include_resolved=False)
    discovery = _resolve_discovery(args.data_root, config=config)
    if args.manifest:
        manifest_paths = [Path(path).expanduser().resolve() for path in args.manifest]
    else:
        manifests = _ensure_manifests(discovery, config)
        manifest_paths = [manifests.train_manifest, manifests.val_manifest]
    rows = [row for path in manifest_paths for row in read_manifest(path)]
    result = write_roi_grid_audit(
        rows,
        discovery.selected_root,
        args.output_dir,
        mosaic_count=args.mosaic_count,
        border_width=args.border_width,
    )
    result["manifests"] = [str(path) for path in manifest_paths]
    return result


def command_pretrain_dapi(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_effective(args)
    project_config = config.setdefault("project", {})
    if Path(project_config.get("output_root", "outputs")) == Path("outputs"):
        project_config["output_root"] = "outputs/performance_v2"
    _apply_hardware_defaults(config)
    discovery = _resolve_discovery(args.data_root, config=config)
    root = discovery.selected_root
    config["data"]["root"] = str(root)
    manifests = _ensure_manifests(discovery, config)
    run_dir = (
        Path(config["project"].get("output_root", "outputs/performance_v2"))
        / args.run_id
    ).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Pretraining run directory already contains artifacts: {run_dir}. "
            "Use a new run id."
        )
    config["project"]["run_id"] = args.run_id
    train_rows = [dict(row) for row in read_manifest(manifests.train_manifest)]
    validation_rows = [dict(row) for row in read_manifest(manifests.val_manifest)]
    fold_provenance: dict[str, Any] | None = None
    grouped_options = config["data"].get("grouped_inner_folds", {})
    if isinstance(grouped_options, Mapping) and bool(
        grouped_options.get("enabled", False)
    ):
        roi_audit = write_roi_grid_audit(
            [*train_rows, *validation_rows],
            root,
            run_dir / "artifacts" / "roi_grid",
        )
        fold = int(config.get("validation", {}).get("fold", 0))
        assignment_csv = run_dir / "artifacts" / "inner_fold_assignments.csv"
        split = prepare_authoritative_fold_split(
            train_rows,
            fold=fold,
            roi_audit=roi_audit,
            fold_count=int(grouped_options.get("fold_count", 5)),
            seed=int(
                grouped_options.get("seed", config["project"].get("seed", 2026))
            ),
            output_csv=assignment_csv,
        )
        train_rows = [dict(row) for row in split.train_rows]
        fold_provenance = split.to_dict()
        config["project"]["fold"] = fold
        _write_json(run_dir / "artifacts" / "fold_provenance.json", fold_provenance)
    limit = config["data"].get("max_train_samples")
    rows = _limited_rows(
        train_rows,
        int(limit) if limit is not None else None,
    )
    if not rows:
        raise ValueError("DAPI pretraining requires at least one training row")
    resolved_manifest = _write_rows_csv(
        run_dir / "manifests" / "dapi_pretrain_manifest.csv", rows
    )
    input_channels, _ = _channel_mapping(config, _selected_targets(config))
    dataset = InferenceDataset(
        rows,
        root,
        input_channels=input_channels,
        strict=True,
    )
    generator = set_seed(int(config["project"].get("seed", 2026)))
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"].get("batch_size", 2)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        generator=generator,
    )
    pretrain = config.get("pretrain", {})
    model = DAPIMaskedAutoencoder(
        in_channels=input_channels,
        block_size=int(pretrain.get("block_size", 16)),
        mask_ratio=float(pretrain.get("mask_ratio", 0.5)),
        widths=tuple(pretrain.get("widths", (32, 64, 128, 256))),
        encoder_depths=tuple(pretrain.get("encoder_depths", (1, 1, 2, 2))),
        decoder_depths=tuple(pretrain.get("decoder_depths", (1, 1, 1))),
        masking_enabled=bool(pretrain.get("masking_enabled", True)),
        use_sobel_input=bool(config["model"].get("use_sobel_input", True)),
        use_laplacian_input=bool(
            config["model"].get(
                "use_laplacian_input",
                config["model"].get("local_encoder", {}).get(
                    "use_laplacian_input", False
                ),
            )
        ),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(config, run_dir / "effective_config.yaml")
    epochs = int(args.max_epochs or pretrain.get("epochs", config["train"].get("epochs", 1)))
    pretrainer = DAPIPretrainer(
        model,
        loader,
        device=config["train"].get("device", "auto"),
        learning_rate=float(pretrain.get("lr", config["train"].get("lr", 2e-4))),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
        amp=bool(config["train"].get("amp", True)),
        progress_bar=config["train"].get("progress_bar", "auto"),
        progress_bar_refresh_seconds=float(
            config["train"].get("progress_bar_refresh_seconds", 1.0)
        ),
    )
    image_size = int(config["data"].get("image_size", 256))
    stats = model_statistics(
        model,
        (1, input_channels, image_size, image_size),
        device=pretrainer.device,
    )
    _write_json(run_dir / "model_stats.json", stats)
    checkpoint = run_dir / "checkpoints" / "dapi_mae_last.ckpt"
    train_manifest_hash = manifest_sha1(resolved_manifest)
    pretrain_scope = (
        "authoritative_inner_fold" if fold_provenance is not None else "outer_train_only"
    )
    history = pretrainer.fit(
        epochs,
        checkpoint,
        config=config,
        manifest_hash=train_manifest_hash,
        provenance={
            "pretrain_scope": pretrain_scope,
            "fold_provenance": fold_provenance,
            "resolved_manifest": str(resolved_manifest),
        },
    )
    _write_json(run_dir / "pretrain_metrics.json", history)
    result = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "best_ssim_checkpoint": str(checkpoint),
        "manifest_hash": train_manifest_hash,
        "epochs_completed": len(history),
        "training_samples": len(rows),
        "uses_target_labels": False,
        "allowed_split": "train_only",
        "pretrain_scope": pretrain_scope,
        "fold_provenance": fold_provenance,
        "resolved_manifest": str(resolved_manifest),
    }
    registry = ExperimentRegistry(
        Path(config["project"].get("artifact_root", "artifacts/performance_v2"))
        / "experiment_registry.csv"
    )
    registry.upsert(_v2_registry_record(config, result, status="completed_pretrain"))
    return result


def _configure_finetune_from_checkpoint(
    config: dict[str, Any],
    checkpoint: str | Path,
    stage: str,
) -> bool:
    """Bind a fine-tune config to its immutable parent architecture and loss."""

    source = load_checkpoint(checkpoint, map_location="cpu")
    source_config = source.get("config", {})
    source_model = source_config.get("model") if isinstance(source_config, Mapping) else None
    if not isinstance(source_model, Mapping):
        raise ValueError("Fine-tuning checkpoint does not contain a model configuration")
    config["model"] = copy.deepcopy(dict(source_model))
    source_data = source_config.get("data") if isinstance(source_config, Mapping) else None
    if isinstance(source_data, Mapping):
        for key in ("input_channels", "target_channels"):
            if key in source_data:
                config["data"][key] = copy.deepcopy(source_data[key])
    source_loss = source_config.get("loss") if isinstance(source_config, Mapping) else None
    if stage != "metric_finetune" and isinstance(source_loss, Mapping):
        config["loss"] = copy.deepcopy(dict(source_loss))
    if stage == "organ_finetune":
        conditioning = dict(config["model"].get("conditioning", {}))
        conditioning["organ_embedding"] = True
        config["model"]["conditioning"] = conditioning
        adapters = dict(config["model"].get("adapters", {}))
        adapters["organ"] = True
        config["model"]["adapters"] = adapters
    source_targets = tuple(source.get("targets") or ())
    return stage != "organ_finetune" and (
        not source_targets or source_targets == _selected_targets(config)
    )


def _command_finetune(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    config = _load_effective(args, apply_target_to_model=True)
    strict_initialization = _configure_finetune_from_checkpoint(
        config,
        args.checkpoint,
        stage,
    )
    config["project"]["output_root"] = "outputs/performance_v2"
    config["project"]["artifact_root"] = "artifacts/performance_v2"
    result = _run_train(
        config,
        args.data_root,
        args.run_id,
        initial_checkpoint=args.checkpoint,
        initial_strict=strict_initialization,
        stage=stage,
    )
    registry = ExperimentRegistry(
        Path(config["project"].get("artifact_root", "artifacts/performance_v2"))
        / "experiment_registry.csv"
    )
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    parent_run = (
        checkpoint_path.parent.parent.name
        if checkpoint_path.parent.name.casefold() == "checkpoints"
        else checkpoint_path.stem
    )
    registry.upsert(_v2_registry_record(config, result, parent_run=parent_run))
    return result


def command_train_v2(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_effective(args, apply_target_to_model=True)
    if str(config["model"].get("name", "")).casefold() != "camp_vs_v2":
        raise ValueError("train-v2 requires model.name=camp_vs_v2")
    pretrain_checkpoint = getattr(args, "pretrain_checkpoint", None)
    if pretrain_checkpoint is not None:
        config.setdefault("pretrain", {})["enabled"] = True
        config["pretrain"]["checkpoint"] = str(
            Path(pretrain_checkpoint).expanduser().resolve()
        )
    result = _run_train(
        config,
        args.data_root,
        args.run_id,
        pretrain_checkpoint=pretrain_checkpoint,
    )
    registry = ExperimentRegistry(
        Path(config["project"].get("artifact_root", "artifacts/performance_v2"))
        / "experiment_registry.csv"
    )
    registry.upsert(_v2_registry_record(config, result))
    return result


def command_finetune_target(args: argparse.Namespace) -> dict[str, Any]:
    return _command_finetune(args, "target_finetune")


def command_finetune_organ(args: argparse.Namespace) -> dict[str, Any]:
    return _command_finetune(args, "organ_finetune")


def command_finetune_metric(args: argparse.Namespace) -> dict[str, Any]:
    return _command_finetune(args, "metric_finetune")


def _ablation_stage_slice(
    order: Sequence[str], start: str | None, stop: str | None
) -> list[str]:
    values = list(order)
    if start is not None:
        if start not in values:
            raise ValueError(f"Unknown --from-stage {start}")
        values = values[values.index(start) :]
    if stop is not None:
        if stop not in values:
            raise ValueError(f"Unknown --through-stage {stop}")
        values = values[: values.index(stop) + 1]
    return values


def _ablation_run_id(
    suite_name: str,
    stage_name: str,
    budget: str,
    seed: int | str,
) -> str:
    return f"{suite_name}_{stage_name}_{budget}_seed{seed}"


def _ablation_evidence_run_id(
    suite_name: str,
    stage_name: str,
    plan: AblationBudgetPlan,
    evidence: EvidenceRun,
) -> str:
    """Return a stable run id while preserving the historical one-run spelling."""

    if (
        plan.evidence_count == 1
        and evidence.fold == 0
        and not plan.uses_official_outer_validation
    ):
        return _ablation_run_id(
            suite_name,
            stage_name,
            plan.budget,
            evidence.seed,
        )
    return evidence.make_run_id(f"{suite_name}_{stage_name}_{plan.budget}")


def _declared_ablation_parent_run(
    suite_name: str,
    stage_spec: Mapping[str, Any],
    budget: str,
    seed: int | str,
) -> str:
    parent_stage = str(stage_spec.get("parent", "")).strip()
    if not parent_stage:
        return ""
    return _ablation_run_id(suite_name, parent_stage, budget, seed)


def _declared_ablation_parent_evidence_run(
    suite_name: str,
    stage_spec: Mapping[str, Any],
    plan: AblationBudgetPlan,
    evidence: EvidenceRun,
) -> str:
    parent_stage = str(stage_spec.get("parent", "")).strip()
    if not parent_stage:
        return ""
    return _ablation_evidence_run_id(
        suite_name,
        parent_stage,
        plan,
        evidence,
    )


def _roi_context_audit_failures(audit: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every failed condition in the promotable ROI/context gate."""

    failures: list[str] = []
    total_rows = int(audit.get("total_rows", 0))
    parsed_rows = int(audit.get("parsed_rows", 0))
    if total_rows <= 0:
        failures.append("empty_manifest")
    if parsed_rows != total_rows:
        failures.append("incomplete_filename_coordinates")
    if not bool(audit.get("filename_grid_verified", False)):
        failures.append("unverified_filename_coordinates")
    if audit.get("duplicate_coordinates"):
        failures.append("duplicate_coordinates")
    if audit.get("train_val_shared_rois"):
        failures.append("train_val_roi_overlap")
    if audit.get("cross_split_adjacent_pairs"):
        failures.append("cross_split_adjacent_patches")

    boundary = audit.get("boundary")
    if not isinstance(boundary, Mapping):
        failures.append("missing_boundary_audit")
    else:
        if not bool(boundary.get("direction_verified", False)):
            failures.append("coordinate_direction_not_verified")
        if not bool(boundary.get("continuity_verified", False)):
            failures.append("boundary_continuity_not_verified")

    gate_reasons = audit.get("context_gate_reasons", ())
    if isinstance(gate_reasons, Sequence) and not isinstance(gate_reasons, (str, bytes)):
        failures.extend(str(reason) for reason in gate_reasons if str(reason))
    if not bool(audit.get("context_enabled", False)):
        failures.append("context_gate_disabled")
    return tuple(dict.fromkeys(failures))


def _context_audit_for_config(config: Mapping[str, Any], data_root: str | Path | None) -> dict[str, Any]:
    discovery = _resolve_discovery(data_root, config=config)
    manifests = _ensure_manifests(discovery, config)
    rows = [
        *read_manifest(manifests.train_manifest),
        *read_manifest(manifests.val_manifest),
    ]
    return write_roi_grid_audit(
        rows,
        discovery.selected_root,
        config["project"].get("artifact_root", "artifacts/performance_v2"),
    )


def _last_run_validation_records(run_dir: str | Path) -> list[dict[str, Any]]:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    records_path = resolved_run_dir / "validation" / "per_image.csv"
    if records_path.is_file():
        with records_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        if rows:
            return rows
    metrics_path = resolved_run_dir / "metrics.json"
    if not metrics_path.is_file():
        return []
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else [payload]
    for entry in reversed(entries):
        if not isinstance(entry, Mapping):
            continue
        validation = entry.get("validation", entry)
        if not isinstance(validation, Mapping):
            continue
        records = validation.get("records")
        if isinstance(records, list):
            return [dict(record) for record in records if isinstance(record, Mapping)]
    return []


def _promotion_across_evidence(
    parent_results: Mapping[str, Mapping[str, Any]],
    candidate_results: Mapping[str, Mapping[str, Any]],
    *,
    evidence_runs: Sequence[EvidenceRun],
    roi_audit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    parent_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    parent_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    for evidence in evidence_runs:
        parent_result = parent_results.get(evidence.evidence_id)
        candidate_result = candidate_results.get(evidence.evidence_id)
        if parent_result is None or candidate_result is None:
            return None
        parent_record_dir = str(
            candidate_result.get("promotion_parent_run_dir")
            or parent_result.get("run_dir", "")
        )
        evidence_parent_records = _last_run_validation_records(parent_record_dir)
        evidence_candidate_records = _last_run_validation_records(
            str(candidate_result.get("run_dir", ""))
        )
        if not evidence_parent_records or not evidence_candidate_records:
            return None
        parent_records.extend(bind_evidence_records(evidence_parent_records, evidence))
        candidate_records.extend(
            bind_evidence_records(evidence_candidate_records, evidence)
        )
        parent_hashes[evidence.evidence_id] = str(
            candidate_result.get("promotion_parent_validation_manifest_hash")
            or parent_result.get("validation_manifest_hash", "")
        )
        candidate_hashes[evidence.evidence_id] = str(
            candidate_result.get("validation_manifest_hash", "")
        )
    if any(not value for value in (*parent_hashes.values(), *candidate_hashes.values())):
        return None
    provenance = build_promotion_provenance(
        evidence_runs,
        parent_validation_hashes=parent_hashes,
        candidate_validation_hashes=candidate_hashes,
    )
    return evaluate_roi_jpg_promotion(
        parent_records,
        candidate_records,
        roi_audit=roi_audit,
        fold_seed_provenance=provenance,
        bootstrap_samples=int(
            config.get("validation", {}).get("bootstrap_samples", 1000)
        ),
        seed=int(config.get("project", {}).get("seed", 2026)),
    )


def _roi_grouped_budget_ready(audit: Mapping[str, Any]) -> bool:
    total_rows = int(audit.get("total_rows", 0))
    return bool(
        total_rows > 0
        and int(audit.get("parsed_rows", 0)) == total_rows
        and audit.get("filename_grid_verified") is True
        and not audit.get("duplicate_coordinates")
        and not audit.get("train_val_shared_rois")
        and not audit.get("cross_split_adjacent_pairs")
    )


def _configure_ablation_evidence(
    source: Mapping[str, Any],
    *,
    plan: AblationBudgetPlan,
    evidence: EvidenceRun,
    audit: Mapping[str, Any],
    max_epochs: int | None,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> dict[str, Any]:
    """Resolve one immutable fold/seed config from a declared budget plan."""

    config = copy.deepcopy(dict(source))
    config.setdefault("project", {})["seed"] = int(evidence.seed)
    config["project"]["fold"] = evidence.record_fold
    config.setdefault("validation", {})["fold"] = evidence.record_fold
    config.setdefault("train", {})["epochs"] = int(
        max_epochs if max_epochs is not None else plan.epochs
    )
    budget_entry = config.get("budget", {}).get(plan.budget, {})
    if not isinstance(budget_entry, Mapping):
        budget_entry = {}
    train_limit = (
        max_train_samples
        if max_train_samples is not None
        else budget_entry.get("max_train_samples")
    )
    val_limit = (
        max_val_samples
        if max_val_samples is not None
        else budget_entry.get("max_val_samples")
    )
    if train_limit is not None:
        config.setdefault("data", {})["max_train_samples"] = int(train_limit)
    if val_limit is not None:
        config.setdefault("data", {})["max_val_samples"] = int(val_limit)

    grouped = config.setdefault("data", {}).setdefault(
        "grouped_inner_folds",
        {
            "enabled": False,
            "fold_count": 5,
            "seed": 2026,
            "require_official_train": True,
            "coordinate_source": "filename",
        },
    )
    if not isinstance(grouped, dict):
        raise TypeError("data.grouped_inner_folds must be a mapping")
    if plan.budget in {"screen", "confirm"}:
        grouped["enabled"] = _roi_grouped_budget_ready(audit)
    elif plan.uses_official_outer_validation:
        grouped["enabled"] = False
    return config


def _ablation_parent_checkpoint(parent_result: Mapping[str, Any]) -> Path:
    for field in ("best_ssim_checkpoint", "checkpoint", "last_checkpoint"):
        value = str(parent_result.get(field, "")).strip()
        if value and Path(value).expanduser().is_file():
            return Path(value).expanduser().resolve()
    raise FileNotFoundError("Ablation parent does not expose an existing checkpoint")


def _run_ablation_training_evidence(
    stage_name: str,
    config: dict[str, Any],
    data_root: str | Path | None,
    run_id: str,
    parent_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if stage_name not in {"A5", "A6"}:
        return _run_train(config, data_root, run_id)
    if parent_result is None:
        raise ValueError(f"{stage_name} requires its parent checkpoint")
    checkpoint = _ablation_parent_checkpoint(parent_result)
    stage = "target_finetune" if stage_name == "A5" else "organ_finetune"
    strict = _configure_finetune_from_checkpoint(config, checkpoint, stage)
    return _run_train(
        config,
        data_root,
        run_id,
        initial_checkpoint=checkpoint,
        initial_strict=strict,
        stage=stage,
    )


def _run_ablation_tta_evidence(
    config: dict[str, Any],
    data_root: str | Path | None,
    run_id: str,
    parent_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate A7 without retraining and preserve a same-checkpoint no-TTA parent."""

    checkpoint = _ablation_parent_checkpoint(parent_result)
    config["project"]["output_root"] = "outputs/performance_v2"
    config["project"]["artifact_root"] = "artifacts/performance_v2"
    output_root = Path(config["project"]["output_root"])
    run_dir = (output_root / run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already contains artifacts and cannot be overwritten: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    config["project"]["run_id"] = run_id
    discovery = _resolve_discovery(data_root, config=config)
    manifests = _ensure_manifests(discovery, config)
    model, checkpoint_payload = _load_model_for_evaluation(config, checkpoint)
    runtime_config = _checkpoint_runtime_config(config, checkpoint_payload)
    loader = _make_loader(
        manifests.val_manifest,
        discovery.selected_root,
        runtime_config,
        train=False,
    )
    task_name = runtime_config.get("train", {}).get("task_name")
    baseline = Validator(
        model,
        loader,
        config=runtime_config,
        task_name=task_name,
        tta=None,
        image_spec=checkpoint_payload.get("image_spec"),
    ).evaluate(
        records_path=run_dir / "promotion_parent" / "validation" / "per_image.csv"
    )
    candidate = Validator(
        model,
        loader,
        config=runtime_config,
        task_name=task_name,
        tta="d4",
        image_spec=checkpoint_payload.get("image_spec"),
    ).evaluate(records_path=run_dir / "validation" / "per_image.csv")
    _write_json(
        run_dir / "promotion_parent" / "metrics.json",
        [{"validation": baseline, "selected_weight_source": checkpoint_payload["loaded_weight_source"]}],
    )
    _write_json(
        run_dir / "metrics.json",
        [{"validation": candidate, "selected_weight_source": checkpoint_payload["loaded_weight_source"]}],
    )
    save_effective_config(runtime_config, run_dir / "effective_config.yaml")
    parent_stats = checkpoint.parent.parent / "model_stats.json"
    if parent_stats.is_file():
        stats = json.loads(parent_stats.read_text(encoding="utf-8"))
    else:
        stats = {"parameters": count_parameters(model), "approximate_macs": None}
    stats["evaluation_only"] = True
    _write_json(run_dir / "model_stats.json", stats)
    validation_hash = manifest_sha1(manifests.val_manifest)
    return {
        "run_dir": str(run_dir),
        "targets": list(_selected_targets(runtime_config)),
        "model_parameters": count_parameters(model),
        "manifest_hash": parent_result.get("manifest_hash", ""),
        "validation_manifest_hash": validation_hash,
        "promotion_parent_validation_manifest_hash": validation_hash,
        "promotion_parent_run_dir": str(run_dir / "promotion_parent"),
        "epochs_completed": 0,
        "best_ssim_checkpoint": str(checkpoint),
        "checkpoint": str(checkpoint),
        "stage": "d4_tta_evaluation",
        "loaded_weight_source": checkpoint_payload["loaded_weight_source"],
    }


def command_run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    suite_path = Path(args.suite).expanduser().resolve()
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    if not isinstance(suite, Mapping):
        raise TypeError("Ablation suite must be a YAML mapping")
    order = _ablation_stage_slice(
        [str(value) for value in suite.get("order", ())],
        args.from_stage,
        args.through_stage,
    )
    stages = suite.get("stages", {})
    if not order or not isinstance(stages, Mapping):
        raise ValueError("Ablation suite has no runnable stages")
    registry_path = Path(args.registry).expanduser().resolve()
    registry = ExperimentRegistry(registry_path)
    reports: dict[str, Any] = {}
    suite_name = str(suite.get("suite", "ablation"))
    completed_results: dict[str, dict[str, Mapping[str, Any]]] = {}
    blocked_parent = False
    planned_evidence_ids: tuple[str, ...] | None = None
    report_output: Path | None = None
    for stage_name in order:
        stage_spec = stages.get(stage_name)
        if not isinstance(stage_spec, Mapping):
            raise ValueError(f"Missing stage specification for {stage_name}")
        config_path = str(stage_spec.get("config", ""))
        base_config = load_config(config_path, args.set, include_resolved=True)
        base_config["project"]["output_root"] = "outputs/performance_v2"
        base_config["project"]["artifact_root"] = "artifacts/performance_v2"
        default_seed = int(base_config["project"].get("seed", 2026))
        default_fold_value = base_config["project"].get("fold", 0)
        default_fold = int(default_fold_value) if str(default_fold_value).isdigit() else 0
        plan = resolve_ablation_budget(
            base_config.get("budget", {}),
            args.budget,
            default_seed=default_seed,
            default_fold=default_fold,
        )
        stage_evidence_ids = tuple(
            evidence.evidence_id for evidence in plan.evidence_runs
        )
        if planned_evidence_ids is None:
            planned_evidence_ids = stage_evidence_ids
            evidence_label = (
                stage_evidence_ids[0]
                if len(stage_evidence_ids) == 1
                else f"{len(stage_evidence_ids)}evidence_"
                f"{config_hash({'evidence_ids': stage_evidence_ids})}"
            )
            report_output = registry_path.parent / (
                f"{registry_path.stem}_{suite_name}_{args.budget}_"
                f"{evidence_label}_report.json"
            )
            if report_output.exists():
                raise FileExistsError(
                    "Ablation report already exists and cannot be overwritten: "
                    f"{report_output}"
                )
        elif stage_evidence_ids != planned_evidence_ids:
            raise ValueError(
                "All stages in one ablation invocation must resolve the same evidence grid"
            )
        audit = _context_audit_for_config(base_config, args.data_root)
        gate_failures = _roi_context_audit_failures(audit)
        declared_parent = str(stage_spec.get("parent", "")).strip()
        evidence_configs = {
            evidence.evidence_id: _configure_ablation_evidence(
                base_config,
                plan=plan,
                evidence=evidence,
                audit=audit,
                max_epochs=args.max_epochs,
                max_train_samples=args.max_train_samples,
                max_val_samples=args.max_val_samples,
            )
            for evidence in plan.evidence_runs
        }

        stage_block_reasons: list[str] = []
        stage_block_status = "blocked_unverified_grid"
        if blocked_parent:
            stage_block_reasons.append("parent_context_or_promotion_stage_was_blocked")
        if plan.budget == "confirm" and not _roi_grouped_budget_ready(audit):
            stage_block_reasons.append("confirm_requires_verified_grouped_roi_grid")
        if bool(stage_spec.get("requires_verified_grid", False)) and gate_failures:
            stage_block_reasons.extend(gate_failures)
        if stage_name == "A8" and not plan.uses_official_outer_validation:
            stage_block_status = "blocked_requires_oof_ensemble"
            stage_block_reasons.append(
                "A8_requires_validation_or_oof_safe_ensemble_members"
            )
        if stage_name == "A8" and plan.evidence_count < 2:
            stage_block_status = "blocked_insufficient_ensemble_members"
            stage_block_reasons.append("A8_requires_at_least_two_independent_members")
        if stage_name == "A8" and not stage_block_reasons:
            stage_block_status = "blocked_requires_explicit_validation_ensemble"
            stage_block_reasons.append(
                "A8_must_use_build_model_soup_or_optimize_ensemble_with_explicit_val_oof_inputs"
            )

        if stage_block_reasons:
            unique_reasons = list(dict.fromkeys(stage_block_reasons))
            blocked_evidence: list[dict[str, Any]] = []
            for evidence in plan.evidence_runs:
                config = evidence_configs[evidence.evidence_id]
                run_id = _ablation_evidence_run_id(
                    suite_name,
                    stage_name,
                    plan,
                    evidence,
                )
                parent_run = _declared_ablation_parent_evidence_run(
                    suite_name,
                    stage_spec,
                    plan,
                    evidence,
                )
                record = _v2_registry_record(
                    config,
                    {"run_id": run_id},
                    parent_run=parent_run,
                    status=stage_block_status,
                    failure_reason=",".join(unique_reasons),
                )
                record["run_id"] = run_id
                registry.upsert(record)
                blocked_evidence.append(
                    {
                        "run_id": run_id,
                        "parent_run": parent_run,
                        "evidence": evidence.to_dict(),
                    }
                )
            reports[stage_name] = {
                "status": stage_block_status,
                "reasons": unique_reasons,
                "budget_plan": plan.to_dict(),
                "evidence_runs": blocked_evidence,
            }
            blocked_parent = True
            continue

        stage_results: dict[str, Mapping[str, Any]] = {}
        parent_results = completed_results.get(declared_parent, {})
        parent_runs: dict[str, str] = {}
        for evidence in plan.evidence_runs:
            config = evidence_configs[evidence.evidence_id]
            if bool(stage_spec.get("requires_verified_grid", False)):
                config["model"]["context"]["enabled"] = True
            run_id = _ablation_evidence_run_id(
                suite_name,
                stage_name,
                plan,
                evidence,
            )
            parent_result = parent_results.get(evidence.evidence_id)
            parent_run = (
                Path(str(parent_result.get("run_dir", ""))).name
                if parent_result is not None
                else _declared_ablation_parent_evidence_run(
                    suite_name,
                    stage_spec,
                    plan,
                    evidence,
                )
            )
            if stage_name == "A7":
                if parent_result is None:
                    raise ValueError("A7 requires the corresponding A6 evidence run")
                result = _run_ablation_tta_evidence(
                    config,
                    args.data_root,
                    run_id,
                    parent_result,
                )
            else:
                result = _run_ablation_training_evidence(
                    stage_name,
                    config,
                    args.data_root,
                    run_id,
                    parent_result,
                )
            stage_results[evidence.evidence_id] = result
            parent_runs[evidence.evidence_id] = parent_run

        promotion = (
            _promotion_across_evidence(
                parent_results,
                stage_results,
                evidence_runs=plan.evidence_runs,
                roi_audit=audit,
                config=base_config,
            )
            if parent_results
            else None
        )
        if promotion is not None:
            for result in stage_results.values():
                write_promotion_report(
                    promotion,
                    Path(str(result["run_dir"]))
                    / "artifacts"
                    / "promotion_vs_parent.json",
                )
        confirmation_only_reason = "insufficient_independent_fold_seed_evidence"
        promotion_blockers = (
            [
                reason
                for reason in promotion["reasons"]
                if not (
                    reason == confirmation_only_reason
                    and plan.budget in {"smoke", "screen"}
                )
            ]
            if promotion is not None
            else []
        )
        promotable = not gate_failures and not promotion_blockers
        status = "completed" if promotable else "not_promotable"
        failure_reasons = list(gate_failures)
        failure_reasons.extend(
            reason for reason in promotion_blockers if reason not in failure_reasons
        )
        evidence_reports: list[dict[str, Any]] = []
        for evidence in plan.evidence_runs:
            result = stage_results[evidence.evidence_id]
            config = evidence_configs[evidence.evidence_id]
            parent_run = parent_runs[evidence.evidence_id]
            registry.upsert(
                _v2_registry_record(
                    config,
                    result,
                    parent_run=parent_run,
                    status=status,
                    failure_reason="" if promotable else ",".join(failure_reasons),
                )
            )
            evidence_reports.append(
                {
                    **result,
                    "parent_run": parent_run,
                    "evidence": evidence.to_dict(),
                }
            )
        stage_report: dict[str, Any] = {
            "status": status,
            "failure_reasons": failure_reasons,
            "budget_plan": plan.to_dict(),
            "evidence_runs": evidence_reports,
            "screen_promotion": promotion,
            "final_confirmation_required": bool(
                promotion is not None
                and confirmation_only_reason in promotion["reasons"]
            ),
        }
        if len(evidence_reports) == 1:
            stage_report.update(evidence_reports[0])
        reports[stage_name] = stage_report
        completed_results[stage_name] = stage_results
        if stage_name in {"A3", "A4", "A5", "A6", "A7"} and not promotable:
            blocked_parent = True
    result = {
        "suite": str(suite_path),
        "budget": args.budget,
        "stages": reports,
        "registry": str(registry_path),
        "report": str(report_output) if report_output is not None else None,
    }
    if report_output is None:
        raise RuntimeError("Ablation did not resolve an output report")
    _write_json(report_output, result)
    return result


def command_compare_runs(args: argparse.Namespace) -> dict[str, Any]:
    registry = ExperimentRegistry(args.registry)
    rows = registry.read()
    if args.stages:
        wanted = {str(stage) for stage in args.stages}
        rows = [
            row
            for row in rows
            if any(f"_{stage}_" in str(row.get("run_id", "")) for stage in wanted)
        ]
    comparison = compare_experiments(rows, primary_metric=args.metric)
    if args.output:
        _write_json(args.output, comparison)
    return comparison


def _model_soup_output_path(
    requested: str | Path,
    *,
    unsafe_lineage_override: bool,
    unsafe_engineering_validation: bool = False,
) -> Path:
    output = Path(requested).expanduser().resolve()
    markers: list[str] = []
    if unsafe_lineage_override:
        markers.append("UNSAFE_LINEAGE_OVERRIDE")
    if unsafe_engineering_validation:
        markers.append("UNSAFE_ENGINEERING_VALIDATION")
    missing = [marker for marker in markers if marker.casefold() not in output.stem.casefold()]
    if missing:
        output = output.with_name(
            f"{output.stem}_{'_'.join(missing)}{output.suffix or '.ckpt'}"
        )
    return output


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_soup_jpg_score(
    evaluation: Mapping[str, Any],
    *,
    target_marker: str,
    metric: str,
) -> float:
    domains = evaluation.get("domains", {})
    jpg = domains.get("jpg", {}) if isinstance(domains, Mapping) else {}
    per_target = jpg.get("per_target", {}) if isinstance(jpg, Mapping) else {}
    target_values: Mapping[str, Any] | None = None
    if isinstance(per_target, Mapping):
        for name, values in per_target.items():
            try:
                matches = normalize_marker(str(name)) == target_marker
            except ValueError:
                matches = False
            if matches and isinstance(values, Mapping):
                target_values = values
                break
    if target_values is None:
        raise ValueError(
            f"JPG validation has no per-target metrics for {target_marker}"
        )
    value = float(target_values.get(metric, math.nan))
    if not math.isfinite(value):
        raise ValueError(
            f"JPG validation metric {metric!r} is missing or non-finite for "
            f"{target_marker}"
        )
    return value


def _model_soup_target_record_count(
    records: Sequence[Mapping[str, Any]], target_marker: str
) -> int:
    count = 0
    for record in records:
        try:
            matches = normalize_marker(str(record.get("target", ""))) == target_marker
        except ValueError:
            matches = False
        count += int(matches)
    return count


def command_build_model_soup(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_effective(args)
    _apply_hardware_defaults(config)
    discovery = _resolve_discovery(args.data_root, config=config)
    manifests = _ensure_manifests(discovery, config)
    scores = args.validation_scores or [0.0] * len(args.checkpoints)
    if len(scores) != len(args.checkpoints):
        raise ValueError("Validation score count must match checkpoint count")
    allow_unsafe_lineage_mismatch = bool(
        getattr(args, "allow_unsafe_lineage_mismatch", False)
        or config.get("ensemble", {}).get(
            "allow_unsafe_model_soup_lineage", False
        )
    )
    allow_unsafe_engineering_validation = bool(
        getattr(args, "allow_unsafe_engineering_validation", False)
        or config.get("ensemble", {}).get(
            "allow_unsafe_model_soup_validation", False
        )
    )
    output = _model_soup_output_path(
        args.output,
        unsafe_lineage_override=allow_unsafe_lineage_mismatch,
        unsafe_engineering_validation=allow_unsafe_engineering_validation,
    )
    evidence_dir = output.parent / f"{output.stem}_validation_evidence"
    train_rows = read_manifest(manifests.train_manifest)
    val_rows = read_manifest(manifests.val_manifest)
    roi_audit = write_roi_grid_audit(
        [*train_rows, *val_rows],
        discovery.selected_root,
        evidence_dir / "roi_audit",
        mosaic_count=0,
    )
    early_failures = list(_roi_context_audit_failures(roi_audit))
    configured_limit = config.get("data", {}).get("max_val_samples")
    if configured_limit is not None:
        early_failures.append("validation_manifest_was_configured_with_a_sample_limit")
    configured_domain = str(
        config.get("validation", {}).get("primary_domain", "float")
    ).casefold()
    if configured_domain not in {"jpg", "jpeg", "jpg_roundtrip"}:
        early_failures.append("validation.primary_domain_is_not_jpg")
    if early_failures and not allow_unsafe_engineering_validation:
        raise ValueError(
            "Strict model soup requires full authoritative ROI-grouped JPG validation: "
            + ", ".join(dict.fromkeys(early_failures))
        )
    selected_targets = _selected_targets(config)
    requested_target = getattr(args, "target_marker", None)
    if requested_target is None:
        if len(selected_targets) != 1:
            raise ValueError(
                "--target-marker is required when the config contains multiple targets"
            )
        target_marker = selected_targets[0]
    else:
        target_marker = normalize_marker(str(requested_target))
        if target_marker not in selected_targets:
            raise ValueError("--target-marker must be present in data.targets")
    members: list[SoupMember] = []
    member_provenance: list[dict[str, Any]] = []
    template_payload: Mapping[str, Any] | None = None
    for index, checkpoint_value in enumerate(args.checkpoints):
        checkpoint_path = Path(checkpoint_value).expanduser().resolve()
        checkpoint_payload = load_checkpoint(checkpoint_path, map_location="cpu")
        provenance = extract_checkpoint_provenance(
            checkpoint_payload,
            checkpoint_path,
            weight_source=args.weight_source,
            require_complete=not allow_unsafe_lineage_mismatch,
        )
        member = SoupMember(
            name=checkpoint_path.stem + f"_{index}",
            state_dict=load_checkpoint_state(
                checkpoint_path, weight_source=args.weight_source
            ),
            validation_score=float(scores[index]),
            weight_source=args.weight_source,
            lineage=provenance.initialization_lineage,
            checkpoint_sha256=provenance.checkpoint_sha256,
            parent_checkpoint_sha256=provenance.parent_checkpoint_sha256,
            pretrain_checkpoint_sha256=provenance.pretrain_checkpoint_sha256,
            checkpoint_path=provenance.checkpoint_path,
            architecture_sha256=provenance.architecture_sha256,
        )
        members.append(member)
        member_provenance.append(
            {
                "member_name": member.name,
                **provenance.to_dict(),
            }
        )
        if template_payload is None:
            template_payload = checkpoint_payload
    checkpoint_hashes = [str(member.checkpoint_sha256) for member in members]
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("Model soup input checkpoints must be unique")
    if template_payload is None:
        raise ValueError("At least one checkpoint is required")
    model, template_payload = _load_model_for_evaluation(config, args.checkpoints[0])
    runtime_config = _checkpoint_runtime_config(config, template_payload)
    loader = _make_loader(
        manifests.val_manifest,
        discovery.selected_root,
        runtime_config,
        train=False,
    )
    loader_sample_count = len(loader.dataset)
    if loader_sample_count != len(val_rows) and not allow_unsafe_engineering_validation:
        raise ValueError(
            "Strict model soup must evaluate every validation manifest row: "
            f"loader={loader_sample_count}, manifest={len(val_rows)}"
        )
    validation_calls: list[dict[str, Any]] = []
    validation_counter = 0

    def validate_state(
        state: Mapping[str, torch.Tensor], *, label: str | None = None
    ) -> float:
        nonlocal validation_counter
        validation_counter += 1
        model.load_state_dict(dict(state), strict=True)
        resolved_label = label or f"greedy_{validation_counter:03d}"
        records_path = evidence_dir / "trials" / f"{resolved_label}.csv"
        evaluation = Validator(
            model,
            loader,
            config=runtime_config,
            use_ema=False,
            image_spec=template_payload.get("image_spec"),
        ).evaluate(records_path=records_path)
        score = _model_soup_jpg_score(
            evaluation,
            target_marker=target_marker,
            metric=args.metric,
        )
        validation_calls.append(
            {
                "label": resolved_label,
                "score": score,
                "records_path": str(records_path.resolve()),
                "records_sha256": _sha256_file(records_path),
                "target_record_count": _model_soup_target_record_count(
                    evaluation.get("records", []), target_marker
                ),
            }
        )
        return score

    # User-supplied scores are never trusted for ordering.  Every member is first
    # re-evaluated on the same full JPG protocol used by the greedy combinations.
    members = [
        replace(
            member,
            validation_score=validate_state(
                member.state_dict,
                label=f"individual_{index:03d}_{member.name}",
            ),
        )
        for index, member in enumerate(members)
    ]

    result = greedy_model_soup(
        members,
        validate_state,
        min_improvement=float(args.min_improvement),
        require_matching_lineage=True,
        allow_unsafe_lineage_mismatch=allow_unsafe_lineage_mismatch,
    )
    model.load_state_dict(result.state_dict, strict=True)
    final_records_path = evidence_dir / "final" / "per_image.csv"
    final_evaluation = Validator(
        model,
        loader,
        config=runtime_config,
        use_ema=False,
        image_spec=template_payload.get("image_spec"),
    ).evaluate(records_path=final_records_path)
    final_score = _model_soup_jpg_score(
        final_evaluation,
        target_marker=target_marker,
        metric=args.metric,
    )
    if not math.isclose(
        final_score, result.validation_score, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            "Final soup JPG revalidation did not reproduce the greedy selection score"
        )
    validation_contract = build_soup_validation_contract(
        manifests.val_manifest,
        roi_audit["audit_path"],
        final_records_path,
        target_marker=target_marker,
        evaluated_sample_count=_model_soup_target_record_count(
            final_evaluation.get("records", []), target_marker
        ),
        metric_domain="jpg_roundtrip",
        primary_metric=args.metric,
        max_val_samples=config.get("data", {}).get("max_val_samples"),
        audited_manifest_paths=(manifests.train_manifest, manifests.val_manifest),
        allow_unsafe_engineering_override=allow_unsafe_engineering_validation,
    )
    from virtual_staining.engine.checkpoint import save_checkpoint

    accepted_names = set(result.member_names)
    for record in member_provenance:
        record["accepted"] = record["member_name"] in accepted_names
    soup_provenance = {
        "version": 2,
        "safety_mode": (
            "UNSAFE_ENGINEERING_OVERRIDE"
            if allow_unsafe_lineage_mismatch
            or allow_unsafe_engineering_validation
            else "strict"
        ),
        "unsafe_lineage_override_used": result.unsafe_lineage_override_used,
        "unsafe_validation_override_used": False,
        "unsafe_engineering_override_used": False,
        "architecture_sha256": result.architecture_sha256,
        "common_initialization_lineage": result.common_initialization_lineage,
        "common_parent_checkpoint_sha256": (
            result.common_parent_checkpoint_sha256
        ),
        "common_pretrain_checkpoint_sha256": (
            result.common_pretrain_checkpoint_sha256
        ),
        "weight_source": result.weight_source,
        "input_checkpoints": member_provenance,
        "accepted_members": list(result.member_names),
    }
    soup_provenance = bind_soup_validation_contract(
        soup_provenance, validation_contract
    )
    save_checkpoint(
        output,
        model,
        config=runtime_config,
        manifest_hash=manifest_sha1(manifests.val_manifest),
        image_spec=template_payload.get("image_spec"),
        targets=list(_selected_targets(config)),
        extra={
            "soup_members": list(result.member_names),
            "validation_score": final_score,
            "weight_source": result.weight_source,
            "model_soup_provenance": soup_provenance,
        },
    )
    report = {
        "checkpoint": str(output),
        "checkpoint_sha256": checkpoint_file_sha256(output),
        "members": list(result.member_names),
        "validation_score": final_score,
        "model_soup_provenance": soup_provenance,
        "validation_contract": validation_contract.to_dict(),
        "validation_calls": validation_calls,
        "provided_validation_scores_ignored_for_selection": list(scores),
        "trials": [
            {
                "member_name": trial.member_name,
                "validation_score": trial.validation_score,
                "accepted": trial.accepted,
                "member_count": trial.member_count,
            }
            for trial in result.trials
        ],
    }
    report_path = (
        _model_soup_output_path(
            args.report,
            unsafe_lineage_override=allow_unsafe_lineage_mismatch,
            unsafe_engineering_validation=allow_unsafe_engineering_validation,
        )
        if getattr(args, "report", None)
        else output.with_suffix(".report.json")
    )
    _write_json(report_path, report)
    report["report"] = str(report_path)
    return report


def command_optimize_ensemble(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, getattr(args, "set", None))
    configured_cross_validation = bool(
        config.get("ensemble", {}).get("cross_validate_weights", True)
    )
    requested_cross_validation = getattr(args, "cross_validate_weights", None)
    cross_validate_weights = (
        configured_cross_validation
        if requested_cross_validation is None
        else bool(requested_cross_validation)
    )
    predictions, target, provenance = load_verified_ensemble_inputs(
        args.predictions,
        args.target_array,
        prediction_sidecars=getattr(args, "prediction_sidecars", None),
        target_sidecar=getattr(args, "target_sidecar", None),
        expected_source=args.source,
        manifest_path=args.manifest,
        target_marker=args.target_marker,
        metric_domain=args.metric_domain,
        fold_assignment_path=getattr(args, "fold_assignment", None),
        roi_audit_path=args.roi_audit,
        audited_manifest_paths=args.audited_manifests,
        allow_unsafe_engineering_manifest=bool(
            getattr(args, "allow_unsafe_engineering_manifest", False)
        ),
    )
    optimizer_name = str(config.get("ensemble", {}).get("optimizer", "coordinate"))
    optimizer_failure_policy = str(
        config.get("ensemble", {}).get("optimizer_failure_policy", "uniform")
    )
    result = optimize_ensemble_weights(
        predictions,
        target,
        source=args.source,
        provenance=provenance,
        cross_validate_weights=cross_validate_weights,
        optimizer=optimizer_name,
        optimizer_failure_policy=optimizer_failure_policy,
        min_gain_over_uniform=float(args.min_gain_over_uniform),
        uniform_shrinkage=float(args.uniform_shrinkage),
    )
    payload = {
        "weights": list(result.weights),
        "score": result.score,
        "uniform_score": result.uniform_score,
        "source": result.source,
        "evaluations": result.evaluations,
        "used_learned_weights": result.used_learned_weights,
        "optimizer": result.optimizer,
        "optimizer_failure_policy": optimizer_failure_policy,
        "fallback_reason": result.fallback_reason,
        "cross_validate_weights": cross_validate_weights,
        "cross_validated": result.cross_validated,
        "cross_validated_score": result.cross_validated_score,
        "cross_validated_uniform_score": result.cross_validated_uniform_score,
        "cross_validation_folds": [
            {
                "fold_id": fold.fold_id,
                "train_samples": fold.train_samples,
                "held_out_samples": fold.held_out_samples,
                "held_out_groups": fold.held_out_groups,
                "weights": list(fold.weights),
                "score": fold.score,
                "uniform_score": fold.uniform_score,
                "evaluations": fold.evaluations,
                "optimizer": fold.optimizer,
                "fallback_reason": fold.fallback_reason,
            }
            for fold in result.fold_results
        ],
        "provenance": provenance.summary(),
        "prediction_files": [str(Path(path).resolve()) for path in args.predictions],
        "target_array": str(Path(args.target_array).resolve()),
    }
    output = _ensemble_output_path(
        args.output,
        unsafe_engineering_manifest=bool(
            getattr(args, "allow_unsafe_engineering_manifest", False)
        ),
    )
    payload["output"] = str(output)
    _write_json(output, payload)
    return payload


def _ensemble_output_path(
    requested: str | Path,
    *,
    unsafe_engineering_manifest: bool,
) -> Path:
    output = Path(requested).expanduser().resolve()
    if (
        unsafe_engineering_manifest
        and "unsafe_engineering_manifest" not in output.stem.casefold()
    ):
        output = output.with_name(
            f"{output.stem}_UNSAFE_ENGINEERING_MANIFEST{output.suffix or '.json'}"
        )
    return output


def command_predict_v2(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_effective(args)
    if str(config["model"].get("name", "")).casefold() != "camp_vs_v2":
        metadata = load_checkpoint(args.checkpoint, map_location="cpu")
        saved = metadata.get("config", {}).get("model", {})
        if str(saved.get("name", "")).casefold() != "camp_vs_v2":
            raise ValueError("predict-v2 requires a CAMP-VS v2 checkpoint")
    return command_predict(args)


def _add_config_arguments(parser: argparse.ArgumentParser, *, target: bool = False) -> None:
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--set", action="append", default=[])
    if target:
        parser.add_argument("--target", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="virtual_staining", description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-root",
        default="log",
        help="Root directory for lightweight downloadable experiment logs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("env", help="Report Python, PyTorch, CUDA, GPU, CPU, and RAM")
    env_parser.add_argument("--output", default="artifacts/environment.json")
    env_parser.set_defaults(handler=command_env)

    discover = subparsers.add_parser("discover-data", help="Discover and score bounded data roots")
    discover.add_argument("--data-root", default="AUTO")
    discover.add_argument("--config", default="configs/default.yaml")
    discover.add_argument("--output", default="artifacts/data_discovery.json")
    discover.set_defaults(handler=command_discover)

    manifest = subparsers.add_parser("build-manifest", help="Build leakage-aware train/val/test manifests")
    _add_config_arguments(manifest)
    manifest.set_defaults(handler=command_manifest)

    audit = subparsers.add_parser("audit-data", help="Run streaming data statistics and alignment audit")
    audit.add_argument("--data-root", default="AUTO")
    audit.add_argument("--correlation-samples", type=int, default=64)
    audit.add_argument("--alignment-samples", type=int, default=32)
    audit.add_argument("--figure-count", type=int, default=16)
    audit.set_defaults(handler=command_audit)

    train = subparsers.add_parser("train", help="Train a model from scratch")
    _add_config_arguments(train, target=True)
    train.add_argument("--run-id", default="train_seed2026")
    train.add_argument("--max-epochs", type=int, default=None)
    train.set_defaults(handler=command_train)

    resume = subparsers.add_parser("resume", help="Resume all training state from a checkpoint")
    _add_config_arguments(resume, target=True)
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--max-epochs", type=int, default=None)
    resume.set_defaults(handler=command_resume)

    validate = subparsers.add_parser("validate", help="Calculate reference SSIM/PSNR on validation data")
    _add_config_arguments(validate, target=True)
    validate.add_argument("--checkpoint", required=True)
    validate.add_argument("--max-epochs", type=int, default=None)
    validate.set_defaults(handler=command_validate)

    predict = subparsers.add_parser("predict", help="Run deterministic inference")
    _add_config_arguments(predict, target=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--manifest", default=None)
    predict.add_argument("--output-dir", default=None)
    predict.add_argument("--max-epochs", type=int, default=None)
    predict.set_defaults(handler=command_predict)

    ensemble = subparsers.add_parser("ensemble", help="Average several compatible checkpoints")
    _add_config_arguments(ensemble, target=True)
    ensemble.add_argument("--checkpoints", nargs="+", required=True)
    ensemble.add_argument("--weights", nargs="+", type=float, default=None)
    ensemble.add_argument("--validation-scores", nargs="+", type=float, default=None)
    ensemble.add_argument("--weight-temperature", type=float, default=0.02)
    ensemble.add_argument("--manifest", default=None)
    ensemble.add_argument("--output-dir", default=None)
    ensemble.add_argument("--max-epochs", type=int, default=None)
    ensemble.set_defaults(handler=command_ensemble)

    make = subparsers.add_parser("make-submission", help="Build results hierarchy and ZIP")
    make.add_argument("--config", default="configs/infer.yaml")
    make.add_argument("--set", action="append", default=[])
    make.add_argument("--pred-dir", required=True)
    make.add_argument("--test-manifest", required=True)
    make.add_argument("--target", required=True)
    make.add_argument("--output-dir", default=".")
    make.add_argument("--max-epochs", type=int, default=None)
    make.add_argument("--submit-target", action="store_true")
    make.set_defaults(handler=command_make_submission)

    check = subparsers.add_parser("validate-submission", help="Strictly validate results and optional ZIP")
    check.add_argument("--submission-dir", required=True)
    check.add_argument("--test-manifest", required=True)
    check.add_argument("--target", required=True)
    check.add_argument("--zip-path", default=None)
    check.add_argument("--artifact-dir", default="artifacts")
    check.add_argument("--expected-mode", choices=("auto", "L", "RGB"), default="auto")
    check.set_defaults(handler=command_validate_submission)

    pipeline = subparsers.add_parser("run-pipeline", help="Run the complete audited smoke workflow")
    _add_config_arguments(pipeline)
    pipeline.add_argument("--run-id", default="smoke_pipeline_acceptance")
    pipeline.set_defaults(handler=command_pipeline, config="configs/smoke.yaml")

    autodl_run = subparsers.add_parser(
        "autodl-run",
        help="One-command training + validation + log for AutoDL CUDA GPUs",
    )
    _add_config_arguments(autodl_run, target=True)
    autodl_run.add_argument("--run-id", default="autodl_run_seed2026")
    autodl_run.add_argument("--max-epochs", type=int, default=None)
    autodl_run.set_defaults(
        handler=command_autodl_run,
        config="configs/initial_round_cd68_retrain_v2.yaml",
    )

    autodl_submit = subparsers.add_parser(
        "autodl-submit",
        help="One-command predict + make-submission + validate-submission -> ZIP",
    )
    _add_config_arguments(autodl_submit, target=True)
    autodl_submit.add_argument("--checkpoint", default=None)
    autodl_submit.set_defaults(
        handler=command_autodl_submit,
        config="configs/initial_round_cd68_retrain_v2.yaml",
    )

    autodl_ensemble_submit = subparsers.add_parser(
        "autodl-ensemble-submit",
        help="D4-average multiple checkpoints and build one validated official ZIP",
    )
    _add_config_arguments(autodl_ensemble_submit, target=True)
    autodl_ensemble_submit.add_argument("--checkpoints", nargs="+", required=True)
    autodl_ensemble_submit.add_argument("--weights", nargs="+", type=float, default=None)
    autodl_ensemble_submit.add_argument(
        "--validation-scores", nargs="+", type=float, default=None
    )
    autodl_ensemble_submit.add_argument("--weight-temperature", type=float, default=0.02)
    autodl_ensemble_submit.add_argument(
        "--output-dir",
        default="outputs/initial_round_max_v3/ensemble_seed2026_seed3407",
    )
    autodl_ensemble_submit.set_defaults(
        handler=command_autodl_ensemble_submit,
        config="configs/initial_round_cd68_max_v3.yaml",
    )

    benchmark = subparsers.add_parser(
        "benchmark-baseline", help="Freeze and re-evaluate raw/EMA baseline weights in three domains"
    )
    _add_config_arguments(benchmark, target=True)
    benchmark.add_argument("--checkpoint", default=None)
    benchmark.add_argument("--evaluate-weights", nargs="+", default=["raw", "ema"])
    benchmark.add_argument("--tta", nargs="+", default=["none", "d4"])
    benchmark.add_argument("--output-dir", default="artifacts/performance_v2")
    benchmark.add_argument("--max-samples", type=int, default=None)
    benchmark.add_argument("--max-epochs", type=int, default=None)
    benchmark.set_defaults(handler=command_benchmark_baseline)

    roi_audit = subparsers.add_parser(
        "audit-roi-grid", help="Audit authoritative ROI_row_col grids and boundary directions"
    )
    _add_config_arguments(roi_audit)
    roi_audit.add_argument("--manifest", nargs="+", default=None)
    roi_audit.add_argument("--output-dir", default="artifacts/performance_v2")
    roi_audit.add_argument("--mosaic-count", type=int, default=8)
    roi_audit.add_argument("--border-width", type=int, default=8)
    roi_audit.set_defaults(handler=command_audit_roi_grid)

    pretrain = subparsers.add_parser(
        "pretrain-dapi", help="Run optional fold-local DAPI-only masked reconstruction"
    )
    _add_config_arguments(pretrain)
    pretrain.add_argument("--run-id", default="dapi_mae_seed2026")
    pretrain.add_argument("--max-epochs", type=int, default=None)
    pretrain.set_defaults(
        handler=command_pretrain_dapi,
        config="configs/performance_v2/dapi_pretrain.yaml",
    )

    train_v2 = subparsers.add_parser("train-v2", help="Train CAMP-VS v2 from scratch")
    _add_config_arguments(train_v2, target=True)
    train_v2.add_argument("--run-id", default="camp_v2_seed2026")
    train_v2.add_argument(
        "--pretrain-checkpoint",
        default=None,
        help="Optional fold-matched DAPI-MAE checkpoint; only local_encoder is transferred",
    )
    train_v2.add_argument("--max-epochs", type=int, default=None)
    train_v2.set_defaults(
        handler=command_train_v2,
        config="configs/performance_v2/camp_smoke.yaml",
    )

    for name, handler, default_config in (
        (
            "finetune-target",
            command_finetune_target,
            "configs/performance_v2/camp_target_finetune.yaml",
        ),
        (
            "finetune-organ",
            command_finetune_organ,
            "configs/performance_v2/camp_organ_finetune.yaml",
        ),
        (
            "finetune-metric",
            command_finetune_metric,
            "configs/performance_v2/camp_metric_finetune.yaml",
        ),
    ):
        fine = subparsers.add_parser(name, help=f"Run the {name.removeprefix('finetune-')} stage")
        _add_config_arguments(fine, target=True)
        fine.add_argument("--checkpoint", required=True)
        fine.add_argument("--run-id", default=name.replace("-", "_") + "_seed2026")
        fine.add_argument("--max-epochs", type=int, default=None)
        fine.set_defaults(handler=handler, config=default_config)

    ablation = subparsers.add_parser("run-ablation", help="Run an ordered gated ablation suite")
    ablation.add_argument("--suite", required=True)
    ablation.add_argument("--budget", choices=("smoke", "screen", "confirm", "full"), default="screen")
    ablation.add_argument("--from-stage", default=None)
    ablation.add_argument("--through-stage", default=None)
    ablation.add_argument("--data-root", default=None)
    ablation.add_argument("--set", action="append", default=[])
    ablation.add_argument("--max-epochs", type=int, default=None)
    ablation.add_argument("--max-train-samples", type=int, default=None)
    ablation.add_argument("--max-val-samples", type=int, default=None)
    ablation.add_argument(
        "--registry", default="artifacts/performance_v2/experiment_registry.csv"
    )
    ablation.set_defaults(handler=command_run_ablation)

    compare = subparsers.add_parser("compare-runs", help="Rank registered v2 experiments")
    compare.add_argument("--registry", required=True)
    compare.add_argument("--stages", nargs="+", default=None)
    compare.add_argument("--metric", default="jpg_ssim")
    compare.add_argument("--output", default=None)
    compare.set_defaults(handler=command_compare_runs)

    soup = subparsers.add_parser(
        "build-model-soup", help="Build an architecture-compatible validation-gated greedy soup"
    )
    _add_config_arguments(soup)
    soup.add_argument("--checkpoints", nargs="+", required=True)
    soup.add_argument("--validation-scores", nargs="+", type=float, default=None)
    soup.add_argument("--weight-source", choices=("raw", "ema", "swa"), default="raw")
    soup.add_argument(
        "--target-marker",
        default=None,
        help="Marker-specific JPG validation target; required for multi-target configs",
    )
    soup.add_argument(
        "--allow-unsafe-lineage-mismatch",
        action="store_true",
        help=(
            "UNSAFE: allow missing or mismatched initialization provenance; "
            "the output is forcibly marked UNSAFE_LINEAGE_OVERRIDE"
        ),
    )
    soup.add_argument(
        "--allow-unsafe-engineering-validation",
        action="store_true",
        help=(
            "UNSAFE engineering smoke only: bypass full authoritative ROI/JPG "
            "validation; checkpoint, report, and evidence paths are permanently marked"
        ),
    )
    soup.add_argument("--metric", default="local_proxy_score")
    soup.add_argument("--min-improvement", type=float, default=0.0)
    soup.add_argument("--output", default="outputs/performance_v2/model_soup.ckpt")
    soup.add_argument("--report", default=None)
    soup.set_defaults(handler=command_build_model_soup)

    optimize = subparsers.add_parser(
        "optimize-ensemble", help="Fit global nonnegative weights from validation/OOF arrays"
    )
    optimize.add_argument("--config", default="configs/performance_v2/ensemble.yaml")
    optimize.add_argument("--set", action="append", default=[])
    optimize.add_argument("--predictions", nargs="+", required=True)
    optimize.add_argument(
        "--prediction-sidecars",
        nargs="+",
        default=None,
        help="Optional explicit sidecars; otherwise <array>.meta.json is required",
    )
    optimize.add_argument("--target-array", required=True)
    optimize.add_argument(
        "--target-sidecar",
        default=None,
        help="Optional explicit sidecar; otherwise <target array>.meta.json is required",
    )
    optimize.add_argument("--source", required=True, choices=("validation", "val", "oof"))
    optimize.add_argument(
        "--manifest",
        required=True,
        help="Actual full validation/train manifest used as the external array anchor",
    )
    optimize.add_argument(
        "--target-marker",
        required=True,
        help="Target marker represented by every prediction and target array",
    )
    optimize.add_argument(
        "--metric-domain",
        default="jpg_roundtrip",
        choices=("jpg_roundtrip",),
    )
    optimize.add_argument(
        "--fold-assignment",
        default=None,
        help="Required authoritative ROI-grouped assignment CSV for source=oof",
    )
    optimize.add_argument(
        "--roi-audit",
        required=True,
        help="ROI grid/direction/boundary/leakage audit JSON for the bound manifests",
    )
    optimize.add_argument(
        "--audited-manifests",
        nargs=2,
        required=True,
        metavar=("TRAIN_MANIFEST", "VAL_MANIFEST"),
        help="Exact train and validation manifests covered by the ROI audit",
    )
    optimize.add_argument(
        "--allow-unsafe-engineering-manifest",
        action="store_true",
        help=(
            "UNSAFE engineering smoke only: accept a non-authoritative non-test "
            "manifest and permanently mark the output"
        ),
    )
    optimize.add_argument(
        "--cross-validate-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override ensemble.cross_validate_weights from config",
    )
    optimize.add_argument("--min-gain-over-uniform", type=float, default=0.0)
    optimize.add_argument("--uniform-shrinkage", type=float, default=0.0)
    optimize.add_argument("--output", default="artifacts/performance_v2/ensemble_weights.json")
    optimize.set_defaults(handler=command_optimize_ensemble)

    predict_v2 = subparsers.add_parser("predict-v2", help="Run CAMP-VS v2 inference")
    _add_config_arguments(predict_v2, target=True)
    predict_v2.add_argument("--checkpoint", required=True)
    predict_v2.add_argument("--manifest", default=None)
    predict_v2.add_argument("--output-dir", default=None)
    predict_v2.add_argument("--max-epochs", type=int, default=None)
    predict_v2.set_defaults(
        handler=command_predict_v2,
        config="configs/performance_v2/camp_infer.yaml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    session = ExperimentLogSession.from_args(args)
    logger = configure_logging(session.console_path, verbose=bool(args.verbose))
    logger.info("Command log directory: %s", session.directory)
    with activate_experiment_log(session):
        try:
            result = args.handler(args)
        except KeyboardInterrupt as error:
            logger.error("Interrupted by user")
            session.fail(error)
            return 130
        except Exception as error:
            logger.exception("Command failed: %s", error)
            session.fail(error)
            return 1
    if isinstance(result, Mapping):
        enriched_result = dict(result)
        enriched_result.update(session.complete(enriched_result))
        result = enriched_result
    else:
        session.complete({"result": _json_safe(result)})
    _print_json(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
