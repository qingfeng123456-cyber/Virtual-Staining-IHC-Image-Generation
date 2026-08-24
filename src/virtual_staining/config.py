"""Configuration loading, validation, and deterministic override handling."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is malformed."""


_ALLOWED_KEYS: dict[str, set[str]] = {
    "project": {
        "name", "output_root", "artifact_root", "seed", "run_id", "fold",
        "initialization_provenance",
    },
    "data": {
        "root", "organ", "train_split", "val_split", "test_split", "targets",
        "submit_targets", "group_key", "val_ratio", "surrogate_group_size",
        "smoke_count", "image_size", "input_mode", "target_modes", "num_workers",
        "pin_memory", "persistent_workers", "max_train_samples", "max_val_samples",
        "input_channels", "target_channels", "grouped_inner_folds", "activity_sampler",
        "split_seed",
        "prefetch_factor",
    },
    "model": {
        "name", "base_channels", "encoder_depths", "decoder_depths", "use_sobel_input",
        "use_task_adapters", "use_prototypes", "shared_prototypes", "task_prototypes",
        "prototype_temperature", "prototype_fusion_weight", "deep_supervision",
        "decoder_mode", "output_activation", "local_encoder", "context", "global_mixer",
        "conditioning", "adapters", "prototypes", "output", "intensity_calibrator",
        "organ_names", "context_stop_gradient", "use_laplacian_input", "base_detail",
        "base_detail_residual", "max_detail_amplitude", "spatial_frequency",
        "lightweight_unet", "heavy_unet", "multi_route_fusion",
    },
    "loss": {
        "charbonnier", "ssim", "ms_ssim", "gradient", "frequency",
        "structure_weight_alpha", "correlation", "prototype_activation",
        "prototype_diversity", "deep_supervision_weights", "task_weights",
        "deep_supervision_final_weights", "competition_proxy",
        "auto_balance", "auto_balance_decay", "data_range", "mse", "pyramid",
        "statistics", "schedule", "phase_a_ratio", "phase_a", "phase_b",
        "prototype", "shift_tolerant", "fluorescence_foreground", "transition",
        "interpolation",
    },
    "train": {
        "device", "hardware_profile", "resolved_hardware_profile", "epochs", "batch_size",
        "gradient_accumulation", "optimizer", "lr", "weight_decay", "warmup_epochs",
        "scheduler", "amp", "amp_dtype", "grad_clip", "ema", "ema_decay",
        "early_stopping_patience", "save_top_k", "resume", "deterministic",
        "max_oom_retries", "task_name", "stages", "stage", "freeze_encoder_epochs",
        "stage_lr_scale", "metric_finetune_epochs", "evaluate_weight_sources",
        "initial_weight_source", "prototype_monitor", "progress_bar",
        "progress_bar_refresh_seconds", "async_metric_logging",
        "batch_metric_log_interval", "validate_every_n_epochs",
        "float32_matmul_precision", "profiler", "compile",
        "channels_last", "cudnn_benchmark", "retain_validation_records_in_history",
        "checkpoint_metrics",
        "optimizer_options",
        "equivariance",
        "max_wall_time_hours",
        "gpu_monitor",
    },
    "validation": {
        "device", "task_name", "primary_metric", "psnr_norm_min", "psnr_norm_max",
        "psnr_cap", "save_visuals", "worst_k", "tta", "primary_domain",
        "group_by_roi", "bootstrap_by_roi", "bootstrap_samples", "save_predictions",
        "domains", "inner_folds", "fold", "rank_proxy", "organ_weights",
    },
    "inference": {
        "device", "batch_size", "use_ema", "tta", "amp", "ensemble_checkpoints",
        "jpeg_quality", "jpeg_subsampling", "jpeg_optimize", "context", "weight_source",
        "tta_policy",
    },
    "submission": {
        "root_name", "split_name", "fake_suffix", "extension", "create_zip",
        "official", "allow_smoke_manifest",
    },
    "augmentation": {
        "max_translation", "horizontal_flip_probability", "vertical_flip_probability",
        "rotate_probability", "gamma_range", "gamma_probability", "brightness_delta",
        "brightness_probability", "contrast_range", "contrast_probability", "noise_std",
        "noise_probability", "context_shared_intensity",
    },
    "audit": {
        "roi_leakage_verifiable", "official_test_status", "require_verified_grid",
        "allow_numeric_stem_inference", "allow_edge_graph_inference", "edge_strip_width",
        "continuity_threshold",
    },
    "multitask": {
        "optimizer", "gradient_cosine_enabled", "log_gradient_cosine_every",
        "uncertainty_init",
    },
    "pretrain": {
        "enabled", "mask_ratio", "block_size", "epochs", "lr", "context_encoder",
        "neighbor_consistency", "neighbor_consistency_weight", "widths",
        "encoder_depths", "decoder_depths", "masking_enabled", "checkpoint",
    },
    "ensemble": {
        "method", "weights", "validation_only", "optimizer", "cross_validate_weights",
        "allow_range_wise", "weight_source", "allow_unsafe_model_soup_lineage",
        "allow_unsafe_model_soup_validation", "optimizer_failure_policy",
    },
    "budget": {"active", "smoke", "screen", "confirm", "full"},
}


_NESTED_ALLOWED: dict[tuple[str, str], set[str]] = {
    ("project", "initialization_provenance"): {
        "version", "initialization_lineage", "initial_state_sha256",
        "architecture_sha256", "parent_checkpoint_sha256",
        "pretrain_checkpoint_sha256",
    },
    ("data", "grouped_inner_folds"): {
        "enabled", "fold_count", "seed", "require_official_train",
        "coordinate_source", "output_csv",
    },
    ("data", "activity_sampler"): {
        "enabled", "activity_key", "num_bins", "seed", "num_samples",
        "strategy", "start_ratio", "hard_fraction",
    },
    ("model", "local_encoder"): {
        "type", "widths", "depths", "drop_path", "use_laplacian_input",
    },
    ("model", "context"): {
        "enabled", "grid_size", "missing_policy", "include_center", "token_dim",
        "encoder_width", "context_dropout", "fusion_scales",
        "bottleneck_cross_attention", "residual_init", "stop_gradient",
        "require_verified_grid", "coordinate_source", "allow_numeric_stem_inference",
        "allow_edge_graph_inference", "tile_chunk_size", "encoder_depth",
        "cross_attention_heads", "cache_size",
    },
    ("model", "spatial_frequency"): {
        "enabled", "spatial_depth", "spatial_expansion", "gate_reduction",
        "frequency_cutoff", "frequency_transition_width", "residual_init",
        "adaptive_frequency", "adaptive_reduction", "adaptive_cutoff_delta",
        "modulation_init",
    },
    ("model", "lightweight_unet"): {
        "enabled", "widths", "depths", "kernel_size", "expansion",
        "fusion_scales", "residual_init",
    },
    ("model", "heavy_unet"): {
        "enabled", "widths", "encoder_depths", "decoder_depths",
        "local_kernel_size", "large_kernel_size", "expansion",
        "fusion_scales", "checkpoint_blocks",
    },
    ("model", "multi_route_fusion"): {
        "enabled", "fusion_scales", "gate_reduction", "heads",
        "residual_init", "refinements",
    },
    ("model", "global_mixer"): {
        "enabled", "type", "blocks_1_8", "blocks_1_16", "heads", "heads_1_8",
        "heads_1_16", "expansion",
    },
    ("model", "conditioning"): {
        "marker_embedding", "organ_embedding", "film", "zero_init", "embedding_dim",
    },
    ("model", "adapters"): {"marker", "organ", "mixture_of_experts", "reduction"},
    ("model", "prototypes"): {
        "enabled", "scales", "shared_count", "marker_count", "organ_count", "dim",
        "temperature", "residual_init", "reset_dead", "dead_threshold",
        "reset_patience", "reset_seed", "reset_std",
    },
    ("model", "output"): {
        "base_detail", "max_detail_amplitude", "deep_supervision",
        "deep_supervision_scales",
    },
    ("model", "intensity_calibrator"): {
        "enabled", "max_gain_delta", "max_bias",
    },
    ("train", "prototype_monitor"): {
        "enabled",
        "dead_threshold",
        "attention_visuals_enabled",
        "attention_visual_count",
        "attention_visual_seed",
        "attention_visual_size",
    },
    ("train", "profiler"): {
        "enabled", "max_steps", "output_dir", "record_shapes", "with_stack",
    },
    ("train", "compile"): {
        "enabled", "mode", "backend",
    },
    ("train", "optimizer_options"): {
        "enabled", "no_decay_norm_bias", "fused",
    },
    ("train", "equivariance"): {
        "enabled", "probability", "weight", "start_ratio", "ramp_end_ratio",
        "smooth_l1_beta",
    },
    ("train", "gpu_monitor"): {
        "enabled", "interval_seconds",
    },
    ("loss", "phase_a"): {
        "mse", "charbonnier", "ssim", "ms_ssim", "pyramid", "gradient", "statistics",
    },
    ("loss", "phase_b"): {
        "mse", "charbonnier", "ssim", "ms_ssim", "pyramid", "gradient", "statistics",
    },
    ("loss", "prototype"): {"activation", "diversity", "usage_entropy"},
    ("loss", "shift_tolerant"): {
        "enabled", "weight", "max_shift", "mode", "temperature",
    },
    ("loss", "fluorescence_foreground"): {
        "enabled", "weight", "threshold_std_scale", "temperature", "mse_weight",
        "dice_weight", "intensity_weight", "min_activity_std",
    },
    ("loss", "competition_proxy"): {
        "enabled", "weight", "start_ratio", "ssim_weight", "window_size",
        "psnr_cap",
    },
}


def _normalized_marker(value: Any) -> str:
    key = "".join(character for character in str(value).casefold() if character.isalnum())
    aliases = {"hladr": "HLA-DR", "cd45ro": "CD45RO", "vimentin": "Vimentin", "cd68": "CD68"}
    if key not in aliases:
        raise ConfigError(f"Unknown prediction target: {value!r}")
    return aliases[key]


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return value


def _parse_scalar(value: str) -> Any:
    return yaml.safe_load(value)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dotted `section.key=value` overrides."""

    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override must contain '=': {item}")
        dotted, raw = item.split("=", 1)
        keys = [key for key in dotted.split(".") if key]
        if not keys:
            raise ConfigError(f"Override has an empty key: {item}")
        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise ConfigError(f"Override crosses a scalar key: {dotted}")
            cursor = child
        cursor[keys[-1]] = _parse_scalar(raw)
    return result


def validate_config(config: dict[str, Any]) -> None:
    """Validate cross-cutting invariants relied upon by every CLI command."""

    required = ("project", "data", "model", "loss", "train", "validation", "inference", "submission")
    unknown_sections = sorted(set(config) - set(_ALLOWED_KEYS))
    if unknown_sections:
        raise ConfigError(f"Unknown configuration section(s): {unknown_sections}")
    for section in required:
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"Missing configuration section: {section}")
    for section, values in config.items():
        if not isinstance(values, dict):
            raise ConfigError(f"Configuration section {section} must be a mapping")
        unknown = sorted(set(values) - _ALLOWED_KEYS[section])
        if unknown:
            raise ConfigError(f"Unknown key(s) in {section}: {unknown}")
        for key, value in values.items():
            allowed_nested = _NESTED_ALLOWED.get((section, key))
            if allowed_nested is None:
                continue
            if not isinstance(value, dict):
                raise ConfigError(f"{section}.{key} must be a mapping")
            nested_unknown = sorted(set(value) - allowed_nested)
            if nested_unknown:
                raise ConfigError(f"Unknown key(s) in {section}.{key}: {nested_unknown}")
    size = int(config["data"].get("image_size", 0))
    if size <= 0:
        raise ConfigError("data.image_size must be positive")
    workers = int(config["data"].get("num_workers", 0))
    if workers < 0:
        raise ConfigError("data.num_workers cannot be negative")
    prefetch_factor = config["data"].get("prefetch_factor", 2)
    if (
        isinstance(prefetch_factor, bool)
        or not isinstance(prefetch_factor, int)
        or prefetch_factor < 1
    ):
        raise ConfigError("data.prefetch_factor must be a positive integer")
    grouped_folds = config["data"].get("grouped_inner_folds", {})
    if grouped_folds:
        if not isinstance(grouped_folds.get("enabled", False), bool):
            raise ConfigError("data.grouped_inner_folds.enabled must be boolean")
        fold_count = grouped_folds.get("fold_count", 5)
        if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 2:
            raise ConfigError("data.grouped_inner_folds.fold_count must be an integer >= 2")
        grouped_seed = grouped_folds.get("seed", 2026)
        if (
            isinstance(grouped_seed, bool)
            or not isinstance(grouped_seed, int)
            or grouped_seed < 0
        ):
            raise ConfigError("data.grouped_inner_folds.seed must be a nonnegative integer")
        if grouped_folds.get("enabled", False):
            if not grouped_folds.get("require_official_train", True):
                raise ConfigError("Grouped inner folds must require official train rows")
            if str(grouped_folds.get("coordinate_source", "filename")) != "filename":
                raise ConfigError("Grouped inner folds require filename coordinates")
    activity_sampler = config["data"].get("activity_sampler", {})
    if activity_sampler:
        if not isinstance(activity_sampler.get("enabled", False), bool):
            raise ConfigError("data.activity_sampler.enabled must be boolean")
        num_bins = activity_sampler.get("num_bins", 4)
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 1:
            raise ConfigError("data.activity_sampler.num_bins must be a positive integer")
        num_samples = activity_sampler.get("num_samples")
        if num_samples is not None and (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples < 1
        ):
            raise ConfigError("data.activity_sampler.num_samples must be a positive integer")
        activity_seed = activity_sampler.get("seed", 2026)
        if (
            isinstance(activity_seed, bool)
            or not isinstance(activity_seed, int)
            or activity_seed < 0
        ):
            raise ConfigError("data.activity_sampler.seed must be a nonnegative integer")
        if activity_sampler.get("enabled", False) and not str(
            activity_sampler.get("activity_key", "")
        ).strip():
            raise ConfigError("data.activity_sampler.activity_key cannot be empty")
        strategy = str(activity_sampler.get("strategy", "balanced")).casefold()
        if strategy not in {"balanced", "hard_mix"}:
            raise ConfigError(
                "data.activity_sampler.strategy must be balanced or hard_mix"
            )
        try:
            start_ratio = float(activity_sampler.get("start_ratio", 0.0))
            hard_fraction = float(activity_sampler.get("hard_fraction", 0.0))
        except (TypeError, ValueError) as error:
            raise ConfigError(
                "data.activity_sampler start_ratio/hard_fraction must be numeric"
            ) from error
        if not 0.0 <= start_ratio < 1.0:
            raise ConfigError("data.activity_sampler.start_ratio must lie in [0, 1)")
        if not 0.0 <= hard_fraction < 1.0:
            raise ConfigError("data.activity_sampler.hard_fraction must lie in [0, 1)")
        if (
            activity_sampler.get("enabled", False)
            and strategy == "hard_mix"
            and hard_fraction <= 0.0
        ):
            raise ConfigError("hard_mix activity sampling requires hard_fraction > 0")
    targets = config["data"].get("targets", [])
    if not isinstance(targets, list) or not targets:
        raise ConfigError("data.targets must be a non-empty list")
    normalized_targets = [_normalized_marker(target) for target in targets]
    if len(set(normalized_targets)) != len(normalized_targets):
        raise ConfigError("data.targets contains duplicates after normalization")
    submit_targets = config["data"].get("submit_targets", [])
    if not isinstance(submit_targets, list) or not submit_targets:
        raise ConfigError("data.submit_targets must be a non-empty list")
    normalized_submit = [_normalized_marker(target) for target in submit_targets]
    if not set(normalized_submit).issubset(normalized_targets):
        raise ConfigError("data.submit_targets must be a subset of data.targets")
    val_ratio = float(config["data"].get("val_ratio", 0.0))
    if not 0.0 < val_ratio < 1.0:
        raise ConfigError("data.val_ratio must be in (0, 1)")
    split_seed = config["data"].get(
        "split_seed", config["project"].get("seed", 2026)
    )
    if isinstance(split_seed, bool) or not isinstance(split_seed, int) or split_seed < 0:
        raise ConfigError("data.split_seed must be a nonnegative integer")
    for section, key in (("train", "epochs"), ("train", "gradient_accumulation")):
        value = config[section].get(key)
        if value != "auto" and int(value) < 1:
            raise ConfigError(f"{section}.{key} must be positive or 'auto'")
    progress_bar = config["train"].get("progress_bar", "auto")
    if not isinstance(progress_bar, bool) and str(progress_bar).casefold() != "auto":
        raise ConfigError("train.progress_bar must be boolean or 'auto'")
    progress_refresh_value = config["train"].get("progress_bar_refresh_seconds", 1.0)
    if isinstance(progress_refresh_value, bool):
        raise ConfigError("train.progress_bar_refresh_seconds must be finite and positive")
    try:
        progress_refresh_seconds = float(progress_refresh_value)
    except (TypeError, ValueError) as error:
        raise ConfigError(
            "train.progress_bar_refresh_seconds must be finite and positive"
        ) from error
    if not math.isfinite(progress_refresh_seconds) or progress_refresh_seconds <= 0.0:
        raise ConfigError("train.progress_bar_refresh_seconds must be finite and positive")
    for section, key in (("train", "batch_size"), ("inference", "batch_size")):
        value = config[section].get(key)
        if value != "auto" and int(value) < 1:
            raise ConfigError(f"{section}.{key} must be positive or 'auto'")
    inference_tta = str(config["inference"].get("tta", "none")).strip().casefold()
    if inference_tta not in {"none", "d4"}:
        raise ConfigError("inference.tta must be none or d4")
    tta_policy = str(
        config["inference"].get("tta_policy", "configured")
    ).strip().casefold()
    if tta_policy not in {"configured", "validation_decision"}:
        raise ConfigError(
            "inference.tta_policy must be configured or validation_decision"
        )
    weight_sources = config["train"].get("evaluate_weight_sources")
    if weight_sources is not None:
        if not isinstance(weight_sources, list) or not weight_sources:
            raise ConfigError("train.evaluate_weight_sources must be a non-empty list")
        normalized_sources = [str(value).strip().casefold() for value in weight_sources]
        if any(value not in {"raw", "ema"} for value in normalized_sources):
            raise ConfigError(
                "train.evaluate_weight_sources accepts only raw and ema"
            )
        if len(set(normalized_sources)) != len(normalized_sources):
            raise ConfigError("train.evaluate_weight_sources cannot contain duplicates")
        if "ema" in normalized_sources and not bool(config["train"].get("ema", True)):
            raise ConfigError(
                "train.evaluate_weight_sources requests ema while train.ema is disabled"
            )
    async_logging = config["train"].get("async_metric_logging", False)
    if not isinstance(async_logging, bool):
        raise ConfigError("train.async_metric_logging must be boolean")
    retain_validation_records = config["train"].get(
        "retain_validation_records_in_history", True
    )
    if not isinstance(retain_validation_records, bool):
        raise ConfigError(
            "train.retain_validation_records_in_history must be boolean"
        )
    checkpoint_metrics = config["train"].get(
        "checkpoint_metrics", ["ssim", "psnr", "proxy"]
    )
    if not isinstance(checkpoint_metrics, list) or not checkpoint_metrics:
        raise ConfigError("train.checkpoint_metrics must be a non-empty list")
    normalized_checkpoint_metrics = [
        str(value).strip().casefold() for value in checkpoint_metrics
    ]
    if any(
        value not in {"ssim", "psnr", "proxy"}
        for value in normalized_checkpoint_metrics
    ):
        raise ConfigError(
            "train.checkpoint_metrics accepts only ssim, psnr, and proxy"
        )
    if len(set(normalized_checkpoint_metrics)) != len(
        normalized_checkpoint_metrics
    ):
        raise ConfigError("train.checkpoint_metrics cannot contain duplicates")
    log_interval = config["train"].get("batch_metric_log_interval", 0)
    if (
        isinstance(log_interval, bool)
        or not isinstance(log_interval, int)
        or log_interval < 0
    ):
        raise ConfigError("train.batch_metric_log_interval must be a nonnegative integer")
    validate_every = config["train"].get("validate_every_n_epochs", 1)
    if (
        isinstance(validate_every, bool)
        or not isinstance(validate_every, int)
        or validate_every < 1
    ):
        raise ConfigError("train.validate_every_n_epochs must be a positive integer")
    max_wall_time = config["train"].get("max_wall_time_hours", 0.0)
    if isinstance(max_wall_time, bool):
        raise ConfigError("train.max_wall_time_hours must be finite and nonnegative")
    try:
        max_wall_time_value = float(max_wall_time)
    except (TypeError, ValueError) as error:
        raise ConfigError(
            "train.max_wall_time_hours must be finite and nonnegative"
        ) from error
    if not math.isfinite(max_wall_time_value) or max_wall_time_value < 0.0:
        raise ConfigError("train.max_wall_time_hours must be finite and nonnegative")
    matmul_precision = str(
        config["train"].get("float32_matmul_precision", "highest")
    ).strip().casefold()
    if matmul_precision not in {"highest", "high", "medium"}:
        raise ConfigError(
            "train.float32_matmul_precision must be one of highest, high, medium"
        )
    profiler_cfg = config["train"].get("profiler", {})
    if profiler_cfg:
        if not isinstance(profiler_cfg.get("enabled", False), bool):
            raise ConfigError("train.profiler.enabled must be boolean")
        profiler_steps = profiler_cfg.get("max_steps", 30)
        if (
            isinstance(profiler_steps, bool)
            or not isinstance(profiler_steps, int)
            or profiler_steps < 1
        ):
            raise ConfigError("train.profiler.max_steps must be a positive integer")
    compile_cfg = config["train"].get("compile", {})
    if compile_cfg:
        if not isinstance(compile_cfg.get("enabled", False), bool):
            raise ConfigError("train.compile.enabled must be boolean")
        compile_mode = str(compile_cfg.get("mode", "default")).strip().casefold()
        if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ConfigError(
                "train.compile.mode must be default, reduce-overhead, or max-autotune"
            )
    optimizer_options = config["train"].get("optimizer_options", {})
    if optimizer_options:
        if not isinstance(optimizer_options.get("enabled", False), bool):
            raise ConfigError("train.optimizer_options.enabled must be boolean")
        if not isinstance(
            optimizer_options.get("no_decay_norm_bias", True), bool
        ):
            raise ConfigError(
                "train.optimizer_options.no_decay_norm_bias must be boolean"
            )
        fused = optimizer_options.get("fused", "auto")
        if not isinstance(fused, bool) and str(fused).casefold() != "auto":
            raise ConfigError("train.optimizer_options.fused must be boolean or auto")
    equivariance = config["train"].get("equivariance", {})
    if equivariance:
        if not isinstance(equivariance.get("enabled", False), bool):
            raise ConfigError("train.equivariance.enabled must be boolean")
        numeric_defaults = {
            "probability": 0.0,
            "weight": 0.0,
            "start_ratio": 0.10,
            "ramp_end_ratio": 0.40,
            "smooth_l1_beta": 0.01,
        }
        try:
            values = {
                key: float(equivariance.get(key, default))
                for key, default in numeric_defaults.items()
            }
        except (TypeError, ValueError) as error:
            raise ConfigError("train.equivariance values must be numeric") from error
        if any(not math.isfinite(value) for value in values.values()):
            raise ConfigError("train.equivariance values must be finite")
        if not 0.0 <= values["probability"] <= 1.0:
            raise ConfigError("train.equivariance.probability must lie in [0, 1]")
        if values["weight"] < 0.0:
            raise ConfigError("train.equivariance.weight must be nonnegative")
        if not 0.0 <= values["start_ratio"] < values["ramp_end_ratio"] <= 1.0:
            raise ConfigError(
                "train.equivariance requires 0 <= start_ratio < ramp_end_ratio <= 1"
            )
        if values["smooth_l1_beta"] <= 0.0:
            raise ConfigError("train.equivariance.smooth_l1_beta must be positive")
        if equivariance.get("enabled", False) and (
            values["probability"] <= 0.0 or values["weight"] <= 0.0
        ):
            raise ConfigError(
                "Enabled equivariance regularization requires positive probability and weight"
            )
    gpu_monitor = config["train"].get("gpu_monitor", {})
    if gpu_monitor:
        if not isinstance(gpu_monitor.get("enabled", False), bool):
            raise ConfigError("train.gpu_monitor.enabled must be boolean")
        try:
            monitor_interval = float(gpu_monitor.get("interval_seconds", 2.0))
        except (TypeError, ValueError) as error:
            raise ConfigError(
                "train.gpu_monitor.interval_seconds must be numeric"
            ) from error
        if not math.isfinite(monitor_interval) or monitor_interval < 0.5:
            raise ConfigError(
                "train.gpu_monitor.interval_seconds must be finite and at least 0.5"
            )
    phase_a_ratio = float(config["loss"].get("phase_a_ratio", 0.7))
    if not 0.0 < phase_a_ratio < 1.0:
        raise ConfigError("loss.phase_a_ratio must be in (0, 1)")
    foreground = config["loss"].get("fluorescence_foreground", {})
    if foreground:
        enabled = foreground.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError("loss.fluorescence_foreground.enabled must be boolean")
        numeric_defaults = {
            "weight": 0.0,
            "threshold_std_scale": 0.25,
            "temperature": 0.05,
            "mse_weight": 1.0,
            "dice_weight": 0.10,
            "intensity_weight": 0.25,
            "min_activity_std": 0.01,
        }
        numeric: dict[str, float] = {}
        for key, default in numeric_defaults.items():
            value = foreground.get(key, default)
            if isinstance(value, bool):
                raise ConfigError(
                    f"loss.fluorescence_foreground.{key} must be numeric"
                )
            try:
                numeric[key] = float(value)
            except (TypeError, ValueError) as error:
                raise ConfigError(
                    f"loss.fluorescence_foreground.{key} must be numeric"
                ) from error
            if not math.isfinite(numeric[key]):
                raise ConfigError(
                    f"loss.fluorescence_foreground.{key} must be finite"
                )
        nonnegative = (
            "weight",
            "threshold_std_scale",
            "mse_weight",
            "dice_weight",
            "intensity_weight",
            "min_activity_std",
        )
        if any(numeric[key] < 0.0 for key in nonnegative):
            raise ConfigError(
                "loss.fluorescence_foreground weights, threshold scale, and "
                "minimum activity must be nonnegative"
            )
        if numeric["temperature"] <= 0.0:
            raise ConfigError(
                "loss.fluorescence_foreground.temperature must be positive"
            )
        if enabled:
            if str(config["loss"].get("schedule", "constant")).casefold() != "two_phase":
                raise ConfigError(
                    "loss.fluorescence_foreground requires loss.schedule=two_phase"
                )
            if numeric["weight"] <= 0.0:
                raise ConfigError(
                    "Enabled fluorescence foreground loss requires a positive weight"
                )
            if sum(
                numeric[key]
                for key in ("mse_weight", "dice_weight", "intensity_weight")
            ) <= 0.0:
                raise ConfigError(
                    "Enabled fluorescence foreground loss requires a positive "
                    "internal loss weight"
                )
    initial_deep = config["loss"].get("deep_supervision_weights", [1.0, 0.5, 0.25])
    final_deep = config["loss"].get("deep_supervision_final_weights")
    if final_deep is not None:
        if (
            not isinstance(final_deep, list)
            or not isinstance(initial_deep, list)
            or len(final_deep) != len(initial_deep)
            or not final_deep
        ):
            raise ConfigError(
                "loss.deep_supervision_final_weights must be a non-empty list "
                "matching loss.deep_supervision_weights"
            )
        try:
            final_values = [float(value) for value in final_deep]
        except (TypeError, ValueError) as error:
            raise ConfigError(
                "loss.deep_supervision_final_weights must contain numeric values"
            ) from error
        if (
            any(not math.isfinite(value) or value < 0.0 for value in final_values)
            or final_values[0] <= 0.0
        ):
            raise ConfigError(
                "Final deep-supervision weights must be finite/nonnegative and "
                "retain a positive full-resolution weight"
            )
    competition_proxy = config["loss"].get("competition_proxy", {})
    if competition_proxy:
        enabled = competition_proxy.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigError("loss.competition_proxy.enabled must be boolean")
        numeric_defaults = {
            "weight": 0.0,
            "start_ratio": 0.55,
            "ssim_weight": 0.7,
            "window_size": 7,
            "psnr_cap": 60.0,
        }
        try:
            proxy_values = {
                key: float(competition_proxy.get(key, default))
                for key, default in numeric_defaults.items()
            }
        except (TypeError, ValueError) as error:
            raise ConfigError("loss.competition_proxy values must be numeric") from error
        if any(not math.isfinite(value) for value in proxy_values.values()):
            raise ConfigError("loss.competition_proxy values must be finite")
        if enabled and proxy_values["weight"] <= 0.0:
            raise ConfigError("Enabled competition proxy requires a positive weight")
        if not 0.0 <= proxy_values["start_ratio"] < 1.0:
            raise ConfigError("loss.competition_proxy.start_ratio must lie in [0, 1)")
        if not 0.0 <= proxy_values["ssim_weight"] <= 1.0:
            raise ConfigError("loss.competition_proxy.ssim_weight must lie in [0, 1]")
        window_size = competition_proxy.get("window_size", 7)
        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or window_size < 3
            or window_size % 2 == 0
        ):
            raise ConfigError(
                "loss.competition_proxy.window_size must be an odd integer >= 3"
            )
        if proxy_values["psnr_cap"] <= 0.0:
            raise ConfigError("loss.competition_proxy.psnr_cap must be positive")
    channels_last_cfg = config["train"].get("channels_last", False)
    if not isinstance(channels_last_cfg, bool):
        raise ConfigError("train.channels_last must be boolean")
    cudnn_benchmark_cfg = config["train"].get("cudnn_benchmark", None)
    if cudnn_benchmark_cfg is not None and not isinstance(cudnn_benchmark_cfg, bool):
        raise ConfigError("train.cudnn_benchmark must be boolean or null")
    base_detail = config["model"].get("base_detail", False)
    base_detail_residual = config["model"].get("base_detail_residual", False)
    if not isinstance(base_detail, bool):
        raise ConfigError("model.base_detail must be boolean")
    if not isinstance(base_detail_residual, bool):
        raise ConfigError("model.base_detail_residual must be boolean")
    if base_detail_residual and not base_detail:
        raise ConfigError("model.base_detail_residual requires model.base_detail=true")
    context = config["model"].get("context", {})
    if context:
        grid_size = int(context.get("grid_size", 3))
        if grid_size < 1 or grid_size % 2 == 0:
            raise ConfigError("model.context.grid_size must be a positive odd integer")
        cache_size = context.get("cache_size", 0)
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ConfigError("model.context.cache_size must be a nonnegative integer")
        if context.get("enabled") and context.get("require_verified_grid", True):
            if context.get("allow_numeric_stem_inference", False):
                raise ConfigError(
                    "Verified context cannot use numeric-stem coordinate inference"
                )
            if context.get("allow_edge_graph_inference", False):
                raise ConfigError(
                    "Verified context cannot use image-edge graph coordinate inference"
                )
    spatial_frequency = config["model"].get("spatial_frequency", {})
    if spatial_frequency:
        if not isinstance(spatial_frequency.get("enabled", False), bool):
            raise ConfigError("model.spatial_frequency.enabled must be boolean")
        for key, default in (
            ("spatial_depth", 1),
            ("spatial_expansion", 1),
            ("gate_reduction", 8),
        ):
            value = spatial_frequency.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(
                    f"model.spatial_frequency.{key} must be a positive integer"
                )
        if not isinstance(
            spatial_frequency.get("adaptive_frequency", False), bool
        ):
            raise ConfigError(
                "model.spatial_frequency.adaptive_frequency must be boolean"
            )
        adaptive_reduction = spatial_frequency.get("adaptive_reduction", 8)
        if (
            isinstance(adaptive_reduction, bool)
            or not isinstance(adaptive_reduction, int)
            or adaptive_reduction < 1
        ):
            raise ConfigError(
                "model.spatial_frequency.adaptive_reduction must be a positive integer"
            )
        try:
            frequency_cutoff = float(
                spatial_frequency.get("frequency_cutoff", 0.35)
            )
            transition_width = float(
                spatial_frequency.get("frequency_transition_width", 0.08)
            )
            residual_init = float(spatial_frequency.get("residual_init", 0.0))
            adaptive_cutoff_delta = float(
                spatial_frequency.get("adaptive_cutoff_delta", 0.12)
            )
            modulation_init = float(
                spatial_frequency.get("modulation_init", 0.0)
            )
        except (TypeError, ValueError) as error:
            raise ConfigError(
                "model.spatial_frequency floating-point options must be numeric"
            ) from error
        if not all(
            math.isfinite(value)
            for value in (
                frequency_cutoff,
                transition_width,
                residual_init,
                adaptive_cutoff_delta,
                modulation_init,
            )
        ):
            raise ConfigError(
                "model.spatial_frequency floating-point options must be finite"
            )
        if not 0.0 < frequency_cutoff < 1.0:
            raise ConfigError(
                "model.spatial_frequency.frequency_cutoff must lie in (0, 1)"
            )
        if transition_width <= 0.0:
            raise ConfigError(
                "model.spatial_frequency.frequency_transition_width must be positive"
            )
        if not 0.0 <= residual_init <= 1.0:
            raise ConfigError(
                "model.spatial_frequency.residual_init must lie in [0, 1]"
            )
        if not 0.0 <= adaptive_cutoff_delta < 0.5:
            raise ConfigError(
                "model.spatial_frequency.adaptive_cutoff_delta must lie in [0, 0.5)"
            )
        if not 0.0 <= modulation_init <= 1.0:
            raise ConfigError(
                "model.spatial_frequency.modulation_init must lie in [0, 1]"
            )
        if spatial_frequency.get("adaptive_frequency", False):
            if not bool(spatial_frequency.get("enabled", False)):
                raise ConfigError(
                    "Adaptive frequency requires model.spatial_frequency.enabled=true"
                )
            max_cutoff_delta = min(frequency_cutoff, 1.0 - frequency_cutoff)
            if adaptive_cutoff_delta >= max_cutoff_delta:
                raise ConfigError(
                    "Enabled adaptive frequency requires adaptive_cutoff_delta "
                    "to keep the learned cutoff strictly inside (0, 1)"
                )
            if adaptive_cutoff_delta <= 0.0:
                raise ConfigError(
                    "Enabled adaptive frequency requires adaptive_cutoff_delta > 0"
                )
            if modulation_init <= 0.0:
                raise ConfigError(
                    "Enabled adaptive frequency requires modulation_init > 0"
                )
    lightweight_unet = config["model"].get("lightweight_unet", {})
    if lightweight_unet:
        if not isinstance(lightweight_unet.get("enabled", False), bool):
            raise ConfigError("model.lightweight_unet.enabled must be boolean")
        for key, default in (
            ("widths", [16, 24, 32, 48]),
            ("depths", [1, 1, 1, 1]),
        ):
            values = lightweight_unet.get(key, default)
            if (
                not isinstance(values, list)
                or len(values) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    for value in values
                )
            ):
                raise ConfigError(
                    f"model.lightweight_unet.{key} must contain four positive integers"
                )
        kernel_size = lightweight_unet.get("kernel_size", 7)
        if (
            isinstance(kernel_size, bool)
            or not isinstance(kernel_size, int)
            or kernel_size < 3
            or kernel_size % 2 == 0
        ):
            raise ConfigError(
                "model.lightweight_unet.kernel_size must be an odd integer >= 3"
            )
        expansion = lightweight_unet.get("expansion", 2)
        if (
            isinstance(expansion, bool)
            or not isinstance(expansion, int)
            or expansion < 1
        ):
            raise ConfigError(
                "model.lightweight_unet.expansion must be a positive integer"
            )
        fusion_scales = lightweight_unet.get("fusion_scales", [1, 2, 4])
        if (
            not isinstance(fusion_scales, list)
            or not fusion_scales
            or any(
                isinstance(scale, bool)
                or not isinstance(scale, int)
                or scale not in {1, 2, 4}
                for scale in fusion_scales
            )
            or len(set(fusion_scales)) != len(fusion_scales)
        ):
            raise ConfigError(
                "model.lightweight_unet.fusion_scales must be unique values from 1/2/4"
            )
        try:
            lightweight_residual_init = float(
                lightweight_unet.get("residual_init", 0.0)
            )
        except (TypeError, ValueError) as error:
            raise ConfigError(
                "model.lightweight_unet.residual_init must be numeric"
            ) from error
        if not math.isfinite(lightweight_residual_init):
            raise ConfigError(
                "model.lightweight_unet.residual_init must be finite"
            )
        if lightweight_residual_init != 0.0:
            raise ConfigError(
                "model.lightweight_unet.residual_init must be zero so enabling "
                "the branch preserves the pretrained/local backbone initially"
            )
    heavy_unet = config["model"].get("heavy_unet", {})
    if heavy_unet:
        heavy_enabled = heavy_unet.get("enabled", False)
        if not isinstance(heavy_enabled, bool):
            raise ConfigError("model.heavy_unet.enabled must be boolean")
        for key, default, expected_length in (
            ("widths", [32, 48, 72, 96], 4),
            ("encoder_depths", [2, 2, 3, 3], 4),
            ("decoder_depths", [1, 1, 2], 3),
        ):
            values = heavy_unet.get(key, default)
            if (
                not isinstance(values, list)
                or len(values) != expected_length
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    for value in values
                )
            ):
                raise ConfigError(
                    f"model.heavy_unet.{key} must contain "
                    f"{expected_length} positive integers"
                )
        for key, default in (
            ("local_kernel_size", 3),
            ("large_kernel_size", 11),
        ):
            value = heavy_unet.get(key, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 3
                or value % 2 == 0
            ):
                raise ConfigError(
                    f"model.heavy_unet.{key} must be an odd integer >= 3"
                )
        heavy_expansion = heavy_unet.get("expansion", 2)
        if (
            isinstance(heavy_expansion, bool)
            or not isinstance(heavy_expansion, int)
            or heavy_expansion < 1
        ):
            raise ConfigError("model.heavy_unet.expansion must be a positive integer")
        heavy_scales = heavy_unet.get("fusion_scales", [1, 2, 4, 8])
        if (
            not isinstance(heavy_scales, list)
            or not heavy_scales
            or any(
                isinstance(scale, bool)
                or not isinstance(scale, int)
                or scale not in {1, 2, 4, 8}
                for scale in heavy_scales
            )
            or len(set(heavy_scales)) != len(heavy_scales)
        ):
            raise ConfigError(
                "model.heavy_unet.fusion_scales must be unique values from 1/2/4/8"
            )
        if not isinstance(heavy_unet.get("checkpoint_blocks", False), bool):
            raise ConfigError("model.heavy_unet.checkpoint_blocks must be boolean")
        if heavy_enabled:
            lightweight_enabled = bool(lightweight_unet.get("enabled", False))
            if lightweight_enabled:
                raise ConfigError(
                    "model.heavy_unet and model.lightweight_unet are mutually exclusive"
                )
            model_name = "".join(
                character
                for character in str(config["model"].get("name", "")).casefold()
                if character.isalnum()
            )
            if model_name != "multimarkerrestorer":
                raise ConfigError(
                    "model.heavy_unet is supported only by MultiMarkerRestorer"
                )
            configured_fusion = config["model"].get("multi_route_fusion", {})
            if not bool(configured_fusion.get("enabled", False)):
                raise ConfigError(
                    "Enabled heavy U-Net requires "
                    "model.multi_route_fusion.enabled=true"
                )
    multi_route_fusion = config["model"].get("multi_route_fusion", {})
    if multi_route_fusion:
        fusion_enabled = multi_route_fusion.get("enabled", False)
        if not isinstance(fusion_enabled, bool):
            raise ConfigError("model.multi_route_fusion.enabled must be boolean")
        route_scales = multi_route_fusion.get("fusion_scales", [1, 2, 4, 8])
        if (
            not isinstance(route_scales, list)
            or not route_scales
            or any(
                isinstance(scale, bool)
                or not isinstance(scale, int)
                or scale not in {1, 2, 4, 8}
                for scale in route_scales
            )
            or len(set(route_scales)) != len(route_scales)
        ):
            raise ConfigError(
                "model.multi_route_fusion.fusion_scales must be unique values "
                "from 1/2/4/8"
            )
        for key, default in (("gate_reduction", 8), ("heads", 8)):
            value = multi_route_fusion.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(
                    f"model.multi_route_fusion.{key} must be a positive integer"
                )
        refinements = multi_route_fusion.get("refinements", 1)
        if (
            isinstance(refinements, bool)
            or not isinstance(refinements, int)
            or not 0 <= refinements <= 2
        ):
            raise ConfigError(
                "model.multi_route_fusion.refinements must be an integer in [0, 2]"
            )
        try:
            route_residual_init = float(
                multi_route_fusion.get("residual_init", 0.02)
            )
        except (TypeError, ValueError) as error:
            raise ConfigError(
                "model.multi_route_fusion.residual_init must be numeric"
            ) from error
        if not math.isfinite(route_residual_init):
            raise ConfigError(
                "model.multi_route_fusion.residual_init must be finite"
            )
        if fusion_enabled and not 0.0 < route_residual_init <= 1.0:
            raise ConfigError(
                "Enabled multi-route fusion requires residual_init in (0, 1]"
            )
        if not fusion_enabled and not 0.0 <= route_residual_init <= 1.0:
            raise ConfigError(
                "model.multi_route_fusion.residual_init must lie in [0, 1]"
            )
        bottleneck_channels = int(config["model"].get("base_channels", 0)) * 8
        fusion_heads = int(multi_route_fusion.get("heads", 8))
        if bottleneck_channels < 1 or bottleneck_channels % fusion_heads != 0:
            raise ConfigError(
                "model.multi_route_fusion.heads must divide "
                "model.base_channels * 8"
            )
        if fusion_enabled:
            if not bool(heavy_unet.get("enabled", False)):
                raise ConfigError(
                    "Enabled multi-route fusion requires model.heavy_unet.enabled=true"
                )
            if not bool(spatial_frequency.get("enabled", False)):
                raise ConfigError(
                    "Enabled multi-route fusion requires "
                    "model.spatial_frequency.enabled=true"
                )
            heavy_scales = heavy_unet.get("fusion_scales", [1, 2, 4, 8])
            if set(route_scales) != set(heavy_scales):
                raise ConfigError(
                    "heavy_unet.fusion_scales and multi_route_fusion.fusion_scales "
                    "must match"
                )
    prototypes = config["model"].get("prototypes", {})
    if prototypes:
        reset_enabled = prototypes.get("reset_dead", False)
        if not isinstance(reset_enabled, bool):
            raise ConfigError("model.prototypes.reset_dead must be boolean")
        dead_threshold = float(prototypes.get("dead_threshold", 1e-4))
        if not math.isfinite(dead_threshold) or dead_threshold < 0.0:
            raise ConfigError(
                "model.prototypes.dead_threshold must be finite and nonnegative"
            )
        reset_patience = prototypes.get("reset_patience", 3)
        if (
            isinstance(reset_patience, bool)
            or not isinstance(reset_patience, int)
            or reset_patience < 2
        ):
            raise ConfigError("model.prototypes.reset_patience must be an integer >= 2")
        reset_seed = prototypes.get("reset_seed", config["project"].get("seed", 2026))
        if isinstance(reset_seed, bool) or not isinstance(reset_seed, int) or reset_seed < 0:
            raise ConfigError(
                "model.prototypes.reset_seed must be a nonnegative integer"
            )
        reset_std = float(prototypes.get("reset_std", 0.02))
        if not math.isfinite(reset_std) or reset_std <= 0.0:
            raise ConfigError("model.prototypes.reset_std must be finite and positive")
        if reset_enabled:
            model_name = "".join(
                character
                for character in str(config["model"].get("name", "")).casefold()
                if character.isalnum()
            )
            if model_name not in {"campvsv2", "campv2"}:
                raise ConfigError(
                    "model.prototypes.reset_dead is supported only by CAMP-VS v2"
                )
            if not bool(prototypes.get("enabled", False)):
                raise ConfigError(
                    "model.prototypes.reset_dead requires model.prototypes.enabled=true"
                )
            monitor = config["train"].get("prototype_monitor", {})
            if not isinstance(monitor, dict) or not bool(monitor.get("enabled", False)):
                raise ConfigError(
                    "model.prototypes.reset_dead requires "
                    "train.prototype_monitor.enabled=true"
                )
    prototype_monitor = config["train"].get("prototype_monitor", {})
    if prototype_monitor:
        visuals_enabled = prototype_monitor.get(
            "attention_visuals_enabled", False
        )
        if not isinstance(visuals_enabled, bool):
            raise ConfigError(
                "train.prototype_monitor.attention_visuals_enabled must be boolean"
            )
        visual_count = prototype_monitor.get("attention_visual_count", 4)
        if (
            isinstance(visual_count, bool)
            or not isinstance(visual_count, int)
            or visual_count < 1
        ):
            raise ConfigError(
                "train.prototype_monitor.attention_visual_count must be a positive integer"
            )
        visual_seed = prototype_monitor.get(
            "attention_visual_seed", config["project"].get("seed", 2026)
        )
        if (
            isinstance(visual_seed, bool)
            or not isinstance(visual_seed, int)
            or visual_seed < 0
        ):
            raise ConfigError(
                "train.prototype_monitor.attention_visual_seed must be a nonnegative integer"
            )
        visual_size = prototype_monitor.get("attention_visual_size", 256)
        if (
            isinstance(visual_size, bool)
            or not isinstance(visual_size, int)
            or not 8 <= visual_size <= 2048
        ):
            raise ConfigError(
                "train.prototype_monitor.attention_visual_size must be an integer in [8, 2048]"
            )
        if visuals_enabled:
            if not bool(prototype_monitor.get("enabled", False)):
                raise ConfigError(
                    "Prototype attention visuals require "
                    "train.prototype_monitor.enabled=true"
                )
            nested_prototypes_enabled = (
                bool(prototypes.get("enabled", False))
                if isinstance(prototypes, dict) and "enabled" in prototypes
                else bool(config["model"].get("use_prototypes", False))
            )
            if not nested_prototypes_enabled:
                raise ConfigError(
                    "Prototype attention visuals require an enabled prototype model"
                )
    if str(config.get("multitask", {}).get("optimizer", "equal")) not in {
        "equal", "famo", "uncertainty"
    }:
        raise ConfigError("multitask.optimizer must be equal, famo, or uncertainty")
    gradient_interval = config.get("multitask", {}).get(
        "log_gradient_cosine_every", 500
    )
    if (
        isinstance(gradient_interval, bool)
        or not isinstance(gradient_interval, int)
        or gradient_interval < 1
    ):
        raise ConfigError("multitask.log_gradient_cosine_every must be positive")
    if not isinstance(
        config.get("multitask", {}).get("gradient_cosine_enabled", False), bool
    ):
        raise ConfigError("multitask.gradient_cosine_enabled must be boolean")
    ensemble = config.get("ensemble", {})
    if not isinstance(ensemble.get("validation_only", True), bool):
        raise ConfigError("ensemble.validation_only must be boolean")
    if not bool(ensemble.get("validation_only", True)):
        raise ConfigError("Ensemble fitting must remain validation/OOF-only")
    if not isinstance(ensemble.get("cross_validate_weights", True), bool):
        raise ConfigError("ensemble.cross_validate_weights must be boolean")
    optimizer_name = str(ensemble.get("optimizer", "coordinate")).strip().casefold()
    if optimizer_name not in {"coordinate", "slsqp"}:
        raise ConfigError("ensemble.optimizer must be coordinate or slsqp")
    failure_policy = str(
        ensemble.get("optimizer_failure_policy", "uniform")
    ).strip().casefold()
    if failure_policy not in {"uniform", "error"}:
        raise ConfigError("ensemble.optimizer_failure_policy must be uniform or error")
    pretrain = config.get("pretrain", {})
    if pretrain.get("checkpoint") and not bool(pretrain.get("enabled", False)):
        raise ConfigError("pretrain.checkpoint requires pretrain.enabled=true")
    unsafe_soup = config.get("ensemble", {}).get(
        "allow_unsafe_model_soup_lineage", False
    )
    if not isinstance(unsafe_soup, bool):
        raise ConfigError(
            "ensemble.allow_unsafe_model_soup_lineage must be boolean"
        )
    unsafe_soup_validation = config.get("ensemble", {}).get(
        "allow_unsafe_model_soup_validation", False
    )
    if not isinstance(unsafe_soup_validation, bool):
        raise ConfigError(
            "ensemble.allow_unsafe_model_soup_validation must be boolean"
        )


def _find_project_root(requested: Path) -> Path:
    for candidate in (requested.parent, *requested.parents):
        if (candidate / "configs" / "default.yaml").is_file():
            return candidate
    return Path.cwd()


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    *,
    include_resolved: bool = True,
) -> dict[str, Any]:
    """Load default, requested, resolved-local, and CLI layers in fixed precedence."""

    requested = Path(path).resolve()
    project_root = _find_project_root(requested)
    default_path = project_root / "configs" / "default.yaml"
    config = _read_yaml(default_path)
    if requested != default_path.resolve():
        config = deep_merge(config, _read_yaml(requested))
    resolved_candidates = [project_root / "configs" / "resolved_local.yaml"]
    if requested.parent.name == "performance_v2":
        resolved_candidates.append(requested.parent / "resolved_local.yaml")
    if include_resolved:
        for resolved in resolved_candidates:
            if resolved.exists() and requested != resolved.resolve():
                config = deep_merge(config, _read_yaml(resolved))
    config = apply_overrides(config, overrides)
    validate_config(config)
    return config


def save_effective_config(config: dict[str, Any], path: str | Path) -> Path:
    """Persist the exact effective configuration used by a run."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable short SHA-256 digest of a configuration."""

    import hashlib

    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
