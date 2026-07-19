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
    },
    "model": {
        "name", "base_channels", "encoder_depths", "decoder_depths", "use_sobel_input",
        "use_task_adapters", "use_prototypes", "shared_prototypes", "task_prototypes",
        "prototype_temperature", "prototype_fusion_weight", "deep_supervision",
        "decoder_mode", "output_activation", "local_encoder", "context", "global_mixer",
        "conditioning", "adapters", "prototypes", "output", "intensity_calibrator",
        "organ_names", "context_stop_gradient", "use_laplacian_input", "base_detail",
        "max_detail_amplitude",
    },
    "loss": {
        "charbonnier", "ssim", "ms_ssim", "gradient", "frequency",
        "structure_weight_alpha", "correlation", "prototype_activation",
        "prototype_diversity", "deep_supervision_weights", "task_weights",
        "auto_balance", "auto_balance_decay", "data_range", "mse", "pyramid",
        "statistics", "schedule", "phase_a_ratio", "phase_a", "phase_b",
        "prototype", "shift_tolerant", "transition", "interpolation",
    },
    "train": {
        "device", "hardware_profile", "resolved_hardware_profile", "epochs", "batch_size",
        "gradient_accumulation", "optimizer", "lr", "weight_decay", "warmup_epochs",
        "scheduler", "amp", "amp_dtype", "grad_clip", "ema", "ema_decay",
        "early_stopping_patience", "save_top_k", "resume", "deterministic",
        "max_oom_retries", "task_name", "stages", "stage", "freeze_encoder_epochs",
        "stage_lr_scale", "metric_finetune_epochs", "evaluate_weight_sources",
        "initial_weight_source", "prototype_monitor",
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
    for section, key in (("train", "epochs"), ("train", "gradient_accumulation")):
        value = config[section].get(key)
        if value != "auto" and int(value) < 1:
            raise ConfigError(f"{section}.{key} must be positive or 'auto'")
    for section, key in (("train", "batch_size"), ("inference", "batch_size")):
        value = config[section].get(key)
        if value != "auto" and int(value) < 1:
            raise ConfigError(f"{section}.{key} must be positive or 'auto'")
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
    phase_a_ratio = float(config["loss"].get("phase_a_ratio", 0.7))
    if not 0.0 < phase_a_ratio < 1.0:
        raise ConfigError("loss.phase_a_ratio must be in (0, 1)")
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
