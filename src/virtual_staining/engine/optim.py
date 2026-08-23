"""Feature-flagged optimizer construction for restoration training."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

_NORMALIZATION_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.LocalResponseNorm,
)
_EXPLICIT_SCALE_NAMES = {
    "beta",
    "gamma",
    "residual_scale",
    "fusion_scale",
    "layer_scale",
}


def _unique_trainable_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    values: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                values.append(parameter)
                seen.add(id(parameter))
    if not values:
        raise ValueError("Training requires at least one trainable parameter")
    return values


def _model_parameter_metadata(
    model: nn.Module,
) -> dict[int, tuple[str, nn.Module, str]]:
    metadata: dict[int, tuple[str, nn.Module, str]] = {}
    for module_name, module in model.named_modules():
        for local_name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            metadata[id(parameter)] = (full_name, module, local_name)
    return metadata


def _should_skip_weight_decay(
    parameter: nn.Parameter,
    module: nn.Module | None,
    local_name: str,
) -> bool:
    normalized = local_name.casefold()
    return (
        normalized == "bias"
        or parameter.ndim <= 1
        or normalized in _EXPLICIT_SCALE_NAMES
        or normalized.endswith("_scale")
        or isinstance(module, _NORMALIZATION_TYPES)
        or module is not None
        and module.__class__.__name__.casefold() == "layernorm2d"
    )


def build_adamw_optimizer(
    model: nn.Module,
    loss_fn: Any,
    *,
    learning_rate: float,
    weight_decay: float,
    options: Mapping[str, Any] | None = None,
) -> torch.optim.AdamW:
    """Build legacy AdamW or an opt-in no-decay/fused variant.

    The feature flag preserves exact parameter grouping for existing runs.
    Learned loss parameters, when present, are conservatively placed in the
    no-decay group.
    """

    settings = dict(options or {})
    enabled = bool(settings.get("enabled", False))
    loss_module = loss_fn if isinstance(loss_fn, nn.Module) else nn.Module()
    parameters = _unique_trainable_parameters(model, loss_module)
    groups: list[dict[str, Any]]
    no_decay_names: list[str] = []
    decay_names: list[str] = []
    if enabled and bool(settings.get("no_decay_norm_bias", True)):
        metadata = _model_parameter_metadata(model)
        loss_ids = {id(parameter) for parameter in loss_module.parameters()}
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for parameter in parameters:
            if id(parameter) in loss_ids:
                no_decay.append(parameter)
                no_decay_names.append("loss_parameter")
                continue
            name, module, local_name = metadata[id(parameter)]
            if _should_skip_weight_decay(parameter, module, local_name):
                no_decay.append(parameter)
                no_decay_names.append(name)
            else:
                decay.append(parameter)
                decay_names.append(name)
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": float(weight_decay)})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
    else:
        groups = [{"params": parameters, "weight_decay": float(weight_decay)}]

    fused_setting = settings.get("fused", False) if enabled else False
    wants_fused = (
        any(parameter.device.type == "cuda" for parameter in parameters)
        if str(fused_setting).strip().casefold() == "auto"
        else bool(fused_setting)
    )
    kwargs: dict[str, Any] = {
        "lr": float(learning_rate),
        "weight_decay": float(weight_decay),
    }
    if wants_fused:
        kwargs["fused"] = True
    fused_enabled = wants_fused
    try:
        optimizer = torch.optim.AdamW(groups, **kwargs)
    except (RuntimeError, TypeError, ValueError) as error:
        if not wants_fused:
            raise
        logging.getLogger("virtual_staining").warning(
            "Fused AdamW is unavailable (%s); falling back to standard AdamW",
            error,
        )
        kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(groups, **kwargs)
        fused_enabled = False
    report = {
        "feature_enabled": enabled,
        "fused_requested": fused_setting,
        "fused_enabled": fused_enabled,
        "parameter_count": len(parameters),
        "decay_parameter_count": len(decay_names) if enabled else len(parameters),
        "no_decay_parameter_count": len(no_decay_names),
        "group_count": len(groups),
    }
    optimizer.virtual_staining_report = report
    return optimizer
