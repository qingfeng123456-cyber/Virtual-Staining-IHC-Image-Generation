"""Model registry, shared output types, and lightweight model statistics."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class RestorationOutput:
    """Structured output shared by the baseline and multi-marker networks.

    ``deep_supervision`` is indexed by the actual output height. For square
    256-pixel inputs the expected keys are 64, 128, and 256. Attention maps
    are grouped by task and then by ``shared``/``task`` prototype bank.
    """

    predictions: dict[str, Tensor]
    deep_supervision: dict[str, dict[int, Tensor]] = field(default_factory=dict)
    prototype_attention: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    prototype_features: dict[str, Tensor] = field(default_factory=dict)
    prototype_banks: dict[str, Tensor] = field(default_factory=dict)
    base_predictions: dict[str, Tensor] = field(default_factory=dict)
    detail_predictions: dict[str, Tensor] = field(default_factory=dict)
    logits: dict[str, Tensor] = field(default_factory=dict)
    calibration_parameters: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    context_attention: dict[str, Tensor] = field(default_factory=dict)
    prototype_usage: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    intermediate_features: dict[str, Tensor] = field(default_factory=dict)
    fusion_diagnostics: dict[str, Tensor] = field(default_factory=dict)

    @property
    def prediction(self) -> Tensor:
        """Return the only prediction, raising when multiple tasks are present."""

        if len(self.predictions) != 1:
            names = ", ".join(self.predictions)
            raise ValueError(f"Expected one task prediction, found: {names}")
        return next(iter(self.predictions.values()))

    @property
    def base_prediction(self) -> Tensor:
        """Return the only low-frequency prediction."""

        return self._only_auxiliary(self.base_predictions, "base prediction")

    @property
    def detail_prediction(self) -> Tensor:
        """Return the only bounded detail component."""

        return self._only_auxiliary(self.detail_predictions, "detail prediction")

    @property
    def base(self) -> Tensor:
        """Alias used by visualization code for a single-task base image."""

        return self.base_prediction

    @property
    def detail(self) -> Tensor:
        """Alias used by visualization code for a single-task detail image."""

        return self.detail_prediction

    @staticmethod
    def _only_auxiliary(values: dict[str, Tensor], description: str) -> Tensor:
        if len(values) != 1:
            names = ", ".join(values)
            raise ValueError(f"Expected one {description}, found: {names or 'none'}")
        return next(iter(values.values()))

    def for_task(self, task_name: str) -> Tensor:
        """Return a task prediction using punctuation-insensitive matching."""

        if task_name in self.predictions:
            return self.predictions[task_name]
        normalized = _normalize_name(task_name)
        matches = [value for name, value in self.predictions.items() if _normalize_name(name) == normalized]
        if len(matches) != 1:
            available = ", ".join(self.predictions)
            raise KeyError(f"Unknown or ambiguous task {task_name!r}; available: {available}")
        return matches[0]

    def __getitem__(self, task_name: str) -> Tensor:
        return self.for_task(task_name)


ModelBuilder = Callable[..., nn.Module]
MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def _normalize_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def register_model(*names: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Register a model constructor under one or more normalized aliases."""

    if not names:
        raise ValueError("At least one model name is required")

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        for name in names:
            key = _normalize_name(name)
            existing = MODEL_REGISTRY.get(key)
            if existing is not None and existing is not builder:
                raise ValueError(f"Model alias {name!r} is already registered")
            MODEL_REGISTRY[key] = builder
        return builder

    return decorator


def _load_builtin_models() -> None:
    # Local imports keep registry types independent and avoid circular imports.
    from . import baseline_unet, camp_vs_v2, multi_marker_restorer

    _ = (baseline_unet, camp_vs_v2, multi_marker_restorer)


def available_models() -> tuple[str, ...]:
    """Return normalized names of all built-in registered models."""

    _load_builtin_models()
    return tuple(sorted(MODEL_REGISTRY))


def _config_to_mapping(config: object | None) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config) and not isinstance(config, type):
        values = asdict(config)
    elif isinstance(config, Mapping):
        values = dict(config)
    elif hasattr(config, "__dict__"):
        values = dict(vars(config))
    else:
        raise TypeError(f"Unsupported model config type: {type(config).__name__}")

    nested = values.get("model")
    if nested is not None:
        values = _config_to_mapping(nested)
    return values


def build_model(
    config: object | None = None,
    *,
    name: str | None = None,
    **overrides: Any,
) -> nn.Module:
    """Build a registered model from a mapping/dataclass/object plus overrides."""

    _load_builtin_models()
    options = _config_to_mapping(config)
    options.update(overrides)
    selected_name = name or options.pop("name", None) or "multi_marker_restorer"
    key = _normalize_name(str(selected_name))
    if key not in MODEL_REGISTRY:
        choices = ", ".join(available_models())
        raise KeyError(f"Unknown model {selected_name!r}; available: {choices}")

    aliases = {
        "input_channels": "in_channels",
        "targets": "target_names",
        "output_channels": "out_channels",
        "prototype_temperature": "prototype_temperature",
        "shared_prototypes": "shared_prototypes",
        "task_prototypes": "task_prototypes",
    }
    for source, destination in aliases.items():
        if source in options and destination not in options:
            options[destination] = options.pop(source)

    builder = MODEL_REGISTRY[key]
    signature = inspect.signature(builder)
    if "target_names" in options and "target_names" not in signature.parameters:
        target_names_value = options.pop("target_names")
        target_names = (
            (target_names_value,) if isinstance(target_names_value, str) else tuple(target_names_value)
        )
        if len(target_names) != 1:
            raise ValueError(f"{selected_name} supports exactly one target, got {target_names}")
        options.setdefault("target_name", target_names[0])
        output_channels = options.get("out_channels")
        if isinstance(output_channels, Mapping):
            options["out_channels"] = output_channels[target_names[0]]

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs:
        unsupported = sorted(set(options).difference(signature.parameters))
        if unsupported:
            known_options: set[str] = set()
            for registered_builder in set(MODEL_REGISTRY.values()):
                known_options.update(inspect.signature(registered_builder).parameters)
            unknown = sorted(set(unsupported).difference(known_options))
            if unknown:
                raise TypeError(f"Unsupported options for {selected_name}: {', '.join(unknown)}")
            for option in unsupported:
                options.pop(option)
    return builder(**options)


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Count model parameters without requiring an optional profiling package."""

    parameters = (parameter for parameter in model.parameters() if not trainable_only or parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def model_statistics(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
    *,
    device: torch.device | str | None = None,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Estimate convolution/linear MACs with a deterministic dummy forward."""

    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    macs = 0
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def convolution_hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal macs
        layer = module
        if not isinstance(layer, nn.Conv2d):
            return
        batch, out_channels, out_height, out_width = output.shape
        kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * (layer.in_channels // layer.groups)
        macs += batch * out_channels * out_height * out_width * kernel_ops

    def linear_hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal macs
        layer = module
        if not isinstance(layer, nn.Linear):
            return
        elements = output.numel() // layer.out_features
        macs += elements * layer.in_features * layer.out_features

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(convolution_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_hook))

    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(
                torch.zeros(input_shape, device=target_device),
                **dict(forward_kwargs or {}),
            )
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    return {
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "approximate_macs": macs,
    }


def iter_task_predictions(output: RestorationOutput) -> Iterator[tuple[str, Tensor]]:
    """Yield predictions in their deterministic task insertion order."""

    yield from output.predictions.items()
