"""Small interface adapters shared by engine components."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import torch
from torch import Tensor


def config_get(config: Any, path: str, default: Any = None) -> Any:
    """Read a dotted key from mappings, dataclasses, or attribute objects."""
    if config is None:
        return default
    current = config
    for component in path.split("."):
        if is_dataclass(current) and not isinstance(current, type):
            current = asdict(current)
        if isinstance(current, Mapping):
            if component not in current:
                return default
            current = current[component]
        elif hasattr(current, component):
            current = getattr(current, component)
        else:
            return default
    return current


def move_to_device(value: Any, device: torch.device) -> Any:
    """Move nested tensors to a device without touching string metadata."""
    if isinstance(value, Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def unpack_batch(batch: Any) -> tuple[Tensor, dict[str, Tensor], dict[str, Any]]:
    """Normalize common dataset batches to input, targets, and metadata."""
    if isinstance(batch, Mapping):
        input_value = next(
            (batch[key] for key in ("input", "dapi", "image", "images", "x") if key in batch),
            None,
        )
        if input_value is None or not isinstance(input_value, Tensor):
            raise TypeError("Batch mapping must contain a tensor under input/dapi/image/images/x")
        target_value = batch.get("targets")
        if isinstance(target_value, Mapping):
            targets = {
                str(key): value for key, value in target_value.items() if isinstance(value, Tensor)
            }
        else:
            single = next(
                (batch[key] for key in ("target", "label", "y") if key in batch), None
            )
            targets = {"output": single} if isinstance(single, Tensor) else {}
        excluded = {
            "input",
            "dapi",
            "image",
            "images",
            "x",
            "targets",
            "target",
            "label",
            "y",
        }
        metadata = {str(key): value for key, value in batch.items() if key not in excluded}
        return input_value, targets, metadata
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        if not isinstance(batch[0], Tensor):
            raise TypeError("The first tuple batch element must be a tensor")
        if isinstance(batch[1], Mapping):
            targets = {str(key): value for key, value in batch[1].items() if isinstance(value, Tensor)}
        elif isinstance(batch[1], Tensor):
            targets = {"output": batch[1]}
        else:
            raise TypeError("The second tuple batch element must be a tensor or target mapping")
        metadata = dict(batch[2]) if len(batch) > 2 and isinstance(batch[2], Mapping) else {}
        return batch[0], targets, metadata
    raise TypeError("Unsupported batch type")


def extract_predictions(output: Any) -> dict[str, Tensor]:
    """Extract task predictions from Tensor, mapping, or RestorationOutput."""
    if isinstance(output, Tensor):
        return {"output": output}
    predictions = getattr(output, "predictions", None)
    if isinstance(predictions, Mapping):
        return {str(key): value for key, value in predictions.items() if isinstance(value, Tensor)}
    if isinstance(output, Mapping):
        nested = output.get("predictions")
        if isinstance(nested, Mapping):
            return {str(key): value for key, value in nested.items() if isinstance(value, Tensor)}
        for key in ("prediction", "pred", "output", "logits"):
            if isinstance(output.get(key), Tensor):
                return {"output": output[key]}
        tensor_values = {str(key): value for key, value in output.items() if isinstance(value, Tensor)}
        if tensor_values:
            return tensor_values
    prediction = getattr(output, "prediction", None)
    if isinstance(prediction, Tensor):
        return {"output": prediction}
    raise TypeError(f"Cannot extract predictions from output type {type(output).__name__}")


def loss_total(
    loss_output: Any, *, keep_on_device: bool = False
) -> tuple[Tensor, dict[str, Any]]:
    """Extract a scalar total and logging components from common loss APIs.

    When ``keep_on_device`` is ``False`` (default, legacy behaviour) the returned
    component mapping holds Python floats; converting each scalar to a float
    forces a CUDA->CPU synchronization per component, which on small batches can
    starve the GPU. When ``keep_on_device`` is ``True`` the component values stay
    as detached 0-dim ``Tensor``\\s on the loss device, so the caller can
    accumulate statistics on-device and synchronize once at epoch end. The total
    tensor is always returned as-is (on-device) so the caller can run backward.
    """
    if isinstance(loss_output, Tensor):
        total = loss_output
        components: Mapping[str, Any] = {}
    else:
        total = getattr(loss_output, "total", None)
        components = getattr(loss_output, "components", {})
        if total is None and isinstance(loss_output, Mapping):
            total = loss_output.get("total", loss_output.get("loss"))
            components = loss_output.get("components", {})
        if total is None and isinstance(loss_output, (tuple, list)) and loss_output:
            total = loss_output[0]
            components = loss_output[1] if len(loss_output) > 1 else {}
        if not isinstance(total, Tensor):
            raise TypeError(
                "Loss callable must return a Tensor or an object/mapping with Tensor total"
            )
    if keep_on_device:
        numeric: dict[str, Any] = {"loss": total.detach()}
        if isinstance(components, Mapping):
            for key, value in components.items():
                if isinstance(value, Tensor) and value.numel() == 1:
                    numeric[str(key)] = value.detach()
                elif isinstance(value, (int, float)):
                    numeric[str(key)] = torch.as_tensor(
                        float(value), dtype=total.dtype, device=total.device
                    )
        return total, numeric
    numeric = {"loss": float(total.detach().cpu())}
    if isinstance(components, Mapping):
        for key, value in components.items():
            if isinstance(value, Tensor) and value.numel() == 1:
                numeric[str(key)] = float(value.detach().cpu())
            elif isinstance(value, (int, float)):
                numeric[str(key)] = float(value)
    return total, numeric


def model_kwargs_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract audited conditioning values without forwarding arbitrary metadata.

    The context dataset already exposes ``organ_id`` explicitly, while the
    ordinary manifest dataset exposes the same identity as ``organ``.  PyTorch's
    default collate keeps a batch of organ strings as a sequence of strings, a
    representation accepted by the conditioned v2 models.  Prefer an explicit
    ``organ_id`` when supplied and otherwise promote ``organ`` to that keyword.
    Legacy models remain compatible because :func:`call_model` filters keywords
    against their forward signatures.
    """

    allowed = (
        "context_tiles",
        "context_valid_mask",
        "context_offsets",
    )
    kwargs = {key: metadata[key] for key in allowed if key in metadata}
    organ_id = metadata.get("organ_id")
    if organ_id is None:
        organ_id = metadata.get("organ")
    if organ_id is not None:
        kwargs["organ_id"] = organ_id
    return kwargs


def call_model(
    model: torch.nn.Module,
    inputs: Tensor,
    task_name: str | None = None,
    *,
    model_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Call a legacy or v2 model with only explicitly supported keywords."""

    supplied = dict(model_kwargs or {})
    if task_name is not None:
        supplied["task_name"] = task_name
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        # Some wrappers (e.g. torch.compile's OptimizedModule on some versions)
        # do not expose an inspectable signature. Fall back to forwarding all
        # supplied keywords and let the callable raise if it disagrees; this
        # keeps compiled models usable without weakening the legacy path.
        return model(inputs, **supplied)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        supported = supplied
    else:
        supported = {key: value for key, value in supplied.items() if key in signature.parameters}
    return model(inputs, **supported)


def resolve_target_pairs(
    predictions: Mapping[str, Tensor], targets: Mapping[str, Tensor]
) -> list[tuple[str, Tensor, Tensor]]:
    """Match task predictions and targets, including single-task aliases."""
    common = [key for key in predictions if key in targets]
    if common:
        return [(key, predictions[key], targets[key]) for key in common]
    if len(predictions) == 1 and len(targets) == 1:
        pred_key, prediction = next(iter(predictions.items()))
        target_key, target = next(iter(targets.items()))
        display_key = target_key if pred_key in {"output", "prediction", "pred"} else pred_key
        return [(display_key, prediction, target)]
    missing = sorted(set(predictions).symmetric_difference(targets))
    raise KeyError(f"Could not match prediction and target tasks: {missing}")


def infer_batch_size(value: Any) -> int:
    """Find the leading tensor batch dimension in a nested structure."""
    if isinstance(value, Tensor) and value.ndim > 0:
        return int(value.shape[0])
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return infer_batch_size(item)
            except ValueError:
                continue
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return infer_batch_size(item)
            except ValueError:
                continue
    raise ValueError("No batched tensor found")


def slice_batch(value: Any, start: int, stop: int, batch_size: int) -> Any:
    """Slice tensors and collated metadata that share a leading batch axis."""
    if isinstance(value, Tensor):
        return value[start:stop] if value.ndim > 0 and value.shape[0] == batch_size else value
    if isinstance(value, Mapping):
        return {key: slice_batch(item, start, stop, batch_size) for key, item in value.items()}
    if isinstance(value, tuple):
        if len(value) == batch_size and not any(isinstance(item, Tensor) for item in value):
            return value[start:stop]
        return tuple(slice_batch(item, start, stop, batch_size) for item in value)
    if isinstance(value, list):
        if len(value) == batch_size and not any(isinstance(item, Tensor) for item in value):
            return value[start:stop]
        return [slice_batch(item, start, stop, batch_size) for item in value]
    return value


def metadata_item(metadata: Mapping[str, Any], key: str, index: int, default: Any = None) -> Any:
    """Read one item from default-collated metadata."""
    value = metadata.get(key, default)
    if isinstance(value, Tensor) and value.ndim > 0:
        item = value[index]
        return item.item() if item.numel() == 1 else item
    if isinstance(value, (list, tuple)):
        return value[index]
    return value
