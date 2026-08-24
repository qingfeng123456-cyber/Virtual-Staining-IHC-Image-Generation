"""Generic deterministic trainer with AMP, EMA, accumulation, and OOM recovery."""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from virtual_staining.data.transforms import apply_context_d4, apply_d4, invert_d4
from virtual_staining.utils.gpu_monitor import NvidiaSmiMonitor

from .checkpoint import save_checkpoint
from .common import (
    call_model,
    config_get,
    extract_predictions,
    infer_batch_size,
    loss_total,
    model_kwargs_from_metadata,
    move_to_device,
    resolve_target_pairs,
    slice_batch,
    unpack_batch,
)
from .ema import ExponentialMovingAverage
from .gradient_monitor import GradientCosineMonitor
from .optim import build_adamw_optimizer
from .prototype_monitor import PrototypeUsageMonitor


def set_global_seed(seed: int, *, deterministic: bool = True, benchmark: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without modifying import paths."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(benchmark)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(bool(deterministic))


def dataloader_worker_init(worker_id: int) -> None:
    """Seed NumPy/Python from the worker-specific PyTorch seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed + int(worker_id))


def _is_cuda_oom(error: BaseException) -> bool:
    text = str(error).lower()
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom_type is not None and isinstance(error, oom_type)) or (
        isinstance(error, RuntimeError) and "cuda" in text and "out of memory" in text
    )


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return resolved


def _without_validation_records(value: Any) -> Any:
    """Drop repeated per-image rows while retaining aggregate validation evidence."""

    if isinstance(value, Mapping):
        return {
            key: _without_validation_records(item)
            for key, item in value.items()
            if str(key) != "records"
        }
    if isinstance(value, list):
        return [_without_validation_records(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_validation_records(item) for item in value)
    return value


def _resolve_amp_dtype(device: torch.device, requested: str) -> torch.dtype:
    normalized = requested.lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized != "auto":
        raise ValueError(f"Unsupported AMP dtype: {requested}")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.bfloat16 if device.type == "cpu" else torch.float16


def _make_scaler(device: torch.device, enabled: bool, dtype: torch.dtype) -> Any:
    scaler_enabled = bool(enabled and device.type == "cuda" and dtype == torch.float16)
    try:
        return torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def _automatic_accumulation(device: torch.device) -> int:
    if device.type != "cuda":
        return 1
    memory_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if memory_gib <= 8.5:
        return 4
    if memory_gib <= 16.5:
        return 2
    return 1


def _build_configured_scheduler(
    optimizer: torch.optim.Optimizer, config: Any
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(config_get(config, "train.scheduler", "none")).casefold()
    if name in {"none", "null", "false"}:
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported configured scheduler: {name}")
    total_epochs = max(1, int(config_get(config, "train.epochs", 1)))
    warmup_epochs = max(0, int(config_get(config, "train.warmup_epochs", 0)))

    def multiplier(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        decay_epochs = max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (epoch - warmup_epochs) / decay_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def loss_optimizer_parameters(loss_fn: Any) -> tuple[nn.Parameter, ...]:
    """Return unique trainable parameters owned by a loss callable."""

    provider = getattr(loss_fn, "optimizer_parameters", None)
    if callable(provider):
        candidates = tuple(provider())
    elif isinstance(loss_fn, nn.Module):
        candidates = tuple(loss_fn.parameters())
    else:
        return ()
    unique: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in candidates:
        if not isinstance(parameter, nn.Parameter):
            raise TypeError("Loss optimizer_parameters() must return nn.Parameter values")
        if parameter.requires_grad and id(parameter) not in seen:
            unique.append(parameter)
            seen.add(id(parameter))
    return tuple(unique)


def collect_optimizer_parameters(model: nn.Module, loss_fn: Any = None) -> tuple[nn.Parameter, ...]:
    """Collect deduplicated trainable model and learned-loss parameters."""

    unique: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in (*tuple(model.parameters()), *loss_optimizer_parameters(loss_fn)):
        if parameter.requires_grad and id(parameter) not in seen:
            unique.append(parameter)
            seen.add(id(parameter))
    if not unique:
        raise ValueError("Training requires at least one trainable parameter")
    return tuple(unique)


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> tuple[nn.Parameter, ...]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if isinstance(parameter, nn.Parameter) and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return tuple(parameters)


class _AsyncMetricBuffer:
    """Accumulate 0-dim on-device loss tensors without per-batch CPU sync.

    The legacy training hot path converts every microbatch loss and every loss
    component to a Python float via ``float(tensor.detach().cpu())``. Each such
    conversion forces the CPU to wait for all prior CUDA kernels to complete,
    serializing CPU and GPU and starving the GPU on small batches. This buffer
    keeps the detached 0-dim tensors on their original device and performs a
    single synchronized reduction when :meth:`sync` is called (typically once per
    epoch, or every ``batch_metric_log_interval`` batches when live logging is
    requested). Numerically, the epoch-mean loss and component means match the
    legacy float-path means up to floating-point summation order.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._loss_tensors: list[Tensor] = []
        self._component_tensors: dict[str, list[Tensor]] = defaultdict(list)

    def __bool__(self) -> bool:
        return bool(self._loss_tensors)

    def __len__(self) -> int:
        return len(self._loss_tensors)

    def append(
        self, loss: Tensor, components: Mapping[str, Tensor]
    ) -> None:
        """Record one microbatch's detached 0-dim loss and component tensors."""
        detached_loss = loss.detach()
        if detached_loss.device != self._device:
            detached_loss = detached_loss.to(self._device)
        self._loss_tensors.append(detached_loss)
        for key, value in components.items():
            if not isinstance(value, Tensor) or value.numel() != 1:
                continue
            detached = value.detach()
            if detached.device != self._device:
                detached = detached.to(self._device)
            self._component_tensors[str(key)].append(detached)

    def running_mean_loss(self) -> float | None:
        """Synchronize the running mean loss for progress-bar display.

        This intentionally performs a CPU sync and should only be called when a
        live progress bar is active (i.e. an interactive terminal). It returns
        ``None`` when no batches have been recorded yet.
        """
        if not self._loss_tensors:
            return None
        stacked = torch.stack(self._loss_tensors)
        return float(stacked.mean().cpu().item())

    def sync(self) -> tuple[float, dict[str, float]]:
        """Reduce all accumulated tensors to Python floats in one sync point.

        Returns ``(mean_loss, component_means)``. The buffer is cleared after
        syncing so it can be reused for the next epoch.
        """
        if not self._loss_tensors:
            return float("nan"), {}
        loss_mean_tensor = torch.stack(self._loss_tensors).mean()
        component_mean_tensors = {
            key: torch.stack(values).mean()
            for key, values in self._component_tensors.items()
            if values
        }
        # Single synchronization point: pull all scalars to CPU together.
        loss_float = float(loss_mean_tensor.cpu().item())
        component_floats = {
            key: float(value.cpu().item())
            for key, value in component_mean_tensors.items()
        }
        self._loss_tensors.clear()
        self._component_tensors.clear()
        return loss_float, component_floats


class _AsyncScalarBuffer:
    """Accumulate detached scalar diagnostics on-device until epoch end.

    Unlike loss-component logging, this buffer is always asynchronous. This is
    important for content-adaptive fusion diagnostics: reading gate statistics
    with ``.item()`` after every forward would serialize CUDA execution even
    when legacy synchronous loss logging is selected. Detached scalars stay on
    the training device; float32 means are formed and transferred to CPU
    together once per epoch.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._values: dict[str, list[Tensor]] = defaultdict(list)

    def append(self, values: Mapping[str, Tensor]) -> None:
        """Add scalar tensors without forcing a device synchronization."""

        for key, value in values.items():
            if not isinstance(value, Tensor) or value.numel() != 1:
                continue
            detached = value.detach()
            if detached.device != self._device:
                detached = detached.to(device=self._device)
            name = str(key)
            self._values[name].append(detached)

    def sync(self) -> dict[str, float]:
        """Return epoch means after one batched device-to-host transfer."""

        names = sorted(name for name, values in self._values.items() if values)
        if not names:
            return {}
        means = torch.stack(
            [torch.stack(self._values[name]).float().mean() for name in names]
        )
        host_values = means.cpu().tolist()
        self._values.clear()
        return {name: float(value) for name, value in zip(names, host_values, strict=True)}


def _fusion_diagnostic_scalars(output: Any) -> dict[str, Tensor]:
    """Flatten scalar tensors exposed by ``output.fusion_diagnostics``.

    Mapping outputs are accepted for compatibility with small custom models.
    Non-scalar tensors are intentionally ignored: feature maps and per-sample
    gates belong in opt-in visualisation artifacts, not an epoch scalar log.
    """

    if isinstance(output, Mapping):
        diagnostics = output.get("fusion_diagnostics", {})
    else:
        diagnostics = getattr(output, "fusion_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return {}

    scalars: dict[str, Tensor] = {}

    def collect(values: Mapping[str, Any], prefix: str = "") -> None:
        for key, value in values.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                collect(value, name)
            elif isinstance(value, Tensor) and value.numel() == 1:
                scalars[name] = value.detach()

    collect(diagnostics)
    return scalars


class Trainer:
    """Train an arbitrary restoration model against a compatible loss callable."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: Any,
        loss_fn: Any,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        device: str | torch.device | None = None,
        config: Any = None,
        ema: ExponentialMovingAverage | None = None,
        scaler: Any = None,
        stage_controller: Any = None,
    ) -> None:
        self.config = config
        configured_device = config_get(config, "train.device", config_get(config, "device", "auto"))
        self.device = _resolve_device(device if device is not None else configured_device)
        self.model = model.to(self.device)
        # float32 matmul precision: "highest" is the PyTorch default and leaves
        # numerics untouched (legacy). "high"/"medium" enable TF32 inner products
        # on Ampere+ and are A/B candidates for throughput; they may slightly
        # change reduction order, so they stay opt-in and never affect the
        # default configuration.
        matmul_precision = str(
            config_get(config, "train.float32_matmul_precision", "highest")
        ).strip().casefold()
        if matmul_precision != "highest":
            torch.set_float32_matmul_precision(matmul_precision)
        self.float32_matmul_precision = matmul_precision
        # Optional torch.compile of the model forward. Disabled by default; the
        # first epochs pay compile cost so always A/B over >=5 epochs. On any
        # failure (unsupported op, signature issues, missing inductor) we fall
        # back to eager so a compile problem can never silently break training.
        compile_options = config_get(config, "train.compile", {}) or {}
        self.compile_enabled = bool(compile_options.get("enabled", False)) if isinstance(
            compile_options, Mapping
        ) else False
        self._compiled_model: nn.Module | None = None
        if self.compile_enabled:
            compile_mode = str(compile_options.get("mode", "default"))
            compile_backend = compile_options.get("backend", "inductor")
            try:
                self.model = torch.compile(
                    self.model, mode=compile_mode, backend=compile_backend
                )
                self._compiled_model = self.model
                logging.getLogger("virtual_staining").info(
                    "torch.compile enabled (mode=%s, backend=%s); first epochs pay "
                    "compile cost, evaluate throughput over >=5 epochs",
                    compile_mode,
                    compile_backend,
                )
            except Exception as error:  # pragma: no cover - depends on backend
                logging.getLogger("virtual_staining").warning(
                    "torch.compile failed (%s); falling back to eager model", error
                )
                self.compile_enabled = False
                self._compiled_model = None
        # channels_last memory format: opt-in flag for convolution-dense models
        # on Tensor Core GPUs (Ampere+ / Blackwell). It reorders physical memory
        # to NHWC so conv kernels hit the Tensor Core's 8-channel tile path,
        # yielding a notable throughput win with AMP. Default false preserves the
        # legacy NCHW layout. Applied to the model here; input tensors are
        # converted per-batch in _backward_microbatch via _to_channels_last.
        self.channels_last_enabled = bool(
            config_get(config, "train.channels_last", False)
        ) and self.device.type == "cuda"
        if self.channels_last_enabled:
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception as error:  # pragma: no cover - model-dependent
                logging.getLogger("virtual_staining").warning(
                    "channels_last not applicable to model (%s); using NCHW", error
                )
                self.channels_last_enabled = False
        self.dataloader = dataloader
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, nn.Module) else loss_fn
        learning_rate = float(config_get(config, "train.lr", 2e-4))
        weight_decay = float(config_get(config, "train.weight_decay", 1e-4))
        learned_loss_parameters = loss_optimizer_parameters(self.loss_fn)
        if optimizer is None:
            optimizer_options = config_get(config, "train.optimizer_options", {})
            self.optimizer = build_adamw_optimizer(
                self.model,
                self.loss_fn,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                options=(
                    optimizer_options
                    if isinstance(optimizer_options, Mapping)
                    else None
                ),
            )
        else:
            self.optimizer = optimizer
            existing = {id(parameter) for parameter in _optimizer_parameters(self.optimizer)}
            missing = [
                parameter
                for parameter in learned_loss_parameters
                if id(parameter) not in existing
            ]
            if missing:
                self.optimizer.add_param_group({"params": missing})
        self._optimizer_parameters = _optimizer_parameters(self.optimizer)
        self.optimizer_report = dict(
            getattr(self.optimizer, "virtual_staining_report", {})
        )
        self.scheduler = scheduler if scheduler is not None else _build_configured_scheduler(
            self.optimizer, config
        )
        accumulation_value = config_get(config, "train.gradient_accumulation", 1)
        self.gradient_accumulation = (
            _automatic_accumulation(self.device)
            if str(accumulation_value).casefold() == "auto"
            else max(1, int(accumulation_value or 1))
        )
        self.grad_clip = float(config_get(config, "train.grad_clip", 1.0) or 0.0)
        self.task_name = config_get(config, "train.task_name", config_get(config, "target", None))
        self.amp_enabled = bool(config_get(config, "train.amp", True))
        self.amp_dtype = _resolve_amp_dtype(
            self.device, str(config_get(config, "train.amp_dtype", "auto"))
        )
        if self.device.type == "cpu" and self.amp_dtype == torch.float16:
            self.amp_dtype = torch.bfloat16
        self.scaler = scaler or _make_scaler(self.device, self.amp_enabled, self.amp_dtype)
        self.stage_controller = stage_controller
        ema_enabled = bool(config_get(config, "train.ema", True))
        decay = float(config_get(config, "train.ema_decay", 0.999))
        self.ema = ema if ema is not None else (
            ExponentialMovingAverage(self.model, decay=decay) if ema_enabled else None
        )
        self.global_step = 0
        self.oom_retries = 0
        self.microbatch_factor = 1
        self._oom_cleared_accumulated_gradients = False
        self.metric_history: list[dict[str, Any]] = []
        self._top_checkpoints: list[tuple[tuple[float, float], Path]] = []
        monitor_enabled = bool(config_get(config, "train.prototype_monitor.enabled", False))
        monitor_threshold = float(
            config_get(
                config,
                "train.prototype_monitor.dead_threshold",
                config_get(config, "model.prototypes.dead_threshold", 1e-4),
            )
        )
        self.prototype_monitor = (
            PrototypeUsageMonitor(dead_threshold=monitor_threshold)
            if monitor_enabled
            else None
        )
        self.prototype_reset_enabled = bool(
            config_get(config, "model.prototypes.reset_dead", False)
        )
        self.prototype_reset_patience = int(
            config_get(config, "model.prototypes.reset_patience", 3)
        )
        self.prototype_reset_seed = int(
            config_get(
                config,
                "model.prototypes.reset_seed",
                config_get(config, "project.seed", 2026),
            )
        )
        self.prototype_reset_std = float(
            config_get(config, "model.prototypes.reset_std", 0.02)
        )
        if self.prototype_reset_enabled:
            required_methods = (
                "resolve_prototype_bank_key",
                "prototype_bank_parameters",
                "reset_prototype_rows",
            )
            missing_methods = [
                name
                for name in required_methods
                if not callable(getattr(self.model, name, None))
            ]
            if self.prototype_monitor is None:
                raise ValueError(
                    "Prototype dead-row reset requires prototype monitoring"
                )
            if missing_methods:
                raise TypeError(
                    "Prototype dead-row reset is unsupported by this model: "
                    + ", ".join(missing_methods)
                )
        gradient_interval = int(
            config_get(config, "multitask.log_gradient_cosine_every", 500)
        )
        self.gradient_monitor = GradientCosineMonitor(
            enabled=bool(
                config_get(config, "multitask.gradient_cosine_enabled", False)
            ),
            interval=gradient_interval,
        )
        self._last_gradient_monitor_step: int | None = None
        self._epoch_callbacks: list[Callable[[Mapping[str, Any]], None]] = []
        self.progress_bar_enabled = self._resolve_progress_bar_enabled(
            config_get(config, "train.progress_bar", "auto")
        )
        self.progress_bar_refresh_seconds = float(
            config_get(config, "train.progress_bar_refresh_seconds", 1.0)
        )
        # Async metric logging: when True, the hot path keeps loss/components as
        # on-device 0-dim tensors and synchronizes once per epoch (or every
        # batch_metric_log_interval batches for live logging). Default False
        # preserves the legacy per-microbatch CUDA->CPU sync behaviour.
        self.async_metric_logging = bool(
            config_get(config, "train.async_metric_logging", False)
        )
        self.batch_metric_log_interval = int(
            config_get(config, "train.batch_metric_log_interval", 0)
        )
        # Profiler trace directory is populated lazily by _maybe_start_profiler;
        # initialize it so attribute access is always safe.
        self._profiler_trace_dir: Path | None = None
        equivariance = config_get(config, "train.equivariance", {}) or {}
        if not isinstance(equivariance, Mapping):
            equivariance = {}
        self.equivariance_enabled = bool(equivariance.get("enabled", False))
        self.equivariance_probability = float(equivariance.get("probability", 0.0))
        self.equivariance_max_weight = float(equivariance.get("weight", 0.0))
        self.equivariance_start_ratio = float(equivariance.get("start_ratio", 0.10))
        self.equivariance_ramp_end_ratio = float(
            equivariance.get("ramp_end_ratio", 0.40)
        )
        self.equivariance_beta = float(equivariance.get("smooth_l1_beta", 0.01))
        self._equivariance_weight = 0.0
        gpu_monitor = config_get(config, "train.gpu_monitor", {}) or {}
        if not isinstance(gpu_monitor, Mapping):
            gpu_monitor = {}
        self.gpu_monitor_enabled = bool(gpu_monitor.get("enabled", False))
        self.gpu_monitor_interval = float(gpu_monitor.get("interval_seconds", 2.0))

    def _equivariance_weight_at(self, progress: float) -> float:
        if not self.equivariance_enabled or progress <= self.equivariance_start_ratio:
            return 0.0
        span = self.equivariance_ramp_end_ratio - self.equivariance_start_ratio
        fraction = min(1.0, max(0.0, (progress - self.equivariance_start_ratio) / span))
        return self.equivariance_max_weight * (
            0.5 - 0.5 * math.cos(math.pi * fraction)
        )

    @staticmethod
    def _transform_context_kwargs(
        model_kwargs: Mapping[str, Any], transform_id: int
    ) -> dict[str, Any]:
        transformed = dict(model_kwargs)
        tiles = transformed.get("context_tiles")
        mask = transformed.get("context_valid_mask")
        offsets = transformed.get("context_offsets")
        if not all(isinstance(value, Tensor) for value in (tiles, mask, offsets)):
            return transformed
        if tiles.ndim != 5 or mask.ndim != 2 or offsets.ndim != 3:
            raise ValueError(
                "Batched context must be [B,N,C,H,W], [B,N], and [B,N,2]"
            )
        batches = [
            apply_context_d4(
                tiles[index], mask[index], offsets[index], transform_id
            )
            for index in range(tiles.shape[0])
        ]
        transformed["context_tiles"] = torch.stack([item[0] for item in batches])
        transformed["context_valid_mask"] = torch.stack([item[1] for item in batches])
        transformed["context_offsets"] = torch.stack([item[2] for item in batches])
        return transformed

    def _equivariance_loss(
        self,
        base_predictions: Mapping[str, Tensor],
        inputs: Tensor,
        task_name: str | None,
        model_kwargs: Mapping[str, Any],
        transform_id: int,
    ) -> Tensor:
        transformed_inputs = self._to_channels_last(apply_d4(inputs, transform_id))
        transformed_kwargs = self._transform_context_kwargs(
            model_kwargs, transform_id
        )
        transformed_output = call_model(
            self.model,
            transformed_inputs,
            task_name,
            model_kwargs=transformed_kwargs,
        )
        transformed_predictions = extract_predictions(transformed_output)
        if set(transformed_predictions) != set(base_predictions):
            raise KeyError(
                "Equivariance branch changed prediction tasks: "
                f"{sorted(base_predictions)} != {sorted(transformed_predictions)}"
            )
        values = [
            F.smooth_l1_loss(
                invert_d4(transformed_predictions[name], transform_id).float(),
                base_predictions[name].float(),
                beta=self.equivariance_beta,
            )
            for name in base_predictions
        ]
        return torch.stack(values).mean()

    @staticmethod
    def _resolve_progress_bar_enabled(value: Any) -> bool:
        """Resolve the terminal progress-bar setting without polluting log files."""

        if isinstance(value, bool):
            return value
        normalized = str(value).strip().casefold()
        if normalized == "auto":
            isatty = getattr(sys.stderr, "isatty", None)
            return bool(isatty()) if callable(isatty) else False
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError("train.progress_bar must be true, false, or auto")

    def _create_progress_bar(self, *, epoch: int, total_epochs: int | None, total: int | None) -> Any:
        """Create a live batch bar only for an interactive terminal session."""

        if not self.progress_bar_enabled:
            return None
        from tqdm.auto import tqdm

        epoch_label = f"{epoch + 1}/{total_epochs}" if total_epochs is not None else str(epoch + 1)
        return tqdm(
            total=total,
            desc=f"Train {epoch_label}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=self.progress_bar_refresh_seconds,
            leave=True,
        )

    def add_epoch_callback(self, callback: Callable[[Mapping[str, Any]], None]) -> None:
        """Register a best-effort callback invoked after every completed epoch."""

        if not callable(callback):
            raise TypeError("epoch callback must be callable")
        self._epoch_callbacks.append(callback)

    def _notify_epoch_callbacks(self, entry: Mapping[str, Any]) -> None:
        """Write diagnostics without allowing a log I/O failure to stop training."""

        for callback in self._epoch_callbacks:
            try:
                callback(entry)
            except Exception:
                import logging

                logging.getLogger("virtual_staining").exception(
                    "Epoch diagnostic callback failed; training will continue"
                )

    def sync_ema_from_model(self, *, reset_num_updates: bool = True) -> bool:
        """Synchronize EMA after an initial-weight load and report whether it exists."""

        if self.ema is None:
            return False
        self.ema.sync_from(self.model, reset_num_updates=reset_num_updates)
        return True

    def prototype_reset_contract(self) -> dict[str, Any]:
        """Return the immutable feature-flagged reset settings for checkpoints."""

        return {
            "enabled": self.prototype_reset_enabled,
            "patience": self.prototype_reset_patience,
            "seed": self.prototype_reset_seed,
            "std": self.prototype_reset_std,
        }

    def _repair_reset_optimizer_and_ema(
        self, reset_rows: list[dict[str, Any]]
    ) -> None:
        banks = self.model.prototype_bank_parameters()
        if not isinstance(banks, Mapping):
            raise TypeError("prototype_bank_parameters() must return a mapping")
        parameter_names = {
            id(parameter): name for name, parameter in self.model.named_parameters()
        }
        for row in reset_rows:
            bank_key = str(row["bank_key"])
            index = int(row["prototype_index"])
            parameter = banks.get(bank_key)
            if not isinstance(parameter, nn.Parameter):
                raise KeyError(f"Reset prototype bank is unavailable: {bank_key}")
            cleared = 0
            for state_value in self.optimizer.state.get(parameter, {}).values():
                if (
                    isinstance(state_value, Tensor)
                    and state_value.ndim >= 1
                    and state_value.shape[0] == parameter.shape[0]
                ):
                    state_value[index].zero_()
                    cleared += 1
            row["optimizer_state_tensors_cleared"] = cleared
            parameter_name = parameter_names.get(id(parameter))
            if parameter_name is None:
                raise KeyError(f"Reset bank parameter has no model name: {bank_key}")
            ema_synchronized = False
            if self.ema is not None:
                if parameter_name not in self.ema.shadow:
                    raise KeyError(
                        f"EMA shadow is missing reset bank parameter: {parameter_name}"
                    )
                shadow = self.ema.shadow[parameter_name]
                shadow[index].copy_(
                    parameter.detach()[index].to(
                        device=shadow.device, dtype=shadow.dtype
                    )
                )
                ema_synchronized = True
            row["ema_synchronized"] = ema_synchronized

    def _maybe_reset_dead_prototypes(self, epoch: int) -> dict[str, Any] | None:
        if not self.prototype_reset_enabled:
            return None
        if self.prototype_monitor is None:
            raise RuntimeError("Prototype reset is enabled without a usage monitor")
        resolver = self.model.resolve_prototype_bank_key
        resetter = self.model.reset_prototype_rows
        grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        skipped: list[dict[str, Any]] = []
        for row in self.prototype_monitor.latest_rows_with_streaks():
            diagnostic = str(row["diagnostic"])
            streak = int(row["dead_streak"])
            bank_key = resolver(diagnostic)
            if bank_key is None:
                if streak >= self.prototype_reset_patience:
                    skipped.append(
                        {
                            "diagnostic": diagnostic,
                            "prototype_index": int(row["prototype_index"]),
                            "dead_streak": streak,
                            "reason": "ambiguous_or_unsupported_bank_mapping",
                        }
                    )
                continue
            grouped[str(bank_key)][int(row["prototype_index"])].append(row)

        rows_by_bank: dict[str, list[int]] = {}
        evidence: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for bank_key, by_index in grouped.items():
            for index, rows in by_index.items():
                if rows and all(
                    int(row["dead_streak"]) >= self.prototype_reset_patience
                    for row in rows
                ):
                    rows_by_bank.setdefault(bank_key, []).append(index)
                    evidence[(bank_key, index)] = rows
        if not rows_by_bank and not skipped:
            return None

        event_seed = self.prototype_reset_seed + int(epoch)
        raw_reset_rows = (
            resetter(
                rows_by_bank,
                seed=event_seed,
                std=self.prototype_reset_std,
            )
            if rows_by_bank
            else []
        )
        reset_rows: list[dict[str, Any]] = []
        for raw in raw_reset_rows:
            row = dict(raw)
            key = (str(row["bank_key"]), int(row["prototype_index"]))
            source_rows = evidence[key]
            row["diagnostics"] = sorted(
                {str(source["diagnostic"]) for source in source_rows}
            )
            row["dead_streaks"] = [
                int(source["dead_streak"]) for source in source_rows
            ]
            reset_rows.append(row)
        self._repair_reset_optimizer_and_ema(reset_rows)
        return self.prototype_monitor.record_reset_event(
            epoch=epoch,
            seed=event_seed,
            reset_rows=reset_rows,
            skipped_rows=skipped,
        )

    def _validation_weight_sources(self, validator: Any) -> tuple[str, ...]:
        configured = config_get(self.config, "train.evaluate_weight_sources", None)
        if configured is None:
            default = "ema" if bool(getattr(validator, "use_ema", False)) else "raw"
            return (default,)
        values = [configured] if isinstance(configured, str) else list(configured)
        normalized: list[str] = []
        for value in values:
            source = str(value).strip().casefold()
            if source not in {"raw", "ema"}:
                raise ValueError(
                    "train.evaluate_weight_sources accepts only raw and ema"
                )
            if source not in normalized:
                normalized.append(source)
        if not normalized:
            raise ValueError("train.evaluate_weight_sources cannot be empty")
        if "ema" in normalized and self.ema is None:
            raise ValueError("EMA validation was requested but train.ema is disabled")
        return tuple(normalized)

    @staticmethod
    def _validation_rank(validation: Mapping[str, Any]) -> tuple[float, float]:
        domains = validation.get("domains", {})
        jpg = domains.get("jpg", {}) if isinstance(domains, Mapping) else {}
        macro = jpg.get("macro", {}) if isinstance(jpg, Mapping) else {}
        if not isinstance(macro, Mapping) or "mean_ssim" not in macro:
            macro = validation.get("macro", {})
        if not isinstance(macro, Mapping):
            return (-float("inf"), -float("inf"))
        return (
            float(macro.get("mean_ssim", -float("inf"))),
            float(macro.get("mean_psnr", -float("inf"))),
        )

    def _evaluate_weight_sources(
        self,
        validator: Any,
        *,
        epoch_number: int | None = None,
        total_epochs: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
        sources = self._validation_weight_sources(validator)
        original_use_ema = bool(getattr(validator, "use_ema", False))
        original_diagnostic_source = getattr(
            validator, "prototype_diagnostics_weight_source", None
        )
        original_progress_label = getattr(validator, "progress_label", None)
        results: dict[str, dict[str, Any]] = {}
        try:
            for source_index, source in enumerate(sources, start=1):
                if source == "ema" and getattr(validator, "ema", None) is None:
                    validator.ema = self.ema
                validator.use_ema = source == "ema"
                if hasattr(validator, "prototype_diagnostics_weight_source"):
                    validator.prototype_diagnostics_weight_source = source
                if hasattr(validator, "progress_label"):
                    epoch_label = (
                        f"epoch {epoch_number}/{total_epochs}"
                        if epoch_number is not None and total_epochs is not None
                        else "training validation"
                    )
                    validator.progress_label = (
                        f"{epoch_label}, weights={source}, "
                        f"pass={source_index}/{len(sources)}"
                    )
                result = validator.evaluate()
                if not isinstance(result, Mapping):
                    raise TypeError("Validator.evaluate() must return a mapping")
                results[source] = dict(result)
        finally:
            validator.use_ema = original_use_ema
            if hasattr(validator, "prototype_diagnostics_weight_source"):
                validator.prototype_diagnostics_weight_source = (
                    original_diagnostic_source
                )
            if hasattr(validator, "progress_label"):
                validator.progress_label = original_progress_label
        selected = max(sources, key=lambda source: self._validation_rank(results[source]))
        self.selected_weight_source = selected
        return results[selected], results, selected

    def _autocast(self) -> Any:
        enabled = self.amp_enabled and self.device.type in {"cpu", "cuda"}
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=enabled,
        )

    def _to_channels_last(self, value: Any) -> Any:
        """Convert 4D float tensors to channels_last when the flag is on.

        Channels-last must be applied to BOTH the model parameters and the
        input/feature tensors for the conv kernels to take the optimized NHWC
        path. Non-4D tensors and non-floating tensors are left untouched. The
        conversion is a metadata-only stride change (no copy) when the tensor is
        already contiguous in NCHW, so the overhead is negligible.
        """
        if not self.channels_last_enabled:
            return value
        if isinstance(value, Tensor) and value.ndim == 4 and torch.is_floating_point(value):
            return value.to(memory_format=torch.channels_last)
        return value

    def _compute_loss(
        self, output: Any, targets: dict[str, Tensor], *, keep_on_device: bool = False
    ) -> tuple[Tensor, dict[str, Any], Mapping[str, Tensor]]:
        if isinstance(self.loss_fn, nn.modules.loss._Loss):
            pairs = resolve_target_pairs(extract_predictions(output), targets)
            if len(pairs) != 1:
                raise ValueError("A standard PyTorch loss can only train one output task at a time")
            _, prediction, reference = pairs[0]
            result = self.loss_fn(prediction, reference)
        elif isinstance(output, Tensor) and len(targets) == 1:
            result = self.loss_fn(output, next(iter(targets.values())))
        else:
            result = self.loss_fn(output, targets)
        total, components = loss_total(result, keep_on_device=keep_on_device)
        raw_per_task = getattr(result, "per_task", {})
        per_task = (
            {
                str(key): value
                for key, value in raw_per_task.items()
                if isinstance(value, Tensor)
            }
            if isinstance(raw_per_task, Mapping)
            else {}
        )
        return total, components, per_task

    def _shared_gradient_parameters(self) -> tuple[nn.Parameter, ...]:
        local_encoder = getattr(self.model, "local_encoder", None)
        if isinstance(local_encoder, nn.Module):
            return tuple(
                parameter for parameter in local_encoder.parameters() if parameter.requires_grad
            )
        modules = [
            module
            for name in ("stem", "encoder_stages")
            if isinstance((module := getattr(self.model, name, None)), nn.Module)
        ]
        if modules:
            seen: set[int] = set()
            parameters: list[nn.Parameter] = []
            for module in modules:
                for parameter in module.parameters():
                    if parameter.requires_grad and id(parameter) not in seen:
                        parameters.append(parameter)
                        seen.add(id(parameter))
            return tuple(parameters)
        return tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )

    def _backward_microbatch(
        self, batch: Any, *, loss_divisor: float, keep_on_device: bool = False
    ) -> tuple[Any, dict[str, Any], dict[str, Tensor]]:
        inputs, targets, metadata = unpack_batch(batch)
        inputs = move_to_device(inputs, self.device)
        inputs = self._to_channels_last(inputs)
        targets = move_to_device(targets, self.device)
        metadata = move_to_device(metadata, self.device)
        if not targets:
            raise ValueError("Training batch contains no target tensors")
        effective_task = self.task_name
        if effective_task is None and len(targets) == 1:
            effective_task = next(iter(targets))
        model_kwargs = model_kwargs_from_metadata(metadata)
        apply_equivariance = (
            self.equivariance_enabled
            and self._equivariance_weight > 0.0
            and random.random() < self.equivariance_probability
        )
        with self._autocast():
            output = call_model(
                self.model,
                inputs,
                effective_task,
                model_kwargs=model_kwargs,
            )
            fusion_diagnostics = _fusion_diagnostic_scalars(output)
            total, components, per_task = self._compute_loss(
                output, targets, keep_on_device=keep_on_device
            )
            base_predictions = (
                {
                    name: prediction.detach()
                    for name, prediction in extract_predictions(output).items()
                }
                if apply_equivariance
                else {}
            )
            if (
                len(per_task) > 1
                and self._last_gradient_monitor_step != self.global_step
            ):
                gradient_report = self.gradient_monitor.maybe_measure(
                    per_task,
                    self._shared_gradient_parameters(),
                    step=self.global_step,
                    retain_graph=True,
                )
                if gradient_report is not None:
                    self._last_gradient_monitor_step = self.global_step
                    for key, value in gradient_report.summary.items():
                        if isinstance(value, (int, float)):
                            components[f"gradient_cosine/{key}"] = float(value)
        # Finiteness guard: this is the one intentional sync on the hot path. It
        # protects the optimizer from NaN/Inf losses propagating into gradients.
        # We keep it even in async mode so a single bad microbatch is caught
        # before backward corrupts model state; the cost is one sync per
        # microbatch, which is far smaller than the per-component syncs removed
        # by keep_on_device.
        if total.numel() != 1 or not torch.isfinite(total.detach()).all():
            raise FloatingPointError(f"Training loss must be one finite scalar, got {total}")
        scaled_loss = total / float(loss_divisor)
        if self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        equivariance_value = total.detach().new_zeros(())
        if apply_equivariance:
            transform_id = random.randint(1, 7)
            with self._autocast():
                equivariance_value = self._equivariance_loss(
                    base_predictions,
                    inputs,
                    effective_task,
                    model_kwargs,
                    transform_id,
                )
            if not torch.isfinite(equivariance_value.detach()).all():
                raise FloatingPointError(
                    "Equivariance consistency loss must be finite"
                )
            scaled_equivariance = (
                self._equivariance_weight * equivariance_value / float(loss_divisor)
            )
            if self.scaler.is_enabled():
                self.scaler.scale(scaled_equivariance).backward()
            else:
                scaled_equivariance.backward()
        weighted_equivariance = self._equivariance_weight * equivariance_value.detach()
        logged_total = total.detach() + weighted_equivariance
        if keep_on_device:
            components["equivariance/raw"] = equivariance_value.detach()
            components["equivariance/weighted"] = weighted_equivariance
            components["equivariance/applied"] = total.detach().new_tensor(
                float(apply_equivariance)
            )
            components["equivariance/weight"] = total.detach().new_tensor(
                self._equivariance_weight
            )
        else:
            components["equivariance/raw"] = float(equivariance_value.detach().cpu())
            components["equivariance/weighted"] = float(
                weighted_equivariance.detach().cpu()
            )
            components["equivariance/applied"] = float(apply_equivariance)
            components["equivariance/weight"] = self._equivariance_weight
        if self.prototype_monitor is not None:
            self.prototype_monitor.observe(output)
        if keep_on_device:
            # Return detached 0-dim on-device tensors; the caller accumulates
            # them and synchronizes once per epoch (or every N batches).
            return logged_total, components, fusion_diagnostics
        return float(logged_total.cpu()), components, fusion_diagnostics

    def _run_batch_with_oom_recovery(
        self, batch: Any, *, keep_on_device: bool = False
    ) -> tuple[Any, dict[str, Any], dict[str, Tensor]]:
        batch_size = infer_batch_size(batch)
        while True:
            chunk_size = max(1, math.ceil(batch_size / self.microbatch_factor))
            chunks = [
                slice_batch(batch, start, min(start + chunk_size, batch_size), batch_size)
                for start in range(0, batch_size, chunk_size)
            ]
            try:
                chunk_sizes = [infer_batch_size(chunk) for chunk in chunks]
                chunk_weights = [size / float(batch_size) for size in chunk_sizes]
                if keep_on_device:
                    loss_tensors: list[Tensor] = []
                    component_lists: dict[str, list[Tensor]] = defaultdict(list)
                    diagnostic_lists: dict[str, list[Tensor]] = defaultdict(list)
                    diagnostic_weights: dict[str, list[float]] = defaultdict(list)
                    for chunk, size, weight in zip(
                        chunks, chunk_sizes, chunk_weights, strict=True
                    ):
                        divisor = float(
                            self.gradient_accumulation * batch_size / size
                        )
                        value, components, diagnostics = self._backward_microbatch(
                            chunk, loss_divisor=divisor, keep_on_device=True
                        )
                        loss_tensors.append(value)
                        for key, component in components.items():
                            component_lists[str(key)].append(component)
                        for key, diagnostic in diagnostics.items():
                            diagnostic_lists[str(key)].append(diagnostic.detach())
                            diagnostic_weights[str(key)].append(weight)
                    weight_tensor = torch.as_tensor(
                        chunk_weights,
                        device=loss_tensors[0].device,
                        dtype=loss_tensors[0].dtype,
                    )
                    loss_mean = torch.sum(torch.stack(loss_tensors) * weight_tensor)
                    component_means: dict[str, Any] = {
                        key: torch.sum(torch.stack(values) * weight_tensor)
                        for key, values in component_lists.items()
                        if values
                    }
                    if len(chunks) == 1:
                        diagnostic_means = {
                            key: values[0] for key, values in diagnostic_lists.items() if values
                        }
                    else:
                        diagnostic_means = {
                            key: torch.sum(
                                torch.stack(values).float()
                                * torch.as_tensor(
                                    diagnostic_weights[key],
                                    device=values[0].device,
                                    dtype=torch.float32,
                                )
                            )
                            / float(sum(diagnostic_weights[key]))
                            for key, values in diagnostic_lists.items()
                            if values
                        }
                    return loss_mean, component_means, diagnostic_means
                losses: list[float] = []
                component_sums: dict[str, float] = defaultdict(float)
                diagnostic_lists: dict[str, list[Tensor]] = defaultdict(list)
                diagnostic_weights: dict[str, list[float]] = defaultdict(list)
                for chunk, size, weight in zip(
                    chunks, chunk_sizes, chunk_weights, strict=True
                ):
                    divisor = float(self.gradient_accumulation * batch_size / size)
                    value, components, diagnostics = self._backward_microbatch(
                        chunk, loss_divisor=divisor
                    )
                    losses.append(value * weight)
                    for key, component in components.items():
                        component_sums[key] += component * weight
                    for key, diagnostic in diagnostics.items():
                        diagnostic_lists[str(key)].append(diagnostic.detach())
                        diagnostic_weights[str(key)].append(weight)
                if len(chunks) == 1:
                    diagnostic_means = {
                        key: values[0] for key, values in diagnostic_lists.items() if values
                    }
                else:
                    diagnostic_means = {
                        key: torch.sum(
                            torch.stack(values).float()
                            * torch.as_tensor(
                                diagnostic_weights[key],
                                device=values[0].device,
                                dtype=torch.float32,
                            )
                        )
                        / float(sum(diagnostic_weights[key]))
                        for key, values in diagnostic_lists.items()
                        if values
                    }
                return float(np.sum(losses)), dict(component_sums), diagnostic_means
            except BaseException as error:
                if self.device.type != "cuda" or not _is_cuda_oom(error):
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                # A failed microbatch may already have contributed gradients,
                # and earlier DataLoader batches may have been accumulating.
                # Retry the complete batch from a clean state and tell the
                # epoch loop to restart its accumulation window.  Splitting a
                # DataLoader batch must not alter the configured effective
                # batch size.
                self._oom_cleared_accumulated_gradients = True
                torch.cuda.empty_cache()
                self.oom_retries += 1
                if self.oom_retries > 3 or chunk_size == 1:
                    raise RuntimeError(
                        "CUDA OOM persisted after at most three batch-size reductions"
                    ) from error
                self.microbatch_factor *= 2

    def _optimizer_step(self) -> None:
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._optimizer_parameters, self.grad_clip)
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema is not None:
            self.ema.update(self.model)
        self.global_step += 1

    def _maybe_start_profiler(self, epoch: int) -> Any:
        """Start a torch.profiler trace over the first epoch's first N batches.

        Returns an entered profiler object, or ``None`` when profiling is
        disabled, already consumed, or unavailable. The profiler is only ever
        started on the very first epoch (``global_step == 0``) so resumed or
        later epochs never pay tracing cost.
        """
        profiler_cfg = config_get(self.config, "train.profiler", {}) or {}
        if not isinstance(profiler_cfg, Mapping) or not bool(
            profiler_cfg.get("enabled", False)
        ):
            return None
        # Only profile the first epoch ever trained by this trainer instance.
        if self.global_step != 0:
            return None
        max_steps = int(profiler_cfg.get("max_steps", 30))
        if max_steps < 1:
            return None
        record_shapes = bool(profiler_cfg.get("record_shapes", False))
        with_stack = bool(profiler_cfg.get("with_stack", False))
        trace_dir_value = profiler_cfg.get("output_dir")
        if trace_dir_value:
            trace_dir = Path(str(trace_dir_value)).expanduser().resolve()
        else:
            trace_dir = Path(".") / "log" / "profiler"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._profiler_trace_dir = trace_dir
        try:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=0, warmup=0, active=max_steps, repeat=1
                ),
                record_shapes=record_shapes,
                with_stack=with_stack,
            )
            profiler.__enter__()
            logging.getLogger("virtual_staining").info(
                "torch.profiler started: tracing first %d batches of epoch %d to %s",
                max_steps,
                epoch + 1,
                trace_dir,
            )
            return profiler
        except Exception as error:  # pragma: no cover - depends on torch build
            logging.getLogger("virtual_staining").warning(
                "torch.profiler could not be started (%s); training continues "
                "without profiling",
                error,
            )
            return None

    def _finalize_profiler(self, profiler: Any, epoch: int) -> None:
        """Stop the profiler and export a Chrome trace; never raise into training."""
        try:
            profiler.__exit__(None, None, None)
            trace_path = self._profiler_trace_dir / f"profiler_epoch{epoch + 1:04d}.json"
            profiler.export_chrome_trace(str(trace_path))
            logging.getLogger("virtual_staining").info(
                "torch.profiler trace written to %s", trace_path
            )
        except Exception as error:  # pragma: no cover - depends on torch build
            logging.getLogger("virtual_staining").warning(
                "torch.profiler finalization failed (%s); training continues", error
            )

    def train_epoch(
        self, epoch: int = 0, *, total_epochs: int | None = None
    ) -> dict[str, float | int]:
        """Run one complete training epoch and return real measured statistics."""
        activity_plan = getattr(self, "activity_sampling_plan", None)
        set_sampler_epoch = getattr(activity_plan, "set_epoch", None)
        if callable(set_sampler_epoch):
            set_sampler_epoch(int(epoch))
        schedule_progress = (
            float(epoch) / max(1, int(total_epochs or epoch + 1) - 1)
        )
        self._equivariance_weight = self._equivariance_weight_at(schedule_progress)
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if self.prototype_monitor is not None:
            self.prototype_monitor.start_epoch(epoch)
        self.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        losses: list[float] = []
        components: dict[str, list[float]] = defaultdict(list)
        async_buffer: _AsyncMetricBuffer | None = (
            _AsyncMetricBuffer(self.device) if self.async_metric_logging else None
        )
        fusion_buffer = _AsyncScalarBuffer(self.device)
        batch_count = len(self.dataloader) if hasattr(self.dataloader, "__len__") else None
        seen_batches = 0
        seen_samples = 0
        accumulated_batches = 0
        progress = self._create_progress_bar(
            epoch=epoch, total_epochs=total_epochs, total=batch_count
        )
        # Optional torch.profiler over the first N batches of the first epoch
        # only. Disabled by default so production training never pays profiling
        # overhead; when enabled it records CPU/CUDA time, DataLoader waits and
        # memory, then writes a Chrome trace for offline analysis on AutoDL.
        profiler = self._maybe_start_profiler(epoch)
        gpu_monitor = NvidiaSmiMonitor(
            enabled=self.gpu_monitor_enabled and self.device.type == "cuda",
            interval_seconds=self.gpu_monitor_interval,
            device_index=int(self.device.index or 0),
        )
        gpu_monitor.start()
        epoch_loop_completed = False
        try:
            for batch_index, batch in enumerate(self.dataloader):
                batch_samples = infer_batch_size(batch)
                seen_samples += batch_samples
                if async_buffer is not None:
                    (
                        loss_tensor,
                        component_tensors,
                        fusion_diagnostics,
                    ) = self._run_batch_with_oom_recovery(batch, keep_on_device=True)
                    async_buffer.append(loss_tensor, component_tensors)
                    loss_value: float | None = None
                else:
                    (
                        loss_value,
                        logged,
                        fusion_diagnostics,
                    ) = self._run_batch_with_oom_recovery(batch)
                    losses.append(loss_value)
                    for key, value in logged.items():
                        components[key].append(float(value))
                fusion_buffer.append(fusion_diagnostics)
                seen_batches += 1
                if self._oom_cleared_accumulated_gradients:
                    accumulated_batches = 0
                    self._oom_cleared_accumulated_gradients = False
                accumulated_batches += 1
                is_last = batch_count is not None and batch_index + 1 == batch_count
                if accumulated_batches >= self.gradient_accumulation or is_last:
                    self._optimizer_step()
                    accumulated_batches = 0
                if profiler is not None:
                    profiler.step()
                if progress is not None:
                    elapsed = max(time.perf_counter() - started, 1e-12)
                    postfix: dict[str, str] = {
                        "img/s": f"{seen_samples / elapsed:.2f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    }
                    if async_buffer is not None:
                        # Only pay for a CUDA->CPU sync when the user explicitly
                        # requested live interval logging; otherwise keep loss
                        # on-device until epoch end.
                        if (
                            self.batch_metric_log_interval > 0
                            and seen_batches % self.batch_metric_log_interval == 0
                        ):
                            running_loss = async_buffer.running_mean_loss()
                            postfix["loss"] = (
                                f"{running_loss:.4f}" if running_loss is not None else "n/a"
                            )
                        else:
                            postfix["loss"] = "async"
                    else:
                        postfix["loss"] = f"{loss_value:.4f}"
                    if self.device.type == "cuda":
                        postfix["vram"] = (
                            f"{torch.cuda.max_memory_allocated(self.device) / 1024**2:.0f}MiB"
                        )
                    progress.set_postfix(postfix, refresh=False)
                    progress.update(1)
            epoch_loop_completed = True
        finally:
            if progress is not None:
                progress.close()
            if profiler is not None:
                self._finalize_profiler(profiler, epoch)
            if not epoch_loop_completed:
                gpu_monitor.stop()
        if seen_batches == 0:
            raise ValueError("Training dataloader produced no batches")
        if batch_count is None and accumulated_batches:
            self._optimizer_step()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        gpu_monitor_summary = gpu_monitor.stop()
        duration_seconds = float(time.perf_counter() - started)
        if async_buffer is not None:
            mean_loss, component_means = async_buffer.sync()
            epoch_loss = mean_loss
        else:
            epoch_loss = float(np.mean(losses))
            component_means = {
                key: float(np.mean(values)) for key, values in components.items()
            }
        fusion_means = fusion_buffer.sync()
        result: dict[str, float | int] = {
            "epoch": int(epoch),
            "epoch_number": int(epoch) + 1,
            "total_epochs": int(total_epochs) if total_epochs is not None else int(epoch) + 1,
            "loss": epoch_loss,
            "duration_seconds": duration_seconds,
            "seen_samples": seen_samples,
            "seen_batches": seen_batches,
            "images_per_second": float(seen_samples / duration_seconds),
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            ),
            "peak_vram_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(self.device))
                if self.device.type == "cuda"
                else 0
            ),
            "process_rss_bytes": int(psutil.Process().memory_info().rss),
            "system_ram_available_bytes": int(psutil.virtual_memory().available),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "global_step": self.global_step,
            "oom_retries": self.oom_retries,
            "effective_microbatch_factor": self.microbatch_factor,
            "gradient_accumulation": self.gradient_accumulation,
            "async_metric_logging": bool(self.async_metric_logging),
            "float32_matmul_precision": self.float32_matmul_precision,
            "torch_compile_enabled": bool(self.compile_enabled),
            "channels_last_enabled": bool(self.channels_last_enabled),
            "optimizer/fused_enabled": bool(
                self.optimizer_report.get("fused_enabled", False)
            ),
            "optimizer/group_count": int(
                self.optimizer_report.get("group_count", len(self.optimizer.param_groups))
            ),
        }
        for key, value in component_means.items():
            result[f"loss/{key}"] = value
        for key, value in fusion_means.items():
            result[f"fusion/{key}"] = value
        result.update(gpu_monitor_summary)
        if self.prototype_monitor is not None:
            prototype_summary = self.prototype_monitor.finalize_epoch(epoch)
            if prototype_summary["total_prototypes"]:
                for key in (
                    "observed_outputs",
                    "diagnostic_count",
                    "total_prototypes",
                    "dead_prototypes",
                    "dead_fraction",
                    "mean_entropy",
                    "nonfinite_values",
                ):
                    result[f"prototype/{key}"] = prototype_summary[key]
            reset_event = self._maybe_reset_dead_prototypes(epoch)
            if reset_event is not None:
                reset_rows = reset_event["reset_rows"]
                skipped_rows = reset_event["skipped_rows"]
                result["prototype/reset_rows"] = len(reset_rows)
                result["prototype/reset_banks"] = len(
                    {str(row["bank_key"]) for row in reset_rows}
                )
                result["prototype/reset_skipped_rows"] = len(skipped_rows)
        return result

    def fit(
        self,
        *,
        epochs: int | None = None,
        start_epoch: int = 0,
        validator: Any = None,
        checkpoint_dir: str | Path | None = None,
        manifest_hash: str | None = None,
        image_spec: Any = None,
        targets: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Train several epochs, validate, and save best/last checkpoints."""
        total_epochs = int(epochs if epochs is not None else config_get(self.config, "train.epochs", 1))
        if total_epochs <= start_epoch:
            raise ValueError("epochs must be greater than start_epoch")
        logging.getLogger("virtual_staining").info(
            "Starting supervised training: epoch %d through %d, batch progress bar=%s",
            start_epoch + 1,
            total_epochs,
            "enabled" if self.progress_bar_enabled else "disabled (non-interactive terminal or config)",
        )
        output_dir = Path(checkpoint_dir).resolve() if checkpoint_dir is not None else None
        fit_started = time.perf_counter()
        max_wall_time_hours = float(
            config_get(self.config, "train.max_wall_time_hours", 0.0) or 0.0
        )
        time_budget_seconds = max_wall_time_hours * 3600.0
        completed_epoch_durations: list[float] = []
        activity_plan = getattr(self, "activity_sampling_plan", None)
        if self.prototype_monitor is not None and output_dir is not None:
            self.prototype_monitor.bind_output_dir(output_dir.parent / "artifacts")
        if output_dir is not None:
            self.gradient_monitor.bind_output_dir(output_dir.parent / "artifacts")
        best = {"ssim": -float("inf"), "psnr": -float("inf"), "proxy": -float("inf")}
        epochs_without_improvement = 0
        patience = max(0, int(config_get(self.config, "train.early_stopping_patience", 0) or 0))
        save_top_k = max(0, int(config_get(self.config, "train.save_top_k", 3) or 0))
        checkpoint_metrics = {
            str(value).strip().casefold()
            for value in config_get(
                self.config,
                "train.checkpoint_metrics",
                ["ssim", "psnr", "proxy"],
            )
        }
        # Validate every N epochs. Default 1 preserves the legacy behaviour of
        # validating every epoch. Screen runs may set a larger value to skip
        # validation on intermediate epochs; skipped epochs never update best
        # checkpoints (they are recorded as unvalidated) and the final epoch is
        # always validated when a validator is provided.
        validate_every = max(
            1, int(config_get(self.config, "train.validate_every_n_epochs", 1) or 1)
        )
        retain_validation_records = bool(
            config_get(
                self.config,
                "train.retain_validation_records_in_history",
                True,
            )
        )
        if output_dir is not None and start_epoch == 0:
            top_directory = output_dir / "top_k"
            if top_directory.is_dir():
                for stale_checkpoint in top_directory.glob("*.ckpt"):
                    stale_checkpoint.unlink()
        # Resume must retain the historical best scores and early-stopping
        # streak. Otherwise the first resumed epoch can overwrite a genuinely
        # better checkpoint merely because the in-memory sentinels were reset.
        historical_best_ssim = -float("inf")
        for item in self.metric_history:
            macro = item.get("validation", {}).get("macro", {})
            # Skip epochs that were not validated (e.g. validate_every_n_epochs
            # skip-epochs) or that had no validator: they carry no SSIM signal
            # and must not inflate the early-stopping streak.
            if not macro:
                continue
            values = {
                "ssim": float(macro.get("mean_ssim", -float("inf"))),
                "psnr": float(macro.get("mean_psnr", -float("inf"))),
                "proxy": float(macro.get("local_proxy_score", -float("inf"))),
            }
            for metric, value in values.items():
                best[metric] = max(best[metric], value)
            if values["ssim"] > historical_best_ssim:
                historical_best_ssim = values["ssim"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        if output_dir is not None and save_top_k:
            for item in self.metric_history:
                macro = item.get("validation", {}).get("macro", {})
                train = item.get("train", {})
                if "mean_ssim" not in macro or "epoch" not in train:
                    continue
                epoch_number = int(train["epoch"])
                ssim = float(macro["mean_ssim"])
                psnr = float(macro.get("mean_psnr", -float("inf")))
                path = output_dir / "top_k" / f"epoch_{epoch_number:04d}_ssim_{ssim:.6f}.ckpt"
                if path.is_file():
                    self._top_checkpoints.append(((ssim, psnr), path))
            self._top_checkpoints.sort(key=lambda item: item[0], reverse=True)
            self._top_checkpoints = self._top_checkpoints[:save_top_k]
        for epoch in range(start_epoch, total_epochs):
            epoch_wall_started = time.perf_counter()
            set_progress = getattr(self.loss_fn, "set_progress", None)
            if callable(set_progress):
                set_progress(epoch=epoch, total_epochs=total_epochs)
            train_metrics = self.train_epoch(epoch, total_epochs=total_epochs)
            if self.stage_controller is not None:
                self.stage_controller.advance_epoch()
            entry: dict[str, Any] = {"train": train_metrics}
            # A validation epoch is one where we actually run the validator. The
            # last epoch is always validated (when a validator exists) so a
            # final candidate is always produced; intermediate epochs honour
            # validate_every_n_epochs to trade validation cost for screen speed.
            is_validation_epoch = validator is not None and (
                (epoch + 1) % validate_every == 0 or epoch == total_epochs - 1
            )
            if validator is not None and is_validation_epoch:
                if hasattr(validator, "prototype_diagnostics_epoch"):
                    validator.prototype_diagnostics_epoch = int(epoch)
                validation_start = time.perf_counter()
                validation, source_results, selected_source = (
                    self._evaluate_weight_sources(
                        validator,
                        epoch_number=epoch + 1,
                        total_epochs=total_epochs,
                    )
                )
                entry["validation_duration_seconds"] = float(
                    time.perf_counter() - validation_start
                )
                entry["validation"] = (
                    validation
                    if retain_validation_records
                    else _without_validation_records(validation)
                )
                entry["validation_weight_sources"] = (
                    source_results
                    if retain_validation_records
                    else _without_validation_records(source_results)
                )
                entry["selected_weight_source"] = selected_source
                entry["validated"] = True
                macro = validation.get("macro", {})
                current = {
                    "ssim": float(macro.get("mean_ssim", -float("inf"))),
                    "psnr": float(macro.get("mean_psnr", -float("inf"))),
                    "proxy": float(macro.get("local_proxy_score", -float("inf"))),
                }
            else:
                # Either no validator at all, or a skip-epoch: do not touch best.
                current = best.copy()
                if validator is not None:
                    entry["validated"] = False
                    entry["validation_skipped"] = True
                    entry["validation_every_n_epochs"] = validate_every
            self.metric_history.append(entry)
            previous_best_ssim = -float("inf")
            if validator is not None and is_validation_epoch:
                previous_best_ssim = max(
                    (
                        float(
                            item.get("validation", {})
                            .get("macro", {})
                            .get("mean_ssim", -float("inf"))
                        )
                        for item in self.metric_history[:-1]
                        if item.get("validated", False)
                    ),
                    default=-float("inf"),
                )
                if current["ssim"] > previous_best_ssim:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            early_stopping_triggered = bool(
                validator is not None
                and is_validation_epoch
                and patience
                and epochs_without_improvement >= patience
                and epoch < total_epochs - 1
            )
            entry["validation_checks_without_improvement"] = (
                epochs_without_improvement
            )
            entry["early_stopping_patience"] = patience
            if validator is None:
                logging.getLogger("virtual_staining").info(
                    "Epoch %d/%d complete | loss=%.6f | %.3f img/s",
                    epoch + 1,
                    total_epochs,
                    float(train_metrics["loss"]),
                    float(train_metrics["images_per_second"]),
                )
            elif is_validation_epoch:
                logging.getLogger("virtual_staining").info(
                    "Epoch %d/%d complete | loss=%.6f | %.3f img/s | val_ssim=%.6f | val_psnr=%.4f",
                    epoch + 1,
                    total_epochs,
                    float(train_metrics["loss"]),
                    float(train_metrics["images_per_second"]),
                    current["ssim"],
                    current["psnr"],
                )
            else:
                logging.getLogger("virtual_staining").info(
                    "Epoch %d/%d complete | loss=%.6f | %.3f img/s | validation=skipped (every=%d)",
                    epoch + 1,
                    total_epochs,
                    float(train_metrics["loss"]),
                    float(train_metrics["images_per_second"]),
                    validate_every,
                )
            epoch_wall_seconds = float(time.perf_counter() - epoch_wall_started)
            completed_epoch_durations.append(epoch_wall_seconds)
            fit_elapsed_seconds = float(time.perf_counter() - fit_started)
            recent = completed_epoch_durations[-3:]
            estimated_next_epoch_seconds = float(np.median(recent))
            epochs_until_next_safe_stop = validate_every if validator is not None else 1
            estimated_next_safe_stop_seconds = (
                estimated_next_epoch_seconds * epochs_until_next_safe_stop
            )
            time_budget_stop = bool(
                time_budget_seconds > 0.0
                and epoch < total_epochs - 1
                and (validator is None or is_validation_epoch)
                and fit_elapsed_seconds + estimated_next_safe_stop_seconds
                >= time_budget_seconds
            )
            entry["fit_elapsed_seconds"] = fit_elapsed_seconds
            entry["epoch_wall_seconds"] = epoch_wall_seconds
            entry["time_budget_seconds"] = time_budget_seconds
            entry["estimated_next_epoch_seconds"] = estimated_next_epoch_seconds
            entry["estimated_next_safe_stop_seconds"] = (
                estimated_next_safe_stop_seconds
            )
            entry["stopped_by_time_budget"] = time_budget_stop
            entry["stopped_by_early_stopping"] = bool(
                early_stopping_triggered and not time_budget_stop
            )
            if time_budget_stop:
                entry["stop_reason"] = "time_budget"
            elif early_stopping_triggered:
                entry["stop_reason"] = "early_stopping"
            elif epoch == total_epochs - 1:
                entry["stop_reason"] = "completed_epochs"
            else:
                entry["stop_reason"] = None
            self._notify_epoch_callbacks(entry)
            if output_dir is not None:
                checkpoint_start = time.perf_counter()
                output_dir.mkdir(parents=True, exist_ok=True)
                save_kwargs = {
                    "ema": self.ema,
                    "optimizer": self.optimizer,
                    "scheduler": self.scheduler,
                    "scaler": self.scaler,
                    "loss_fn": self.loss_fn,
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "config": self.config,
                    "manifest_hash": manifest_hash,
                    "image_spec": image_spec,
                    "targets": targets,
                    "metric_history": self.metric_history,
                    "dataloader_generator_state": (
                        self.dataloader.generator.get_state()
                        if isinstance(
                            getattr(self.dataloader, "generator", None), torch.Generator
                        )
                        else None
                    ),
                    "extra": {
                        **(
                            {"multistage": self.stage_controller.state_dict()}
                            if self.stage_controller is not None
                            else {}
                        ),
                        **(
                            {"prototype_monitor": self.prototype_monitor.state_dict()}
                            if self.prototype_monitor is not None
                            else {}
                        ),
                        **(
                            {"prototype_reset": self.prototype_reset_contract()}
                            if self.prototype_reset_enabled
                            else {}
                        ),
                        **(
                            {"activity_sampler": activity_plan.state_dict()}
                            if activity_plan is not None
                            else {}
                        ),
                        **(
                            {"fold_provenance": self.fold_provenance}
                            if getattr(self, "fold_provenance", None) is not None
                            else {}
                        ),
                        **(
                            {"activity_report": self.activity_report}
                            if getattr(self, "activity_report", None) is not None
                            else {}
                        ),
                        **(
                            {
                                "resolved_manifests": {
                                    "train": str(self.training_manifest_path),
                                    "validation": str(self.validation_manifest_path),
                                    "train_hash": self.training_manifest_hash,
                                    "validation_hash": self.validation_manifest_hash,
                                    "dapi_pretrain": str(
                                        self.dapi_pretrain_manifest_path
                                    ),
                                    "dapi_pretrain_hash": (
                                        self.dapi_pretrain_manifest_hash
                                    ),
                                }
                            }
                            if hasattr(self, "training_manifest_path")
                            else {}
                        ),
                        **(
                            {"pretrain_transfer": self.pretrain_transfer_report}
                            if getattr(self, "pretrain_transfer_report", None)
                            is not None
                            else {}
                        ),
                        **(
                            {"selected_weight_source": self.selected_weight_source}
                            if getattr(self, "selected_weight_source", None) is not None
                            else {}
                        ),
                        **(
                            {
                                "gradient_cosine_monitor": {
                                    "enabled": self.gradient_monitor.enabled,
                                    "interval": self.gradient_monitor.interval,
                                    "last_step": self._last_gradient_monitor_step,
                                    "report_count": len(self.gradient_monitor.history),
                                }
                            }
                            if self.gradient_monitor.enabled
                            else {}
                        ),
                    }
                    or None,
                }
                save_checkpoint(output_dir / "last.ckpt", self.model, **save_kwargs)
                # Best checkpoints and top-k may only be updated on epochs that
                # were actually validated. Skipped epochs must never overwrite a
                # genuine best with an unvalidated sentinel, and must never pose
                # as a top-k candidate.
                if is_validation_epoch:
                    for metric, filename in (
                        ("ssim", "best_ssim.ckpt"),
                        ("psnr", "best_psnr.ckpt"),
                        ("proxy", "best_proxy.ckpt"),
                    ):
                        if (
                            metric in checkpoint_metrics
                            and current[metric] > best[metric]
                        ):
                            best[metric] = current[metric]
                            save_checkpoint(output_dir / filename, self.model, **save_kwargs)
                    if save_top_k:
                        rank = (current["ssim"], current["psnr"])
                        qualifies = (
                            len(self._top_checkpoints) < save_top_k
                            or rank > self._top_checkpoints[-1][0]
                        )
                        if qualifies:
                            top_path = output_dir / "top_k" / (
                                f"epoch_{epoch:04d}_ssim_{current['ssim']:.6f}.ckpt"
                            )
                            save_checkpoint(top_path, self.model, **save_kwargs)
                            self._top_checkpoints.append((rank, top_path))
                            self._top_checkpoints.sort(key=lambda item: item[0], reverse=True)
                            while len(self._top_checkpoints) > save_top_k:
                                _, stale = self._top_checkpoints.pop()
                                stale.unlink(missing_ok=True)
                entry["checkpoint_duration_seconds"] = float(
                    time.perf_counter() - checkpoint_start
                )
            if time_budget_stop:
                logging.getLogger("virtual_staining").warning(
                    "Stopping after epoch %d to respect the %.2f-hour training "
                    "budget (elapsed %.2f hours; next safe validation/checkpoint "
                    "estimate %.1f seconds)",
                    epoch + 1,
                    max_wall_time_hours,
                    float(time.perf_counter() - fit_started) / 3600.0,
                    estimated_next_safe_stop_seconds,
                )
                break
            if early_stopping_triggered:
                logging.getLogger("virtual_staining").warning(
                    "Early stopping after epoch %d/%d: validation SSIM did not "
                    "improve for %d consecutive validation checks "
                    "(patience=%d, validate_every=%d, best=%.6f, current=%.6f)",
                    epoch + 1,
                    total_epochs,
                    epochs_without_improvement,
                    patience,
                    validate_every,
                    max(previous_best_ssim, current["ssim"]),
                    current["ssim"],
                )
                break
        return self.metric_history
