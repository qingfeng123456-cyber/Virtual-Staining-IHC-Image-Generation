"""Generic deterministic trainer with AMP, EMA, accumulation, and OOM recovery."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

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
        self.dataloader = dataloader
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, nn.Module) else loss_fn
        learning_rate = float(config_get(config, "train.lr", 2e-4))
        weight_decay = float(config_get(config, "train.weight_decay", 1e-4))
        learned_loss_parameters = loss_optimizer_parameters(self.loss_fn)
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                collect_optimizer_parameters(self.model, self.loss_fn),
                lr=learning_rate,
                weight_decay=weight_decay,
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
        self, validator: Any
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
        sources = self._validation_weight_sources(validator)
        original_use_ema = bool(getattr(validator, "use_ema", False))
        original_diagnostic_source = getattr(
            validator, "prototype_diagnostics_weight_source", None
        )
        results: dict[str, dict[str, Any]] = {}
        try:
            for source in sources:
                if source == "ema" and getattr(validator, "ema", None) is None:
                    validator.ema = self.ema
                validator.use_ema = source == "ema"
                if hasattr(validator, "prototype_diagnostics_weight_source"):
                    validator.prototype_diagnostics_weight_source = source
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

    def _compute_loss(
        self, output: Any, targets: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float], Mapping[str, Tensor]]:
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
        total, components = loss_total(result)
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
        self, batch: Any, *, loss_divisor: float
    ) -> tuple[float, dict[str, float]]:
        inputs, targets, metadata = unpack_batch(batch)
        inputs = move_to_device(inputs, self.device)
        targets = move_to_device(targets, self.device)
        metadata = move_to_device(metadata, self.device)
        if not targets:
            raise ValueError("Training batch contains no target tensors")
        effective_task = self.task_name
        if effective_task is None and len(targets) == 1:
            effective_task = next(iter(targets))
        with self._autocast():
            output = call_model(
                self.model,
                inputs,
                effective_task,
                model_kwargs=model_kwargs_from_metadata(metadata),
            )
            total, components, per_task = self._compute_loss(output, targets)
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
        if total.numel() != 1 or not torch.isfinite(total.detach()).all():
            raise FloatingPointError(f"Training loss must be one finite scalar, got {total}")
        scaled_loss = total / float(loss_divisor)
        if self.scaler.is_enabled():
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if self.prototype_monitor is not None:
            self.prototype_monitor.observe(output)
        return float(total.detach().cpu()), components

    def _run_batch_with_oom_recovery(self, batch: Any) -> tuple[float, dict[str, float]]:
        batch_size = infer_batch_size(batch)
        while True:
            chunk_size = max(1, math.ceil(batch_size / self.microbatch_factor))
            chunks = [
                slice_batch(batch, start, min(start + chunk_size, batch_size), batch_size)
                for start in range(0, batch_size, chunk_size)
            ]
            losses: list[float] = []
            component_sums: dict[str, float] = defaultdict(float)
            try:
                divisor = float(self.gradient_accumulation * len(chunks))
                for chunk in chunks:
                    value, components = self._backward_microbatch(chunk, loss_divisor=divisor)
                    losses.append(value)
                    for key, component in components.items():
                        component_sums[key] += component
                averaged = {key: value / len(chunks) for key, value in component_sums.items()}
                return float(np.mean(losses)), averaged
            except BaseException as error:
                if self.device.type != "cuda" or not _is_cuda_oom(error):
                    raise
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                self.oom_retries += 1
                if self.oom_retries > 3 or chunk_size == 1:
                    raise RuntimeError(
                        "CUDA OOM persisted after at most three batch-size reductions"
                    ) from error
                self.microbatch_factor *= 2
                self.gradient_accumulation *= 2

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

    def train_epoch(self, epoch: int = 0) -> dict[str, float | int]:
        """Run one complete training epoch and return real measured statistics."""
        activity_plan = getattr(self, "activity_sampling_plan", None)
        set_sampler_epoch = getattr(activity_plan, "set_epoch", None)
        if callable(set_sampler_epoch):
            set_sampler_epoch(int(epoch))
        self.model.train()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if self.prototype_monitor is not None:
            self.prototype_monitor.start_epoch(epoch)
        self.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        losses: list[float] = []
        components: dict[str, list[float]] = defaultdict(list)
        batch_count = len(self.dataloader) if hasattr(self.dataloader, "__len__") else None
        seen_batches = 0
        seen_samples = 0
        for batch_index, batch in enumerate(self.dataloader):
            seen_samples += infer_batch_size(batch)
            loss_value, logged = self._run_batch_with_oom_recovery(batch)
            losses.append(loss_value)
            for key, value in logged.items():
                components[key].append(float(value))
            seen_batches += 1
            is_last = batch_count is not None and batch_index + 1 == batch_count
            if (batch_index + 1) % self.gradient_accumulation == 0 or is_last:
                self._optimizer_step()
        if seen_batches == 0:
            raise ValueError("Training dataloader produced no batches")
        if batch_count is None and seen_batches % self.gradient_accumulation:
            self._optimizer_step()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        duration_seconds = float(time.perf_counter() - started)
        result: dict[str, float | int] = {
            "epoch": int(epoch),
            "loss": float(np.mean(losses)),
            "duration_seconds": duration_seconds,
            "seen_samples": seen_samples,
            "images_per_second": float(seen_samples / duration_seconds),
            "peak_vram_bytes": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            ),
            "global_step": self.global_step,
            "oom_retries": self.oom_retries,
            "effective_microbatch_factor": self.microbatch_factor,
            "gradient_accumulation": self.gradient_accumulation,
        }
        for key, values in components.items():
            result[f"loss/{key}"] = float(np.mean(values))
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
        output_dir = Path(checkpoint_dir).resolve() if checkpoint_dir is not None else None
        activity_plan = getattr(self, "activity_sampling_plan", None)
        if self.prototype_monitor is not None and output_dir is not None:
            self.prototype_monitor.bind_output_dir(output_dir.parent / "artifacts")
        if output_dir is not None:
            self.gradient_monitor.bind_output_dir(output_dir.parent / "artifacts")
        best = {"ssim": -float("inf"), "psnr": -float("inf"), "proxy": -float("inf")}
        epochs_without_improvement = 0
        patience = max(0, int(config_get(self.config, "train.early_stopping_patience", 0) or 0))
        save_top_k = max(0, int(config_get(self.config, "train.save_top_k", 3) or 0))
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
            set_progress = getattr(self.loss_fn, "set_progress", None)
            if callable(set_progress):
                set_progress(epoch=epoch, total_epochs=total_epochs)
            train_metrics = self.train_epoch(epoch)
            if self.stage_controller is not None:
                self.stage_controller.advance_epoch()
            entry: dict[str, Any] = {"train": train_metrics}
            if validator is not None:
                if hasattr(validator, "prototype_diagnostics_epoch"):
                    validator.prototype_diagnostics_epoch = int(epoch)
                validation, source_results, selected_source = (
                    self._evaluate_weight_sources(validator)
                )
                entry["validation"] = validation
                entry["validation_weight_sources"] = source_results
                entry["selected_weight_source"] = selected_source
                macro = validation.get("macro", {})
                current = {
                    "ssim": float(macro.get("mean_ssim", -float("inf"))),
                    "psnr": float(macro.get("mean_psnr", -float("inf"))),
                    "proxy": float(macro.get("local_proxy_score", -float("inf"))),
                }
            else:
                current = best.copy()
            self.metric_history.append(entry)
            if output_dir is not None:
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
                for metric, filename in (
                    ("ssim", "best_ssim.ckpt"),
                    ("psnr", "best_psnr.ckpt"),
                    ("proxy", "best_proxy.ckpt"),
                ):
                    if current[metric] > best[metric]:
                        best[metric] = current[metric]
                        save_checkpoint(output_dir / filename, self.model, **save_kwargs)
                if validator is not None and save_top_k:
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
            if validator is not None:
                previous_best = max(
                    (
                        float(item.get("validation", {}).get("macro", {}).get("mean_ssim", -float("inf")))
                        for item in self.metric_history[:-1]
                    ),
                    default=-float("inf"),
                )
                if current["ssim"] > previous_best:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if patience and epochs_without_improvement >= patience:
                    break
        return self.metric_history
