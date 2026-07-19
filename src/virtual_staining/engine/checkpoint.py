"""Complete, atomic training checkpoints and deterministic resume helpers."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .ema import ExponentialMovingAverage

CHECKPOINT_VERSION = 2


def _state_dict_without_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[7:]: value for key, value in state.items()}
    return state


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch random-number generator state."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore an RNG state captured by :func:`capture_rng_state`."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def detect_git_commit(workdir: str | Path | None = None) -> str | None:
    """Return the current Git commit, or ``None`` outside a usable repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(workdir).resolve() if workdir is not None else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    value = result.stdout.strip()
    return value or None


def _object_state(value: Any) -> Any:
    return value.state_dict() if value is not None and hasattr(value, "state_dict") else None


def _seed_from_config(config: Any) -> int | None:
    if isinstance(config, dict):
        project = config.get("project", {})
        train = config.get("train", {})
        value = project.get("seed", train.get("seed")) if isinstance(project, dict) else None
        if value is not None:
            return int(value)
    return None


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    ema: ExponentialMovingAverage | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    loss_fn: Any = None,
    epoch: int = -1,
    global_step: int = 0,
    config: Any = None,
    manifest_hash: str | None = None,
    image_spec: Any = None,
    targets: list[str] | tuple[str, ...] | None = None,
    metric_history: Any = None,
    dataloader_generator_state: torch.Tensor | None = None,
    seed: int | None = None,
    git_commit: str | None = None,
    workdir: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically save all state needed to reproduce or resume a run."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": _object_state(optimizer),
        "scheduler": _object_state(scheduler),
        "scaler": _object_state(scaler),
        "loss_fn": _object_state(loss_fn),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
        "manifest_hash": manifest_hash,
        "image_spec": image_spec,
        "targets": list(targets or []),
        "metric_history": metric_history if metric_history is not None else [],
        "seed": int(seed) if seed is not None else _seed_from_config(config),
        "git_commit": git_commit if git_commit is not None else detect_git_commit(workdir),
        "rng_state": capture_rng_state(),
        "dataloader_generator_state": dataloader_generator_state,
    }
    if extra:
        payload["extra"] = dict(extra)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    *,
    ema: ExponentialMovingAverage | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    loss_fn: Any = None,
    dataloader_generator: torch.Generator | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
    restore_rng: bool = False,
    use_ema_as_model: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore supplied runtime objects."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a mapping")
    model_state = payload.get("model", payload.get("state_dict"))
    if model is not None:
        if use_ema_as_model and payload.get("ema") is not None:
            ema_state = payload["ema"]
            model_state = ema_state.get("shadow", ema_state) if isinstance(ema_state, dict) else ema_state
        if not isinstance(model_state, dict):
            raise KeyError("Checkpoint does not contain model weights")
        model.load_state_dict(_state_dict_without_module_prefix(model_state), strict=strict)
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    for obj, key in (
        (optimizer, "optimizer"),
        (scheduler, "scheduler"),
        (scaler, "scaler"),
        (loss_fn, "loss_fn"),
    ):
        if obj is not None and payload.get(key) is not None:
            obj.load_state_dict(payload[key])
    if restore_rng:
        restore_rng_state(payload.get("rng_state"))
    generator_state = payload.get("dataloader_generator_state")
    if dataloader_generator is not None and generator_state is not None:
        dataloader_generator.set_state(generator_state.cpu())
    return payload


def resume_from_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    ema: ExponentialMovingAverage | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    loss_fn: Any = None,
    dataloader_generator: torch.Generator | None = None,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore training objects and return resume counters plus raw metadata."""
    payload = load_checkpoint(
        path,
        model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loss_fn=loss_fn,
        dataloader_generator=dataloader_generator,
        map_location=map_location,
        strict=strict,
        restore_rng=restore_rng,
    )
    return {
        "start_epoch": int(payload.get("epoch", -1)) + 1,
        "global_step": int(payload.get("global_step", 0)),
        "metric_history": payload.get("metric_history", []),
        "config": payload.get("config"),
        "manifest_hash": payload.get("manifest_hash"),
        "image_spec": payload.get("image_spec"),
        "targets": payload.get("targets", []),
        "seed": payload.get("seed"),
        "checkpoint": payload,
    }
