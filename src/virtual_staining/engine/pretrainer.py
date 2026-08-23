"""Fold-local DAPI-only masked reconstruction pretraining."""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from virtual_staining.losses.ssim import SSIMLoss

from .checkpoint import load_checkpoint, save_checkpoint
from .common import metadata_item, move_to_device, unpack_batch


class DAPIPretrainer:
    """Train a DAPI MAE while explicitly rejecting validation/test samples."""

    _ALLOWED_SPLITS = {"train", "official_train", "final_train"}

    def __init__(
        self,
        model: nn.Module,
        dataloader: Any,
        *,
        device: str | torch.device = "auto",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        amp: bool = True,
        progress_bar: bool | str = "auto",
        progress_bar_refresh_seconds: float = 1.0,
    ) -> None:
        requested = str(device)
        self.device = torch.device(
            "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu"
            if requested == "auto"
            else requested
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA pretraining was requested but unavailable")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.ssim = SSIMLoss()
        self.amp = bool(amp)
        self.amp_dtype = (
            torch.bfloat16
            if self.device.type == "cpu" or torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.global_step = 0
        self.progress_bar_enabled = self._resolve_progress_bar_enabled(progress_bar)
        self.progress_bar_refresh_seconds = float(progress_bar_refresh_seconds)

    @staticmethod
    def _resolve_progress_bar_enabled(value: bool | str) -> bool:
        """Mirror supervised training's interactive-terminal progress behavior."""

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
        raise ValueError("progress_bar must be true, false, or auto")

    def _create_progress_bar(self, *, epoch: int, total_epochs: int | None, total: int | None) -> Any:
        if not self.progress_bar_enabled:
            return None
        from tqdm.auto import tqdm

        epoch_label = f"{epoch + 1}/{total_epochs}" if total_epochs is not None else str(epoch + 1)
        return tqdm(
            total=total,
            desc=f"DAPI pretrain {epoch_label}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=self.progress_bar_refresh_seconds,
            leave=True,
        )

    @staticmethod
    def _masked_l1(prediction: Tensor, reference: Tensor, mask: Tensor) -> Tensor:
        expanded = mask.expand_as(reference)
        denominator = expanded.sum().clamp_min(1.0)
        return torch.sum(torch.abs(prediction - reference) * expanded) / denominator

    def train_epoch(
        self, epoch: int, *, total_epochs: int | None = None
    ) -> dict[str, float | int]:
        self.model.train()
        started = time.perf_counter()
        losses: list[float] = []
        batch_count = len(self.dataloader) if hasattr(self.dataloader, "__len__") else None
        seen_samples = 0
        progress = self._create_progress_bar(
            epoch=epoch, total_epochs=total_epochs, total=batch_count
        )
        try:
            for batch in self.dataloader:
                inputs, _, metadata = unpack_batch(batch)
                batch_size = int(inputs.shape[0])
                for index in range(batch_size):
                    split = str(metadata_item(metadata, "split", index, "")).casefold()
                    if split not in self._ALLOWED_SPLITS:
                        raise ValueError(
                            f"DAPI pretraining accepts training splits only, received {split!r}"
                        )
                inputs = move_to_device(inputs, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp and self.device.type in {"cpu", "cuda"},
                ):
                    output = self.model(inputs)
                    reconstruction = output.reconstruction.float()
                    reference = inputs.float()
                    masked_l1 = self._masked_l1(
                        reconstruction, reference, output.mask.float()
                    )
                    ssim = self.ssim(reconstruction, reference)
                    loss = 0.8 * masked_l1 + 0.2 * ssim
                if not torch.isfinite(loss):
                    raise FloatingPointError("DAPI pretraining loss became non-finite")
                loss.backward()
                self.optimizer.step()
                self.global_step += 1
                seen_samples += batch_size
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                if progress is not None:
                    elapsed = max(time.perf_counter() - started, 1e-12)
                    postfix = {
                        "loss": f"{loss_value:.4f}",
                        "img/s": f"{seen_samples / elapsed:.2f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    }
                    if self.device.type == "cuda":
                        postfix["vram"] = (
                            f"{torch.cuda.max_memory_allocated(self.device) / 1024**2:.0f}MiB"
                        )
                    progress.set_postfix(postfix, refresh=False)
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()
        if not losses:
            raise ValueError("DAPI pretraining dataloader produced no batches")
        return {
            "epoch": int(epoch),
            "loss": float(sum(losses) / len(losses)),
            "global_step": self.global_step,
            "duration_seconds": float(time.perf_counter() - started),
        }

    def fit(
        self,
        epochs: int,
        checkpoint_path: str | Path,
        *,
        config: Any = None,
        manifest_hash: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> list[dict[str, float | int]]:
        if epochs < 1:
            raise ValueError("Pretraining epochs must be positive")
        logging.getLogger("virtual_staining").info(
            "Starting DAPI pretraining: %d epoch(s), batch progress bar=%s",
            epochs,
            "enabled" if self.progress_bar_enabled else "disabled (non-interactive terminal or config)",
        )
        history: list[dict[str, float | int]] = []
        for epoch in range(epochs):
            metrics = self.train_epoch(epoch, total_epochs=epochs)
            history.append(metrics)
            logging.getLogger("virtual_staining").info(
                "DAPI pretrain epoch %d/%d complete | loss=%.6f | %.3f seconds",
                epoch + 1,
                epochs,
                float(metrics["loss"]),
                float(metrics["duration_seconds"]),
            )
        save_checkpoint(
            checkpoint_path,
            self.model,
            optimizer=self.optimizer,
            epoch=epochs - 1,
            global_step=self.global_step,
            config=config,
            manifest_hash=manifest_hash,
            metric_history=history,
            extra={
                **dict(provenance or {}),
                "pretrain_state_version": 1,
                "training_type": "fold_local_dapi_mae",
                "uses_target_labels": False,
                "allowed_splits": sorted(self._ALLOWED_SPLITS),
                "transfer_scope": ["local_encoder"],
            },
        )
        return history


def _copy_local_encoder_state(
    source_state: Mapping[str, Tensor], destination_encoder: nn.Module
) -> dict[str, Any]:
    destination_state = destination_encoder.state_dict()
    compatible = {
        str(key): value
        for key, value in source_state.items()
        if isinstance(value, Tensor)
        and key in destination_state
        and destination_state[key].shape == value.shape
    }
    if not compatible:
        raise ValueError("No compatible local-encoder tensors were found")
    shape_mismatch = sorted(
        str(key)
        for key, value in source_state.items()
        if isinstance(value, Tensor)
        and key in destination_state
        and destination_state[key].shape != value.shape
    )
    source_only = sorted(str(key) for key in source_state if key not in destination_state)
    destination_missing = sorted(key for key in destination_state if key not in compatible)
    destination_encoder.load_state_dict(compatible, strict=False)
    return {
        "transferred_tensors": len(compatible),
        "source_tensors": sum(isinstance(value, Tensor) for value in source_state.values()),
        "destination_tensors": len(destination_state),
        "transferred_keys": sorted(compatible),
        "shape_mismatch_keys": shape_mismatch,
        "source_only_keys": source_only,
        "destination_missing_keys": destination_missing,
    }


def transfer_local_encoder(source: nn.Module, destination: nn.Module) -> dict[str, Any]:
    """Copy matching local-encoder tensors and fail if nothing is transferable."""

    source_encoder = getattr(source, "local_encoder", None)
    destination_encoder = getattr(destination, "local_encoder", None)
    if not isinstance(source_encoder, nn.Module) or not isinstance(destination_encoder, nn.Module):
        raise TypeError("Both models must expose a local_encoder module")
    report = _copy_local_encoder_state(source_encoder.state_dict(), destination_encoder)
    report.update(
        {
            "transfer_type": "module_local_encoder",
            "source_class": type(source).__name__,
            "destination_class": type(destination).__name__,
        }
    )
    return report


def _checkpoint_local_encoder_state(
    payload: Mapping[str, Any], destination_encoder: nn.Module
) -> dict[str, Tensor]:
    state = payload.get("model", payload.get("state_dict"))
    if not isinstance(state, Mapping):
        raise KeyError("Pretraining checkpoint does not contain model weights")
    normalized = {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state.items()
        if isinstance(value, Tensor)
    }
    prefixed = {
        key[len("local_encoder.") :]: value
        for key, value in normalized.items()
        if key.startswith("local_encoder.")
    }
    if prefixed:
        return prefixed
    destination_keys = set(destination_encoder.state_dict())
    direct = {key: value for key, value in normalized.items() if key in destination_keys}
    if not direct:
        raise KeyError("Pretraining checkpoint has no local_encoder tensor namespace")
    return direct


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transfer_local_encoder_from_checkpoint(
    checkpoint_path: str | Path,
    destination: nn.Module,
    *,
    map_location: str | torch.device | None = "cpu",
    require_pretrain_provenance: bool = True,
    expected_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Load a fold-local DAPI checkpoint and transfer only its local encoder.

    The returned mapping is JSON/checkpoint-safe provenance that callers can
    attach to a fine-tuning run.  With the default safety gate, a checkpoint
    must explicitly identify itself as fold-local DAPI pretraining and affirm
    that no target labels were used.
    """

    destination_encoder = getattr(destination, "local_encoder", None)
    if not isinstance(destination_encoder, nn.Module):
        raise TypeError("Destination model must expose a local_encoder module")
    path = Path(checkpoint_path).expanduser().resolve()
    payload = load_checkpoint(path, map_location=map_location)
    extra = payload.get("extra", {})
    if not isinstance(extra, Mapping):
        extra = {}
    training_type = str(extra.get("training_type", ""))
    uses_target_labels = extra.get("uses_target_labels")
    source_manifest_hash = str(payload.get("manifest_hash", "")).strip()
    if require_pretrain_provenance:
        if training_type != "fold_local_dapi_mae":
            raise ValueError("Checkpoint is not marked as fold-local DAPI pretraining")
        if uses_target_labels is not False:
            raise ValueError("DAPI pretraining checkpoint must affirm uses_target_labels=false")
        if not source_manifest_hash:
            raise ValueError("DAPI pretraining checkpoint has no training-manifest hash")
    if expected_manifest_hash is not None and source_manifest_hash != str(
        expected_manifest_hash
    ).strip():
        raise ValueError(
            "DAPI pretraining checkpoint manifest hash does not match the destination fold"
        )
    source_state = _checkpoint_local_encoder_state(payload, destination_encoder)
    report = _copy_local_encoder_state(source_state, destination_encoder)
    report.update(
        {
            "transfer_type": "dapi_mae_local_encoder",
            "source_checkpoint": str(path),
            "source_checkpoint_sha256": _file_sha256(path),
            "source_checkpoint_version": payload.get("checkpoint_version"),
            "source_manifest_hash": source_manifest_hash or None,
            "expected_manifest_hash": expected_manifest_hash,
            "manifest_hash_verified": expected_manifest_hash is not None,
            "source_training_type": training_type or None,
            "uses_target_labels": uses_target_labels,
            "pretrain_state_version": extra.get("pretrain_state_version"),
            "allowed_splits": list(extra.get("allowed_splits", []))
            if isinstance(extra.get("allowed_splits"), (list, tuple))
            else [],
            "transfer_scope": list(extra.get("transfer_scope", []))
            if isinstance(extra.get("transfer_scope"), (list, tuple))
            else [],
            "destination_class": type(destination).__name__,
        }
    )
    return report
