"""Deterministic inference, strict D4 TTA, and lossless tensor handling."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from virtual_staining.data.transforms import apply_context_d4

from .common import (
    call_model,
    config_get,
    extract_predictions,
    infer_batch_size,
    metadata_item,
    model_kwargs_from_metadata,
    move_to_device,
    slice_batch,
    unpack_batch,
)
from .ema import ExponentialMovingAverage

D4_TRANSFORMS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "hflip",
    "vflip",
    "transpose",
    "anti_transpose",
)

_D4_CONTEXT_IDS = {
    "identity": 0,
    "rot90": 1,
    "rot180": 2,
    "rot270": 3,
    "hflip": 4,
    "transpose": 5,
    "vflip": 6,
    "anti_transpose": 7,
}


def _portable_stem(value: str) -> str:
    return Path(str(value).replace("\\", "/").rsplit("/", 1)[-1]).stem


def apply_d4(tensor: Tensor, transform: str | int) -> Tensor:
    """Apply one deterministic D4 transform to the final two dimensions."""
    name = D4_TRANSFORMS[int(transform)] if isinstance(transform, int) else transform.lower()
    if name == "identity":
        return tensor
    if name == "hflip":
        return torch.flip(tensor, dims=(-1,))
    if name == "vflip":
        return torch.flip(tensor, dims=(-2,))
    if name == "rot90":
        return torch.rot90(tensor, k=1, dims=(-2, -1))
    if name == "rot180":
        return torch.rot90(tensor, k=2, dims=(-2, -1))
    if name == "rot270":
        return torch.rot90(tensor, k=3, dims=(-2, -1))
    if name == "transpose":
        return tensor.transpose(-2, -1)
    if name == "anti_transpose":
        return torch.flip(tensor.transpose(-2, -1), dims=(-2, -1))
    raise ValueError(f"Unknown D4 transform: {transform}")


def invert_d4(tensor: Tensor, transform: str | int) -> Tensor:
    """Exactly invert :func:`apply_d4`, including rectangular tensors."""
    name = D4_TRANSFORMS[int(transform)] if isinstance(transform, int) else transform.lower()
    inverse = {"rot90": "rot270", "rot270": "rot90"}.get(name, name)
    return apply_d4(tensor, inverse)


def d4_context_transform_id(transform: str | int) -> int:
    """Map inference transform names to the synchronized context convention."""

    name = D4_TRANSFORMS[int(transform)] if isinstance(transform, int) else transform.lower()
    if name not in _D4_CONTEXT_IDS:
        raise ValueError(f"Unknown D4 transform: {transform}")
    return _D4_CONTEXT_IDS[name]


def _is_cuda_oom(error: BaseException) -> bool:
    text = str(error).lower()
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom_type is not None and isinstance(error, oom_type)) or (
        isinstance(error, RuntimeError) and "cuda" in text and "out of memory" in text
    )


def tensor_to_uint8_image(
    tensor: Tensor, *, target_mode: str | None = None
) -> tuple[np.ndarray, str]:
    """Round one CHW tensor in [0, 1] to uint8 L or RGB pixels."""
    array = tensor.detach().float().clamp(0.0, 1.0).cpu().numpy()
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[0] not in (1, 3):
        raise ValueError(f"Only one- or three-channel predictions can be saved, got {array.shape}")
    pixels = np.rint(array * 255.0).astype(np.uint8)
    inferred_mode = "L" if pixels.shape[0] == 1 else "RGB"
    mode = target_mode or inferred_mode
    if mode not in {"L", "RGB"}:
        raise ValueError(f"Competition JPEG output mode must be L or RGB, got {mode!r}")
    if mode == "L":
        if pixels.shape[0] == 3:
            pixels = np.rint(pixels.astype(np.float32).mean(axis=0)).astype(np.uint8)[None]
        return pixels[0], mode
    if pixels.shape[0] == 1:
        pixels = np.repeat(pixels, 3, axis=0)
    return np.moveaxis(pixels, 0, -1), mode


def save_prediction_jpeg(
    tensor: Tensor,
    path: str | Path,
    *,
    quality: int = 100,
    subsampling: int = 0,
    target_mode: str | None = None,
) -> Path:
    """Encode one prediction exactly once as a high-quality JPEG."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels, mode = tensor_to_uint8_image(tensor, target_mode=target_mode)
    Image.fromarray(pixels, mode=mode).save(
        destination,
        format="JPEG",
        quality=int(quality),
        subsampling=int(subsampling),
        optimize=False,
    )
    return destination


class Inferencer:
    """Run stable ordered inference for Tensor or RestorationOutput models."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device | None = None,
        config: Any = None,
        ema: ExponentialMovingAverage | None = None,
        use_ema: bool | None = None,
        tta: str | bool | None = None,
        image_spec: Any = None,
    ) -> None:
        configured_device = config_get(
            config, "inference.device", config_get(config, "train.device", "auto")
        )
        requested_device = configured_device if device is None else device
        if str(requested_device).lower() == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference was requested but CUDA is unavailable")
        self.model = model.to(self.device)
        self.config = config
        self.ema = ema
        self.image_spec = image_spec
        self.use_ema = bool(
            config_get(config, "inference.use_ema", True) if use_ema is None else use_ema
        )
        tta_setting = config_get(config, "inference.tta", "none") if tta is None else tta
        self.tta = str(tta_setting).lower() in {"d4", "true", "1"} or tta_setting is True
        self.amp = bool(config_get(config, "inference.amp", config_get(config, "train.amp", True)))
        requested = str(config_get(config, "train.amp_dtype", "auto")).lower()
        self.amp_dtype = (
            torch.bfloat16
            if requested in {"auto", "bfloat16", "bf16"}
            and (self.device.type == "cpu" or torch.cuda.is_bf16_supported())
            else torch.float16
        )
        self.jpeg_quality = int(config_get(config, "inference.jpeg_quality", 100))
        self.jpeg_subsampling = int(config_get(config, "inference.jpeg_subsampling", 0))
        self.oom_retries = 0

    @classmethod
    def from_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        config: Any = None,
        use_ema: bool = True,
        tta: str | bool | None = None,
    ) -> Inferencer:
        """Restore model/EMA plus checkpoint ImageSpec for deterministic inference."""
        from .checkpoint import load_checkpoint

        ema = ExponentialMovingAverage(model)
        payload = load_checkpoint(
            checkpoint_path,
            model,
            ema=ema,
            map_location="cpu",
            use_ema_as_model=False,
        )
        if payload.get("ema") is None:
            ema = None
        effective_config = config if config is not None else payload.get("config")
        return cls(
            model,
            device=device,
            config=effective_config,
            ema=ema,
            use_ema=use_ema,
            tta=tta,
            image_spec=payload.get("image_spec"),
        )

    def _target_mode(self, target: str) -> str | None:
        spec = self.image_spec
        if spec is None:
            return None
        if hasattr(spec, "mode"):
            return str(spec.mode)
        if not isinstance(spec, dict):
            return None
        candidate: Any = spec
        for container_key in ("targets", "target_specs", "outputs"):
            nested = spec.get(container_key)
            if isinstance(nested, dict) and target in nested:
                candidate = nested[target]
                break
        if target in spec:
            candidate = spec[target]
        if hasattr(candidate, "mode"):
            return str(candidate.mode)
        if isinstance(candidate, dict):
            mode = candidate.get("mode") or candidate.get("storage_mode")
            if mode:
                return str(mode)
            channels = candidate.get("storage_channels", candidate.get("channels"))
            if channels is not None:
                return "L" if int(channels) == 1 else "RGB"
        return None

    def _autocast(self) -> Any:
        return torch.autocast(
            self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp and self.device.type in {"cpu", "cuda"},
        )

    def _forward(
        self,
        inputs: Tensor,
        target: str | None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        with self._autocast():
            output = call_model(self.model, inputs, target, model_kwargs=model_kwargs)
        predictions = extract_predictions(output)
        if target is not None:
            if target in predictions:
                predictions = {target: predictions[target]}
            elif len(predictions) == 1:
                predictions = {target: next(iter(predictions.values()))}
            else:
                raise KeyError(f"Model did not produce requested target {target!r}")
        return {name: value.float().clamp(0.0, 1.0) for name, value in predictions.items()}

    @staticmethod
    def _transform_context_kwargs(
        model_kwargs: dict[str, Any], transform: str | int
    ) -> dict[str, Any]:
        transformed = dict(model_kwargs)
        tiles = transformed.get("context_tiles")
        mask = transformed.get("context_valid_mask")
        offsets = transformed.get("context_offsets")
        if not all(isinstance(value, Tensor) for value in (tiles, mask, offsets)):
            return transformed
        if tiles.ndim != 5 or mask.ndim != 2 or offsets.ndim != 3:
            raise ValueError("Batched context must be [B,N,C,H,W], [B,N], and [B,N,2]")
        transform_id = d4_context_transform_id(transform)
        batches = [
            apply_context_d4(tiles[index], mask[index], offsets[index], transform_id)
            for index in range(tiles.shape[0])
        ]
        transformed["context_tiles"] = torch.stack([item[0] for item in batches])
        transformed["context_valid_mask"] = torch.stack([item[1] for item in batches])
        transformed["context_offsets"] = torch.stack([item[2] for item in batches])
        return transformed

    def predict_tensor(
        self,
        inputs: Tensor,
        *,
        target: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """Predict a tensor batch, optionally averaging all eight D4 views."""
        inputs = move_to_device(inputs, self.device)
        kwargs = move_to_device(model_kwargs or {}, self.device)
        if not self.tta:
            return self._forward(inputs, target, kwargs)
        accumulated: dict[str, Tensor] = {}
        for transform in D4_TRANSFORMS:
            transformed = apply_d4(inputs, transform)
            transformed_kwargs = self._transform_context_kwargs(kwargs, transform)
            predictions = self._forward(transformed, target, transformed_kwargs)
            for name, prediction in predictions.items():
                restored = invert_d4(prediction, transform)
                accumulated[name] = accumulated.get(name, torch.zeros_like(restored)) + restored
        return {name: value.div(len(D4_TRANSFORMS)).clamp(0.0, 1.0) for name, value in accumulated.items()}

    def _predict_with_oom_recovery(
        self,
        inputs: Tensor,
        target: str | None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        batch_size = int(inputs.shape[0])
        factor = 1
        while True:
            chunk_size = max(1, math.ceil(batch_size / factor))
            bounds = [
                (start, min(start + chunk_size, batch_size))
                for start in range(0, batch_size, chunk_size)
            ]
            collected: dict[str, list[Tensor]] = defaultdict(list)
            try:
                for start, stop in bounds:
                    chunk = inputs[start:stop]
                    chunk_kwargs = slice_batch(
                        model_kwargs or {}, start, stop, batch_size
                    )
                    for name, prediction in self.predict_tensor(
                        chunk,
                        target=target,
                        model_kwargs=chunk_kwargs,
                    ).items():
                        collected[name].append(prediction.cpu())
                return {name: torch.cat(values, dim=0) for name, values in collected.items()}
            except BaseException as error:
                if self.device.type != "cuda" or not _is_cuda_oom(error):
                    raise
                torch.cuda.empty_cache()
                self.oom_retries += 1
                if self.oom_retries > 3 or chunk_size == 1:
                    raise RuntimeError(
                        "CUDA OOM persisted after at most three inference batch reductions"
                    ) from error
                factor *= 2

    @torch.inference_mode()
    def predict_loader(
        self,
        dataloader: Any,
        *,
        output_dir: str | Path | None = None,
        target: str | None = None,
    ) -> dict[str, Any]:
        """Predict an ordered dataloader and optionally save per-target JPEGs."""
        if getattr(dataloader, "sampler", None).__class__.__name__ == "RandomSampler":
            raise ValueError("Inference dataloader must not shuffle samples")
        was_training = self.model.training
        self.model.eval()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        saved: list[str] = []
        count = 0
        root = Path(output_dir).expanduser().resolve() if output_dir is not None else None
        ema_context = (
            self.ema.average_parameters(self.model)
            if self.use_ema and self.ema is not None
            else nullcontext()
        )
        try:
            with ema_context:
                for batch in dataloader:
                    inputs, _, metadata = unpack_batch(batch)
                    predictions = self._predict_with_oom_recovery(
                        inputs,
                        target,
                        model_kwargs_from_metadata(metadata),
                    )
                    batch_size = infer_batch_size(inputs)
                    for index in range(batch_size):
                        stem_value = metadata_item(metadata, "stem", index, None)
                        if stem_value is None:
                            dapi_path = metadata_item(metadata, "dapi_path", index, None)
                            stem_value = _portable_stem(str(dapi_path)) if dapi_path else f"{count:05d}"
                        stem = _portable_stem(str(stem_value))
                        for target_name, prediction in predictions.items():
                            if root is not None:
                                destination = root / target_name / f"{stem}.jpg"
                                saved.append(
                                    str(
                                        save_prediction_jpeg(
                                            prediction[index],
                                            destination,
                                            quality=self.jpeg_quality,
                                            subsampling=self.jpeg_subsampling,
                                            target_mode=self._target_mode(target_name),
                                        )
                                    )
                                )
                        count += 1
        finally:
            self.model.train(was_training)
        elapsed = float(time.perf_counter() - started)
        if count == 0:
            raise ValueError(
                "Inference dataloader produced zero inputs; an empty official test is not valid."
            )
        peak = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        return {
            "count": count,
            "files": saved,
            "duration_seconds": elapsed,
            "seconds_per_image": elapsed / max(1, count),
            "peak_vram_bytes": peak,
            "oom_retries": self.oom_retries,
            "tta": "d4" if self.tta else "none",
        }

    predict = predict_loader
