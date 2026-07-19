"""Fixed-kernel Gaussian and Laplacian pyramid reconstruction losses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .charbonnier import float32_context


def _validate_image_pair(prediction: Tensor, target: Tensor) -> None:
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("Pyramid losses expect BCHW tensors")
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    if prediction.shape[1] < 1:
        raise ValueError("Pyramid losses require at least one image channel")


def gaussian_kernel(reference: Tensor, channels: int) -> Tensor:
    """Return the normalized fixed 5x5 binomial kernel for grouped convolution."""

    if channels < 1:
        raise ValueError("channels must be positive")
    vector = reference.new_tensor((1.0, 4.0, 6.0, 4.0, 1.0), dtype=torch.float32)
    vector = vector / vector.sum()
    kernel = torch.outer(vector, vector)
    return kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)


def gaussian_blur(image: Tensor) -> Tensor:
    """Blur a BCHW image without changing its spatial dimensions."""

    if image.ndim != 4:
        raise ValueError("gaussian_blur expects a BCHW tensor")
    padding_mode = "reflect" if min(image.shape[-2:]) > 2 else "replicate"
    padded = F.pad(image, (2, 2, 2, 2), mode=padding_mode)
    return F.conv2d(padded, gaussian_kernel(image, image.shape[1]), groups=image.shape[1])


def build_gaussian_pyramid(image: Tensor, levels: int = 4) -> tuple[Tensor, ...]:
    """Build full, half, quarter, and lower Gaussian levels as available."""

    if image.ndim != 4:
        raise ValueError("Gaussian pyramids expect BCHW tensors")
    if levels < 1:
        raise ValueError("levels must be positive")
    result = [image]
    for _ in range(1, levels):
        previous = result[-1]
        if min(previous.shape[-2:]) <= 1:
            break
        blurred = gaussian_blur(previous)
        result.append(blurred[:, :, ::2, ::2])
    return tuple(result)


def build_laplacian_pyramid(
    image: Tensor,
    levels: int = 4,
) -> tuple[Tensor, ...]:
    """Build residual levels followed by the coarsest Gaussian image."""

    gaussian = build_gaussian_pyramid(image, levels)
    laplacian: list[Tensor] = []
    for current, lower in zip(gaussian[:-1], gaussian[1:], strict=True):
        expanded = F.interpolate(lower, size=current.shape[-2:], mode="bilinear", align_corners=False)
        laplacian.append(current - gaussian_blur(expanded))
    laplacian.append(gaussian[-1])
    return tuple(laplacian)


def _level_weights(weights: Sequence[float], count: int) -> tuple[float, ...]:
    if not weights or any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in weights):
        raise ValueError("Pyramid level weights must be finite and nonnegative")
    if sum(weights) <= 0:
        raise ValueError("At least one pyramid level weight must be positive")
    values = list(float(value) for value in weights[:count])
    while len(values) < count:
        values.append(values[-1] * 0.5)
    return tuple(values)


class GaussianPyramidLoss(nn.Module):
    """Weighted L1 loss across fixed Gaussian pyramid levels."""

    def __init__(
        self,
        levels: int = 4,
        level_weights: Sequence[float] = (1.0, 0.5, 0.25, 0.125),
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        self.levels = int(levels)
        self.level_weights = tuple(float(value) for value in level_weights)
        _level_weights(self.level_weights, self.levels)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        _validate_image_pair(prediction, target)
        with float32_context(prediction):
            prediction_levels = build_gaussian_pyramid(prediction.float(), self.levels)
            target_levels = build_gaussian_pyramid(target.float(), self.levels)
            weights = _level_weights(self.level_weights, len(prediction_levels))
            losses = [
                weight * F.l1_loss(prediction_level, target_level)
                for weight, prediction_level, target_level in zip(
                    weights, prediction_levels, target_levels, strict=True
                )
            ]
            return torch.stack(losses).sum() / sum(weights)


class LaplacianPyramidLoss(nn.Module):
    """Weighted loss over Gaussian structure and Laplacian residual levels."""

    def __init__(
        self,
        levels: int = 4,
        level_weights: Sequence[float] = (1.0, 0.5, 0.25, 0.125),
        *,
        gaussian_weight: float = 0.25,
        laplacian_weight: float = 0.75,
        mode: str = "l1",
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        if gaussian_weight < 0 or laplacian_weight < 0 or gaussian_weight + laplacian_weight <= 0:
            raise ValueError("Gaussian/Laplacian weights must be nonnegative with a positive sum")
        if mode not in {"l1", "mse"}:
            raise ValueError("Pyramid loss mode must be 'l1' or 'mse'")
        self.levels = int(levels)
        self.level_weights = tuple(float(value) for value in level_weights)
        _level_weights(self.level_weights, self.levels)
        self.gaussian_weight = float(gaussian_weight)
        self.laplacian_weight = float(laplacian_weight)
        self.mode = mode

    def _distance(self, prediction: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(prediction, target) if self.mode == "l1" else F.mse_loss(prediction, target)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        _validate_image_pair(prediction, target)
        with float32_context(prediction):
            prediction_float = prediction.float()
            target_float = target.float()
            prediction_gaussian = build_gaussian_pyramid(prediction_float, self.levels)
            target_gaussian = build_gaussian_pyramid(target_float, self.levels)
            prediction_laplacian = build_laplacian_pyramid(prediction_float, self.levels)
            target_laplacian = build_laplacian_pyramid(target_float, self.levels)
            weights = _level_weights(self.level_weights, len(prediction_gaussian))
            gaussian = sum(
                weight * self._distance(prediction_level, target_level)
                for weight, prediction_level, target_level in zip(
                    weights, prediction_gaussian, target_gaussian, strict=True
                )
            ) / sum(weights)
            laplacian = sum(
                weight * self._distance(prediction_level, target_level)
                for weight, prediction_level, target_level in zip(
                    weights, prediction_laplacian, target_laplacian, strict=True
                )
            ) / sum(weights)
            normalizer = self.gaussian_weight + self.laplacian_weight
            return (
                self.gaussian_weight * gaussian + self.laplacian_weight * laplacian
            ) / normalizer
