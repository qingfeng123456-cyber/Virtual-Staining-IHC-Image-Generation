"""Self-contained differentiable SSIM and adaptive small-image MS-SSIM."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .charbonnier import float32_context


def _effective_window(requested: int, height: int, width: int) -> int:
    size = min(requested, height, width)
    if size % 2 == 0:
        size -= 1
    return max(size, 1)


def _gaussian_kernel(window_size: int, sigma: float, channels: int, reference: Tensor) -> Tensor:
    if window_size == 1:
        one_dimensional = reference.new_ones(1)
    else:
        coordinates = torch.arange(window_size, device=reference.device, dtype=torch.float32)
        coordinates = coordinates - (window_size - 1) / 2
        effective_sigma = min(sigma, max(window_size / 3.0, 0.5))
        one_dimensional = torch.exp(-(coordinates.square()) / (2.0 * effective_sigma**2))
        one_dimensional = one_dimensional / one_dimensional.sum()
    kernel = torch.outer(one_dimensional, one_dimensional)
    return kernel.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def _ssim_components(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float,
    window_size: int,
    sigma: float,
    k1: float,
    k2: float,
) -> tuple[Tensor, Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "SSIM expects equal BCHW tensors, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    channels = prediction.shape[1]
    window = _effective_window(window_size, prediction.shape[-2], prediction.shape[-1])
    kernel = _gaussian_kernel(window, sigma, channels, prediction)
    padding = window // 2
    mu_prediction = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_prediction_squared = mu_prediction.square()
    mu_target_squared = mu_target.square()
    mu_product = mu_prediction * mu_target

    variance_prediction = (
        F.conv2d(prediction.square(), kernel, padding=padding, groups=channels)
        - mu_prediction_squared
    ).clamp_min(0.0)
    variance_target = (
        F.conv2d(target.square(), kernel, padding=padding, groups=channels)
        - mu_target_squared
    ).clamp_min(0.0)
    covariance = (
        F.conv2d(prediction * target, kernel, padding=padding, groups=channels) - mu_product
    )

    constant1 = (k1 * data_range) ** 2
    constant2 = (k2 * data_range) ** 2
    luminance = (2.0 * mu_product + constant1) / (
        mu_prediction_squared + mu_target_squared + constant1
    )
    contrast_structure = (2.0 * covariance + constant2) / (
        variance_prediction + variance_target + constant2
    )
    ssim_map = luminance * contrast_structure
    dimensions = (1, 2, 3)
    return ssim_map.mean(dim=dimensions), contrast_structure.mean(dim=dimensions)


def differentiable_ssim(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    size_average: bool = True,
) -> Tensor:
    """Compute differentiable SSIM in float32, including under outer AMP."""

    if data_range <= 0.0 or window_size < 1 or sigma <= 0.0:
        raise ValueError("data_range, window_size, and sigma must be positive")
    with float32_context(prediction):
        scores, _ = _ssim_components(
            prediction.float(),
            target.float(),
            data_range=data_range,
            window_size=window_size,
            sigma=sigma,
            k1=k1,
            k2=k2,
        )
        scores = scores.clamp(min=-1.0, max=1.0)
        return scores.mean() if size_average else scores


def differentiable_ms_ssim(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    weights: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    size_average: bool = True,
) -> Tensor:
    """Compute MS-SSIM while automatically reducing levels for tiny tensors."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("MS-SSIM expects equal BCHW tensors")
    if not weights or any(weight <= 0.0 for weight in weights):
        raise ValueError("MS-SSIM weights must be a non-empty positive sequence")
    minimum_size = min(prediction.shape[-2:])
    available_levels = max(1, int(math.floor(math.log2(max(minimum_size, 1)))) + 1)
    level_count = min(len(weights), available_levels)
    level_weights = prediction.new_tensor(tuple(weights[:level_count]), dtype=torch.float32)
    level_weights = level_weights / level_weights.sum()

    with float32_context(prediction):
        current_prediction = prediction.float()
        current_target = target.float()
        ssim_values: list[Tensor] = []
        contrast_values: list[Tensor] = []
        for level in range(level_count):
            score, contrast = _ssim_components(
                current_prediction,
                current_target,
                data_range=data_range,
                window_size=window_size,
                sigma=sigma,
                k1=0.01,
                k2=0.03,
            )
            ssim_values.append(score.clamp_min(1e-6))
            contrast_values.append(contrast.clamp_min(1e-6))
            if level + 1 < level_count:
                current_prediction = F.avg_pool2d(current_prediction, kernel_size=2, stride=2)
                current_target = F.avg_pool2d(current_target, kernel_size=2, stride=2)

        result = ssim_values[-1].pow(level_weights[-1])
        for level, contrast in enumerate(contrast_values[:-1]):
            result = result * contrast.pow(level_weights[level])
        result = result.clamp(min=0.0, max=1.0)
        return result.mean() if size_average else result


class DifferentiableSSIM(nn.Module):
    """Module wrapper around :func:`differentiable_ssim`."""

    def __init__(self, data_range: float = 1.0, window_size: int = 11) -> None:
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return differentiable_ssim(
            prediction,
            target,
            data_range=self.data_range,
            window_size=self.window_size,
        )


class SSIMLoss(DifferentiableSSIM):
    """``1 - SSIM`` reconstruction loss."""

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return 1.0 - super().forward(prediction, target)


class MultiScaleSSIM(nn.Module):
    """Module wrapper for adaptive MS-SSIM."""

    def __init__(
        self,
        data_range: float = 1.0,
        window_size: int = 11,
        weights: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    ) -> None:
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size
        self.weights = tuple(float(weight) for weight in weights)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return differentiable_ms_ssim(
            prediction,
            target,
            data_range=self.data_range,
            window_size=self.window_size,
            weights=self.weights,
        )


class MSSSIMLoss(MultiScaleSSIM):
    """``1 - MS-SSIM`` reconstruction loss."""

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return 1.0 - super().forward(prediction, target)
