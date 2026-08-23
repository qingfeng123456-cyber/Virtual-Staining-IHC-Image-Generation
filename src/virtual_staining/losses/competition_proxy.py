"""Differentiable float-domain proxy for per-image validation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .charbonnier import float32_context


def uniform_window_ssim(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    """Return per-image SSIM using scikit-image's default statistics.

    Scikit-image uses a uniform 7x7 window, sample covariance, and excludes the
    filter radius at the boundary. A valid average pool therefore reproduces
    exactly the pixels that enter its final mean.
    """

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Competition SSIM expects equal BCHW tensors")
    if data_range <= 0.0:
        raise ValueError("data_range must be positive")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least three")
    if min(prediction.shape[-2:]) < window_size:
        raise ValueError("Competition SSIM window is larger than the image")

    with float32_context(prediction):
        prediction_float = prediction.float()
        target_float = target.float()
        mean_prediction = F.avg_pool2d(
            prediction_float, kernel_size=window_size, stride=1
        )
        mean_target = F.avg_pool2d(
            target_float, kernel_size=window_size, stride=1
        )
        mean_prediction_squared = mean_prediction.square()
        mean_target_squared = mean_target.square()
        mean_product = mean_prediction * mean_target

        covariance_normalization = (window_size**2) / (window_size**2 - 1.0)
        variance_prediction = covariance_normalization * (
            F.avg_pool2d(
                prediction_float.square(), kernel_size=window_size, stride=1
            )
            - mean_prediction_squared
        )
        variance_target = covariance_normalization * (
            F.avg_pool2d(target_float.square(), kernel_size=window_size, stride=1)
            - mean_target_squared
        )
        covariance = covariance_normalization * (
            F.avg_pool2d(
                prediction_float * target_float,
                kernel_size=window_size,
                stride=1,
            )
            - mean_product
        )

        constant1 = (k1 * data_range) ** 2
        constant2 = (k2 * data_range) ** 2
        numerator = (2.0 * mean_product + constant1) * (
            2.0 * covariance + constant2
        )
        denominator = (
            mean_prediction_squared + mean_target_squared + constant1
        ) * (variance_prediction + variance_target + constant2)
        score_map = numerator / denominator.clamp_min(torch.finfo(torch.float32).eps)
        return score_map.mean(dim=(1, 2, 3)).clamp(min=-1.0, max=1.0)


def capped_per_image_psnr_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    psnr_cap: float = 60.0,
) -> Tensor:
    """Return a stable loss equivalent to maximizing mean capped PSNR.

    PSNR is calculated per image by the validator. The returned value is
    normalized to roughly ``[0, 1]`` and reaches zero at the configured cap.
    """

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Competition PSNR proxy expects equal BCHW tensors")
    if data_range <= 0.0 or not math.isfinite(float(data_range)):
        raise ValueError("data_range must be finite and positive")
    if psnr_cap <= 0.0 or not math.isfinite(float(psnr_cap)):
        raise ValueError("psnr_cap must be finite and positive")
    with float32_context(prediction):
        error = (prediction.float() - target.float()).square().mean(dim=(1, 2, 3))
        normalized_error = error / float(data_range**2)
        floor = 10.0 ** (-float(psnr_cap) / 10.0)
        capped = normalized_error.clamp_min(floor)
        return torch.log10(capped / floor) / (-math.log10(floor))


@dataclass(frozen=True, slots=True)
class CompetitionProxyOutput:
    """Per-batch components of :class:`CompetitionProxyLoss`."""

    total: Tensor
    ssim_loss: Tensor
    psnr_loss: Tensor
    mean_ssim: Tensor
    mean_psnr: Tensor


class CompetitionProxyLoss(nn.Module):
    """Blend float SSIM and per-image PSNR before uint8/JPEG conversion."""

    def __init__(
        self,
        *,
        ssim_weight: float = 0.7,
        data_range: float = 1.0,
        window_size: int = 7,
        psnr_cap: float = 60.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(ssim_weight) <= 1.0:
            raise ValueError("ssim_weight must lie in [0, 1]")
        self.ssim_weight = float(ssim_weight)
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.psnr_cap = float(psnr_cap)

    def forward(self, prediction: Tensor, target: Tensor) -> CompetitionProxyOutput:
        scores = uniform_window_ssim(
            prediction,
            target,
            data_range=self.data_range,
            window_size=self.window_size,
        )
        per_image_psnr_loss = capped_per_image_psnr_loss(
            prediction,
            target,
            data_range=self.data_range,
            psnr_cap=self.psnr_cap,
        )
        ssim_loss = 1.0 - scores.mean()
        psnr_loss = per_image_psnr_loss.mean()
        total = self.ssim_weight * ssim_loss + (1.0 - self.ssim_weight) * psnr_loss
        mean_psnr = self.psnr_cap * (1.0 - psnr_loss)
        return CompetitionProxyOutput(
            total=total,
            ssim_loss=ssim_loss,
            psnr_loss=psnr_loss,
            mean_ssim=scores.mean(),
            mean_psnr=mean_psnr,
        )
