"""Small-weight per-image intensity-statistics alignment."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from .charbonnier import float32_context


class IntensityStatisticsLoss(nn.Module):
    """Match channel means, standard deviations, and optional low-frequency maps."""

    def __init__(
        self,
        mean_weight: float = 1.0,
        std_weight: float = 0.5,
        pooled_weight: float = 0.0,
        pooled_size: int = 16,
    ) -> None:
        super().__init__()
        values = (mean_weight, std_weight, pooled_weight)
        if any(value < 0 for value in values) or sum(values) <= 0:
            raise ValueError("Statistics weights must be nonnegative with a positive sum")
        if pooled_size < 1:
            raise ValueError("pooled_size must be positive")
        self.mean_weight = float(mean_weight)
        self.std_weight = float(std_weight)
        self.pooled_weight = float(pooled_weight)
        self.pooled_size = int(pooled_size)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.ndim != 4 or prediction.shape != target.shape:
            raise ValueError("IntensityStatisticsLoss expects equal BCHW tensors")
        with float32_context(prediction):
            prediction_float = prediction.float()
            target_float = target.float()
            spatial = (-2, -1)
            mean_loss = F.l1_loss(
                prediction_float.mean(dim=spatial), target_float.mean(dim=spatial)
            )
            std_loss = F.l1_loss(
                prediction_float.std(dim=spatial, unbiased=False),
                target_float.std(dim=spatial, unbiased=False),
            )
            total = self.mean_weight * mean_loss + self.std_weight * std_loss
            if self.pooled_weight > 0:
                size = (
                    min(self.pooled_size, prediction.shape[-2]),
                    min(self.pooled_size, prediction.shape[-1]),
                )
                prediction_pooled = F.adaptive_avg_pool2d(prediction_float, size)
                target_pooled = F.adaptive_avg_pool2d(target_float, size)
                total = total + self.pooled_weight * F.l1_loss(
                    prediction_pooled, target_pooled
                )
            return total
