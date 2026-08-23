"""Protein-expression-aware supervision for bright fluorescence targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .charbonnier import float32_context


@dataclass(slots=True)
class FluorescenceForegroundOutput:
    """Scalar foreground objective and detached-friendly diagnostics."""

    total: Tensor
    mse: Tensor
    dice: Tensor
    intensity: Tensor
    foreground_fraction: Tensor


class FluorescenceForegroundLoss(nn.Module):
    """Emphasize bright protein signal without discarding background pixels.

    A per-image threshold is computed from the target mean and standard
    deviation.  The resulting soft mask is detached: it is supervision derived
    only from the official training target, not a learnable shortcut.  The
    global restoration loss remains active elsewhere; this module is a small
    additive term inspired by foreground-aware fluorescence staining and
    protein-distribution consistency objectives.
    """

    def __init__(
        self,
        *,
        threshold_std_scale: float = 0.25,
        temperature: float = 0.05,
        mse_weight: float = 1.0,
        dice_weight: float = 0.10,
        intensity_weight: float = 0.25,
        min_activity_std: float = 0.01,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if threshold_std_scale < 0.0:
            raise ValueError("threshold_std_scale cannot be negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        weights = (mse_weight, dice_weight, intensity_weight)
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError(
                "Foreground weights must be nonnegative with a positive sum"
            )
        if min_activity_std < 0.0:
            raise ValueError("min_activity_std cannot be negative")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.threshold_std_scale = float(threshold_std_scale)
        self.temperature = float(temperature)
        self.mse_weight = float(mse_weight)
        self.dice_weight = float(dice_weight)
        self.intensity_weight = float(intensity_weight)
        self.min_activity_std = float(min_activity_std)
        self.epsilon = float(epsilon)

    @staticmethod
    def _weighted_batch_mean(values: Tensor, activity: Tensor, epsilon: float) -> Tensor:
        activity_flat = activity.flatten()
        values_flat = values.flatten()
        return (values_flat * activity_flat).sum() / activity_flat.sum().clamp_min(epsilon)

    def forward(self, prediction: Tensor, target: Tensor) -> FluorescenceForegroundOutput:
        if prediction.ndim != 4 or prediction.shape != target.shape:
            raise ValueError(
                "FluorescenceForegroundLoss expects equal BCHW prediction and target tensors"
            )
        with float32_context(prediction):
            prediction_float = prediction.float()
            target_float = target.float()
            spatial = (-2, -1)
            target_mean = target_float.mean(dim=spatial, keepdim=True)
            target_std = target_float.std(dim=spatial, unbiased=False, keepdim=True)
            threshold = (
                target_mean + self.threshold_std_scale * target_std
            ).clamp(0.0, 1.0).detach()
            target_mask = torch.sigmoid(
                (target_float - threshold) / self.temperature
            ).detach()

            if self.min_activity_std == 0.0:
                activity = torch.ones_like(target_std)
            else:
                activity = (
                    (target_std - self.min_activity_std)
                    / max(self.min_activity_std, self.epsilon)
                ).clamp(0.0, 1.0).detach()

            mask_sum = target_mask.sum(dim=spatial).clamp_min(self.epsilon)
            squared_error = (prediction_float - target_float).square()
            masked_mse = (squared_error * target_mask).sum(dim=spatial) / mask_sum
            mse = self._weighted_batch_mean(masked_mse, activity, self.epsilon)

            prediction_mask = torch.sigmoid(
                (prediction_float - threshold) / self.temperature
            )
            intersection = (prediction_mask * target_mask).sum(dim=spatial)
            denominator = (
                prediction_mask.sum(dim=spatial) + target_mask.sum(dim=spatial)
            ).clamp_min(self.epsilon)
            dice_per_channel = 1.0 - (2.0 * intersection + self.epsilon) / (
                denominator + self.epsilon
            )
            dice = self._weighted_batch_mean(
                dice_per_channel, activity, self.epsilon
            )

            prediction_intensity = (
                prediction_float * target_mask
            ).sum(dim=spatial) / mask_sum
            target_intensity = (target_float * target_mask).sum(dim=spatial) / mask_sum
            intensity_per_channel = F.smooth_l1_loss(
                prediction_intensity,
                target_intensity,
                reduction="none",
                beta=0.02,
            )
            intensity = self._weighted_batch_mean(
                intensity_per_channel, activity, self.epsilon
            )
            total = (
                self.mse_weight * mse
                + self.dice_weight * dice
                + self.intensity_weight * intensity
            )
            return FluorescenceForegroundOutput(
                total=total,
                mse=mse,
                dice=dice,
                intensity=intensity,
                foreground_fraction=target_mask.mean().detach(),
            )
