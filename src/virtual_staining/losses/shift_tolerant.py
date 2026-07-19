"""Optional training-only loss robust to audited one- or two-pixel shifts."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .charbonnier import CharbonnierLoss, float32_context
from .ssim import SSIMLoss


def overlapping_shift_pair(
    prediction: Tensor,
    target: Tensor,
    dy: int,
    dx: int,
) -> tuple[Tensor, Tensor]:
    """Return equal, valid-overlap crops for a candidate target displacement."""

    if prediction.ndim != 4 or prediction.shape != target.shape:
        raise ValueError("Shift comparison expects equal BCHW tensors")
    height, width = prediction.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError("Shift leaves no overlapping pixels")
    prediction_y = slice(max(0, -dy), min(height, height - dy))
    target_y = slice(max(0, dy), min(height, height + dy))
    prediction_x = slice(max(0, -dx), min(width, width - dx))
    target_x = slice(max(0, dx), min(width, width + dx))
    return (
        prediction[:, :, prediction_y, prediction_x],
        target[:, :, target_y, target_x],
    )


class ShiftTolerantLoss(nn.Module):
    """Choose or softly weight the best low-resolution registered training loss."""

    def __init__(
        self,
        max_shift: int = 1,
        *,
        mode: str = "hard",
        temperature: float = 0.05,
        downsample: int = 2,
        charbonnier_weight: float = 0.7,
        ssim_weight: float = 0.3,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        if max_shift < 0:
            raise ValueError("max_shift cannot be negative")
        if mode not in {"hard", "softmin"}:
            raise ValueError("mode must be 'hard' or 'softmin'")
        if temperature <= 0 or downsample < 1:
            raise ValueError("temperature and downsample must be positive")
        if charbonnier_weight < 0 or ssim_weight < 0 or charbonnier_weight + ssim_weight <= 0:
            raise ValueError("Shift loss weights must be nonnegative with a positive sum")
        self.max_shift = int(max_shift)
        self.mode = mode
        self.temperature = float(temperature)
        self.downsample = int(downsample)
        self.charbonnier_weight = float(charbonnier_weight)
        self.ssim_weight = float(ssim_weight)
        self.loss_fn = loss_fn
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()

    def _loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.loss_fn is not None:
            return self.loss_fn(prediction, target)
        normalizer = self.charbonnier_weight + self.ssim_weight
        return (
            self.charbonnier_weight * self.charbonnier(prediction, target)
            + self.ssim_weight * self.ssim(prediction, target)
        ) / normalizer

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.ndim != 4 or prediction.shape != target.shape:
            raise ValueError("ShiftTolerantLoss expects equal BCHW tensors")
        with float32_context(prediction):
            prediction_float = prediction.float()
            target_float = target.float()
            if self.downsample > 1:
                size = (
                    max(1, prediction.shape[-2] // self.downsample),
                    max(1, prediction.shape[-1] // self.downsample),
                )
                prediction_float = F.interpolate(prediction_float, size=size, mode="area")
                target_float = F.interpolate(target_float, size=size, mode="area")
            effective_shift = min(
                self.max_shift,
                prediction_float.shape[-2] - 1,
                prediction_float.shape[-1] - 1,
            )
            losses: list[Tensor] = []
            for dy in range(-effective_shift, effective_shift + 1):
                for dx in range(-effective_shift, effective_shift + 1):
                    candidate, reference = overlapping_shift_pair(
                        prediction_float, target_float, dy, dx
                    )
                    losses.append(self._loss(candidate, reference))
            stacked = torch.stack(losses)
            if self.mode == "hard":
                index = int(stacked.detach().argmin())
                return stacked[index]
            weights = torch.softmax(-stacked.detach() / self.temperature, dim=0)
            return torch.sum(weights * stacked)
