"""Sobel-domain reconstruction losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .charbonnier import float32_context


def sobel_gradients(image: Tensor) -> tuple[Tensor, Tensor]:
    """Return channel-wise Sobel x/y gradients for a BCHW image."""

    if image.ndim != 4:
        raise ValueError(f"Expected BCHW image, got {tuple(image.shape)}")
    with float32_context(image):
        values = image.float()
        channels = values.shape[1]
        kernel_x = values.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        kernel_y = kernel_x.transpose(-1, -2).contiguous()
        kernel_x = kernel_x.repeat(channels, 1, 1, 1)
        kernel_y = kernel_y.repeat(channels, 1, 1, 1)
        padded = F.pad(values, (1, 1, 1, 1), mode="replicate")
        return (
            F.conv2d(padded, kernel_x, groups=channels),
            F.conv2d(padded, kernel_y, groups=channels),
        )


class GradientLoss(nn.Module):
    """Compare prediction and target Sobel gradients with L1 or Charbonnier."""

    def __init__(self, mode: str = "charbonnier", epsilon: float = 1e-3) -> None:
        super().__init__()
        if mode not in {"l1", "charbonnier"}:
            raise ValueError("Gradient loss mode must be 'l1' or 'charbonnier'")
        if epsilon <= 0.0:
            raise ValueError("Gradient epsilon must be positive")
        self.mode = mode
        self.epsilon = float(epsilon)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError("Gradient loss expects equal prediction and target shapes")
        prediction_x, prediction_y = sobel_gradients(prediction)
        target_x, target_y = sobel_gradients(target)
        difference_x = prediction_x - target_x
        difference_y = prediction_y - target_y
        if self.mode == "l1":
            return 0.5 * (difference_x.abs().mean() + difference_y.abs().mean())
        loss_x = torch.sqrt(difference_x.square() + self.epsilon**2).mean()
        loss_y = torch.sqrt(difference_y.square() + self.epsilon**2).mean()
        return 0.5 * (loss_x + loss_y)
