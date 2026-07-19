"""Robust pixel reconstruction loss and polarity-neutral structure weighting."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def float32_context(tensor: Tensor) -> AbstractContextManager[None]:
    """Disable surrounding AMP for numerically sensitive loss operations."""

    if tensor.device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=tensor.device.type, enabled=False)
    return nullcontext()


def structure_weight_map(target: Tensor, alpha: float = 1.0, epsilon: float = 1e-6) -> Tensor:
    """Build a soft target-gradient/Laplacian weight independent of bright/dark polarity."""

    if target.ndim != 4:
        raise ValueError(f"Expected BCHW target, got shape {tuple(target.shape)}")
    if alpha < 0.0:
        raise ValueError("Structure weight alpha cannot be negative")
    with float32_context(target):
        image = target.float().mean(dim=1, keepdim=True)
        kernel_x = image.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        kernel_y = kernel_x.transpose(-1, -2).contiguous()
        laplacian = image.new_tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3) / 4.0
        padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
        gradient_x = F.conv2d(padded, kernel_x)
        gradient_y = F.conv2d(padded, kernel_y)
        gradient = torch.sqrt(gradient_x.square() + gradient_y.square() + epsilon)
        curvature = F.conv2d(padded, laplacian).abs()
        structure = gradient + 0.5 * curvature
        normalizer = structure.mean(dim=(-2, -1), keepdim=True).clamp_min(epsilon)
        normalized = (structure / normalizer).clamp(max=4.0)
        return (1.0 + alpha * normalized).detach()


class CharbonnierLoss(nn.Module):
    """Differentiable robust L1 loss with optional spatial weights."""

    def __init__(self, epsilon: float = 1e-3, reduction: str = "mean") -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError("Charbonnier epsilon must be positive")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("Reduction must be 'none', 'mean', or 'sum'")
        self.epsilon = float(epsilon)
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        with float32_context(prediction):
            difference = prediction.float() - target.float()
            loss = torch.sqrt(difference.square() + self.epsilon**2)
            expanded_weight: Tensor | None = None
            if weight is not None:
                try:
                    expanded_weight = torch.broadcast_to(weight.float(), loss.shape)
                except RuntimeError as error:
                    raise ValueError(
                        f"Weight shape {tuple(weight.shape)} cannot broadcast to {tuple(loss.shape)}"
                    ) from error
                loss = loss * expanded_weight
            if self.reduction == "none":
                return loss
            if self.reduction == "sum":
                return loss.sum()
            if expanded_weight is not None:
                return loss.sum() / expanded_weight.sum().clamp_min(1e-12)
            return loss.mean()
