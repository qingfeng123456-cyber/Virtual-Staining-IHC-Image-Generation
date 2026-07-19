"""Low-weight log-amplitude frequency reconstruction loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .charbonnier import float32_context


class FrequencyAmplitudeLoss(nn.Module):
    """Compare log FFT magnitudes without imposing an unstable phase penalty."""

    def __init__(self, mode: str = "l1", epsilon: float = 1e-3) -> None:
        super().__init__()
        if mode not in {"l1", "charbonnier"}:
            raise ValueError("Frequency loss mode must be 'l1' or 'charbonnier'")
        if epsilon <= 0.0:
            raise ValueError("Frequency epsilon must be positive")
        self.mode = mode
        self.epsilon = float(epsilon)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("Frequency loss expects equal BCHW tensors")
        with float32_context(prediction):
            prediction_spectrum = torch.fft.rfft2(prediction.float(), norm="ortho")
            target_spectrum = torch.fft.rfft2(target.float(), norm="ortho")
            prediction_parts = torch.view_as_real(prediction_spectrum)
            target_parts = torch.view_as_real(target_spectrum)
            prediction_amplitude = torch.log1p(
                torch.sqrt(prediction_parts.square().sum(dim=-1) + 1e-12)
            )
            target_amplitude = torch.log1p(
                torch.sqrt(target_parts.square().sum(dim=-1) + 1e-12)
            )
            difference = prediction_amplitude - target_amplitude
            if self.mode == "l1":
                return difference.abs().mean()
            return torch.sqrt(difference.square() + self.epsilon**2).mean()
