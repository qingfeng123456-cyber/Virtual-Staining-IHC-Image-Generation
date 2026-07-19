"""Explicit low-frequency base and bounded high-frequency detail output head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(slots=True)
class BaseDetailOutput:
    """Logit decomposition and display-space intermediates."""

    base_logits: Tensor
    detail_logits: Tensor
    final_logits: Tensor
    base: Tensor
    detail: Tensor
    prediction: Tensor


class LaplacianBaseDetailHead(nn.Module):
    """Combine an upsampled coarse logit field with a bounded detail residual."""

    def __init__(
        self,
        low_channels: int,
        full_channels: int,
        out_channels: int = 1,
        *,
        max_detail_amplitude: float = 1.0,
    ) -> None:
        super().__init__()
        if min(low_channels, full_channels, out_channels) < 1:
            raise ValueError("Base-detail channel counts must be positive")
        if max_detail_amplitude < 0.0:
            raise ValueError("max_detail_amplitude cannot be negative")
        self.max_detail_amplitude = float(max_detail_amplitude)
        self.low_resolution_head = nn.Conv2d(low_channels, out_channels, kernel_size=1)
        self.detail_head = nn.Sequential(
            nn.Conv2d(full_channels, full_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(full_channels, out_channels, kernel_size=1),
        )

    def forward(
        self,
        low_resolution_features: Tensor,
        full_resolution_features: Tensor,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> BaseDetailOutput:
        size = output_size or tuple(full_resolution_features.shape[-2:])
        base_logits = self.low_resolution_head(low_resolution_features)
        base_logits = F.interpolate(
            base_logits,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        full_features = full_resolution_features
        if full_features.shape[-2:] != size:
            full_features = F.interpolate(
                full_features,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        detail_logits = self.max_detail_amplitude * torch.tanh(self.detail_head(full_features))
        final_logits = base_logits + detail_logits
        return BaseDetailOutput(
            base_logits=base_logits,
            detail_logits=detail_logits,
            final_logits=final_logits,
            base=torch.sigmoid(base_logits),
            detail=detail_logits,
            prediction=torch.sigmoid(final_logits),
        )


LaplacianBaseDetailDecoder = LaplacianBaseDetailHead
