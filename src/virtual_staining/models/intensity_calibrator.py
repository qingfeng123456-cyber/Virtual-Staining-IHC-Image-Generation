"""Bounded trainable global intensity calibration in logit space."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class CalibrationOutput:
    """Calibrated logits and bounded per-image parameters."""

    logits: Tensor
    gain: Tensor
    bias: Tensor


class GlobalIntensityCalibrator(nn.Module):
    """Predict a mild gain and bias, initialized to exact identity."""

    def __init__(
        self,
        bottleneck_channels: int,
        embedding_dim: int,
        out_channels: int = 1,
        *,
        max_gain_delta: float = 0.15,
        max_bias: float = 0.15,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if min(bottleneck_channels, embedding_dim, out_channels) < 1:
            raise ValueError("Calibrator dimensions must be positive")
        if max_gain_delta < 0.0 or max_bias < 0.0:
            raise ValueError("Calibration bounds cannot be negative")
        self.bottleneck_channels = int(bottleneck_channels)
        self.embedding_dim = int(embedding_dim)
        self.out_channels = int(out_channels)
        self.max_gain_delta = float(max_gain_delta)
        self.max_bias = float(max_bias)
        hidden = int(hidden_dim or max(32, embedding_dim))
        input_dim = bottleneck_channels + embedding_dim * 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels * 2),
        )
        output_layer = self.predictor[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("Calibrator output layer must be linear")
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(
        self,
        logits: Tensor,
        bottleneck: Tensor,
        marker_embedding: Tensor,
        organ_embedding: Tensor,
    ) -> CalibrationOutput:
        batch = logits.shape[0]
        if bottleneck.ndim != 4 or bottleneck.shape[:2] != (
            batch,
            self.bottleneck_channels,
        ):
            raise ValueError("Bottleneck shape does not match calibrator configuration")
        expected_embedding = (batch, self.embedding_dim)
        if marker_embedding.shape != expected_embedding or organ_embedding.shape != expected_embedding:
            raise ValueError(f"Expected marker and organ embeddings shaped {expected_embedding}")
        pooled = self.pool(bottleneck).flatten(1)
        parameters = self.predictor(
            torch.cat((pooled, marker_embedding, organ_embedding), dim=-1)
        )
        raw_gain, raw_bias = parameters.chunk(2, dim=-1)
        gain = 1.0 + self.max_gain_delta * torch.tanh(raw_gain)
        bias = self.max_bias * torch.tanh(raw_bias)
        gain = gain[:, :, None, None].to(dtype=logits.dtype)
        bias = bias[:, :, None, None].to(dtype=logits.dtype)
        return CalibrationOutput(logits=gain * logits + bias, gain=gain, bias=bias)


IntensityCalibrator = GlobalIntensityCalibrator
