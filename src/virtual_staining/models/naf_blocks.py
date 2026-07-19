"""Compact NAF/ConvNeXt-style building blocks used by the main model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """Per-pixel channel LayerNorm for BCHW tensors."""

    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.epsilon = epsilon

    def forward(self, inputs: Tensor) -> Tensor:
        features = inputs.permute(0, 2, 3, 1)
        features = F.layer_norm(
            features,
            (features.shape[-1],),
            self.weight,
            self.bias,
            self.epsilon,
        )
        return features.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    """Parameter-free multiplicative gate from NAFNet."""

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[1] % 2 != 0:
            raise ValueError("SimpleGate requires an even channel count")
        first, second = inputs.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """Depthwise gated residual block with channel attention and residual scaling."""

    def __init__(
        self,
        channels: int,
        depthwise_expansion: int = 2,
        feedforward_expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        depthwise_channels = channels * depthwise_expansion
        feedforward_channels = channels * feedforward_expansion
        if depthwise_channels % 2 or feedforward_channels % 2:
            raise ValueError("Expanded channel counts must be even")

        self.norm1 = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, depthwise_channels, kernel_size=1)
        self.depthwise = nn.Conv2d(
            depthwise_channels,
            depthwise_channels,
            kernel_size=3,
            padding=1,
            groups=depthwise_channels,
        )
        self.gate1 = SimpleGate()
        gated_channels = depthwise_channels // 2
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gated_channels, gated_channels, kernel_size=1),
        )
        self.project = nn.Conv2d(gated_channels, channels, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.norm2 = LayerNorm2d(channels)
        self.feedforward_expand = nn.Conv2d(channels, feedforward_channels, kernel_size=1)
        self.gate2 = SimpleGate()
        self.feedforward_project = nn.Conv2d(feedforward_channels // 2, channels, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.depthwise(self.expand(self.norm1(inputs)))
        features = self.gate1(features)
        features = features * self.channel_attention(features)
        features = self.dropout1(self.project(features))
        residual = inputs + features * self.beta

        features = self.feedforward_expand(self.norm2(residual))
        features = self.gate2(features)
        features = self.dropout2(self.feedforward_project(features))
        return residual + features * self.gamma


class NAFStage(nn.Module):
    """A configurable stack of NAF blocks."""

    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("NAF stage depth must be at least one")
        self.blocks = nn.Sequential(*(NAFBlock(channels) for _ in range(depth)))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.blocks(inputs)


class SobelMagnitude(nn.Module):
    """Fixed differentiable Sobel magnitude derived from the DAPI input."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 8.0
        kernel_y = kernel_x.transpose(0, 1).contiguous()
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3), persistent=False)
        self.epsilon = epsilon

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[1] == 3:
            weights = inputs.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            grayscale = (inputs * weights).sum(dim=1, keepdim=True)
        else:
            grayscale = inputs.mean(dim=1, keepdim=True)
        padded = F.pad(grayscale, (1, 1, 1, 1), mode="replicate")
        kernel_x = self.kernel_x.to(dtype=inputs.dtype)
        kernel_y = self.kernel_y.to(dtype=inputs.dtype)
        gradient_x = F.conv2d(padded, kernel_x)
        gradient_y = F.conv2d(padded, kernel_y)
        return torch.sqrt(gradient_x.square() + gradient_y.square() + self.epsilon)


class TaskAdapter(nn.Module):
    """Lightweight task-specific residual adapter with embedding-derived FiLM."""

    def __init__(self, channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.normalization = LayerNorm2d(channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.film = nn.Linear(embedding_dim, channels * 2)
        self.residual_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, inputs: Tensor, embedding: Tensor) -> Tensor:
        scale, shift = self.film(embedding).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        conditioned = inputs * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift
        features = self.normalization(conditioned)
        features = self.activation(self.depthwise(features))
        features = self.pointwise(features)
        return conditioned + features * self.residual_scale


class DecoderStage(nn.Module):
    """Bilinear upsampling, skip projection, and NAF refinement."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, depth: int) -> None:
        super().__init__()
        self.upsample_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_fusion = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1)
        self.refinement = NAFStage(out_channels, depth)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        features = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        features = self.upsample_projection(features)
        features = self.skip_fusion(torch.cat((features, skip), dim=1))
        return self.refinement(features)
