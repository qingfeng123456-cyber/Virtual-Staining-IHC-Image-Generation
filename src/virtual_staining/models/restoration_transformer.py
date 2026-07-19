"""Memory-bounded Restormer-lite blocks for low-resolution feature maps."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .naf_blocks import LayerNorm2d, SimpleGate


class RestorationTransformerBlock(nn.Module):
    """Restormer-style channel attention followed by a gated feed-forward path."""

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if channels < 1 or heads < 1 or expansion < 1:
            raise ValueError("Transformer dimensions must be positive")
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        hidden_channels = channels * expansion
        if hidden_channels % 2:
            raise ValueError("Expanded channels must be even")
        self.channels = int(channels)
        self.heads = int(heads)

        self.norm1 = LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_depthwise = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.attention_output = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.attention_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

        self.norm2 = LayerNorm2d(channels)
        self.ffn_expand = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1)
        self.ffn_depthwise = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
        )
        self.gate = SimpleGate()
        self.ffn_project = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.ffn_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def _channel_attention(self, inputs: Tensor) -> Tensor:
        batch, channels, height, width = inputs.shape
        head_channels = channels // self.heads
        query, key, value = self.qkv_depthwise(self.qkv(inputs)).chunk(3, dim=1)
        query = query.reshape(batch, self.heads, head_channels, height * width)
        key = key.reshape(batch, self.heads, head_channels, height * width)
        value = value.reshape(batch, self.heads, head_channels, height * width)
        query = torch.nn.functional.normalize(query.float(), dim=-1)
        key = torch.nn.functional.normalize(key.float(), dim=-1)
        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = torch.softmax(attention * self.temperature.float(), dim=-1)
        features = torch.matmul(attention, value.float())
        return features.to(dtype=inputs.dtype).reshape(batch, channels, height, width)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self._channel_attention(self.norm1(inputs))
        features = self.attention_output(features)
        residual = inputs + features * self.attention_scale.to(dtype=inputs.dtype)
        features = self.ffn_depthwise(self.ffn_expand(self.norm2(residual)))
        features = self.ffn_project(self.gate(features))
        return residual + features * self.ffn_scale.to(dtype=inputs.dtype)


class RestormerLiteStage(nn.Module):
    """Stack of global blocks used only at 1/8 or 1/16 resolution."""

    def __init__(
        self,
        channels: int,
        blocks: int,
        *,
        heads: int,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if blocks < 0:
            raise ValueError("blocks cannot be negative")
        self.blocks = nn.Sequential(
            *(
                RestorationTransformerBlock(channels, heads=heads, expansion=expansion)
                for _ in range(blocks)
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.blocks(inputs)
