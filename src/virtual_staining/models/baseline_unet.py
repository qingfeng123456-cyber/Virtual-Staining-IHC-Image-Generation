"""Residual U-Net baseline for deterministic single-marker restoration."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .registry import RestorationOutput, register_model


def _group_count(channels: int, maximum: int = 8) -> int:
    """Choose the largest valid GroupNorm group count up to ``maximum``."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """Two-convolution residual block using small-batch-safe normalization."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        features = self.activation(self.norm1(self.conv1(inputs)))
        features = self.norm2(self.conv2(features))
        return self.activation(features + residual)


class Downsample(nn.Module):
    """Learned stride-two reduction without max-pooling aliasing."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.normalization = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.normalization(self.projection(inputs)))


class DecoderBlock(nn.Module):
    """Bilinear upsampling followed by skip fusion and residual refinement."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.refinement = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        features = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        features = self.projection(features)
        return self.refinement(torch.cat((features, skip), dim=1))


@register_model("residual_unet", "baseline_unet", "unet")
class ResidualUNet(nn.Module):
    """Four-level residual U-Net with optional three-scale supervision."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        target_name: str = "CD68",
        deep_supervision: bool = True,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, base_channels) < 1:
            raise ValueError("Channel counts must be positive")
        if output_activation != "sigmoid":
            raise ValueError("ResidualUNet currently supports output_activation='sigmoid'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.target_name = target_name
        self.deep_supervision_enabled = deep_supervision

        channels = [base_channels * (2**index) for index in range(5)]
        self.encoder1 = ResidualBlock(in_channels, channels[0])
        self.down1 = Downsample(channels[0], channels[1])
        self.encoder2 = ResidualBlock(channels[1], channels[1])
        self.down2 = Downsample(channels[1], channels[2])
        self.encoder3 = ResidualBlock(channels[2], channels[2])
        self.down3 = Downsample(channels[2], channels[3])
        self.encoder4 = ResidualBlock(channels[3], channels[3])
        self.down4 = Downsample(channels[3], channels[4])
        self.bottleneck = ResidualBlock(channels[4], channels[4])

        self.decoder4 = DecoderBlock(channels[4], channels[3], channels[3])
        self.decoder3 = DecoderBlock(channels[3], channels[2], channels[2])
        self.decoder2 = DecoderBlock(channels[2], channels[1], channels[1])
        self.decoder1 = DecoderBlock(channels[1], channels[0], channels[0])

        self.output_head = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.auxiliary_heads = nn.ModuleList(
            [
                nn.Conv2d(channels[2], out_channels, kernel_size=1),
                nn.Conv2d(channels[1], out_channels, kernel_size=1),
            ]
        )

    def forward(self, inputs: Tensor, task_name: str | None = None) -> RestorationOutput:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected BCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )
        normalized_requested = (
            "".join(character for character in task_name.casefold() if character.isalnum())
            if task_name is not None
            else None
        )
        normalized_available = "".join(
            character for character in self.target_name.casefold() if character.isalnum()
        )
        if normalized_requested is not None and normalized_requested != normalized_available:
            raise KeyError(f"ResidualUNet only provides task {self.target_name!r}")

        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.down1(encoder1))
        encoder3 = self.encoder3(self.down2(encoder2))
        encoder4 = self.encoder4(self.down3(encoder3))
        bottleneck = self.bottleneck(self.down4(encoder4))

        decoder4 = self.decoder4(bottleneck, encoder4)
        decoder3 = self.decoder3(decoder4, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)
        prediction = torch.sigmoid(self.output_head(decoder1))

        supervision: dict[int, Tensor] = {prediction.shape[-2]: prediction}
        if self.deep_supervision_enabled:
            quarter = torch.sigmoid(self.auxiliary_heads[0](decoder3))
            half = torch.sigmoid(self.auxiliary_heads[1](decoder2))
            supervision = {
                quarter.shape[-2]: quarter,
                half.shape[-2]: half,
                prediction.shape[-2]: prediction,
            }
        return RestorationOutput(
            predictions={self.target_name: prediction},
            deep_supervision={self.target_name: supervision},
        )
