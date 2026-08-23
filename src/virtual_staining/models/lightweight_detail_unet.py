"""Small large-kernel U-Net branch for high-resolution restoration detail."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .naf_blocks import LayerNorm2d


class LargeKernelGatedResidualBlock(nn.Module):
    """GPU-friendly depthwise/pointwise block with multiplicative gating."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("Block channels must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least three")
        if expansion < 1:
            raise ValueError("expansion must be positive")

        hidden_channels = channels * expansion
        self.normalization = LayerNorm2d(channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.expand = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1)
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.depthwise(self.normalization(inputs))
        value, gate = self.expand(features).chunk(2, dim=1)
        update = self.project(F.silu(value, inplace=False) * gate)
        return inputs + update * self.residual_scale


class LightweightDetailUNet(nn.Module):
    """Four-scale U-Net producing zero-initialized updates for fine main features.

    The fourth (1/8-resolution) level only supplies U-Net context.  Updates are
    emitted for 1, 1/2 and 1/4 resolution, leaving bottleneck global/frequency
    processing to the main restoration network.
    """

    def __init__(
        self,
        in_channels: int,
        main_channels: Sequence[int],
        *,
        widths: Sequence[int] = (16, 24, 32, 48),
        depths: Sequence[int] = (1, 1, 1, 1),
        kernel_size: int = 7,
        expansion: int = 2,
        fusion_scales: Sequence[int] = (1, 2, 4),
        residual_init: float = 0.0,
    ) -> None:
        super().__init__()
        detail_widths = tuple(int(width) for width in widths)
        detail_depths = tuple(int(depth) for depth in depths)
        destination_channels = tuple(int(channel) for channel in main_channels)
        selected_fusion_scales = tuple(int(scale) for scale in fusion_scales)
        if in_channels < 1:
            raise ValueError("LightweightDetailUNet in_channels must be positive")
        if len(detail_widths) != 4 or any(width < 1 for width in detail_widths):
            raise ValueError("widths must contain four positive integers")
        if len(detail_depths) != 4 or any(depth < 1 for depth in detail_depths):
            raise ValueError("depths must contain four positive integers")
        if len(destination_channels) != 3 or any(
            channel < 1 for channel in destination_channels
        ):
            raise ValueError("main_channels must contain three positive integers")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least three")
        if expansion < 1:
            raise ValueError("expansion must be positive")
        if not selected_fusion_scales or len(set(selected_fusion_scales)) != len(
            selected_fusion_scales
        ):
            raise ValueError("fusion_scales must contain unique scales")
        invalid_scales = sorted(set(selected_fusion_scales).difference({1, 2, 4}))
        if invalid_scales:
            raise ValueError(
                f"fusion_scales must be drawn from 1/2/4, got {invalid_scales}"
            )
        if float(residual_init) != 0.0:
            raise ValueError(
                "residual_init must be zero so the detail branch starts as an identity"
            )

        self.fusion_scales = selected_fusion_scales

        self.input_projection = nn.Conv2d(
            int(in_channels), detail_widths[0], kernel_size=3, padding=1
        )
        self.encoder_blocks = nn.ModuleList(
            nn.Sequential(
                *(
                    LargeKernelGatedResidualBlock(
                        width,
                        kernel_size=kernel_size,
                        expansion=expansion,
                    )
                    for _ in range(depth)
                )
            )
            for width, depth in zip(detail_widths, detail_depths, strict=True)
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(
                detail_widths[index],
                detail_widths[index + 1],
                kernel_size=3,
                stride=2,
                padding=1,
            )
            for index in range(3)
        )
        self.decoder_projections = nn.ModuleList(
            nn.Conv2d(detail_widths[index + 1], detail_widths[index], kernel_size=1)
            for index in range(2, -1, -1)
        )
        scale_to_index = {1: 0, 2: 1, 4: 2}
        self.residual_projections = nn.ModuleDict(
            {
                f"scale_{scale}": nn.Conv2d(
                    detail_widths[scale_to_index[scale]],
                    destination_channels[scale_to_index[scale]],
                    kernel_size=1,
                )
                for scale in selected_fusion_scales
            }
        )
        for projection in self.residual_projections.values():
            nn.init.zeros_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def forward(self, inputs: Tensor) -> dict[int, Tensor]:
        if inputs.ndim != 4:
            raise ValueError(
                f"LightweightDetailUNet expects BCHW input, got {tuple(inputs.shape)}"
            )

        encoder_features: list[Tensor] = []
        features = self.input_projection(inputs)
        for index, block in enumerate(self.encoder_blocks):
            features = block(features)
            encoder_features.append(features)
            if index < len(self.downsamples):
                features = self.downsamples[index](features)

        decoded = encoder_features[-1]
        decoded_features: dict[int, Tensor] = {}
        for target_index, projection in zip(
            range(2, -1, -1), self.decoder_projections, strict=True
        ):
            skip = encoder_features[target_index]
            decoded = F.interpolate(
                decoded,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded = projection(decoded) + skip
            decoded_features[target_index] = decoded

        scale_to_index = {1: 0, 2: 1, 4: 2}
        return {
            scale: self.residual_projections[f"scale_{scale}"](
                decoded_features[scale_to_index[scale]]
            )
            for scale in self.fusion_scales
        }
