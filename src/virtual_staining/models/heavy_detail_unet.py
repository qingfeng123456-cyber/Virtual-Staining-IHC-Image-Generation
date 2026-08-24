"""Heavy GPU-efficient U-Net branch for dense virtual-staining detail."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .naf_blocks import LayerNorm2d


class DenseHighResolutionBlock(nn.Module):
    """Depthwise spatial mixing followed by a dense gated channel mixer.

    The expensive dense operations are 1x1 convolutions, which map efficiently
    to CUDA tensor cores.  Spatial processing remains depthwise at the two
    highest-resolution levels so this branch can be made substantially deeper
    without paying for repeated dense 3x3 convolutions.
    """

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 3,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("Block channels must be positive")
        if expansion < 1:
            raise ValueError("expansion must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least three")

        hidden_channels = channels * expansion
        self.normalization = LayerNorm2d(channels)
        self.spatial_mixer = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.expand = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1)
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.normalization(inputs)
        mixed = self.spatial_mixer(normalized)
        value, gate = self.expand(mixed).chunk(2, dim=1)
        update = self.project(F.silu(value, inplace=False) * gate)
        update = update * self.channel_gate(normalized)
        return inputs + update * self.residual_scale


class OmniKernelResidualBlock(nn.Module):
    """Local, axial-large-kernel and dilated depthwise mixing at low resolution."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 3,
        large_kernel_size: int = 11,
        dilation: int = 3,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("Block channels must be positive")
        if expansion < 1:
            raise ValueError("expansion must be positive")
        if large_kernel_size < 3 or large_kernel_size % 2 == 0:
            raise ValueError("large_kernel_size must be an odd integer of at least three")
        if dilation < 1:
            raise ValueError("dilation must be positive")

        hidden_channels = channels * expansion
        large_padding = large_kernel_size // 2
        self.normalization = LayerNorm2d(channels)
        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.large_horizontal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, large_kernel_size),
            padding=(0, large_padding),
            groups=channels,
        )
        self.large_vertical = nn.Conv2d(
            channels,
            channels,
            kernel_size=(large_kernel_size, 1),
            padding=(large_padding, 0),
            groups=channels,
        )
        self.dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.kernel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(16, channels // 4), kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(16, channels // 4), channels * 3, kernel_size=1),
        )
        self.expand = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1)
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

        final_gate = self.kernel_gate[-1]
        if not isinstance(final_gate, nn.Conv2d):
            raise RuntimeError("Unexpected kernel gate layout")
        nn.init.zeros_(final_gate.weight)
        nn.init.zeros_(final_gate.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        normalized = self.normalization(inputs)
        local = self.local(normalized)
        large = self.large_vertical(self.large_horizontal(normalized))
        dilated = self.dilated(normalized)
        batch, channels = inputs.shape[:2]
        logits = self.kernel_gate(normalized).reshape(batch, 3, channels, 1, 1)
        weights = torch.softmax(logits.float(), dim=1).to(dtype=inputs.dtype)
        mixed = (
            local * weights[:, 0]
            + large * weights[:, 1]
            + dilated * weights[:, 2]
        )
        value, gate = self.expand(mixed).chunk(2, dim=1)
        update = self.project(F.silu(value, inplace=False) * gate)
        return inputs + update * self.residual_scale


class _HeavyDecoderStage(nn.Module):
    """Upsample, fuse an encoder skip, and refine at one decoder scale."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        depth: int,
        high_resolution: bool,
        expansion: int,
        local_kernel_size: int,
        large_kernel_size: int,
    ) -> None:
        super().__init__()
        self.upsample_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        block_type = DenseHighResolutionBlock if high_resolution else OmniKernelResidualBlock
        if high_resolution:
            blocks = [
                block_type(
                    out_channels,
                    expansion=expansion,
                    kernel_size=local_kernel_size,
                )
                for _ in range(depth)
            ]
        else:
            blocks = [
                block_type(
                    out_channels,
                    expansion=expansion,
                    large_kernel_size=large_kernel_size,
                )
                for _ in range(depth)
            ]
        self.refinement = nn.ModuleList(blocks)

    def forward(
        self,
        inputs: Tensor,
        skip: Tensor,
        *,
        use_activation_checkpoint: bool,
    ) -> Tensor:
        features = F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features = self.upsample_projection(features)
        features = self.skip_fusion(torch.cat((features, skip), dim=1))
        return _run_blocks(
            self.refinement,
            features,
            use_activation_checkpoint=use_activation_checkpoint,
        )


def _run_blocks(
    blocks: nn.ModuleList,
    inputs: Tensor,
    *,
    use_activation_checkpoint: bool,
) -> Tensor:
    features = inputs
    for block in blocks:
        if use_activation_checkpoint and torch.is_grad_enabled():
            features = checkpoint(block, features, use_reentrant=False)
        else:
            features = block(features)
    return features


class HeavyDetailUNet(nn.Module):
    """Four-scale heavy U-Net producing raw features for adaptive fusion.

    ``decoder_refinement_depths`` is ordered by output scales ``(1, 2, 4)``.
    The encoder's scale-8 feature is retained as the bottleneck route, while
    scales 1/2/4 contain decoded features with matching encoder skips.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        widths: Sequence[int] = (56, 88, 128, 176),
        encoder_depths: Sequence[int] = (3, 4, 6, 6),
        decoder_refinement_depths: Sequence[int] = (1, 2, 3),
        expansion: int = 3,
        local_kernel_size: int = 3,
        large_kernel_size: int = 11,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        heavy_widths = tuple(int(width) for width in widths)
        stage_depths = tuple(int(depth) for depth in encoder_depths)
        decoder_depths = tuple(int(depth) for depth in decoder_refinement_depths)
        if in_channels < 1:
            raise ValueError("HeavyDetailUNet in_channels must be positive")
        if len(heavy_widths) != 4 or any(width < 1 for width in heavy_widths):
            raise ValueError("widths must contain four positive integers")
        if len(stage_depths) != 4 or any(depth < 1 for depth in stage_depths):
            raise ValueError("encoder_depths must contain four positive integers")
        if len(decoder_depths) != 3 or any(depth < 1 for depth in decoder_depths):
            raise ValueError(
                "decoder_refinement_depths must contain three positive integers"
            )
        if expansion < 1:
            raise ValueError("expansion must be positive")
        if local_kernel_size < 3 or local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be an odd integer of at least three")
        if large_kernel_size < 3 or large_kernel_size % 2 == 0:
            raise ValueError("large_kernel_size must be an odd integer of at least three")

        self.widths = heavy_widths
        self.activation_checkpoint = bool(activation_checkpoint)
        self.input_projection = nn.Conv2d(
            int(in_channels), heavy_widths[0], kernel_size=3, padding=1
        )

        encoder_stages: list[nn.ModuleList] = []
        for index, (width, depth) in enumerate(
            zip(heavy_widths, stage_depths, strict=True)
        ):
            if index < 2:
                blocks = [
                    DenseHighResolutionBlock(
                        width,
                        expansion=expansion,
                        kernel_size=local_kernel_size,
                    )
                    for _ in range(depth)
                ]
            else:
                blocks = [
                    OmniKernelResidualBlock(
                        width,
                        expansion=expansion,
                        large_kernel_size=large_kernel_size,
                    )
                    for _ in range(depth)
                ]
            encoder_stages.append(nn.ModuleList(blocks))
        self.encoder_stages = nn.ModuleList(encoder_stages)
        self.downsamples = nn.ModuleList(
            nn.Sequential(
                LayerNorm2d(heavy_widths[index]),
                nn.Conv2d(
                    heavy_widths[index],
                    heavy_widths[index + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            )
            for index in range(3)
        )

        self.decoder_stages = nn.ModuleList(
            _HeavyDecoderStage(
                heavy_widths[target_index + 1],
                heavy_widths[target_index],
                depth=decoder_depths[target_index],
                high_resolution=target_index < 2,
                expansion=expansion,
                local_kernel_size=local_kernel_size,
                large_kernel_size=large_kernel_size,
            )
            for target_index in range(2, -1, -1)
        )

    def forward(self, inputs: Tensor) -> dict[int, Tensor]:
        if inputs.ndim != 4:
            raise ValueError(
                f"HeavyDetailUNet expects BCHW input, got {tuple(inputs.shape)}"
            )

        encoder_features: list[Tensor] = []
        features = self.input_projection(inputs)
        for index, stage in enumerate(self.encoder_stages):
            features = _run_blocks(
                stage,
                features,
                use_activation_checkpoint=self.activation_checkpoint and self.training,
            )
            encoder_features.append(features)
            if index < len(self.downsamples):
                features = self.downsamples[index](features)

        decoded = encoder_features[-1]
        decoded_features: dict[int, Tensor] = {}
        for target_index, stage in zip(
            range(2, -1, -1), self.decoder_stages, strict=True
        ):
            decoded = stage(
                decoded,
                encoder_features[target_index],
                use_activation_checkpoint=self.activation_checkpoint and self.training,
            )
            decoded_features[2**target_index] = decoded

        return {
            1: decoded_features[1],
            2: decoded_features[2],
            4: decoded_features[4],
            8: encoder_features[3],
        }
