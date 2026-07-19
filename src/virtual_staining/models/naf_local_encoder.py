"""High-resolution NAF encoder used by CAMP-VS v2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .naf_blocks import NAFStage, SobelMagnitude


@dataclass(slots=True)
class LocalEncoderOutput:
    """Local features indexed by their spatial downsampling factor."""

    features: dict[int, Tensor]

    @property
    def bottleneck(self) -> Tensor:
        return self.features[16]

    @property
    def skips(self) -> tuple[Tensor, Tensor, Tensor]:
        return (self.features[8], self.features[4], self.features[2])


class LaplacianMagnitude(nn.Module):
    """Fixed single-channel Laplacian magnitude structural feature."""

    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        ) / 4.0
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3), persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        grayscale = inputs.mean(dim=1, keepdim=True)
        padded = F.pad(grayscale, (1, 1, 1, 1), mode="replicate")
        response = F.conv2d(padded, self.kernel.to(dtype=inputs.dtype))
        return response.abs()


class NAFLocalEncoder(nn.Module):
    """Four-stage NAF encoder emitting 1/2, 1/4, 1/8 and 1/16 maps."""

    def __init__(
        self,
        in_channels: int = 1,
        widths: Sequence[int] = (48, 96, 192, 384),
        depths: Sequence[int] = (2, 2, 4, 6),
        *,
        use_sobel_input: bool = True,
        use_laplacian_input: bool = False,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in widths)
        depths = tuple(int(value) for value in depths)
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if len(widths) != 4 or len(depths) != 4:
            raise ValueError("NAFLocalEncoder requires four widths and four depths")
        if any(value < 1 for value in (*widths, *depths)):
            raise ValueError("Encoder widths and depths must be positive")

        self.in_channels = int(in_channels)
        self.widths = widths
        self.use_sobel_input = bool(use_sobel_input)
        self.use_laplacian_input = bool(use_laplacian_input)
        self.sobel = SobelMagnitude() if self.use_sobel_input else None
        self.laplacian = LaplacianMagnitude() if self.use_laplacian_input else None
        structural_channels = int(self.use_sobel_input) + int(self.use_laplacian_input)
        self.stem = nn.Conv2d(
            self.in_channels + structural_channels,
            widths[0],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.stages = nn.ModuleList(
            NAFStage(channels, depth)
            for channels, depth in zip(widths, depths, strict=True)
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(widths[index], widths[index + 1], kernel_size=2, stride=2)
            for index in range(3)
        )

    def _prepare_input(self, inputs: Tensor) -> Tensor:
        structural = [inputs]
        if self.sobel is not None:
            structural.append(self.sobel(inputs))
        if self.laplacian is not None:
            structural.append(self.laplacian(inputs))
        return torch.cat(structural, dim=1)

    def forward(self, inputs: Tensor) -> LocalEncoderOutput:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected BCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )
        features = self.stem(self._prepare_input(inputs))
        outputs: dict[int, Tensor] = {}
        for index, stage in enumerate(self.stages):
            features = stage(features)
            outputs[2 ** (index + 1)] = features
            if index < len(self.downsamples):
                features = self.downsamples[index](features)
        return LocalEncoderOutput(features=outputs)
