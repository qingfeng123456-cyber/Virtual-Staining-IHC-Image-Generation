"""DAPI-native masked autoencoder for optional fold-local pretraining."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .naf_blocks import DecoderStage
from .naf_local_encoder import NAFLocalEncoder


@dataclass(slots=True)
class DAPIMAEOutput:
    """Masked reconstruction and encoder diagnostics.

    ``mask`` uses one for hidden pixels and zero for visible pixels.  It is
    deliberately single-channel even when the DAPI storage representation is
    RGB, so every storage channel receives the same spatial mask.
    """

    reconstruction: Tensor
    mask: Tensor
    masked_input: Tensor
    latent: Tensor

    @property
    def prediction(self) -> Tensor:
        """Expose the reconstruction through the common prediction spelling."""

        return self.reconstruction


class BlockMaskGenerator:
    """Generate exact-count block masks without relying on image content."""

    def __init__(self, block_size: int = 16, mask_ratio: float = 0.5) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1)")
        self.block_size = int(block_size)
        self.mask_ratio = float(mask_ratio)

    @staticmethod
    def _random_scores(
        shape: tuple[int, ...],
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> Tensor:
        if generator is None:
            return torch.rand(shape, device=device)
        generator_device = torch.device(getattr(generator, "device", "cpu"))
        scores = torch.rand(shape, device=generator_device, generator=generator)
        return scores.to(device=device)

    def __call__(
        self,
        batch_size: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
        mask_ratio: float | None = None,
    ) -> Tensor:
        if min(batch_size, height, width) < 1:
            raise ValueError("Mask dimensions must be positive")
        ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        if not 0.0 <= ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1)")
        grid_height = math.ceil(height / self.block_size)
        grid_width = math.ceil(width / self.block_size)
        block_count = grid_height * grid_width
        masked_count = min(block_count - 1, round(block_count * ratio))
        if masked_count == 0:
            return torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)

        scores = self._random_scores(
            (batch_size, block_count),
            device=device,
            generator=generator,
        )
        selected = scores.topk(masked_count, dim=1, largest=False).indices
        coarse = torch.zeros((batch_size, block_count), device=device, dtype=dtype)
        coarse.scatter_(1, selected, 1.0)
        coarse = coarse.view(batch_size, 1, grid_height, grid_width)
        pixel_mask = F.interpolate(
            coarse,
            size=(grid_height * self.block_size, grid_width * self.block_size),
            mode="nearest",
        )
        return pixel_mask[:, :, :height, :width]


class DAPIMaskedAutoencoder(nn.Module):
    """Lightweight convolutional MAE whose local encoder transfers to CAMP-VS.

    Selecting this model is the explicit feature flag for self-supervised
    pretraining.  It consumes DAPI only and has no target-marker branch.
    """

    def __init__(
        self,
        in_channels: int = 1,
        widths: Sequence[int] = (32, 64, 128, 256),
        encoder_depths: Sequence[int] = (1, 1, 2, 2),
        decoder_depths: Sequence[int] = (1, 1, 1),
        *,
        block_size: int = 16,
        mask_ratio: float = 0.5,
        masking_enabled: bool = True,
        use_sobel_input: bool = True,
        use_laplacian_input: bool = False,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in widths)
        decoder_depths = tuple(int(value) for value in decoder_depths)
        if len(widths) != 4 or len(decoder_depths) != 3:
            raise ValueError("DAPI MAE requires four widths and three decoder depths")
        if in_channels < 1 or any(value < 1 for value in (*widths, *decoder_depths)):
            raise ValueError("DAPI MAE channels and depths must be positive")

        self.in_channels = int(in_channels)
        self.masking_enabled = bool(masking_enabled)
        self.mask_generator = BlockMaskGenerator(block_size, mask_ratio)
        self.mask_token = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.local_encoder = NAFLocalEncoder(
            in_channels=self.in_channels,
            widths=widths,
            depths=encoder_depths,
            use_sobel_input=use_sobel_input,
            use_laplacian_input=use_laplacian_input,
        )
        self.decoder = nn.ModuleList(
            (
                DecoderStage(widths[3], widths[2], widths[2], decoder_depths[0]),
                DecoderStage(widths[2], widths[1], widths[1], decoder_depths[1]),
                DecoderStage(widths[1], widths[0], widths[0], decoder_depths[2]),
            )
        )
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(widths[0], widths[0], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(widths[0], self.in_channels, kernel_size=1),
        )

    def _validate_inputs(self, inputs: Tensor) -> None:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected BCHW DAPI input with {self.in_channels} channels, "
                f"got {tuple(inputs.shape)}"
            )
        if not torch.is_floating_point(inputs):
            raise TypeError("DAPI MAE inputs must be floating-point tensors")

    def _prepare_mask(
        self,
        inputs: Tensor,
        mask: Tensor | None,
        generator: torch.Generator | None,
        mask_ratio: float | None,
    ) -> Tensor:
        if mask is None:
            ratio = mask_ratio if self.masking_enabled else 0.0
            return self.mask_generator(
                inputs.shape[0],
                inputs.shape[-2],
                inputs.shape[-1],
                device=inputs.device,
                dtype=inputs.dtype,
                generator=generator,
                mask_ratio=ratio,
            )
        if mask.ndim != 4 or mask.shape[0] != inputs.shape[0] or mask.shape[-2:] != inputs.shape[-2:]:
            raise ValueError("Explicit MAE mask must have shape [B,1,H,W] or [B,C,H,W]")
        if mask.shape[1] not in {1, self.in_channels}:
            raise ValueError("Explicit MAE mask has an incompatible channel count")
        mask = mask.to(device=inputs.device, dtype=inputs.dtype)
        if not torch.isfinite(mask).all() or bool(((mask < 0.0) | (mask > 1.0)).any()):
            raise ValueError("Explicit MAE mask values must be finite and in [0, 1]")
        if mask.shape[1] != 1:
            channel_min = mask.amin(dim=1, keepdim=True)
            channel_max = mask.amax(dim=1, keepdim=True)
            if not torch.allclose(channel_min, channel_max):
                raise ValueError("All channels of an explicit MAE mask must be identical")
            mask = channel_min
        return (mask >= 0.5).to(dtype=inputs.dtype)

    def encode(self, inputs: Tensor) -> dict[int, Tensor]:
        """Encode unmasked DAPI for transfer or optional consistency diagnostics."""

        self._validate_inputs(inputs)
        return self.local_encoder(inputs).features

    def encode_embedding(self, inputs: Tensor) -> Tensor:
        """Return one normalized 1/16 representation per DAPI tile."""

        bottleneck = self.encode(inputs)[16]
        return F.normalize(bottleneck.float().mean(dim=(-2, -1)), dim=-1, eps=1e-6)

    def forward(
        self,
        inputs: Tensor,
        *,
        mask: Tensor | None = None,
        generator: torch.Generator | None = None,
        mask_ratio: float | None = None,
    ) -> DAPIMAEOutput:
        self._validate_inputs(inputs)
        pixel_mask = self._prepare_mask(inputs, mask, generator, mask_ratio)
        expanded_mask = pixel_mask.expand(-1, self.in_channels, -1, -1)
        mask_token = self.mask_token.to(dtype=inputs.dtype).expand_as(inputs)
        masked_input = inputs * (1.0 - expanded_mask) + mask_token * expanded_mask
        encoded = self.local_encoder(masked_input).features
        features = encoded[16]
        for stage, scale in zip(self.decoder, (8, 4, 2), strict=True):
            features = stage(features, encoded[scale])
        features = F.interpolate(
            features,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        reconstruction = torch.sigmoid(self.reconstruction_head(features))
        latent = F.normalize(encoded[16].float().mean(dim=(-2, -1)), dim=-1, eps=1e-6)
        return DAPIMAEOutput(
            reconstruction=reconstruction,
            mask=pixel_mask,
            masked_input=masked_input,
            latent=latent,
        )


DAPIMAE = DAPIMaskedAutoencoder

