"""Shared lightweight DAPI-only context tile encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .naf_blocks import LayerNorm2d


class DepthwiseContextBlock(nn.Module):
    """Small normalized depthwise residual block for context tiles."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.expand = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.activation = nn.GELU()
        self.project = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.depthwise(self.norm(inputs))
        features = self.project(self.activation(self.expand(features)))
        return inputs + features * self.scale


@dataclass(slots=True)
class ContextEncoderOutput:
    """Context tokens and the authoritative validity mask."""

    tokens: Tensor
    valid_mask: Tensor


class SharedTinyContextEncoder(nn.Module):
    """Encode every context tile with shared weights into one token."""

    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        token_dim: int = 192,
        *,
        grid_size: int = 3,
        depth: int = 2,
        stop_gradient: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1 or width < 2 or token_dim < 1 or depth < 1:
            raise ValueError("Context encoder dimensions and depth must be positive")
        if grid_size < 1 or grid_size % 2 == 0:
            raise ValueError("grid_size must be a positive odd integer")
        self.in_channels = int(in_channels)
        self.token_dim = int(token_dim)
        self.grid_size = int(grid_size)
        self.tile_count = grid_size * grid_size
        self.stop_gradient = bool(stop_gradient)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            DepthwiseContextBlock(width),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            *(DepthwiseContextBlock(width * 2) for _ in range(depth)),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.token_projection = nn.Linear(width * 2, token_dim)
        self.offset_embedding = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.center_embedding = nn.Parameter(torch.zeros(token_dim))
        nn.init.trunc_normal_(self.center_embedding, std=0.02)

    def default_offsets(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        radius = self.grid_size // 2
        offsets = [
            (row, column)
            for row in range(-radius, radius + 1)
            for column in range(-radius, radius + 1)
        ]
        return torch.tensor(offsets, device=device, dtype=dtype)

    def _normalize_metadata(
        self,
        tiles: Tensor,
        valid_mask: Tensor | None,
        offsets: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch, tile_count = tiles.shape[:2]
        if valid_mask is None:
            mask = torch.ones(batch, tile_count, device=tiles.device, dtype=torch.bool)
        else:
            mask = valid_mask.to(device=tiles.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(batch, -1)
            if mask.shape != (batch, tile_count):
                raise ValueError(
                    f"Expected context_valid_mask {(batch, tile_count)}, got {tuple(mask.shape)}"
                )
        if offsets is None:
            position = self.default_offsets(device=tiles.device, dtype=tiles.dtype)
            if tile_count != position.shape[0]:
                raise ValueError("context_offsets are required for a non-default tile count")
            position = position.unsqueeze(0).expand(batch, -1, -1)
        else:
            position = offsets.to(device=tiles.device, dtype=tiles.dtype)
            if position.ndim == 2:
                position = position.unsqueeze(0).expand(batch, -1, -1)
            if position.shape != (batch, tile_count, 2):
                raise ValueError(
                    f"Expected context_offsets {(batch, tile_count, 2)}, got {tuple(position.shape)}"
                )
        return mask, position

    def forward(
        self,
        context_tiles: Tensor,
        context_valid_mask: Tensor | None = None,
        context_offsets: Tensor | None = None,
        *,
        organ_embedding: Tensor | None = None,
    ) -> ContextEncoderOutput:
        if context_tiles.ndim != 5 or context_tiles.shape[2] != self.in_channels:
            raise ValueError(
                "Expected context tiles shaped "
                f"[B,N,{self.in_channels},H,W], got {tuple(context_tiles.shape)}"
            )
        batch, tile_count, channels, height, width = context_tiles.shape
        mask, offsets = self._normalize_metadata(
            context_tiles, context_valid_mask, context_offsets
        )
        flat_tiles = context_tiles.reshape(batch * tile_count, channels, height, width)
        features = self.stem(flat_tiles)
        tokens = self.pool(features).flatten(1)
        tokens = self.token_projection(tokens).reshape(batch, tile_count, self.token_dim)
        tokens = tokens + self.offset_embedding(offsets.float()).to(dtype=tokens.dtype)
        center = (offsets == 0).all(dim=-1, keepdim=True).to(dtype=tokens.dtype)
        tokens = tokens + center * self.center_embedding.to(dtype=tokens.dtype)
        if organ_embedding is not None:
            if organ_embedding.shape != (batch, self.token_dim):
                raise ValueError(
                    f"Expected organ embedding {(batch, self.token_dim)}, "
                    f"got {tuple(organ_embedding.shape)}"
                )
            tokens = tokens + organ_embedding[:, None, :].to(dtype=tokens.dtype)
        tokens = tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)
        if self.stop_gradient:
            tokens = tokens.detach()
        return ContextEncoderOutput(tokens=tokens, valid_mask=mask)


ContextTileEncoder = SharedTinyContextEncoder
