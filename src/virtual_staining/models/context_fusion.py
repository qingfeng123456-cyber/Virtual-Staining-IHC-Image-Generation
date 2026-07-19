"""Masked context pooling, identity-initialized FiLM, and cross-attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .naf_blocks import LayerNorm2d


def _masked_softmax(logits: Tensor, valid_mask: Tensor) -> Tensor:
    """Numerically stable softmax that returns zero for an all-invalid row."""

    mask = valid_mask.to(dtype=torch.bool)
    masked_logits = logits.float().masked_fill(~mask, -torch.inf)
    any_valid = mask.any(dim=-1, keepdim=True)
    safe_logits = torch.where(any_valid, masked_logits, torch.zeros_like(masked_logits))
    weights = torch.softmax(safe_logits, dim=-1)
    return weights * mask.to(dtype=weights.dtype)


class MaskedAttentionPool(nn.Module):
    """Pool a short context sequence while respecting missing-neighbor masks."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        if token_dim < 1:
            raise ValueError("token_dim must be positive")
        self.query = nn.Parameter(torch.empty(token_dim))
        self.key = nn.Linear(token_dim, token_dim, bias=False)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 3 or valid_mask.shape != tokens.shape[:2]:
            raise ValueError("Context tokens and valid mask have incompatible shapes")
        keys = self.key(tokens)
        logits = torch.einsum("bnd,d->bn", keys.float(), self.query.float())
        logits = logits / max(tokens.shape[-1] ** 0.5, 1.0)
        weights = _masked_softmax(logits, valid_mask)
        pooled = torch.einsum("bn,bnd->bd", weights.to(tokens.dtype), tokens)
        return pooled, weights


class ZeroInitContextFiLM(nn.Module):
    """Context FiLM whose initialization is exactly the identity transform."""

    def __init__(
        self,
        channels: int,
        token_dim: int,
        *,
        context_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channels < 1 or token_dim < 1:
            raise ValueError("FiLM dimensions must be positive")
        if not 0.0 <= context_dropout < 1.0:
            raise ValueError("context_dropout must be in [0, 1)")
        self.context_dropout = float(context_dropout)
        self.pool = MaskedAttentionPool(token_dim)
        self.affine = nn.Linear(token_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def _drop_context(self, valid_mask: Tensor) -> Tensor:
        if not self.training or self.context_dropout == 0.0:
            return valid_mask
        keep = torch.rand(valid_mask.shape, device=valid_mask.device) >= self.context_dropout
        dropped = valid_mask & keep
        any_original = valid_mask.any(dim=1)
        any_remaining = dropped.any(dim=1)
        needs_fallback = any_original & ~any_remaining
        if needs_fallback.any():
            first_valid = valid_mask.float().argmax(dim=1)
            rows = needs_fallback.nonzero(as_tuple=False).flatten()
            dropped[rows, first_valid[rows]] = True
        return dropped

    def forward(
        self,
        local_features: Tensor,
        context_tokens: Tensor,
        context_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mask = self._drop_context(context_valid_mask.to(dtype=torch.bool).clone())
        pooled, attention = self.pool(context_tokens, mask)
        gamma, beta = self.affine(pooled).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None].to(dtype=local_features.dtype)
        beta = beta[:, :, None, None].to(dtype=local_features.dtype)
        return local_features * (1.0 + gamma) + beta, attention


class MultiScaleContextFusion(nn.Module):
    """Apply independent zero-initialized FiLM at selected local scales."""

    def __init__(
        self,
        channels_by_scale: Mapping[int, int],
        token_dim: int,
        *,
        fusion_scales: Sequence[int] = (4, 8, 16),
        context_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        selected = tuple(int(scale) for scale in fusion_scales)
        missing = sorted(set(selected).difference(channels_by_scale))
        if missing:
            raise ValueError(f"Missing channel definitions for scales: {missing}")
        self.fusion_scales = selected
        self.layers = nn.ModuleDict(
            {
                str(scale): ZeroInitContextFiLM(
                    channels_by_scale[scale],
                    token_dim,
                    context_dropout=context_dropout,
                )
                for scale in selected
            }
        )

    def forward(
        self,
        local_features: Mapping[int, Tensor],
        context_tokens: Tensor,
        context_valid_mask: Tensor,
    ) -> tuple[dict[int, Tensor], dict[str, Tensor]]:
        outputs = dict(local_features)
        attention: dict[str, Tensor] = {}
        for scale in self.fusion_scales:
            outputs[scale], attention[str(scale)] = self.layers[str(scale)](
                outputs[scale], context_tokens, context_valid_mask
            )
        return outputs, attention


class BottleneckContextCrossAttention(nn.Module):
    """Attend low-resolution local tokens to at most nine context tokens."""

    def __init__(
        self,
        local_channels: int,
        token_dim: int,
        *,
        heads: int = 4,
        residual_init: float = 0.0,
    ) -> None:
        super().__init__()
        if local_channels < 1 or token_dim < 1 or heads < 1:
            raise ValueError("Cross-attention dimensions must be positive")
        if token_dim % heads != 0:
            raise ValueError("token_dim must be divisible by attention heads")
        self.local_norm = LayerNorm2d(local_channels)
        self.query_projection = nn.Conv2d(local_channels, token_dim, kernel_size=1)
        self.context_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(token_dim, heads, batch_first=True)
        self.output_projection = nn.Conv2d(token_dim, local_channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(
        self,
        local_features: Tensor,
        context_tokens: Tensor,
        context_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if context_tokens.ndim != 3 or context_valid_mask.shape != context_tokens.shape[:2]:
            raise ValueError("Context tokens and mask have incompatible shapes")
        batch, _, height, width = local_features.shape
        if context_tokens.shape[0] != batch:
            raise ValueError("Local and context batch sizes differ")
        queries = self.query_projection(self.local_norm(local_features))
        queries = queries.flatten(2).transpose(1, 2)
        tokens = self.context_norm(context_tokens)
        mask = context_valid_mask.to(dtype=torch.bool)
        any_valid = mask.any(dim=1)
        safe_mask = mask.clone()
        if (~any_valid).any():
            safe_mask[~any_valid, 0] = True
            tokens = tokens.clone()
            tokens[~any_valid, 0] = 0.0
        attended, weights = self.attention(
            queries,
            tokens,
            tokens,
            key_padding_mask=~safe_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        attended = attended * any_valid[:, None, None].to(dtype=attended.dtype)
        attended = attended.transpose(1, 2).reshape(batch, -1, height, width)
        update = self.output_projection(attended)
        result = local_features + self.residual_scale.to(local_features.dtype) * update
        return result, weights


def resize_context_attention(attention: Tensor, size: tuple[int, int]) -> Tensor:
    """Resize a per-pixel attention diagnostic without changing model behavior."""

    if attention.ndim != 4:
        raise ValueError("Expected BCHW attention maps")
    return F.interpolate(attention, size=size, mode="nearest")
