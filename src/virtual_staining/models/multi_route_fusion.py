"""Content-adaptive fusion for parallel local, detail, and frequency routes."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .naf_blocks import LayerNorm2d


def _validate_route_pair(reference: Tensor, candidate: Tensor, name: str) -> None:
    if reference.ndim != 4 or candidate.ndim != 4:
        raise ValueError(f"{name} fusion expects BCHW tensors")
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError(f"{name} fusion route batch sizes differ")
    if reference.shape[-2:] != candidate.shape[-2:]:
        raise ValueError(f"{name} fusion route spatial dimensions differ")


def _normalized_entropy(weights: Tensor) -> Tensor:
    branch_count = weights.shape[1]
    probabilities = weights.float().clamp_min(torch.finfo(torch.float32).tiny)
    entropy = -(probabilities * probabilities.log()).sum(dim=1).mean()
    return entropy / math.log(float(branch_count))


def _norm_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    numerator_rms = numerator.float().square().mean().sqrt()
    denominator_rms = denominator.float().square().mean().sqrt()
    return numerator_rms / denominator_rms.clamp_min(1e-8)


class _ContentAdaptiveGate(nn.Module):
    """Combine channel and spatial evidence into normalized route weights."""

    def __init__(self, channels: int, routes: int, *, reduction: int = 8) -> None:
        super().__init__()
        if channels < 1 or routes < 2:
            raise ValueError("Gate dimensions must be positive with at least two routes")
        if reduction < 1:
            raise ValueError("reduction must be positive")
        hidden_channels = max(16, channels // reduction)
        concatenated_channels = channels * routes
        self.routes = routes
        self.channels = channels
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(concatenated_channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * routes, kernel_size=1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(concatenated_channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, routes, kernel_size=3, padding=1),
        )
        for final_layer in (self.channel_gate[-1], self.spatial_gate[-1]):
            if not isinstance(final_layer, nn.Conv2d):
                raise RuntimeError("Unexpected adaptive gate layout")
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)

    def forward(self, routes: tuple[Tensor, ...]) -> Tensor:
        if len(routes) != self.routes:
            raise ValueError(f"Expected {self.routes} routes, received {len(routes)}")
        concatenated = torch.cat(routes, dim=1)
        batch, _, height, width = concatenated.shape
        channel_logits = self.channel_gate(concatenated).reshape(
            batch,
            self.routes,
            self.channels,
            1,
            1,
        )
        spatial_logits = self.spatial_gate(concatenated).reshape(
            batch,
            self.routes,
            1,
            height,
            width,
        )
        logits = channel_logits.float() + spatial_logits.float()
        return torch.softmax(logits, dim=1).to(dtype=concatenated.dtype)


class CrossGatedSkipFusion(nn.Module):
    """Fuse main and heavy U-Net routes using spatial-and-channel softmax gates."""

    def __init__(
        self,
        main_channels: int,
        heavy_channels: int,
        *,
        out_channels: int | None = None,
        gate_reduction: int = 8,
        residual_init: float = 0.02,
    ) -> None:
        super().__init__()
        output_channels = int(out_channels or main_channels)
        if main_channels < 1 or heavy_channels < 1 or output_channels < 1:
            raise ValueError("Fusion channel counts must be positive")
        if not 0.0 < residual_init <= 1.0:
            raise ValueError("residual_init must lie in (0, 1]")

        self.main_projection = (
            nn.Identity()
            if main_channels == output_channels
            else nn.Conv2d(main_channels, output_channels, kernel_size=1)
        )
        self.heavy_projection = nn.Conv2d(
            heavy_channels, output_channels, kernel_size=1
        )
        self.main_normalization = LayerNorm2d(output_channels)
        self.heavy_normalization = LayerNorm2d(output_channels)
        self.gate = _ContentAdaptiveGate(
            output_channels,
            2,
            reduction=gate_reduction,
        )
        self.output_projection = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.residual_alpha = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(
        self,
        main_features: Tensor,
        heavy_features: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _validate_route_pair(main_features, heavy_features, "CrossGatedSkipFusion")
        main = self.main_projection(main_features)
        heavy = self.heavy_projection(heavy_features)
        normalized_main = self.main_normalization(main)
        normalized_heavy = self.heavy_normalization(heavy)
        weights = self.gate((normalized_main, normalized_heavy))
        fused = main * weights[:, 0] + heavy * weights[:, 1]
        fused = self.output_projection(fused)
        alpha = self.residual_alpha.to(dtype=main.dtype)
        output = main + alpha * (fused - main)

        diagnostics = {
            "main_gate_mean": weights[:, 0].float().mean().detach(),
            "heavy_gate_mean": weights[:, 1].float().mean().detach(),
            "gate_entropy": _normalized_entropy(weights).detach(),
            "heavy_to_main_norm_ratio": _norm_ratio(heavy, main).detach(),
            "residual_alpha": self.residual_alpha.float().detach(),
        }
        return output, diagnostics


class RestormerSDPARefinement(nn.Module):
    """Low-resolution transposed attention implemented with native PyTorch SDPA."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int = 4,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if channels < 1 or heads < 1 or channels % heads:
            raise ValueError("channels must be positive and divisible by heads")
        if expansion < 1:
            raise ValueError("expansion must be positive")
        hidden_channels = channels * expansion
        self.channels = channels
        self.heads = heads
        self.head_channels = channels // heads
        self.attention_norm = LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.qkv_depthwise = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
        )
        self.attention_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.attention_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

        self.feedforward_norm = LayerNorm2d(channels)
        self.feedforward_expand = nn.Conv2d(
            channels, hidden_channels * 2, kernel_size=1
        )
        self.feedforward_depthwise = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
        )
        self.feedforward_project = nn.Conv2d(
            hidden_channels, channels, kernel_size=1
        )
        self.feedforward_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, height, width = inputs.shape
        qkv = self.qkv_depthwise(self.qkv(self.attention_norm(inputs)))
        query, key, value = qkv.chunk(3, dim=1)

        def reshape(features: Tensor) -> Tensor:
            return features.reshape(
                batch,
                self.heads,
                self.head_channels,
                height * width,
            )

        query = F.normalize(reshape(query).float(), dim=-1).to(dtype=inputs.dtype)
        key = F.normalize(reshape(key).float(), dim=-1).to(dtype=inputs.dtype)
        value = reshape(value)
        query = query * self.temperature.to(dtype=query.dtype)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            scale=1.0,
        )
        attended = attended.reshape(batch, channels, height, width)
        residual = inputs + self.attention_projection(attended) * self.attention_scale

        value, gate = self.feedforward_depthwise(
            self.feedforward_expand(self.feedforward_norm(residual))
        ).chunk(2, dim=1)
        update = self.feedforward_project(F.gelu(value) * gate)
        return residual + update * self.feedforward_scale


class TriPathAdaptiveFusion(nn.Module):
    """Fuse main, heavy U-Net and frequency routes at the low-resolution bottleneck."""

    def __init__(
        self,
        main_channels: int,
        heavy_channels: int,
        frequency_channels: int,
        *,
        out_channels: int | None = None,
        gate_reduction: int = 8,
        residual_init: float = 0.02,
        refinement_depth: int = 1,
        attention_heads: int = 4,
        attention_expansion: int = 2,
    ) -> None:
        super().__init__()
        output_channels = int(out_channels or main_channels)
        route_channels = (main_channels, heavy_channels, frequency_channels)
        if any(channel < 1 for channel in (*route_channels, output_channels)):
            raise ValueError("Fusion channel counts must be positive")
        if not 0.0 < residual_init <= 1.0:
            raise ValueError("residual_init must lie in (0, 1]")
        if not 0 <= refinement_depth <= 2:
            raise ValueError("refinement_depth must be between zero and two")
        if refinement_depth and output_channels % attention_heads:
            raise ValueError(
                "out_channels must be divisible by attention_heads when refinement is enabled"
            )

        self.main_projection = (
            nn.Identity()
            if main_channels == output_channels
            else nn.Conv2d(main_channels, output_channels, kernel_size=1)
        )
        self.heavy_projection = nn.Conv2d(
            heavy_channels, output_channels, kernel_size=1
        )
        self.frequency_projection = nn.Conv2d(
            frequency_channels, output_channels, kernel_size=1
        )
        self.route_normalizations = nn.ModuleList(
            LayerNorm2d(output_channels) for _ in range(3)
        )
        self.gate = _ContentAdaptiveGate(
            output_channels,
            3,
            reduction=gate_reduction,
        )
        self.output_projection = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.refinement = nn.Sequential(
            *(
                RestormerSDPARefinement(
                    output_channels,
                    heads=attention_heads,
                    expansion=attention_expansion,
                )
                for _ in range(refinement_depth)
            )
        )
        self.residual_alpha = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(
        self,
        main_features: Tensor,
        heavy_features: Tensor,
        frequency_features: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _validate_route_pair(main_features, heavy_features, "TriPathAdaptiveFusion")
        _validate_route_pair(main_features, frequency_features, "TriPathAdaptiveFusion")
        projected_routes = (
            self.main_projection(main_features),
            self.heavy_projection(heavy_features),
            self.frequency_projection(frequency_features),
        )
        normalized_routes = tuple(
            normalization(route)
            for normalization, route in zip(
                self.route_normalizations, projected_routes, strict=True
            )
        )
        weights = self.gate(normalized_routes)
        fused = sum(
            route * weights[:, index]
            for index, route in enumerate(projected_routes)
        )
        fused = self.refinement(self.output_projection(fused))
        main = projected_routes[0]
        alpha = self.residual_alpha.to(dtype=main.dtype)
        output = main + alpha * (fused - main)

        diagnostics = {
            "main_gate_mean": weights[:, 0].float().mean().detach(),
            "heavy_gate_mean": weights[:, 1].float().mean().detach(),
            "frequency_gate_mean": weights[:, 2].float().mean().detach(),
            "gate_entropy": _normalized_entropy(weights).detach(),
            "heavy_to_main_norm_ratio": _norm_ratio(projected_routes[1], main).detach(),
            "frequency_to_main_norm_ratio": _norm_ratio(
                projected_routes[2], main
            ).detach(),
            "residual_alpha": self.residual_alpha.float().detach(),
        }
        return output, diagnostics
