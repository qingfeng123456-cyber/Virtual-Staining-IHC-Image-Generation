"""Efficient parallel spatial/frequency mixing for restoration bottlenecks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .naf_blocks import LayerNorm2d, NAFBlock


class FFTLowHighBranch(nn.Module):
    """Split features into smooth/detail bands with a differentiable FFT mask.

    FFT kernels are deliberately evaluated in float32 because CUDA FFT does not
    reliably support every BF16 shape.  The reconstructed bands are cast back to
    the input dtype before the inexpensive learnable projections, so the rest of
    the branch still benefits from AMP.
    """

    def __init__(
        self,
        channels: int,
        *,
        cutoff: float = 0.35,
        transition_width: float = 0.08,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("FFT branch channels must be positive")
        if not 0.0 < cutoff < 1.0:
            raise ValueError("FFT cutoff must lie strictly between zero and one")
        if transition_width <= 0.0:
            raise ValueError("FFT transition_width must be positive")

        self.normalization = LayerNorm2d(channels)
        self.cutoff = float(cutoff)
        self.transition_width = float(transition_width)
        # Each output channel learns how much of its own low/high reconstruction
        # to retain before a pointwise channel mixer exchanges global evidence.
        self.band_mixer = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1,
            groups=channels,
        )
        self.activation = nn.SiLU(inplace=False)
        self.channel_mixer = nn.Conv2d(channels, channels, kernel_size=1)

    def _low_pass_mask(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
    ) -> Tensor:
        vertical = torch.fft.fftfreq(height, device=device, dtype=torch.float32)
        horizontal = torch.fft.rfftfreq(width, device=device, dtype=torch.float32)
        # Divide by the Nyquist frequency and by sqrt(2) so a corner has radius 1.
        radius = torch.sqrt(vertical[:, None].square() + horizontal[None, :].square())
        radius = radius * (2.0 / math.sqrt(2.0))
        return torch.sigmoid((self.cutoff - radius) / self.transition_width)[
            None, None
        ]

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"FFTLowHighBranch expects BCHW, got {tuple(inputs.shape)}")
        height, width = inputs.shape[-2:]
        if height < 2 or width < 2:
            raise ValueError("FFTLowHighBranch requires spatial dimensions of at least 2")

        normalized = self.normalization(inputs)
        spectrum = torch.fft.rfft2(normalized.float(), norm="ortho")
        low_mask = self._low_pass_mask(height, width, device=inputs.device)
        low = torch.fft.irfft2(
            spectrum * low_mask,
            s=(height, width),
            norm="ortho",
        )
        high = torch.fft.irfft2(
            spectrum * (1.0 - low_mask),
            s=(height, width),
            norm="ortho",
        )
        # Interleave bands as low_c0/high_c0/low_c1/high_c1 so the grouped
        # projection receives the matching pair for each channel.
        bands = torch.stack((low, high), dim=2).reshape(
            inputs.shape[0],
            inputs.shape[1] * 2,
            height,
            width,
        )
        bands = bands.to(dtype=inputs.dtype)
        return self.channel_mixer(self.activation(self.band_mixer(bands)))


class AdaptiveFFTLowHighBranch(FFTLowHighBranch):
    """Content-adaptive low/high-frequency route with bidirectional modulation.

    The static radial mask remains the stable prior.  A small image-conditioned
    predictor adjusts its cutoff and transition width, then pooled low-frequency
    evidence gates the high band and pooled high-frequency evidence gates the low
    band.  All FFT operations stay in float32 while the learned projections use
    the surrounding AMP dtype.
    """

    def __init__(
        self,
        channels: int,
        *,
        cutoff: float = 0.35,
        transition_width: float = 0.08,
        reduction: int = 8,
        cutoff_delta: float = 0.12,
        modulation_init: float = 0.1,
    ) -> None:
        super().__init__(
            channels,
            cutoff=cutoff,
            transition_width=transition_width,
        )
        if reduction < 1:
            raise ValueError("Adaptive FFT reduction must be positive")
        if not 0.0 <= cutoff_delta < 0.5:
            raise ValueError("Adaptive FFT cutoff_delta must lie in [0, 0.5)")
        if not 0.0 <= modulation_init <= 1.0:
            raise ValueError("Adaptive FFT modulation_init must lie in [0, 1]")
        hidden_channels = max(16, channels // reduction)
        self.cutoff_delta = float(cutoff_delta)
        self.mask_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        self.low_from_high = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        self.high_from_low = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
        )
        self.modulation_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(modulation_init))
        )
        final_mask = self.mask_predictor[-1]
        if not isinstance(final_mask, nn.Conv2d):
            raise RuntimeError("Unexpected adaptive mask predictor layout")
        nn.init.zeros_(final_mask.weight)
        nn.init.zeros_(final_mask.bias)
        self.last_diagnostics: dict[str, Tensor] = {}

    def _adaptive_low_pass_mask(
        self,
        inputs: Tensor,
        height: int,
        width: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        descriptor = self.mask_predictor(inputs).float()
        cutoff_offset, transition_log_scale = descriptor.chunk(2, dim=1)
        cutoff = self.cutoff + self.cutoff_delta * torch.tanh(cutoff_offset)
        transition = self.transition_width * torch.exp(
            0.5 * torch.tanh(transition_log_scale)
        )
        vertical = torch.fft.fftfreq(height, device=inputs.device, dtype=torch.float32)
        horizontal = torch.fft.rfftfreq(
            width, device=inputs.device, dtype=torch.float32
        )
        radius = torch.sqrt(vertical[:, None].square() + horizontal[None, :].square())
        radius = radius * (2.0 / math.sqrt(2.0))
        mask = torch.sigmoid(
            (cutoff - radius[None, None]) / transition.clamp_min(1e-4)
        )
        return mask, cutoff, transition

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                f"AdaptiveFFTLowHighBranch expects BCHW, got {tuple(inputs.shape)}"
            )
        height, width = inputs.shape[-2:]
        if height < 2 or width < 2:
            raise ValueError(
                "AdaptiveFFTLowHighBranch requires spatial dimensions of at least 2"
            )

        normalized = self.normalization(inputs)
        spectrum = torch.fft.rfft2(normalized.float(), norm="ortho")
        low_mask, cutoff, transition = self._adaptive_low_pass_mask(
            normalized, height, width
        )
        low = torch.fft.irfft2(spectrum * low_mask, s=(height, width), norm="ortho")
        high = torch.fft.irfft2(
            spectrum * (1.0 - low_mask), s=(height, width), norm="ortho"
        )
        low = low.to(dtype=inputs.dtype)
        high = high.to(dtype=inputs.dtype)
        scale = self.modulation_scale.to(dtype=inputs.dtype)
        low_gate = torch.sigmoid(self.low_from_high(high))
        high_gate = torch.sigmoid(self.high_from_low(low))
        low = low * (1.0 + scale * low_gate)
        high = high * (1.0 + scale * high_gate)
        self.last_diagnostics = {
            "adaptive_cutoff_mean": cutoff.float().mean().detach(),
            "adaptive_transition_mean": transition.float().mean().detach(),
            "low_from_high_gate_mean": low_gate.float().mean().detach(),
            "high_from_low_gate_mean": high_gate.float().mean().detach(),
            "modulation_scale_mean": self.modulation_scale.float().mean().detach(),
        }
        bands = torch.stack((low, high), dim=2).reshape(
            inputs.shape[0], inputs.shape[1] * 2, height, width
        )
        return self.channel_mixer(self.activation(self.band_mixer(bands)))


class ParallelSpatialFrequencyMixer(nn.Module):
    """Fuse a local NAF path and global FFT path with channel-wise gates.

    The complete fused update is multiplied by a zero-initialized residual scale.
    Consequently enabling the mixer starts from the surrounding network's identity
    mapping instead of perturbing a stable restoration backbone on the first step.
    """

    def __init__(
        self,
        channels: int,
        *,
        spatial_depth: int = 1,
        spatial_expansion: int = 1,
        gate_reduction: int = 8,
        frequency_cutoff: float = 0.35,
        frequency_transition_width: float = 0.08,
        residual_init: float = 0.0,
        adaptive_frequency: bool = False,
        adaptive_reduction: int = 8,
        adaptive_cutoff_delta: float = 0.12,
        modulation_init: float = 0.1,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("Mixer channels must be positive")
        if spatial_depth < 1:
            raise ValueError("spatial_depth must be at least one")
        if spatial_expansion < 1:
            raise ValueError("spatial_expansion must be positive")
        if (channels * spatial_expansion) % 2:
            raise ValueError("channels * spatial_expansion must be even")
        if gate_reduction < 1:
            raise ValueError("gate_reduction must be positive")

        self.spatial_branch = nn.Sequential(
            *(
                NAFBlock(
                    channels,
                    depthwise_expansion=spatial_expansion,
                    feedforward_expansion=spatial_expansion,
                )
                for _ in range(spatial_depth)
            )
        )
        frequency_type = AdaptiveFFTLowHighBranch if adaptive_frequency else FFTLowHighBranch
        frequency_options: dict[str, int | float] = {}
        if adaptive_frequency:
            frequency_options = {
                "reduction": adaptive_reduction,
                "cutoff_delta": adaptive_cutoff_delta,
                "modulation_init": modulation_init,
            }
        self.frequency_branch = frequency_type(
            channels,
            cutoff=frequency_cutoff,
            transition_width=frequency_transition_width,
            **frequency_options,
        )
        hidden_channels = max(16, channels // gate_reduction)
        self.branch_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * 2, kernel_size=1),
        )
        self.output_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.residual_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(residual_init))
        )
        # Equal branch priors make the initial gate deterministic and neutral.
        final_gate = self.branch_gate[-1]
        if not isinstance(final_gate, nn.Conv2d):
            raise RuntimeError("Unexpected branch gate layout")
        nn.init.zeros_(final_gate.weight)
        nn.init.zeros_(final_gate.bias)
        self.last_diagnostics: dict[str, Tensor] = {}

    def forward_route(self, inputs: Tensor) -> Tensor:
        """Return the learned spatial-frequency route without the identity input.

        This interface reuses exactly the same branches, gates, projection, and
        residual scale as :meth:`forward`, but exposes only their update.  It is
        intended for an outer multi-route fusion whose main route is already
        supplied separately.  No additional parameters or checkpoint keys are
        introduced.
        """

        if inputs.ndim != 4:
            raise ValueError(
                f"ParallelSpatialFrequencyMixer expects BCHW, got {tuple(inputs.shape)}"
            )
        spatial = self.spatial_branch(inputs)
        frequency = self.frequency_branch(inputs)
        descriptors = torch.cat(
            (
                F.adaptive_avg_pool2d(spatial, 1),
                F.adaptive_avg_pool2d(frequency, 1),
            ),
            dim=1,
        )
        batch, channels = inputs.shape[:2]
        weights = self.branch_gate(descriptors).reshape(batch, 2, channels, 1, 1)
        weights = torch.softmax(weights.float(), dim=1).to(dtype=inputs.dtype)
        fused = spatial * weights[:, 0] + frequency * weights[:, 1]
        update = self.output_projection(fused)
        route = update * self.residual_scale
        input_rms = inputs.float().square().mean().sqrt()
        route_rms = route.float().square().mean().sqrt()
        self.last_diagnostics = {
            "spatial_gate_mean": weights[:, 0].float().mean().detach(),
            "frequency_gate_mean": weights[:, 1].float().mean().detach(),
            "residual_scale_mean": self.residual_scale.float().mean().detach(),
            "input_rms": input_rms.detach(),
            "route_rms": route_rms.detach(),
            "route_to_input_norm_ratio": (
                route_rms / input_rms.clamp_min(1e-8)
            ).detach(),
        }
        adaptive_diagnostics = getattr(self.frequency_branch, "last_diagnostics", {})
        if isinstance(adaptive_diagnostics, dict):
            self.last_diagnostics.update(adaptive_diagnostics)
        return route

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the legacy residual mixer transformation."""

        return inputs + self.forward_route(inputs)
