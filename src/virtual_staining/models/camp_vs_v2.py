"""Context-aware multi-prototype virtual staining restorer (CAMP-VS v2)."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .context_fusion import (
    BottleneckContextCrossAttention,
    MultiScaleContextFusion,
)
from .context_tile_encoder import ContextEncoderOutput, SharedTinyContextEncoder
from .hierarchical_prototypes import HierarchicalPrototypeMixer
from .intensity_calibrator import GlobalIntensityCalibrator
from .laplacian_decoder import LaplacianBaseDetailHead
from .naf_blocks import DecoderStage, NAFStage
from .naf_local_encoder import NAFLocalEncoder
from .registry import RestorationOutput, register_model
from .restoration_transformer import RestormerLiteStage
from .task_organ_conditioning import (
    ConditionalResidualAdapter,
    GatedResidualExperts,
    IdentityConditioningFiLM,
    TaskOrganConditioner,
    normalize_identity,
)


def _mapping(value: Mapping[str, Any] | bool | None, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"enabled": value}
    if not isinstance(value, Mapping):
        raise TypeError(f"model.{section} must be a mapping or boolean")
    return dict(value)


def _check_keys(options: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(options).difference(allowed))
    if unknown:
        raise TypeError(f"Unsupported model.{section} options: {', '.join(unknown)}")


def _sequence(value: Sequence[int] | int, length: int, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        values = (value,) * length
    else:
        values = tuple(int(item) for item in value)
    if len(values) != length or any(item < 1 for item in values):
        raise ValueError(f"{name} must contain {length} positive integers")
    return values


def _attention_heads(token_dim: int, requested: int = 4) -> int:
    for heads in range(min(token_dim, requested), 0, -1):
        if token_dim % heads == 0:
            return heads
    return 1


@register_model("camp_vs_v2", "campvsv2", "camp")
class CAMPVSv2(nn.Module):
    """Incremental CAMP-VS model with independently reversible feature flags.

    The center encoder and shared decoder always remain available. Context,
    global mixing, prototypes, organ adapters, experts, base/detail output, and
    calibration are separate optional branches. Context tensors are accepted
    only when the context branch is enabled and represent DAPI channels only.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int | Mapping[str, int] = 1,
        target_names: Sequence[str] = ("CD68",),
        organ_names: Sequence[str] = ("colon", "liver", "stomach"),
        *,
        local_encoder: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | bool | None = None,
        global_mixer: Mapping[str, Any] | bool | None = None,
        conditioning: Mapping[str, Any] | bool | None = None,
        adapters: Mapping[str, Any] | bool | None = None,
        prototypes: Mapping[str, Any] | bool | None = None,
        output: Mapping[str, Any] | bool | None = None,
        intensity_calibrator: Mapping[str, Any] | bool | None = None,
        base_channels: int | None = None,
        encoder_depths: Sequence[int] | None = None,
        decoder_depths: Sequence[int] = (2, 2, 2),
        use_sobel_input: bool = True,
        use_laplacian_input: bool = False,
        use_task_adapters: bool | None = None,
        use_prototypes: bool | None = None,
        shared_prototypes: int | None = None,
        task_prototypes: int | None = None,
        prototype_temperature: float | None = None,
        deep_supervision: bool | None = None,
        output_activation: str = "sigmoid",
        context_stop_gradient: bool | None = None,
        decoder_mode: str = "shared_decoder_with_adapters",
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if output_activation != "sigmoid":
            raise ValueError("CAMPVSv2 supports output_activation='sigmoid' only")
        if decoder_mode != "shared_decoder_with_adapters":
            raise ValueError("CAMPVSv2 supports the shared decoder rollback path only")

        self.in_channels = int(in_channels)
        self.target_names = tuple(str(name) for name in target_names)
        normalized_targets = tuple(normalize_identity(name) for name in self.target_names)
        if not self.target_names or len(set(normalized_targets)) != len(self.target_names):
            raise ValueError("Target names must be non-empty and normalization-unique")
        self._normalized_targets = dict(zip(normalized_targets, self.target_names, strict=True))
        self._task_keys = {
            task: f"task_{index}" for index, task in enumerate(self.target_names)
        }
        self.organ_names = tuple(str(name) for name in organ_names)
        if isinstance(out_channels, int):
            self.output_channels = {task: int(out_channels) for task in self.target_names}
        else:
            self.output_channels = {
                task: int(out_channels[task]) for task in self.target_names
            }
        if any(channels < 1 for channels in self.output_channels.values()):
            raise ValueError("All output channel counts must be positive")

        local_options = _mapping(local_encoder, "local_encoder")
        _check_keys(
            local_options,
            {"type", "widths", "depths", "drop_path", "use_laplacian_input"},
            "local_encoder",
        )
        if str(local_options.get("type", "naf")).casefold() != "naf":
            raise ValueError("CAMPVSv2 local_encoder.type must be 'naf'")
        drop_path = float(local_options.get("drop_path", 0.0))
        if drop_path != 0.0:
            raise ValueError("The stable NAF local encoder requires drop_path=0.0")
        fallback_base = int(base_channels or 48)
        widths = _sequence(
            local_options.get(
                "widths", tuple(fallback_base * (2**index) for index in range(4))
            ),
            4,
            "local_encoder.widths",
        )
        depths = _sequence(
            local_options.get("depths", encoder_depths or (2, 2, 4, 6)),
            4,
            "local_encoder.depths",
        )
        decoder_depth_values = _sequence(decoder_depths, 3, "decoder_depths")
        self.widths = widths
        channels_by_scale = dict(zip((2, 4, 8, 16), widths, strict=True))
        self.local_encoder = NAFLocalEncoder(
            in_channels=self.in_channels,
            widths=widths,
            depths=depths,
            use_sobel_input=bool(use_sobel_input),
            use_laplacian_input=bool(
                local_options.get("use_laplacian_input", use_laplacian_input)
            ),
        )

        conditioning_options = _mapping(conditioning, "conditioning")
        _check_keys(
            conditioning_options,
            {
                "enabled",
                "marker_embedding",
                "organ_embedding",
                "film",
                "zero_init",
                "embedding_dim",
            },
            "conditioning",
        )
        conditioning_enabled = bool(conditioning_options.get("enabled", True))
        marker_conditioning = conditioning_enabled and bool(
            conditioning_options.get("marker_embedding", True)
        )
        organ_conditioning = conditioning_enabled and bool(
            conditioning_options.get("organ_embedding", False)
        )
        film_enabled = conditioning_enabled and bool(conditioning_options.get("film", True))
        if not bool(conditioning_options.get("zero_init", True)):
            raise ValueError("CAMPVSv2 conditioning FiLM must remain zero initialized")
        embedding_dim = int(conditioning_options.get("embedding_dim", max(32, widths[0])))
        if embedding_dim < 1:
            raise ValueError("conditioning.embedding_dim must be positive")
        self.conditioner = TaskOrganConditioner(
            self.target_names,
            self.organ_names,
            embedding_dim=embedding_dim,
            marker_enabled=marker_conditioning,
            organ_enabled=organ_conditioning,
        )
        self.bottleneck_marker_film = (
            IdentityConditioningFiLM(widths[3], embedding_dim)
            if film_enabled and marker_conditioning
            else None
        )
        self.bottleneck_organ_film = (
            IdentityConditioningFiLM(widths[3], embedding_dim)
            if film_enabled and organ_conditioning
            else None
        )

        context_options = _mapping(context, "context")
        _check_keys(
            context_options,
            {
                "enabled",
                "grid_size",
                "missing_policy",
                "include_center",
                "cache_size",
                "token_dim",
                "encoder_width",
                "encoder_depth",
                "context_dropout",
                "fusion_scales",
                "bottleneck_cross_attention",
                "cross_attention_heads",
                "residual_init",
                "stop_gradient",
                "require_verified_grid",
                "coordinate_source",
                "allow_numeric_stem_inference",
                "allow_edge_graph_inference",
                "tile_chunk_size",
            },
            "context",
        )
        self.context_enabled = bool(context_options.get("enabled", False))
        grid_size = int(context_options.get("grid_size", 3))
        context_dim = int(context_options.get("token_dim", 128))
        context_cache_size = context_options.get("cache_size", 0)
        if isinstance(context_cache_size, bool) or not isinstance(
            context_cache_size, int
        ):
            raise ValueError("context.cache_size must be a nonnegative integer")
        if context_cache_size < 0:
            raise ValueError("context.cache_size must be a nonnegative integer")
        # Dataset workers own the tile cache. The model retains this value only
        # so a fully resolved context mapping can be consumed without ambiguity.
        self.context_cache_size = context_cache_size
        self.context_tile_chunk_size = int(
            context_options.get("tile_chunk_size", grid_size * grid_size)
        )
        if self.context_tile_chunk_size < 1:
            raise ValueError("context.tile_chunk_size must be positive")
        self.context_encoder = (
            SharedTinyContextEncoder(
                self.in_channels,
                width=int(context_options.get("encoder_width", 24)),
                token_dim=context_dim,
                grid_size=grid_size,
                depth=int(context_options.get("encoder_depth", 2)),
                stop_gradient=bool(
                    context_options.get(
                        "stop_gradient",
                        False if context_stop_gradient is None else context_stop_gradient,
                    )
                ),
            )
            if self.context_enabled
            else None
        )
        fusion_scales = tuple(int(value) for value in context_options.get("fusion_scales", (4, 8, 16)))
        self.context_fusion = (
            MultiScaleContextFusion(
                channels_by_scale,
                context_dim,
                fusion_scales=fusion_scales,
                context_dropout=float(context_options.get("context_dropout", 0.1)),
            )
            if self.context_enabled
            else None
        )
        cross_attention_enabled = self.context_enabled and bool(
            context_options.get("bottleneck_cross_attention", False)
        )
        cross_heads = int(
            context_options.get(
                "cross_attention_heads", _attention_heads(context_dim)
            )
        )
        self.context_cross_attention = (
            BottleneckContextCrossAttention(
                widths[3],
                context_dim,
                heads=cross_heads,
                residual_init=float(context_options.get("residual_init", 0.0)),
            )
            if cross_attention_enabled
            else None
        )
        self.context_organ_projection = (
            nn.Linear(embedding_dim, context_dim)
            if self.context_enabled and organ_conditioning
            else None
        )

        global_options = _mapping(global_mixer, "global_mixer")
        _check_keys(
            global_options,
            {
                "enabled",
                "type",
                "blocks_1_8",
                "blocks_1_16",
                "heads",
                "heads_1_8",
                "heads_1_16",
                "expansion",
            },
            "global_mixer",
        )
        self.global_mixer_enabled = bool(global_options.get("enabled", True))
        mixer_type = str(global_options.get("type", "restormer_lite")).casefold()
        if mixer_type not in {"restormer_lite", "mambair_v2_lite"}:
            raise ValueError("global_mixer.type must be restormer_lite or mambair_v2_lite")
        if mixer_type == "mambair_v2_lite":
            warnings.warn(
                "mambair_v2_lite is optional; using the pure PyTorch Restormer-lite fallback",
                RuntimeWarning,
                stacklevel=2,
            )
        self.global_mixer_backend = (
            "restormer_lite_fallback" if mixer_type == "mambair_v2_lite" else mixer_type
        )
        if "heads" in global_options:
            requested_heads = _sequence(
                global_options["heads"], 2, "global_mixer.heads"
            )
        else:
            requested_heads = (
                int(global_options.get("heads_1_8", 4)),
                int(global_options.get("heads_1_16", 8)),
            )
            if any(value < 1 for value in requested_heads):
                raise ValueError("global mixer heads must be positive")
        expansion_value = float(global_options.get("expansion", 2))
        if not expansion_value.is_integer():
            raise ValueError("global_mixer.expansion must be an integer value")
        blocks = (
            int(global_options.get("blocks_1_8", 2)),
            int(global_options.get("blocks_1_16", 4)),
        )
        if any(value < 0 for value in blocks):
            raise ValueError("global mixer block counts cannot be negative")
        self.global_stages = nn.ModuleDict()
        if self.global_mixer_enabled:
            for scale, block_count, heads in zip((8, 16), blocks, requested_heads, strict=True):
                self.global_stages[str(scale)] = RestormerLiteStage(
                    channels_by_scale[scale],
                    block_count,
                    heads=heads,
                    expansion=int(expansion_value),
                )

        prototype_options = _mapping(prototypes, "prototypes")
        _check_keys(
            prototype_options,
            {
                "enabled",
                "scales",
                "shared_count",
                "marker_count",
                "organ_count",
                "dim",
                "temperature",
                "residual_init",
                "reset_dead",
                "dead_threshold",
                "reset_patience",
                "reset_seed",
                "reset_std",
            },
            "prototypes",
        )
        prototype_enabled = bool(
            prototype_options.get(
                "enabled", True if use_prototypes is None else use_prototypes
            )
        )
        self.prototype_reset_enabled = bool(
            prototype_options.get("reset_dead", False)
        )
        self.prototype_dead_threshold = float(
            prototype_options.get("dead_threshold", 1e-4)
        )
        organ_prototype_count = int(prototype_options.get("organ_count", 4))
        self.prototype_mixer = (
            HierarchicalPrototypeMixer(
                channels_by_scale,
                self.target_names,
                self.organ_names,
                scales=tuple(int(value) for value in prototype_options.get("scales", (8, 16))),
                shared_count=int(
                    prototype_options.get("shared_count", shared_prototypes or 8)
                ),
                marker_count=int(
                    prototype_options.get("marker_count", task_prototypes or 8)
                ),
                organ_count=max(organ_prototype_count, 1),
                prototype_dim=int(prototype_options.get("dim", 128)),
                temperature=float(
                    prototype_options.get(
                        "temperature", prototype_temperature or 0.1
                    )
                ),
                residual_init=float(prototype_options.get("residual_init", 0.0)),
                organ_enabled=organ_conditioning and organ_prototype_count > 0,
            )
            if prototype_enabled
            else None
        )

        self.decoder = nn.ModuleList(
            (
                DecoderStage(widths[3], widths[2], widths[2], decoder_depth_values[0]),
                DecoderStage(widths[2], widths[1], widths[1], decoder_depth_values[1]),
                DecoderStage(widths[1], widths[0], widths[0], decoder_depth_values[2]),
            )
        )
        adapter_options = _mapping(adapters, "adapters")
        _check_keys(
            adapter_options,
            {"enabled", "marker", "organ", "mixture_of_experts", "reduction", "expert_count"},
            "adapters",
        )
        adapters_enabled = bool(adapter_options.get("enabled", True))
        marker_adapters_enabled = adapters_enabled and bool(
            adapter_options.get(
                "marker", True if use_task_adapters is None else use_task_adapters
            )
        )
        organ_adapters_enabled = adapters_enabled and bool(adapter_options.get("organ", False))
        expert_enabled = adapters_enabled and bool(
            adapter_options.get("mixture_of_experts", False)
        )
        reduction = int(adapter_options.get("reduction", 4))
        decoder_channels = (widths[2], widths[1], widths[0])
        self.marker_adapters = (
            nn.ModuleList(
                ConditionalResidualAdapter(channels, embedding_dim, reduction)
                for channels in decoder_channels
            )
            if marker_adapters_enabled
            else None
        )
        self.organ_adapters = (
            nn.ModuleList(
                ConditionalResidualAdapter(channels, embedding_dim, reduction)
                for channels in decoder_channels
            )
            if organ_adapters_enabled
            else None
        )
        self.residual_experts = (
            nn.ModuleList(
                GatedResidualExperts(
                    channels,
                    embedding_dim,
                    expert_count=int(adapter_options.get("expert_count", 2)),
                )
                for channels in decoder_channels
            )
            if expert_enabled
            else None
        )
        self.full_resolution_refinement = nn.Sequential(
            nn.Conv2d(widths[0], widths[0], kernel_size=3, padding=1),
            NAFStage(widths[0], 1),
        )

        output_options = _mapping(output, "output")
        _check_keys(
            output_options,
            {
                "enabled",
                "base_detail",
                "max_detail_amplitude",
                "deep_supervision",
                "deep_supervision_scales",
            },
            "output",
        )
        output_enabled = bool(output_options.get("enabled", True))
        if not output_enabled:
            raise ValueError("The CAMPVSv2 output branch cannot be disabled")
        self.base_detail_enabled = bool(output_options.get("base_detail", True))
        self.deep_supervision_enabled = bool(
            output_options.get(
                "deep_supervision", True if deep_supervision is None else deep_supervision
            )
        )
        self.deep_supervision_factors = tuple(
            int(value)
            for value in output_options.get("deep_supervision_scales", (1, 2, 4, 8))
        )
        invalid_factors = sorted(set(self.deep_supervision_factors).difference({1, 2, 4, 8}))
        if invalid_factors:
            raise ValueError(f"Unsupported deep supervision factors: {invalid_factors}")
        self.supervision_heads = nn.ModuleDict(
            {
                self._task_keys[task]: nn.ModuleList(
                    nn.Conv2d(channels, self.output_channels[task], kernel_size=1)
                    for channels in decoder_channels
                )
                for task in self.target_names
            }
        )
        self.base_detail_heads = nn.ModuleDict(
            {
                self._task_keys[task]: LaplacianBaseDetailHead(
                    widths[1],
                    widths[0],
                    self.output_channels[task],
                    max_detail_amplitude=float(
                        output_options.get("max_detail_amplitude", 1.0)
                    ),
                )
                for task in self.target_names
            }
            if self.base_detail_enabled
            else {}
        )
        self.output_heads = nn.ModuleDict(
            {
                self._task_keys[task]: nn.Conv2d(
                    widths[0], self.output_channels[task], kernel_size=1
                )
                for task in self.target_names
            }
            if not self.base_detail_enabled
            else {}
        )

        calibrator_options = _mapping(intensity_calibrator, "intensity_calibrator")
        _check_keys(
            calibrator_options,
            {"enabled", "max_gain_delta", "max_bias", "hidden_dim"},
            "intensity_calibrator",
        )
        self.calibrator_enabled = bool(calibrator_options.get("enabled", True))
        self.calibrators = nn.ModuleDict(
            {
                self._task_keys[task]: GlobalIntensityCalibrator(
                    widths[3],
                    embedding_dim,
                    self.output_channels[task],
                    max_gain_delta=float(calibrator_options.get("max_gain_delta", 0.15)),
                    max_bias=float(calibrator_options.get("max_bias", 0.15)),
                    hidden_dim=(
                        int(calibrator_options["hidden_dim"])
                        if "hidden_dim" in calibrator_options
                        else None
                    ),
                )
                for task in self.target_names
            }
            if self.calibrator_enabled
            else {}
        )

        self.feature_flags = {
            "context": self.context_enabled,
            "context_cross_attention": self.context_cross_attention is not None,
            "global_mixer": self.global_mixer_enabled,
            "prototypes": self.prototype_mixer is not None,
            "marker_adapters": self.marker_adapters is not None,
            "organ_adapters": self.organ_adapters is not None,
            "mixture_of_experts": self.residual_experts is not None,
            "base_detail": self.base_detail_enabled,
            "intensity_calibrator": self.calibrator_enabled,
        }

    def resolve_task(self, task_name: str) -> str:
        """Resolve case and punctuation variants to a configured marker."""

        normalized = normalize_identity(task_name)
        if normalized not in self._normalized_targets:
            available = ", ".join(self.target_names)
            raise KeyError(f"Unknown task {task_name!r}; available: {available}")
        return self._normalized_targets[normalized]

    def resolve_prototype_bank_key(self, diagnostic: str) -> str | None:
        """Resolve a usage diagnostic without guessing an ambiguous bank."""

        if self.prototype_mixer is None:
            return None
        return self.prototype_mixer.resolve_bank_key(diagnostic)

    def prototype_bank_parameters(self) -> dict[str, nn.Parameter]:
        """Expose hierarchical bank parameters for training-only maintenance."""

        if self.prototype_mixer is None:
            return {}
        return self.prototype_mixer.bank_parameters()

    def reset_prototype_rows(
        self,
        rows_by_bank: Mapping[str, Sequence[int]],
        *,
        seed: int,
        std: float = 0.02,
    ) -> list[dict[str, int | str | float]]:
        """Delegate deterministic row resets to the hierarchical mixer."""

        if self.prototype_mixer is None:
            raise RuntimeError("Prototype reset requested while prototypes are disabled")
        return self.prototype_mixer.reset_prototype_rows(
            rows_by_bank,
            seed=seed,
            std=std,
        )

    def _selected_tasks(
        self, task_name: str | Sequence[str] | None
    ) -> tuple[str, ...]:
        if task_name is None:
            return self.target_names
        if isinstance(task_name, str):
            return (self.resolve_task(task_name),)
        selected = tuple(self.resolve_task(name) for name in task_name)
        if not selected:
            raise ValueError("At least one task must be selected")
        if len(set(selected)) != len(selected):
            raise ValueError("Selected tasks must be unique")
        return selected

    def _encode_context(
        self,
        context_tiles: Tensor,
        context_valid_mask: Tensor | None,
        context_offsets: Tensor | None,
        organ_embedding: Tensor | None,
    ) -> ContextEncoderOutput:
        encoder = self.context_encoder
        if encoder is None:
            raise RuntimeError("Context encoder is disabled")
        if context_tiles.ndim != 5 or context_tiles.shape[2] != self.in_channels:
            raise ValueError(
                "context_tiles must have shape "
                f"[B,N,{self.in_channels},H,W], got {tuple(context_tiles.shape)}"
            )
        batch, tile_count = context_tiles.shape[:2]
        if tile_count != encoder.tile_count:
            raise ValueError(
                f"Expected {encoder.tile_count} context tiles, got {tile_count}"
            )
        if context_valid_mask is None:
            mask = torch.ones(
                batch, tile_count, device=context_tiles.device, dtype=torch.bool
            )
        else:
            mask = context_valid_mask.to(device=context_tiles.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(batch, -1)
            if mask.shape != (batch, tile_count):
                raise ValueError(
                    f"Expected context_valid_mask {(batch, tile_count)}, got {tuple(mask.shape)}"
                )
        if context_offsets is None:
            offsets = encoder.default_offsets(
                device=context_tiles.device, dtype=context_tiles.dtype
            ).unsqueeze(0).expand(batch, -1, -1)
        else:
            offsets = context_offsets.to(
                device=context_tiles.device, dtype=context_tiles.dtype
            )
            if offsets.ndim == 2:
                offsets = offsets.unsqueeze(0).expand(batch, -1, -1)
            if offsets.shape != (batch, tile_count, 2):
                raise ValueError(
                    f"Expected context_offsets {(batch, tile_count, 2)}, got {tuple(offsets.shape)}"
                )

        chunk_size = min(self.context_tile_chunk_size, tile_count)
        if chunk_size == tile_count:
            return encoder(
                context_tiles,
                mask,
                offsets,
                organ_embedding=organ_embedding,
            )
        tokens: list[Tensor] = []
        masks: list[Tensor] = []
        for start in range(0, tile_count, chunk_size):
            stop = min(start + chunk_size, tile_count)
            encoded = encoder(
                context_tiles[:, start:stop],
                mask[:, start:stop],
                offsets[:, start:stop],
                organ_embedding=organ_embedding,
            )
            tokens.append(encoded.tokens)
            masks.append(encoded.valid_mask)
        return ContextEncoderOutput(
            tokens=torch.cat(tokens, dim=1),
            valid_mask=torch.cat(masks, dim=1),
        )

    def _shared_features(
        self,
        inputs: Tensor,
        selected_tasks: tuple[str, ...],
        context_tiles: Tensor | None,
        context_valid_mask: Tensor | None,
        context_offsets: Tensor | None,
        organ_id: str | Sequence[str] | None,
    ) -> tuple[dict[int, Tensor], ContextEncoderOutput | None, dict[str, Tensor]]:
        features = self.local_encoder(inputs).features
        context_output: ContextEncoderOutput | None = None
        context_attention: dict[str, Tensor] = {}
        if self.context_enabled:
            if context_tiles is None:
                raise ValueError("context_tiles are required when model.context.enabled=true")
            context_organ: Tensor | None = None
            if self.context_organ_projection is not None:
                condition = self.conditioner(
                    selected_tasks[0],
                    organ_id,
                    batch_size=inputs.shape[0],
                    device=inputs.device,
                )
                context_organ = self.context_organ_projection(condition.organ)
            context_output = self._encode_context(
                context_tiles,
                context_valid_mask,
                context_offsets,
                context_organ,
            )
            if self.context_fusion is None:
                raise RuntimeError("Context fusion is missing while context is enabled")
            features, context_attention = self.context_fusion(
                features,
                context_output.tokens,
                context_output.valid_mask,
            )

        for scale, stage in self.global_stages.items():
            features[int(scale)] = stage(features[int(scale)])
        if self.context_cross_attention is not None:
            if context_output is None:
                raise RuntimeError("Cross-attention requires encoded context")
            features[16], cross_attention = self.context_cross_attention(
                features[16], context_output.tokens, context_output.valid_mask
            )
            context_attention["cross_16"] = cross_attention
        return features, context_output, context_attention

    def _decode_task(
        self,
        shared_features: Mapping[int, Tensor],
        task: str,
        organ_id: str | Sequence[str] | None,
        output_size: tuple[int, int],
    ) -> tuple[
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor,
        dict[int, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
        Tensor | None,
        dict[str, Tensor],
    ]:
        condition = self.conditioner(
            task,
            organ_id,
            batch_size=shared_features[16].shape[0],
            device=shared_features[16].device,
        )
        task_features = dict(shared_features)
        if self.bottleneck_marker_film is not None:
            task_features[16] = self.bottleneck_marker_film(
                task_features[16], condition.marker
            )
        if self.bottleneck_organ_film is not None:
            task_features[16] = self.bottleneck_organ_film(
                task_features[16], condition.organ
            )

        prototype_attention: dict[str, Tensor] = {}
        prototype_usage: dict[str, Tensor] = {}
        prototype_features: Tensor | None = None
        if self.prototype_mixer is not None:
            task_features, diagnostics = self.prototype_mixer(
                task_features, task, organ_id
            )
            for scale, diagnostic in diagnostics.items():
                for bank_name, attention in diagnostic.attention.items():
                    prototype_attention[f"{scale}/{bank_name}"] = attention
                for bank_name, usage in diagnostic.usage.items():
                    prototype_usage[f"{scale}/{bank_name}"] = usage
            prototype_features = diagnostics[max(diagnostics)].normalized_tokens

        decoded_by_factor: dict[int, Tensor] = {}
        features = task_features[16]
        for index, (stage, scale) in enumerate(
            zip(self.decoder, (8, 4, 2), strict=True)
        ):
            features = stage(features, task_features[scale])
            if self.marker_adapters is not None:
                features = self.marker_adapters[index](features, condition.marker)
            if self.organ_adapters is not None:
                features = self.organ_adapters[index](features, condition.organ)
            if self.residual_experts is not None:
                features = self.residual_experts[index](features, condition.combined)
            decoded_by_factor[scale] = features

        full_features = F.interpolate(
            decoded_by_factor[2], size=output_size, mode="bilinear", align_corners=False
        )
        full_features = self.full_resolution_refinement(full_features)
        task_key = self._task_keys[task]
        base_prediction: Tensor | None = None
        detail_prediction: Tensor | None = None
        if self.base_detail_enabled:
            decomposition = self.base_detail_heads[task_key](
                decoded_by_factor[4], full_features, output_size=output_size
            )
            logits = decomposition.final_logits
            base_prediction = decomposition.base
            detail_prediction = decomposition.detail
        else:
            logits = self.output_heads[task_key](full_features)

        calibration: dict[str, Tensor] = {}
        if self.calibrator_enabled:
            calibrated = self.calibrators[task_key](
                logits,
                task_features[16],
                condition.marker,
                condition.organ,
            )
            logits = calibrated.logits
            calibration = {"gain": calibrated.gain, "bias": calibrated.bias}
        prediction = torch.sigmoid(logits)

        supervision: dict[int, Tensor] = {}
        if self.deep_supervision_enabled:
            for head, factor in zip(
                self.supervision_heads[task_key], (8, 4, 2), strict=True
            ):
                if factor in self.deep_supervision_factors:
                    auxiliary = torch.sigmoid(head(decoded_by_factor[factor]))
                    supervision[auxiliary.shape[-2]] = auxiliary
        if 1 in self.deep_supervision_factors or not self.deep_supervision_enabled:
            supervision[prediction.shape[-2]] = prediction
        return (
            prediction,
            base_prediction,
            detail_prediction,
            logits,
            supervision,
            prototype_attention,
            prototype_usage,
            prototype_features,
            calibration,
        )

    def forward(
        self,
        input: Tensor,
        *,
        context_tiles: Tensor | None = None,
        context_valid_mask: Tensor | None = None,
        context_offsets: Tensor | None = None,
        task_name: str | Sequence[str] | None = None,
        organ_id: str | Sequence[str] | None = None,
    ) -> RestorationOutput:
        """Restore one or more markers from a center DAPI and optional neighbors."""

        if input.ndim != 4 or input.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected BCHW input with {self.in_channels} channels, got {tuple(input.shape)}"
            )
        if not torch.is_floating_point(input):
            raise TypeError("CAMPVSv2 inputs must be floating-point tensors")
        if min(input.shape[-2:]) < 16:
            raise ValueError("CAMPVSv2 requires spatial dimensions of at least 16 pixels")
        selected_tasks = self._selected_tasks(task_name)
        features, _, context_attention = self._shared_features(
            input,
            selected_tasks,
            context_tiles,
            context_valid_mask,
            context_offsets,
            organ_id,
        )

        predictions: dict[str, Tensor] = {}
        base_predictions: dict[str, Tensor] = {}
        detail_predictions: dict[str, Tensor] = {}
        logits: dict[str, Tensor] = {}
        deep_supervision: dict[str, dict[int, Tensor]] = {}
        prototype_attention: dict[str, dict[str, Tensor]] = {}
        prototype_usage: dict[str, dict[str, Tensor]] = {}
        prototype_features: dict[str, Tensor] = {}
        calibration_parameters: dict[str, dict[str, Tensor]] = {}
        intermediate_features = {
            f"shared/{scale}": tensor for scale, tensor in features.items()
        }
        for task in selected_tasks:
            (
                prediction,
                base_prediction,
                detail_prediction,
                task_logits,
                supervision,
                task_attention,
                task_usage,
                task_prototype_features,
                calibration,
            ) = self._decode_task(features, task, organ_id, tuple(input.shape[-2:]))
            predictions[task] = prediction
            if base_prediction is not None:
                base_predictions[task] = base_prediction
            if detail_prediction is not None:
                detail_predictions[task] = detail_prediction
            logits[task] = task_logits
            deep_supervision[task] = supervision
            if task_attention:
                prototype_attention[task] = task_attention
            if task_usage:
                prototype_usage[task] = task_usage
            if task_prototype_features is not None:
                prototype_features[task] = task_prototype_features
            if calibration:
                calibration_parameters[task] = calibration

        prototype_banks = (
            self.prototype_mixer.banks() if self.prototype_mixer is not None else {}
        )
        return RestorationOutput(
            predictions=predictions,
            deep_supervision=deep_supervision,
            prototype_attention=prototype_attention,
            prototype_features=prototype_features,
            prototype_banks=prototype_banks,
            base_predictions=base_predictions,
            detail_predictions=detail_predictions,
            logits=logits,
            calibration_parameters=calibration_parameters,
            context_attention=context_attention,
            prototype_usage=prototype_usage,
            intermediate_features=intermediate_features,
        )


CAMPV2 = CAMPVSv2
