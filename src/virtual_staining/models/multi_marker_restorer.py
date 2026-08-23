"""Deterministic NAF-style multi-marker restoration network."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .context_fusion import (
    BottleneckContextCrossAttention,
    MultiScaleContextFusion,
)
from .context_tile_encoder import SharedTinyContextEncoder
from .laplacian_decoder import LaplacianBaseDetailHead
from .lightweight_detail_unet import LightweightDetailUNet
from .naf_blocks import DecoderStage, NAFStage, SobelMagnitude, TaskAdapter
from .prototype_mixer import MultiTaskPrototypeMixer
from .registry import RestorationOutput, register_model
from .spatial_frequency_mixer import ParallelSpatialFrequencyMixer


def _normalized_task_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _attention_heads(token_dim: int, requested: int = 4) -> int:
    for heads in range(min(token_dim, requested), 0, -1):
        if token_dim % heads == 0:
            return heads
    return 1


@register_model("multi_marker_restorer", "multimarkerrestorer", "main")
class MultiMarkerRestorer(nn.Module):
    """Shared four-scale encoder/decoder with task adapters and prototype mixing."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int | Mapping[str, int] = 1,
        base_channels: int = 48,
        encoder_depths: Sequence[int] = (2, 2, 4, 6),
        decoder_depths: Sequence[int] = (2, 2, 2),
        target_names: Sequence[str] = ("HLA-DR", "CD45RO", "Vimentin", "CD68"),
        use_sobel_input: bool = True,
        use_task_adapters: bool = True,
        use_prototypes: bool = True,
        shared_prototypes: int = 8,
        task_prototypes: int = 8,
        prototype_temperature: float = 0.1,
        prototype_fusion_weight: float = 0.1,
        deep_supervision: bool = True,
        decoder_mode: str = "shared_decoder_with_adapters",
        output_activation: str = "sigmoid",
        base_detail: bool = False,
        base_detail_residual: bool = False,
        max_detail_amplitude: float = 1.0,
        context: Mapping[str, object] | None = None,
        spatial_frequency: Mapping[str, object] | None = None,
        lightweight_unet: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        if in_channels < 1 or base_channels < 1:
            raise ValueError("Input and base channels must be positive")
        if len(encoder_depths) != 4 or len(decoder_depths) != 3:
            raise ValueError("Four encoder depths and three decoder depths are required")
        if any(depth < 1 for depth in (*encoder_depths, *decoder_depths)):
            raise ValueError("All encoder and decoder depths must be positive")
        if decoder_mode not in {"shared_decoder_with_adapters", "separate_decoders"}:
            raise ValueError(
                "decoder_mode must be 'shared_decoder_with_adapters' or 'separate_decoders'"
            )
        if output_activation != "sigmoid":
            raise ValueError("MultiMarkerRestorer currently supports output_activation='sigmoid'")

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.target_names = tuple(str(name) for name in target_names)
        if not self.target_names or len(set(self.target_names)) != len(self.target_names):
            raise ValueError("Target names must be non-empty and unique")
        normalized_names = [_normalized_task_name(name) for name in self.target_names]
        if len(set(normalized_names)) != len(normalized_names):
            raise ValueError("Target names must remain unique after punctuation normalization")

        if isinstance(out_channels, int):
            output_channels = {task: out_channels for task in self.target_names}
        else:
            output_channels = {task: int(out_channels[task]) for task in self.target_names}
        if any(channels < 1 for channels in output_channels.values()):
            raise ValueError("All target output channel counts must be positive")

        self.output_channels = output_channels
        self.use_sobel_input = bool(use_sobel_input)
        self.use_task_adapters = bool(use_task_adapters)
        self.use_prototypes = bool(use_prototypes)
        self.deep_supervision_enabled = bool(deep_supervision)
        self.decoder_mode = decoder_mode
        self.base_detail_enabled = bool(base_detail)
        self.base_detail_residual_enabled = bool(base_detail_residual)
        if self.base_detail_residual_enabled and not self.base_detail_enabled:
            raise ValueError("base_detail_residual requires base_detail=True")
        if max_detail_amplitude < 0.0:
            raise ValueError("max_detail_amplitude cannot be negative")
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("context must be a mapping or None")
        if spatial_frequency is not None and not isinstance(spatial_frequency, Mapping):
            raise TypeError("spatial_frequency must be a mapping or None")
        if lightweight_unet is not None and not isinstance(lightweight_unet, Mapping):
            raise TypeError("lightweight_unet must be a mapping or None")
        context_options = dict(context or {})
        spatial_frequency_options = dict(spatial_frequency or {})
        lightweight_unet_options = dict(lightweight_unet or {})
        self.context_enabled = bool(context_options.get("enabled", False))
        self.spatial_frequency_enabled = bool(
            spatial_frequency_options.get("enabled", False)
        )
        self.lightweight_unet_enabled = bool(
            lightweight_unet_options.get("enabled", False)
        )
        cross_attention_requested = bool(
            context_options.get("bottleneck_cross_attention", False)
        )
        if cross_attention_requested and not self.context_enabled:
            raise ValueError(
                "model.context.bottleneck_cross_attention requires context.enabled=true"
            )
        self.feature_flags = {
            "base_detail": self.base_detail_enabled,
            "base_detail_residual": self.base_detail_residual_enabled,
            "context": self.context_enabled,
            "context_cross_attention": cross_attention_requested,
            "spatial_frequency": self.spatial_frequency_enabled,
            "lightweight_unet": self.lightweight_unet_enabled,
        }
        self._task_keys = {task: f"task_{index}" for index, task in enumerate(self.target_names)}
        self._normalized_tasks = {
            _normalized_task_name(task): task for task in self.target_names
        }

        channels = [base_channels * (2**index) for index in range(4)]
        stem_channels = in_channels + (1 if use_sobel_input else 0)
        self.sobel = SobelMagnitude() if use_sobel_input else None
        self.stem = nn.Conv2d(stem_channels, channels[0], kernel_size=3, padding=1)
        self.encoder_stages = nn.ModuleList(
            NAFStage(stage_channels, int(depth))
            for stage_channels, depth in zip(channels, encoder_depths, strict=True)
        )
        self.downsamples = nn.ModuleList(
            nn.Conv2d(channels[index], channels[index + 1], kernel_size=2, stride=2)
            for index in range(3)
        )

        self.lightweight_detail_unet = (
            LightweightDetailUNet(
                stem_channels,
                channels[:3],
                widths=tuple(
                    int(width)
                    for width in lightweight_unet_options.get(
                        "widths", (16, 24, 32, 48)
                    )
                ),
                depths=tuple(
                    int(depth)
                    for depth in lightweight_unet_options.get(
                        "depths", (1, 1, 1, 1)
                    )
                ),
                kernel_size=int(lightweight_unet_options.get("kernel_size", 7)),
                expansion=int(lightweight_unet_options.get("expansion", 2)),
                fusion_scales=tuple(
                    int(scale)
                    for scale in lightweight_unet_options.get(
                        "fusion_scales", (1, 2, 4)
                    )
                ),
                residual_init=float(
                    lightweight_unet_options.get("residual_init", 0.0)
                ),
            )
            if self.lightweight_unet_enabled
            else None
        )

        if self.context_enabled:
            grid_size = int(context_options.get("grid_size", 3))
            token_dim = int(context_options.get("token_dim", max(32, base_channels)))
            encoder_width = int(context_options.get("encoder_width", max(4, base_channels // 2)))
            encoder_depth = int(context_options.get("encoder_depth", 2))
            fusion_scales = tuple(
                int(scale) for scale in context_options.get("fusion_scales", (1, 2, 4, 8))
            )
            available_scales = {1, 2, 4, 8}
            invalid_scales = sorted(set(fusion_scales).difference(available_scales))
            if invalid_scales:
                raise ValueError(
                    "MultiMarkerRestorer context fusion scales must be drawn from "
                    f"1/2/4/8, got {invalid_scales}"
                )
            self.context_encoder = SharedTinyContextEncoder(
                in_channels=in_channels,
                width=encoder_width,
                token_dim=token_dim,
                grid_size=grid_size,
                depth=encoder_depth,
                stop_gradient=bool(context_options.get("stop_gradient", False)),
            )
            self.context_fusion = MultiScaleContextFusion(
                {scale: channels[index] for index, scale in enumerate((1, 2, 4, 8))},
                token_dim,
                fusion_scales=fusion_scales,
                context_dropout=float(context_options.get("context_dropout", 0.1)),
            )
            cross_attention_heads = int(
                context_options.get(
                    "cross_attention_heads",
                    _attention_heads(token_dim),
                )
            )
            self.context_cross_attention = (
                BottleneckContextCrossAttention(
                    channels[-1],
                    token_dim,
                    heads=cross_attention_heads,
                    residual_init=float(context_options.get("residual_init", 0.0)),
                )
                if cross_attention_requested
                else None
            )
        else:
            self.context_encoder = None
            self.context_fusion = None
            self.context_cross_attention = None

        self.spatial_frequency_mixer = (
            ParallelSpatialFrequencyMixer(
                channels[-1],
                spatial_depth=int(spatial_frequency_options.get("spatial_depth", 1)),
                spatial_expansion=int(
                    spatial_frequency_options.get("spatial_expansion", 1)
                ),
                gate_reduction=int(
                    spatial_frequency_options.get("gate_reduction", 8)
                ),
                frequency_cutoff=float(
                    spatial_frequency_options.get("frequency_cutoff", 0.35)
                ),
                frequency_transition_width=float(
                    spatial_frequency_options.get("frequency_transition_width", 0.08)
                ),
                residual_init=float(
                    spatial_frequency_options.get("residual_init", 0.0)
                ),
            )
            if self.spatial_frequency_enabled
            else None
        )

        self.prototype_mixer = (
            MultiTaskPrototypeMixer(
                channels[-1],
                self.target_names,
                shared_prototypes=shared_prototypes,
                task_prototypes=task_prototypes,
                temperature=prototype_temperature,
                fusion_weight=prototype_fusion_weight,
            )
            if use_prototypes
            else None
        )

        decoder_specs = [
            (channels[3], channels[2], channels[2], int(decoder_depths[0])),
            (channels[2], channels[1], channels[1], int(decoder_depths[1])),
            (channels[1], channels[0], channels[0], int(decoder_depths[2])),
        ]
        if decoder_mode == "shared_decoder_with_adapters":
            self.shared_decoder = nn.ModuleList(
                DecoderStage(*specification) for specification in decoder_specs
            )
            self.separate_decoders = nn.ModuleDict()
        else:
            self.shared_decoder = nn.ModuleList()
            self.separate_decoders = nn.ModuleDict(
                {
                    self._task_keys[task]: nn.ModuleList(
                        DecoderStage(*specification) for specification in decoder_specs
                    )
                    for task in self.target_names
                }
            )

        embedding_dim = max(16, base_channels)
        self.task_embeddings = nn.Embedding(len(self.target_names), embedding_dim)
        self.task_indices = {task: index for index, task in enumerate(self.target_names)}
        self.adapters = nn.ModuleDict(
            {
                self._task_keys[task]: nn.ModuleList(
                    TaskAdapter(stage_channels, embedding_dim)
                    for stage_channels in (channels[2], channels[1], channels[0])
                )
                for task in self.target_names
            }
        )
        self.output_heads = nn.ModuleDict(
            {
                self._task_keys[task]: nn.ModuleList(
                    nn.Conv2d(stage_channels, output_channels[task], kernel_size=1)
                    for stage_channels in (channels[2], channels[1], channels[0])
                )
                for task in self.target_names
            }
        )
        self.base_detail_heads = (
            nn.ModuleDict(
                {
                    self._task_keys[task]: LaplacianBaseDetailHead(
                        low_channels=channels[2],
                        full_channels=channels[0],
                        out_channels=output_channels[task],
                        max_detail_amplitude=max_detail_amplitude,
                        residual_to_reference=self.base_detail_residual_enabled,
                    )
                    for task in self.target_names
                }
            )
            if self.base_detail_enabled
            else None
        )

    def resolve_task(self, task_name: str) -> str:
        """Resolve task spelling while accepting hyphen/underscore/case variants."""

        normalized = _normalized_task_name(task_name)
        if normalized not in self._normalized_tasks:
            available = ", ".join(self.target_names)
            raise KeyError(f"Unknown task {task_name!r}; available: {available}")
        return self._normalized_tasks[normalized]

    def _selected_tasks(self, task_name: str | Sequence[str] | None) -> tuple[str, ...]:
        if task_name is None:
            return self.target_names
        if isinstance(task_name, str):
            return (self.resolve_task(task_name),)
        selected = tuple(self.resolve_task(name) for name in task_name)
        if not selected:
            raise ValueError("At least one task must be selected")
        return selected

    def _encode(self, inputs: Tensor) -> list[Tensor]:
        features = self.stem(inputs)
        encoded: list[Tensor] = []
        for index, stage in enumerate(self.encoder_stages):
            features = stage(features)
            encoded.append(features)
            if index < len(self.downsamples):
                features = self.downsamples[index](features)
        return encoded

    def _decode_task(
        self,
        bottleneck: Tensor,
        skips: list[Tensor],
        task: str,
    ) -> tuple[
        Tensor,
        dict[int, Tensor],
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        task_key = self._task_keys[task]
        stages = (
            self.shared_decoder
            if self.decoder_mode == "shared_decoder_with_adapters"
            else self.separate_decoders[task_key]
        )
        task_ids = torch.full(
            (bottleneck.shape[0],),
            self.task_indices[task],
            device=bottleneck.device,
            dtype=torch.long,
        )
        embedding = self.task_embeddings(task_ids)
        features = bottleneck
        predictions: list[Tensor] = []
        prediction_logits: list[Tensor] = []
        decoded_features: list[Tensor] = []
        for index, (stage, skip) in enumerate(zip(stages, skips, strict=True)):
            features = stage(features, skip)
            if self.use_task_adapters:
                features = self.adapters[task_key][index](features, embedding)
            decoded_features.append(features)
            stage_logits = self.output_heads[task_key][index](features)
            prediction_logits.append(stage_logits)
            predictions.append(torch.sigmoid(stage_logits))

        base_prediction: Tensor | None = None
        detail_prediction: Tensor | None = None
        final_logits: Tensor | None = None
        if self.base_detail_heads is not None:
            decomposition = self.base_detail_heads[task_key](
                decoded_features[0],
                decoded_features[-1],
                reference_direct_logits=(
                    prediction_logits[-1]
                    if self.base_detail_residual_enabled
                    else None
                ),
            )
            final_prediction = decomposition.prediction
            predictions[-1] = final_prediction
            base_prediction = decomposition.base
            detail_prediction = decomposition.detail
            final_logits = decomposition.final_logits
        else:
            final_prediction = predictions[-1]
        if self.deep_supervision_enabled:
            supervision = {prediction.shape[-2]: prediction for prediction in predictions}
        else:
            supervision = {final_prediction.shape[-2]: final_prediction}
        return (
            final_prediction,
            supervision,
            base_prediction,
            detail_prediction,
            final_logits,
        )

    def forward(
        self,
        inputs: Tensor,
        task_name: str | Sequence[str] | None = None,
        *,
        context_tiles: Tensor | None = None,
        context_valid_mask: Tensor | None = None,
        context_offsets: Tensor | None = None,
        organ_id: str | Sequence[str] | Tensor | None = None,
    ) -> RestorationOutput:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected BCHW input with {self.in_channels} channels, got {tuple(inputs.shape)}"
            )
        selected_tasks = self._selected_tasks(task_name)
        model_inputs = inputs
        if self.sobel is not None:
            model_inputs = torch.cat((inputs, self.sobel(inputs)), dim=1)
        encoded = self._encode(model_inputs)
        if self.lightweight_detail_unet is not None:
            detail_updates = self.lightweight_detail_unet(model_inputs)
            scale_to_index = {1: 0, 2: 1, 4: 2}
            for scale, update in detail_updates.items():
                index = scale_to_index[scale]
                encoded[index] = encoded[index] + update
        context_attention: dict[str, Tensor] = {}
        context_output = None
        if self.context_enabled:
            if context_tiles is None:
                raise ValueError("context_tiles are required when context is enabled")
            if self.context_encoder is None or self.context_fusion is None:
                raise RuntimeError("Enabled context modules were not initialized")
            context_output = self.context_encoder(
                context_tiles,
                context_valid_mask,
                context_offsets,
            )
            scale_order = (1, 2, 4, 8)
            fused, context_attention = self.context_fusion(
                dict(zip(scale_order, encoded, strict=True)),
                context_output.tokens,
                context_output.valid_mask,
            )
            encoded = [fused[scale] for scale in scale_order]
        if self.context_cross_attention is not None:
            if context_output is None:
                raise RuntimeError("Cross-attention requires encoded context")
            encoded[-1], cross_attention = self.context_cross_attention(
                encoded[-1],
                context_output.tokens,
                context_output.valid_mask,
            )
            context_attention["cross_8"] = cross_attention
        _ = organ_id
        if self.spatial_frequency_mixer is not None:
            encoded[-1] = self.spatial_frequency_mixer(encoded[-1])
        shared_bottleneck = encoded[-1]
        skips = [encoded[2], encoded[1], encoded[0]]

        predictions: dict[str, Tensor] = {}
        deep_supervision: dict[str, dict[int, Tensor]] = {}
        attention: dict[str, dict[str, Tensor]] = {}
        prototype_features: dict[str, Tensor] = {}
        base_predictions: dict[str, Tensor] = {}
        detail_predictions: dict[str, Tensor] = {}
        logits: dict[str, Tensor] = {}
        for task in selected_tasks:
            bottleneck = shared_bottleneck
            if self.prototype_mixer is not None:
                mixed = self.prototype_mixer(shared_bottleneck, task)
                bottleneck = mixed.features
                attention[task] = {
                    "shared": mixed.shared_attention,
                    "task": mixed.task_attention,
                }
                prototype_features[task] = mixed.normalized_tokens
            (
                prediction,
                supervision,
                base_prediction,
                detail_prediction,
                final_logits,
            ) = self._decode_task(bottleneck, skips, task)
            predictions[task] = prediction
            deep_supervision[task] = supervision
            if base_prediction is not None:
                base_predictions[task] = base_prediction
            if detail_prediction is not None:
                detail_predictions[task] = detail_prediction
            if final_logits is not None:
                logits[task] = final_logits

        prototype_banks = self.prototype_mixer.banks() if self.prototype_mixer is not None else {}
        return RestorationOutput(
            predictions=predictions,
            deep_supervision=deep_supervision,
            prototype_attention=attention,
            prototype_features=prototype_features,
            prototype_banks=prototype_banks,
            base_predictions=base_predictions,
            detail_predictions=detail_predictions,
            logits=logits,
            context_attention=context_attention,
        )
