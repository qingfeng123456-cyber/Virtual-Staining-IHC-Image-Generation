"""Neural network architectures for virtual staining."""

from .baseline_unet import ResidualUNet
from .camp_vs_v2 import CAMPVSv2
from .lightweight_detail_unet import (
    LargeKernelGatedResidualBlock,
    LightweightDetailUNet,
)
from .multi_marker_restorer import MultiMarkerRestorer
from .registry import (
    MODEL_REGISTRY,
    RestorationOutput,
    available_models,
    build_model,
    count_parameters,
    model_statistics,
)
from .spatial_frequency_mixer import FFTLowHighBranch, ParallelSpatialFrequencyMixer

__all__ = [
    "MODEL_REGISTRY",
    "CAMPVSv2",
    "FFTLowHighBranch",
    "LargeKernelGatedResidualBlock",
    "LightweightDetailUNet",
    "MultiMarkerRestorer",
    "ParallelSpatialFrequencyMixer",
    "ResidualUNet",
    "RestorationOutput",
    "available_models",
    "build_model",
    "count_parameters",
    "model_statistics",
]
