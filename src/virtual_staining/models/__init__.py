"""Neural network architectures for virtual staining."""

from .baseline_unet import ResidualUNet
from .camp_vs_v2 import CAMPVSv2
from .heavy_detail_unet import (
    DenseHighResolutionBlock,
    HeavyDetailUNet,
    OmniKernelResidualBlock,
)
from .lightweight_detail_unet import (
    LargeKernelGatedResidualBlock,
    LightweightDetailUNet,
)
from .multi_marker_restorer import MultiMarkerRestorer
from .multi_route_fusion import (
    CrossGatedSkipFusion,
    RestormerSDPARefinement,
    TriPathAdaptiveFusion,
)
from .registry import (
    MODEL_REGISTRY,
    RestorationOutput,
    available_models,
    build_model,
    count_parameters,
    model_statistics,
)
from .spatial_frequency_mixer import (
    AdaptiveFFTLowHighBranch,
    FFTLowHighBranch,
    ParallelSpatialFrequencyMixer,
)

__all__ = [
    "MODEL_REGISTRY",
    "CAMPVSv2",
    "AdaptiveFFTLowHighBranch",
    "CrossGatedSkipFusion",
    "DenseHighResolutionBlock",
    "FFTLowHighBranch",
    "HeavyDetailUNet",
    "LargeKernelGatedResidualBlock",
    "LightweightDetailUNet",
    "MultiMarkerRestorer",
    "OmniKernelResidualBlock",
    "ParallelSpatialFrequencyMixer",
    "ResidualUNet",
    "RestorationOutput",
    "RestormerSDPARefinement",
    "TriPathAdaptiveFusion",
    "available_models",
    "build_model",
    "count_parameters",
    "model_statistics",
]
