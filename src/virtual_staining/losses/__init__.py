"""Differentiable objectives for image restoration training."""

from .charbonnier import CharbonnierLoss, structure_weight_map
from .competition_proxy import (
    CompetitionProxyLoss,
    CompetitionProxyOutput,
    capped_per_image_psnr_loss,
    uniform_window_ssim,
)
from .composite import (
    CompositeRestorationLoss,
    LossOutput,
    LossWeights,
    task_correlation_loss,
)
from .fluorescence_foreground import (
    FluorescenceForegroundLoss,
    FluorescenceForegroundOutput,
)
from .frequency import FrequencyAmplitudeLoss
from .gradient import GradientLoss, sobel_gradients
from .prototype import (
    PrototypeLossOutput,
    PrototypeRegularization,
    prototype_activation_loss,
    prototype_diversity_loss,
    prototype_usage_entropy_loss,
)
from .pyramid import (
    GaussianPyramidLoss,
    LaplacianPyramidLoss,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    gaussian_blur,
    gaussian_kernel,
)
from .scheduled_composite import (
    DEFAULT_PHASE_A,
    DEFAULT_PHASE_B,
    ScheduledCompositeLoss,
    ScheduledLossWeights,
    TwoPhaseLossSchedule,
    epoch_progress,
)
from .shift_tolerant import ShiftTolerantLoss, overlapping_shift_pair
from .ssim import (
    DifferentiableSSIM,
    MSSSIMLoss,
    MultiScaleSSIM,
    SSIMLoss,
    differentiable_ms_ssim,
    differentiable_ssim,
)
from .statistics import IntensityStatisticsLoss

__all__ = [
    "CharbonnierLoss",
    "CompetitionProxyLoss",
    "CompetitionProxyOutput",
    "CompositeRestorationLoss",
    "DifferentiableSSIM",
    "FrequencyAmplitudeLoss",
    "FluorescenceForegroundLoss",
    "FluorescenceForegroundOutput",
    "GradientLoss",
    "GaussianPyramidLoss",
    "IntensityStatisticsLoss",
    "LaplacianPyramidLoss",
    "LossOutput",
    "LossWeights",
    "MSSSIMLoss",
    "MultiScaleSSIM",
    "PrototypeLossOutput",
    "PrototypeRegularization",
    "SSIMLoss",
    "ScheduledCompositeLoss",
    "ScheduledLossWeights",
    "ShiftTolerantLoss",
    "TwoPhaseLossSchedule",
    "DEFAULT_PHASE_A",
    "DEFAULT_PHASE_B",
    "build_gaussian_pyramid",
    "build_laplacian_pyramid",
    "capped_per_image_psnr_loss",
    "differentiable_ms_ssim",
    "differentiable_ssim",
    "epoch_progress",
    "gaussian_blur",
    "gaussian_kernel",
    "overlapping_shift_pair",
    "prototype_activation_loss",
    "prototype_diversity_loss",
    "prototype_usage_entropy_loss",
    "sobel_gradients",
    "structure_weight_map",
    "task_correlation_loss",
    "uniform_window_ssim",
]
