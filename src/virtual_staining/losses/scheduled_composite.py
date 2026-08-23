"""Continuous two-phase metric-aligned restoration objective."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .charbonnier import CharbonnierLoss, float32_context
from .competition_proxy import CompetitionProxyLoss
from .composite import (
    LossOutput,
    _align_targets,
    _extract_predictions,
    _zero_from_predictions,
    task_correlation_loss,
)
from .fluorescence_foreground import FluorescenceForegroundLoss
from .frequency import FrequencyAmplitudeLoss
from .gradient import GradientLoss
from .prototype import PrototypeRegularization, prototype_usage_entropy_loss
from .pyramid import LaplacianPyramidLoss
from .shift_tolerant import ShiftTolerantLoss
from .ssim import MSSSIMLoss, SSIMLoss
from .statistics import IntensityStatisticsLoss


@dataclass(frozen=True, slots=True)
class ScheduledLossWeights:
    """Weights for the reconstruction terms in one schedule endpoint."""

    mse: float = 0.10
    charbonnier: float = 0.35
    ssim: float = 0.30
    ms_ssim: float = 0.10
    pyramid: float = 0.10
    gradient: float = 0.03
    statistics: float = 0.02

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Scheduled loss weights must be finite and nonnegative")
        if sum(values.values()) <= 0:
            raise ValueError("At least one scheduled loss weight must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> ScheduledLossWeights:
        return cls(**{key: float(value) for key, value in values.items()})

    def interpolate(
        self,
        other: ScheduledLossWeights,
        fraction: float,
    ) -> ScheduledLossWeights:
        amount = min(1.0, max(0.0, float(fraction)))
        first = asdict(self)
        second = asdict(other)
        return ScheduledLossWeights(
            **{
                name: first[name] + amount * (second[name] - first[name])
                for name in first
            }
        )


DEFAULT_PHASE_A = ScheduledLossWeights()
DEFAULT_PHASE_B = ScheduledLossWeights(
    mse=0.35,
    charbonnier=0.15,
    ssim=0.35,
    ms_ssim=0.05,
    pyramid=0.07,
    gradient=0.02,
    statistics=0.01,
)


class TwoPhaseLossSchedule:
    """Hold Phase A, then continuously interpolate over the final training fraction."""

    def __init__(
        self,
        phase_a: ScheduledLossWeights | Mapping[str, float] = DEFAULT_PHASE_A,
        phase_b: ScheduledLossWeights | Mapping[str, float] = DEFAULT_PHASE_B,
        *,
        phase_a_ratio: float = 0.70,
        interpolation: str = "cosine",
    ) -> None:
        if not 0.0 <= phase_a_ratio < 1.0:
            raise ValueError("phase_a_ratio must be in [0, 1)")
        if interpolation not in {"linear", "cosine"}:
            raise ValueError("interpolation must be 'linear' or 'cosine'")
        self.phase_a = (
            phase_a
            if isinstance(phase_a, ScheduledLossWeights)
            else ScheduledLossWeights.from_mapping(phase_a)
        )
        self.phase_b = (
            phase_b
            if isinstance(phase_b, ScheduledLossWeights)
            else ScheduledLossWeights.from_mapping(phase_b)
        )
        self.phase_a_ratio = float(phase_a_ratio)
        self.interpolation = interpolation

    def transition_fraction(self, progress: float) -> float:
        position = min(1.0, max(0.0, float(progress)))
        fraction = max(0.0, (position - self.phase_a_ratio) / (1.0 - self.phase_a_ratio))
        if self.interpolation == "cosine":
            fraction = 0.5 - 0.5 * math.cos(math.pi * fraction)
        return fraction

    def weights_at(self, progress: float) -> ScheduledLossWeights:
        return self.phase_a.interpolate(self.phase_b, self.transition_fraction(progress))

    def state_dict(self) -> dict[str, Any]:
        return {
            "phase_a": asdict(self.phase_a),
            "phase_b": asdict(self.phase_b),
            "phase_a_ratio": self.phase_a_ratio,
            "interpolation": self.interpolation,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> TwoPhaseLossSchedule:
        return cls(
            phase_a=state["phase_a"],
            phase_b=state["phase_b"],
            phase_a_ratio=float(state["phase_a_ratio"]),
            interpolation=str(state["interpolation"]),
        )


def epoch_progress(epoch: int, total_epochs: int) -> float:
    """Map a zero-based epoch to [0, 1], reaching one on the final epoch."""

    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    return min(1.0, float(epoch) / max(1, total_epochs - 1))


class ScheduledCompositeLoss(nn.Module):
    """Explicit MSE/SSIM objective whose weights are safe to resume mid-schedule."""

    def __init__(
        self,
        schedule: TwoPhaseLossSchedule | None = None,
        *,
        pyramid_levels: int = 4,
        statistics_pooled_weight: float = 0.0,
        task_balancer: nn.Module | None = None,
        deep_supervision_weights: Sequence[float] = (1.0, 0.5, 0.25),
        deep_supervision_final_weights: Sequence[float] | None = None,
        frequency: float = 0.0,
        correlation: float = 0.0,
        prototype_activation: float = 0.0,
        prototype_diversity: float = 0.0,
        prototype_usage_entropy: float = 0.0,
        shift_tolerant_enabled: bool = False,
        shift_tolerant_weight: float = 1.0,
        shift_tolerant_max_shift: int = 1,
        shift_tolerant_mode: str = "hard",
        shift_tolerant_temperature: float = 0.05,
        foreground_enabled: bool = False,
        foreground_weight: float = 0.0,
        foreground_threshold_std_scale: float = 0.25,
        foreground_temperature: float = 0.05,
        foreground_mse_weight: float = 1.0,
        foreground_dice_weight: float = 0.10,
        foreground_intensity_weight: float = 0.25,
        foreground_min_activity_std: float = 0.01,
        competition_proxy_enabled: bool = False,
        competition_proxy_weight: float = 0.0,
        competition_proxy_start_ratio: float = 0.55,
        competition_proxy_ssim_weight: float = 0.7,
        competition_proxy_window_size: int = 7,
        competition_proxy_psnr_cap: float = 60.0,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        if not deep_supervision_weights or any(
            not math.isfinite(float(weight)) or float(weight) <= 0.0
            for weight in deep_supervision_weights
        ):
            raise ValueError("Deep supervision weights must be finite and positive")
        final_deep_weights = (
            tuple(float(weight) for weight in deep_supervision_final_weights)
            if deep_supervision_final_weights is not None
            else tuple(float(weight) for weight in deep_supervision_weights)
        )
        if (
            len(final_deep_weights) != len(deep_supervision_weights)
            or any(not math.isfinite(weight) or weight < 0.0 for weight in final_deep_weights)
            or final_deep_weights[0] <= 0.0
        ):
            raise ValueError(
                "Final deep supervision weights must match the initial length, "
                "be finite/nonnegative, and retain a positive full-resolution weight"
            )
        if (
            not math.isfinite(float(competition_proxy_weight))
            or float(competition_proxy_weight) < 0.0
        ):
            raise ValueError("Competition proxy weight must be finite and nonnegative")
        if not 0.0 <= float(competition_proxy_start_ratio) < 1.0:
            raise ValueError("Competition proxy start_ratio must lie in [0, 1)")
        if competition_proxy_enabled and float(competition_proxy_weight) <= 0.0:
            raise ValueError("Enabled competition proxy requires a positive weight")
        auxiliary_weights = {
            "frequency": float(frequency),
            "correlation": float(correlation),
            "prototype_activation": float(prototype_activation),
            "prototype_diversity": float(prototype_diversity),
            "prototype_usage_entropy": float(prototype_usage_entropy),
            "shift_tolerant": (
                float(shift_tolerant_weight) if shift_tolerant_enabled else 0.0
            ),
            "fluorescence_foreground": (
                float(foreground_weight) if foreground_enabled else 0.0
            ),
        }
        invalid_auxiliary = [
            name
            for name, weight in auxiliary_weights.items()
            if not math.isfinite(weight) or weight < 0.0
        ]
        if invalid_auxiliary:
            raise ValueError(
                "Auxiliary loss weights must be finite and nonnegative: "
                + ", ".join(invalid_auxiliary)
            )
        self.schedule = schedule or TwoPhaseLossSchedule()
        endpoint_weights = (
            asdict(self.schedule.phase_a),
            asdict(self.schedule.phase_b),
        )
        self.active_reconstruction_terms = frozenset(
            name
            for name in endpoint_weights[0]
            if any(endpoint[name] > 0.0 for endpoint in endpoint_weights)
        )
        self.deep_supervision_weights = tuple(
            float(weight) for weight in deep_supervision_weights
        )
        self.deep_supervision_final_weights = final_deep_weights
        self.auxiliary_weights = auxiliary_weights
        self.charbonnier_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss(data_range=data_range)
        self.ms_ssim_loss = MSSSIMLoss(data_range=data_range)
        self.pyramid_loss = LaplacianPyramidLoss(levels=pyramid_levels)
        self.gradient_loss = GradientLoss()
        self.statistics_loss = IntensityStatisticsLoss(
            pooled_weight=statistics_pooled_weight
        )
        self.frequency_loss = FrequencyAmplitudeLoss()
        self.prototype_loss = PrototypeRegularization()
        self.shift_tolerant_loss = (
            ShiftTolerantLoss(
                max_shift=shift_tolerant_max_shift,
                mode=shift_tolerant_mode,
                temperature=shift_tolerant_temperature,
            )
            if shift_tolerant_enabled
            else None
        )
        self.foreground_loss = (
            FluorescenceForegroundLoss(
                threshold_std_scale=foreground_threshold_std_scale,
                temperature=foreground_temperature,
                mse_weight=foreground_mse_weight,
                dice_weight=foreground_dice_weight,
                intensity_weight=foreground_intensity_weight,
                min_activity_std=foreground_min_activity_std,
            )
            if foreground_enabled and foreground_weight > 0.0
            else None
        )
        self.competition_proxy_weight = (
            float(competition_proxy_weight) if competition_proxy_enabled else 0.0
        )
        self.competition_proxy_start_ratio = float(competition_proxy_start_ratio)
        self.competition_proxy_loss = (
            CompetitionProxyLoss(
                ssim_weight=float(competition_proxy_ssim_weight),
                data_range=data_range,
                window_size=int(competition_proxy_window_size),
                psnr_cap=float(competition_proxy_psnr_cap),
            )
            if competition_proxy_enabled
            else None
        )
        self.task_balancer = task_balancer
        self.register_buffer("_progress", torch.tensor(0.0, dtype=torch.float32))
        self.force_phase_b = False

    @property
    def progress(self) -> float:
        return float(self._progress.item())

    def optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        """Expose optional learned loss state that must join the optimizer.

        Equal weighting and FAMO have no trainable parameters, so the default
        path remains unchanged.  Uncertainty weighting owns learned log
        variances; returning them explicitly lets the generic trainer include
        them without coupling the engine to a particular balancer class.
        """

        if self.task_balancer is None:
            return ()
        return tuple(
            parameter
            for parameter in self.task_balancer.parameters()
            if parameter.requires_grad
        )

    def set_progress(
        self,
        progress: float | None = None,
        *,
        epoch: int | None = None,
        total_epochs: int | None = None,
    ) -> None:
        if progress is None:
            if epoch is None or total_epochs is None:
                raise ValueError("Provide progress or both epoch and total_epochs")
            progress = epoch_progress(epoch, total_epochs)
        if not math.isfinite(float(progress)) or not 0.0 <= float(progress) <= 1.0:
            raise ValueError("Loss schedule progress must be finite and in [0, 1]")
        self._progress.fill_(float(progress))

    def current_weights(self) -> ScheduledLossWeights:
        return self.schedule.phase_b if self.force_phase_b else self.schedule.weights_at(self.progress)

    def _terms(self, prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
        if prediction.shape != target.shape:
            raise ValueError("Scheduled loss prediction and target shapes differ")
        with float32_context(prediction):
            prediction_float = prediction.float()
            target_float = target.float()
            zero = prediction_float.sum() * 0.0
            factories = {
                "mse": lambda: F.mse_loss(prediction_float, target_float),
                "charbonnier": lambda: self.charbonnier_loss(
                    prediction_float, target_float
                ),
                "ssim": lambda: self.ssim_loss(prediction_float, target_float),
                "ms_ssim": lambda: self.ms_ssim_loss(
                    prediction_float, target_float
                ),
                "pyramid": lambda: self.pyramid_loss(
                    prediction_float, target_float
                ),
                "gradient": lambda: self.gradient_loss(
                    prediction_float, target_float
                ),
                "statistics": lambda: self.statistics_loss(
                    prediction_float, target_float
                ),
            }
            return {
                name: factory() if name in self.active_reconstruction_terms else zero
                for name, factory in factories.items()
            }

    def _late_fraction(self, start_ratio: float) -> float:
        if self.force_phase_b:
            return 1.0
        if self.progress <= start_ratio:
            return 0.0
        position = (self.progress - start_ratio) / (1.0 - start_ratio)
        return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, position)))

    def current_deep_supervision_weights(self) -> tuple[float, ...]:
        fraction = self.schedule.transition_fraction(self.progress)
        if self.force_phase_b:
            fraction = 1.0
        return tuple(
            initial + fraction * (final - initial)
            for initial, final in zip(
                self.deep_supervision_weights,
                self.deep_supervision_final_weights,
                strict=True,
            )
        )

    def _deep_supervision_weight(self, index: int) -> float:
        weights = self.current_deep_supervision_weights()
        if index < len(weights):
            return weights[index]
        return weights[-1] * 0.5 ** (
            index - len(weights) + 1
        )

    def current_competition_proxy_weight(self) -> float:
        return self.competition_proxy_weight * self._late_fraction(
            self.competition_proxy_start_ratio
        )

    @staticmethod
    def _prototype_usage(output: Any) -> Mapping[str, Any]:
        usage = getattr(output, "prototype_usage", None)
        if isinstance(usage, Mapping):
            return usage
        if isinstance(output, Mapping):
            candidate = output.get("prototype_usage")
            if isinstance(candidate, Mapping):
                return candidate
        return {}

    @staticmethod
    def _aligned_full_target(prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape[-2:] == target.shape[-2:]:
            return target
        return F.interpolate(target, size=prediction.shape[-2:], mode="area")

    def forward(
        self,
        output: Any,
        targets: Mapping[str, Tensor] | Tensor,
        *,
        progress: float | None = None,
    ) -> LossOutput:
        if progress is not None:
            self.set_progress(progress)
        predictions, deep_supervision, prototype_features, prototype_banks = (
            _extract_predictions(output)
        )
        if not predictions:
            raise ValueError("ScheduledCompositeLoss requires at least one prediction")
        aligned = _align_targets(targets, tuple(predictions))
        weights = self.current_weights()
        weight_values = asdict(weights)
        components: dict[str, Tensor] = {}
        per_task: dict[str, Tensor] = {}
        for task, full_prediction in predictions.items():
            scale_predictions = dict(deep_supervision.get(task, {}))
            scale_predictions[full_prediction.shape[-2]] = full_prediction
            ordered_predictions = sorted(
                scale_predictions.values(),
                key=lambda tensor: tensor.shape[-2] * tensor.shape[-1],
                reverse=True,
            )
            task_total = full_prediction.float().sum() * 0.0
            for scale_index, scale_prediction in enumerate(ordered_predictions):
                target = self._aligned_full_target(scale_prediction, aligned[task])
                terms = self._terms(scale_prediction, target)
                resolution = f"{scale_prediction.shape[-2]}x{scale_prediction.shape[-1]}"
                scale_total = scale_prediction.float().sum() * 0.0
                for name, value in terms.items():
                    components[f"{task}/{resolution}/{name}"] = value
                    if scale_prediction is full_prediction:
                        components[f"{task}/{name}"] = value
                    scale_total = scale_total + weight_values[name] * value
                components[f"{task}/{resolution}/reconstruction"] = scale_total
                task_total = (
                    task_total
                    + self._deep_supervision_weight(scale_index) * scale_total
                )
            components[f"{task}/reconstruction"] = task_total
            per_task[task] = task_total

        if self.task_balancer is None:
            reconstruction_total = torch.stack(tuple(per_task.values())).mean()
        else:
            balanced = self.task_balancer(per_task)
            reconstruction_total = (
                balanced.total if hasattr(balanced, "total") else balanced
            )
            balance_weights = getattr(balanced, "weights", None)
            if isinstance(balance_weights, Mapping):
                for task, value in balance_weights.items():
                    components[f"task_balance/{task}"] = value

        zero = _zero_from_predictions(predictions)
        frequency_values: list[Tensor] = []
        shift_values: list[Tensor] = []
        foreground_values: list[Tensor] = []
        for task, prediction in predictions.items():
            target = self._aligned_full_target(prediction, aligned[task])
            frequency_value = (
                self.frequency_loss(prediction, target)
                if self.auxiliary_weights["frequency"] > 0.0
                else zero
            )
            components[f"{task}/auxiliary/frequency"] = frequency_value
            frequency_values.append(frequency_value)
            shift_value = (
                self.shift_tolerant_loss(prediction, target)
                if self.shift_tolerant_loss is not None
                and self.auxiliary_weights["shift_tolerant"] > 0.0
                else zero
            )
            components[f"{task}/auxiliary/shift_tolerant"] = shift_value
            shift_values.append(shift_value)
            if self.foreground_loss is not None:
                foreground = self.foreground_loss(prediction, target)
                foreground_value = foreground.total
                components[f"{task}/auxiliary/foreground_mse"] = foreground.mse
                components[f"{task}/auxiliary/foreground_dice"] = foreground.dice
                components[f"{task}/auxiliary/foreground_intensity"] = (
                    foreground.intensity
                )
                components[f"{task}/diagnostic/foreground_fraction"] = (
                    foreground.foreground_fraction
                )
            else:
                foreground_value = zero
            components[f"{task}/auxiliary/fluorescence_foreground"] = (
                foreground_value
            )
            foreground_values.append(foreground_value)

        frequency_value = torch.stack(frequency_values).mean()
        shift_value = torch.stack(shift_values).mean()
        foreground_value = torch.stack(foreground_values).mean()
        correlation_value = (
            task_correlation_loss(predictions, aligned)
            if self.auxiliary_weights["correlation"] > 0.0
            else zero
        )
        prototype_enabled = (
            self.auxiliary_weights["prototype_activation"] > 0.0
            or self.auxiliary_weights["prototype_diversity"] > 0.0
        )
        if prototype_enabled and prototype_features and prototype_banks:
            prototype_terms = self.prototype_loss(prototype_features, prototype_banks)
            prototype_activation_value = prototype_terms.activation
            prototype_diversity_value = prototype_terms.diversity
        else:
            prototype_activation_value = zero
            prototype_diversity_value = zero
        prototype_usage = self._prototype_usage(output)
        prototype_usage_entropy_value = (
            prototype_usage_entropy_loss(prototype_usage).to(device=zero.device)
            + zero.float()
            if self.auxiliary_weights["prototype_usage_entropy"] > 0.0
            and prototype_usage
            else zero
        )
        auxiliary_values = {
            "frequency": frequency_value,
            "correlation": correlation_value,
            "prototype_activation": prototype_activation_value,
            "prototype_diversity": prototype_diversity_value,
            "prototype_usage_entropy": prototype_usage_entropy_value,
            "shift_tolerant": shift_value,
            "fluorescence_foreground": foreground_value,
        }
        auxiliary_total = zero
        for name, value in auxiliary_values.items():
            components[name] = value
            auxiliary_total = auxiliary_total + self.auxiliary_weights[name] * value
        proxy_weight = self.current_competition_proxy_weight()
        proxy_values: list[Tensor] = []
        if self.competition_proxy_loss is not None and proxy_weight > 0.0:
            for task, prediction in predictions.items():
                target = self._aligned_full_target(prediction, aligned[task])
                proxy = self.competition_proxy_loss(prediction, target)
                proxy_values.append(proxy.total)
                components[f"{task}/competition_proxy/ssim_loss"] = proxy.ssim_loss
                components[f"{task}/competition_proxy/psnr_loss"] = proxy.psnr_loss
                components[f"{task}/competition_proxy/mean_ssim"] = proxy.mean_ssim
                components[f"{task}/competition_proxy/mean_psnr"] = proxy.mean_psnr
        competition_proxy_value = (
            torch.stack(proxy_values).mean() if proxy_values else zero
        )
        competition_proxy_total = proxy_weight * competition_proxy_value
        total = reconstruction_total + auxiliary_total + competition_proxy_total
        components["reconstruction_total"] = reconstruction_total
        components["auxiliary_total"] = auxiliary_total
        components["competition_proxy"] = competition_proxy_value
        components["competition_proxy_total"] = competition_proxy_total
        components["schedule/competition_proxy_weight"] = total.new_tensor(proxy_weight)
        components["schedule/progress"] = total.new_tensor(self.progress)
        for name, value in weight_values.items():
            components[f"schedule/{name}"] = total.new_tensor(value)
        for index, value in enumerate(self.current_deep_supervision_weights()):
            components[f"schedule/deep_supervision_{index}"] = total.new_tensor(value)
        components["total"] = total
        return LossOutput(total=total, components=components, per_task=per_task)
