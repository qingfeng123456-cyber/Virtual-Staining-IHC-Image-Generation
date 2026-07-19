"""Multi-scale, multi-task restoration objective and correlation constraint."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from virtual_staining.models.registry import RestorationOutput

from .charbonnier import CharbonnierLoss, float32_context, structure_weight_map
from .frequency import FrequencyAmplitudeLoss
from .gradient import GradientLoss
from .prototype import PrototypeRegularization
from .ssim import MSSSIMLoss, SSIMLoss


def _normalize_task_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _zero_from_predictions(predictions: Mapping[str, Tensor]) -> Tensor:
    if not predictions:
        raise ValueError("At least one prediction is required")
    return next(iter(predictions.values())).sum() * 0.0


def task_correlation_loss(
    predictions: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    pool_size: int = 16,
) -> Tensor:
    """Match per-sample inter-marker spatial correlation matrices."""

    common_tasks = [task for task in predictions if task in targets]
    if len(common_tasks) < 2:
        return _zero_from_predictions(predictions)
    if pool_size < 1:
        raise ValueError("Correlation pool size must be positive")

    prediction_vectors: list[Tensor] = []
    target_vectors: list[Tensor] = []
    reference = predictions[common_tasks[0]]
    with float32_context(reference):
        for task in common_tasks:
            prediction = predictions[task].float().mean(dim=1, keepdim=True)
            target = targets[task].float().mean(dim=1, keepdim=True)
            if target.shape[-2:] != prediction.shape[-2:]:
                target = F.interpolate(target, size=prediction.shape[-2:], mode="area")
            pooled_size = (
                min(pool_size, prediction.shape[-2]),
                min(pool_size, prediction.shape[-1]),
            )
            prediction = F.adaptive_avg_pool2d(prediction, pooled_size).flatten(1)
            target = F.adaptive_avg_pool2d(target, pooled_size).flatten(1)
            prediction_vectors.append(prediction)
            target_vectors.append(target)

        prediction_matrix = torch.stack(prediction_vectors, dim=1)
        target_matrix = torch.stack(target_vectors, dim=1)
        prediction_matrix = prediction_matrix - prediction_matrix.mean(dim=-1, keepdim=True)
        target_matrix = target_matrix - target_matrix.mean(dim=-1, keepdim=True)
        prediction_matrix = F.normalize(prediction_matrix, dim=-1, eps=1e-6)
        target_matrix = F.normalize(target_matrix, dim=-1, eps=1e-6)
        prediction_correlation = torch.matmul(
            prediction_matrix,
            prediction_matrix.transpose(1, 2),
        )
        target_correlation = torch.matmul(target_matrix, target_matrix.transpose(1, 2))
        return (prediction_correlation - target_correlation).abs().mean()


@dataclass(slots=True)
class LossWeights:
    """Weights for reconstruction and auxiliary objective terms."""

    charbonnier: float = 0.40
    ssim: float = 0.35
    ms_ssim: float = 0.10
    gradient: float = 0.10
    frequency: float = 0.05
    correlation: float = 0.02
    prototype_activation: float = 0.001
    prototype_diversity: float = 0.001

    def validate(self) -> None:
        values = asdict(self)
        negative = [name for name, value in values.items() if value < 0.0]
        if negative:
            raise ValueError(f"Loss weights cannot be negative: {', '.join(negative)}")
        reconstruction_sum = sum(
            values[name] for name in ("charbonnier", "ssim", "ms_ssim", "gradient", "frequency")
        )
        if reconstruction_sum <= 0.0:
            raise ValueError("At least one reconstruction loss must have positive weight")


@dataclass(slots=True)
class LossOutput:
    """Composite scalar together with graph-connected diagnostic components."""

    total: Tensor
    components: dict[str, Tensor]
    per_task: dict[str, Tensor]

    def __iter__(self) -> Iterator[Any]:
        yield self.total
        yield self.components

    def item(self) -> float:
        return self.total.item()


def _extract_predictions(
    output: RestorationOutput | Mapping[str, Any] | Tensor,
) -> tuple[
    dict[str, Tensor],
    dict[str, dict[int, Tensor]],
    Mapping[str, Tensor],
    Mapping[str, Tensor],
]:
    if isinstance(output, Tensor):
        return {"target": output}, {}, {}, {}
    if isinstance(output, RestorationOutput):
        return (
            output.predictions,
            output.deep_supervision,
            output.prototype_features,
            output.prototype_banks,
        )
    if not isinstance(output, Mapping):
        raise TypeError(f"Unsupported loss output type: {type(output).__name__}")
    if "predictions" in output:
        predictions_value = output["predictions"]
        if not isinstance(predictions_value, Mapping):
            raise TypeError("The 'predictions' field must be a mapping")
        predictions = {str(name): value for name, value in predictions_value.items()}
        deep = output.get("deep_supervision", {})
        features = output.get("prototype_features", {})
        banks = output.get("prototype_banks", {})
        return predictions, dict(deep), features, banks
    predictions = {str(name): value for name, value in output.items() if isinstance(value, Tensor)}
    return predictions, {}, {}, {}


def _align_targets(
    targets: Mapping[str, Tensor] | Tensor,
    prediction_tasks: Sequence[str],
) -> dict[str, Tensor]:
    if isinstance(targets, Tensor):
        if len(prediction_tasks) != 1:
            raise ValueError("A tensor target is only valid for a single prediction task")
        return {prediction_tasks[0]: targets}
    if len(prediction_tasks) == 1 and len(targets) == 1:
        return {prediction_tasks[0]: next(iter(targets.values()))}
    normalized_targets = {_normalize_task_name(str(name)): value for name, value in targets.items()}
    aligned: dict[str, Tensor] = {}
    for task in prediction_tasks:
        normalized = _normalize_task_name(task)
        if normalized not in normalized_targets:
            available = ", ".join(str(name) for name in targets)
            raise KeyError(f"Missing target for {task!r}; available: {available}")
        aligned[task] = normalized_targets[normalized]
    return aligned


class CompositeRestorationLoss(nn.Module):
    """Competition-aligned objective for one or more marker predictions."""

    def __init__(
        self,
        charbonnier: float = 0.40,
        ssim: float = 0.35,
        ms_ssim: float = 0.10,
        gradient: float = 0.10,
        frequency: float = 0.05,
        structure_weight_alpha: float = 1.0,
        correlation: float = 0.02,
        prototype_activation: float = 0.001,
        prototype_diversity: float = 0.001,
        deep_supervision_weights: Sequence[float] = (1.0, 0.5, 0.25),
        task_weights: Mapping[str, float] | None = None,
        auto_balance: bool = False,
        auto_balance_decay: float = 0.99,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = LossWeights(
            charbonnier=charbonnier,
            ssim=ssim,
            ms_ssim=ms_ssim,
            gradient=gradient,
            frequency=frequency,
            correlation=correlation,
            prototype_activation=prototype_activation,
            prototype_diversity=prototype_diversity,
        )
        self.weights.validate()
        if structure_weight_alpha < 0.0:
            raise ValueError("structure_weight_alpha cannot be negative")
        if not deep_supervision_weights or any(weight <= 0.0 for weight in deep_supervision_weights):
            raise ValueError("Deep supervision weights must be positive")
        if not 0.0 <= auto_balance_decay < 1.0:
            raise ValueError("auto_balance_decay must be in [0, 1)")
        if task_weights is not None and any(weight <= 0.0 for weight in task_weights.values()):
            raise ValueError("Task weights must be positive")

        self.structure_weight_alpha = float(structure_weight_alpha)
        self.deep_supervision_weights = tuple(float(weight) for weight in deep_supervision_weights)
        self.task_weights = dict(task_weights or {})
        self.auto_balance = bool(auto_balance)
        self.auto_balance_decay = float(auto_balance_decay)
        self._loss_ema: dict[str, float] = {}
        self.charbonnier_loss = CharbonnierLoss()
        self.ssim_loss = SSIMLoss(data_range=data_range)
        self.ms_ssim_loss = MSSSIMLoss(data_range=data_range)
        self.gradient_loss = GradientLoss()
        self.frequency_loss = FrequencyAmplitudeLoss()
        self.prototype_loss = PrototypeRegularization()

    def get_extra_state(self) -> dict[str, dict[str, float]]:
        """Include optional task-loss EMA values in module checkpoints."""

        return {"loss_ema": dict(self._loss_ema)}

    def set_extra_state(self, state: Mapping[str, Mapping[str, float]]) -> None:
        values = state.get("loss_ema", {})
        self._loss_ema = {str(name): float(value) for name, value in values.items()}

    def _task_scale(self, task: str, detached_loss: float, mean_loss: float) -> float:
        configured = self.task_weights.get(task, 1.0)
        if not self.auto_balance:
            return configured
        previous = self._loss_ema.get(task, detached_loss)
        if self.training:
            updated = self.auto_balance_decay * previous + (1.0 - self.auto_balance_decay) * detached_loss
            self._loss_ema[task] = updated
        else:
            updated = previous
        return configured * mean_loss / max(updated, 1e-8)

    def _single_scale_terms(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> dict[str, Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ after resize: "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        zero = prediction.sum() * 0.0
        terms = {name: zero for name in ("charbonnier", "ssim", "ms_ssim", "gradient", "frequency")}
        if self.weights.charbonnier > 0.0:
            structure = structure_weight_map(target, self.structure_weight_alpha)
            terms["charbonnier"] = self.charbonnier_loss(prediction, target, structure)
        if self.weights.ssim > 0.0:
            terms["ssim"] = self.ssim_loss(prediction, target)
        if self.weights.ms_ssim > 0.0:
            terms["ms_ssim"] = self.ms_ssim_loss(prediction, target)
        if self.weights.gradient > 0.0:
            terms["gradient"] = self.gradient_loss(prediction, target)
        if self.weights.frequency > 0.0:
            terms["frequency"] = self.frequency_loss(prediction, target)
        return terms

    def forward(
        self,
        output: RestorationOutput | Mapping[str, Any] | Tensor,
        targets: Mapping[str, Tensor] | Tensor,
    ) -> LossOutput:
        predictions, deep_supervision, prototype_features, prototype_banks = _extract_predictions(
            output
        )
        if not predictions or any(not isinstance(value, Tensor) for value in predictions.values()):
            raise ValueError("Predictions must be a non-empty mapping of tensors")
        aligned_targets = _align_targets(targets, tuple(predictions))
        components: dict[str, Tensor] = {}
        per_task: dict[str, Tensor] = {}

        reconstruction_names = ("charbonnier", "ssim", "ms_ssim", "gradient", "frequency")
        for task, full_prediction in predictions.items():
            scale_predictions = dict(deep_supervision.get(task, {}))
            scale_predictions[full_prediction.shape[-2]] = full_prediction
            ordered = sorted(
                scale_predictions.values(),
                key=lambda tensor: tensor.shape[-2] * tensor.shape[-1],
                reverse=True,
            )
            task_loss = full_prediction.sum() * 0.0
            for scale_index, scale_prediction in enumerate(ordered):
                scale_weight = (
                    self.deep_supervision_weights[scale_index]
                    if scale_index < len(self.deep_supervision_weights)
                    else self.deep_supervision_weights[-1] * 0.5 ** (
                        scale_index - len(self.deep_supervision_weights) + 1
                    )
                )
                target = aligned_targets[task]
                if target.shape[-2:] != scale_prediction.shape[-2:]:
                    target = F.interpolate(target, size=scale_prediction.shape[-2:], mode="area")
                terms = self._single_scale_terms(scale_prediction, target)
                resolution = f"{scale_prediction.shape[-2]}x{scale_prediction.shape[-1]}"
                scale_total = scale_prediction.sum() * 0.0
                for name in reconstruction_names:
                    value = terms[name]
                    components[f"{task}/{resolution}/{name}"] = value
                    scale_total = scale_total + getattr(self.weights, name) * value
                task_loss = task_loss + scale_weight * scale_total
            per_task[task] = task_loss

        detached_losses = {task: float(value.detach()) for task, value in per_task.items()}
        detached_mean = sum(detached_losses.values()) / len(detached_losses)
        scales = {
            task: self._task_scale(task, detached_losses[task], detached_mean)
            for task in per_task
        }
        scale_sum = sum(scales.values())
        total = sum(per_task[task] * scales[task] for task in per_task) / scale_sum

        correlation_value = (
            task_correlation_loss(predictions, aligned_targets)
            if self.weights.correlation > 0.0
            else _zero_from_predictions(predictions)
        )
        components["correlation"] = correlation_value
        total = total + self.weights.correlation * correlation_value

        zero = _zero_from_predictions(predictions)
        prototype_enabled = (
            self.weights.prototype_activation > 0.0 or self.weights.prototype_diversity > 0.0
        )
        if prototype_enabled and prototype_features and prototype_banks:
            prototype_terms = self.prototype_loss(prototype_features, prototype_banks)
            activation_value = prototype_terms.activation
            diversity_value = prototype_terms.diversity
        else:
            activation_value = zero
            diversity_value = zero
        components["prototype_activation"] = activation_value
        components["prototype_diversity"] = diversity_value
        total = total + self.weights.prototype_activation * activation_value
        total = total + self.weights.prototype_diversity * diversity_value
        components["total"] = total
        return LossOutput(total=total, components=components, per_task=per_task)
