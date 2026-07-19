"""Feature-flagged task-loss balancing for multi-marker training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class TaskBalanceOutput:
    total: Tensor
    weights: dict[str, Tensor]


class EqualTaskBalancer(nn.Module):
    """Stable arithmetic mean used as the mandatory reference mode."""

    def forward(self, losses: Mapping[str, Tensor]) -> TaskBalanceOutput:
        if not losses:
            raise ValueError("Task balancing requires at least one loss")
        names = tuple(losses)
        total = torch.stack(tuple(losses.values())).mean()
        weight = total.new_tensor(1.0 / len(names))
        return TaskBalanceOutput(total, {name: weight for name in names})


class UncertaintyTaskBalancer(nn.Module):
    """Learn homoscedastic task weights without using validation or test data."""

    def __init__(self, task_names: Sequence[str]) -> None:
        super().__init__()
        names = tuple(str(name) for name in task_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("task_names must be non-empty and unique")
        self.task_names = names
        self.log_variances = nn.Parameter(torch.zeros(len(names)))

    def forward(self, losses: Mapping[str, Tensor]) -> TaskBalanceOutput:
        missing = [name for name in self.task_names if name not in losses]
        if missing:
            raise KeyError(f"Missing task losses: {', '.join(missing)}")
        precision = torch.exp(-self.log_variances)
        terms = [
            precision[index] * losses[name] + 0.5 * self.log_variances[index]
            for index, name in enumerate(self.task_names)
        ]
        normalized = precision / precision.sum().clamp_min(1e-12)
        return TaskBalanceOutput(
            torch.stack(terms).mean(),
            {name: normalized[index] for index, name in enumerate(self.task_names)},
        )


class FAMOTaskBalancer(nn.Module):
    """Fast adaptive loss reweighting with deterministic, bounded state updates.

    The state uses training losses only.  Detached relative improvement updates
    prevent a second-order graph and keep this optional branch inexpensive.
    """

    def __init__(
        self,
        task_names: Sequence[str],
        *,
        adaptation_rate: float = 0.025,
        maximum_logit: float = 6.0,
    ) -> None:
        super().__init__()
        names = tuple(str(name) for name in task_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("task_names must be non-empty and unique")
        if adaptation_rate <= 0 or maximum_logit <= 0:
            raise ValueError("FAMO rates and bounds must be positive")
        self.task_names = names
        self.adaptation_rate = float(adaptation_rate)
        self.maximum_logit = float(maximum_logit)
        self.register_buffer("logits", torch.zeros(len(names)))
        self.register_buffer("previous_losses", torch.full((len(names),), float("nan")))

    @torch.no_grad()
    def _update(self, current: Tensor) -> None:
        initialized = torch.isfinite(self.previous_losses)
        if bool(initialized.all()):
            improvement = (self.previous_losses - current) / self.previous_losses.clamp_min(1e-8)
            centered = improvement - improvement.mean()
            self.logits.sub_(self.adaptation_rate * centered)
            self.logits.clamp_(-self.maximum_logit, self.maximum_logit)
        self.previous_losses.copy_(current)

    def forward(self, losses: Mapping[str, Tensor]) -> TaskBalanceOutput:
        missing = [name for name in self.task_names if name not in losses]
        if missing:
            raise KeyError(f"Missing task losses: {', '.join(missing)}")
        values = torch.stack([losses[name] for name in self.task_names])
        if not torch.isfinite(values).all():
            raise FloatingPointError("FAMO received a non-finite task loss")
        self._update(values.detach().to(self.logits))
        weights = torch.softmax(self.logits, dim=0).to(values)
        total = torch.sum(weights * values)
        return TaskBalanceOutput(
            total,
            {name: weights[index] for index, name in enumerate(self.task_names)},
        )


def build_task_balancer(mode: str, task_names: Sequence[str]) -> nn.Module:
    normalized = str(mode).casefold()
    if normalized == "equal":
        return EqualTaskBalancer()
    if normalized == "famo":
        return FAMOTaskBalancer(task_names)
    if normalized == "uncertainty":
        return UncertaintyTaskBalancer(task_names)
    raise ValueError(f"Unsupported multitask optimizer: {mode}")
