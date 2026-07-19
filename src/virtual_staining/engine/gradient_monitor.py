"""Side-effect-free multi-task gradient cosine diagnostics.

The monitor deliberately has no dependency on :mod:`trainer` or the project
configuration schema.  A training loop can therefore keep the feature fully
disabled, or explicitly supply ``LossOutput.per_task`` together with the shared
parameters it wants to inspect.  Losses must be passed before ``GradScaler``
scaling; diagnostic gradients are detached and converted to float32.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TaskGradientStats:
    """Serializable statistics for one task's shared-parameter gradient."""

    task: str
    gradient_l2_norm: float | None
    parameter_count: int
    parameter_element_count: int
    parameters_with_gradient: int
    nonzero_elements: int
    nonfinite_elements: int

    @property
    def finite(self) -> bool:
        return self.nonfinite_elements == 0

    @property
    def has_gradient(self) -> bool:
        return self.parameters_with_gradient > 0 and self.nonzero_elements > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "gradient_l2_norm": self.gradient_l2_norm,
            "parameter_count": self.parameter_count,
            "parameter_element_count": self.parameter_element_count,
            "parameters_with_gradient": self.parameters_with_gradient,
            "nonzero_elements": self.nonzero_elements,
            "nonfinite_elements": self.nonfinite_elements,
            "finite": self.finite,
            "has_gradient": self.has_gradient,
        }


@dataclass(frozen=True, slots=True)
class GradientCosinePair:
    """Cosine similarity between two task gradients."""

    task_a: str
    task_b: str
    cosine: float | None
    dot_product: float | None
    norm_a: float | None
    norm_b: float | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_a": self.task_a,
            "task_b": self.task_b,
            "cosine": self.cosine,
            "dot_product": self.dot_product,
            "norm_a": self.norm_a,
            "norm_b": self.norm_b,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class GradientCosineReport:
    """One monitoring observation, containing task and pair diagnostics."""

    step: int
    task_stats: tuple[TaskGradientStats, ...]
    pairs: tuple[GradientCosinePair, ...]
    shared_parameter_count: int
    shared_parameter_element_count: int

    @property
    def summary(self) -> dict[str, int | float | None]:
        defined = [pair.cosine for pair in self.pairs if pair.cosine is not None]
        negative = sum(value < 0.0 for value in defined)
        return {
            "task_count": len(self.task_stats),
            "pair_count": len(self.pairs),
            "defined_pair_count": len(defined),
            "mean_cosine": sum(defined) / len(defined) if defined else None,
            "minimum_cosine": min(defined) if defined else None,
            "maximum_cosine": max(defined) if defined else None,
            "negative_pair_count": negative,
            "negative_pair_fraction": negative / len(defined) if defined else None,
            "zero_gradient_task_count": sum(not item.has_gradient for item in self.task_stats),
            "nonfinite_task_count": sum(not item.finite for item in self.task_stats),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "step": self.step,
            "shared_parameter_count": self.shared_parameter_count,
            "shared_parameter_element_count": self.shared_parameter_element_count,
            "tasks": [item.to_dict() for item in self.task_stats],
            "pairs": [item.to_dict() for item in self.pairs],
            "summary": self.summary,
        }


@dataclass(slots=True)
class _TaskGradient:
    vector: Tensor
    parameters_with_gradient: int
    nonfinite_elements: int


def _trainable_unique_parameters(shared_parameters: Iterable[Tensor]) -> tuple[Tensor, ...]:
    parameters: list[Tensor] = []
    seen: set[int] = set()
    for parameter in shared_parameters:
        if not isinstance(parameter, Tensor):
            raise TypeError("shared_parameters must contain Tensor or Parameter objects")
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        parameters.append(parameter)
    return tuple(parameters)


def _validate_task_losses(task_losses: Mapping[str, Tensor]) -> tuple[tuple[str, Tensor], ...]:
    if not task_losses:
        raise ValueError("Gradient monitoring requires at least one task loss")
    validated: list[tuple[str, Tensor]] = []
    for raw_name, loss in task_losses.items():
        name = str(raw_name)
        if not name:
            raise ValueError("Task names cannot be empty")
        if not isinstance(loss, Tensor):
            raise TypeError(f"Task loss {name!r} must be a Tensor")
        if loss.numel() != 1:
            raise ValueError(f"Task loss {name!r} must be scalar, got shape {tuple(loss.shape)}")
        validated.append((name, loss.reshape(())))
    validated.sort(key=lambda item: item[0])
    return tuple(validated)


def _compute_task_gradients(
    task_losses: Mapping[str, Tensor],
    shared_parameters: Iterable[Tensor],
    *,
    retain_graph: bool,
) -> tuple[dict[str, _TaskGradient], tuple[Tensor, ...]]:
    losses = _validate_task_losses(task_losses)
    parameters = _trainable_unique_parameters(shared_parameters)
    element_count = sum(parameter.numel() for parameter in parameters)
    computed: dict[str, _TaskGradient] = {}

    for index, (task, raw_loss) in enumerate(losses):
        gradients: tuple[Tensor | None, ...]
        if parameters and raw_loss.requires_grad:
            gradients = torch.autograd.grad(
                raw_loss.float(),
                parameters,
                retain_graph=retain_graph or index < len(losses) - 1,
                create_graph=False,
                allow_unused=True,
            )
        else:
            gradients = (None,) * len(parameters)

        flat_parts: list[Tensor] = []
        parameters_with_gradient = 0
        nonfinite_elements = 0
        for parameter, gradient in zip(parameters, gradients, strict=True):
            if gradient is None:
                flat_parts.append(torch.zeros(parameter.numel(), dtype=torch.float32))
                continue
            detached = gradient.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            parameters_with_gradient += 1
            nonfinite_elements += int((~torch.isfinite(detached)).sum().item())
            flat_parts.append(detached)
        vector = (
            torch.cat(flat_parts)
            if flat_parts
            else torch.empty(element_count, dtype=torch.float32)
        )
        computed[task] = _TaskGradient(
            vector=vector,
            parameters_with_gradient=parameters_with_gradient,
            nonfinite_elements=nonfinite_elements,
        )
    return computed, parameters


def compute_task_gradient_vectors(
    task_losses: Mapping[str, Tensor],
    shared_parameters: Iterable[Tensor],
    *,
    retain_graph: bool = True,
) -> dict[str, Tensor]:
    """Return aligned CPU float32 gradient vectors without touching ``.grad``.

    ``task_losses`` should normally be ``LossOutput.per_task`` and must be
    captured before loss scaling.  ``shared_parameters`` must be explicit, for
    example ``model.encoder.parameters()``; selecting the shared module is a
    caller responsibility.  The default retains the graph so the ordinary
    ordinary combined-loss backward call can still run afterwards.
    """

    gradients, _ = _compute_task_gradients(
        task_losses,
        shared_parameters,
        retain_graph=retain_graph,
    )
    return {task: item.vector.clone() for task, item in gradients.items()}


def compute_gradient_cosine_report(
    task_losses: Mapping[str, Tensor],
    shared_parameters: Iterable[Tensor],
    *,
    step: int,
    epsilon: float = 1e-12,
    retain_graph: bool = True,
) -> GradientCosineReport:
    """Compute pairwise shared-gradient cosines for one training step."""

    if step < 0:
        raise ValueError("step cannot be negative")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    gradients, parameters = _compute_task_gradients(
        task_losses,
        shared_parameters,
        retain_graph=retain_graph,
    )
    parameter_count = len(parameters)
    parameter_element_count = sum(parameter.numel() for parameter in parameters)

    stats: list[TaskGradientStats] = []
    finite_vectors: dict[str, Tensor | None] = {}
    norms: dict[str, float | None] = {}
    for task, item in gradients.items():
        finite = item.nonfinite_elements == 0
        vector = item.vector if finite else None
        norm = float(torch.linalg.vector_norm(vector)) if vector is not None else None
        if norm is not None and not math.isfinite(norm):
            norm = None
            finite = False
        finite_vectors[task] = vector if finite else None
        norms[task] = norm
        stats.append(
            TaskGradientStats(
                task=task,
                gradient_l2_norm=norm,
                parameter_count=parameter_count,
                parameter_element_count=parameter_element_count,
                parameters_with_gradient=item.parameters_with_gradient,
                nonzero_elements=int(torch.count_nonzero(item.vector).item()),
                nonfinite_elements=item.nonfinite_elements,
            )
        )

    pairs: list[GradientCosinePair] = []
    names = tuple(gradients)
    for left_index, task_a in enumerate(names):
        for task_b in names[left_index + 1 :]:
            vector_a = finite_vectors[task_a]
            vector_b = finite_vectors[task_b]
            norm_a = norms[task_a]
            norm_b = norms[task_b]
            if vector_a is None or vector_b is None:
                cosine = None
                dot_product = None
                status = "nonfinite_gradient"
            elif norm_a is None or norm_b is None or norm_a <= epsilon or norm_b <= epsilon:
                cosine = None
                dot_product = float(torch.dot(vector_a, vector_b))
                status = "zero_norm"
            else:
                dot_product = float(torch.dot(vector_a, vector_b))
                cosine = max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))
                status = "ok"
            pairs.append(
                GradientCosinePair(
                    task_a=task_a,
                    task_b=task_b,
                    cosine=cosine,
                    dot_product=dot_product,
                    norm_a=norm_a,
                    norm_b=norm_b,
                    status=status,
                )
            )

    return GradientCosineReport(
        step=int(step),
        task_stats=tuple(stats),
        pairs=tuple(pairs),
        shared_parameter_count=parameter_count,
        shared_parameter_element_count=parameter_element_count,
    )


class GradientCosineMonitor:
    """Feature-flag-ready interval monitor with atomic JSON/CSV persistence.

    This class is intentionally not wired into ``Trainer``.  A caller may bind
    an artifact directory and invoke ``maybe_measure`` after obtaining the
    graph-connected per-task losses but before the ordinary backward call.
    """

    schema_version = 1

    def __init__(
        self,
        *,
        enabled: bool = False,
        interval: int = 100,
        include_step_zero: bool = False,
        epsilon: float = 1e-12,
    ) -> None:
        if interval < 1:
            raise ValueError("interval must be at least one")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.enabled = bool(enabled)
        self.interval = int(interval)
        self.include_step_zero = bool(include_step_zero)
        self.epsilon = float(epsilon)
        self.output_dir: Path | None = None
        self.history: list[GradientCosineReport] = []

    def should_measure(self, step: int) -> bool:
        if step < 0:
            raise ValueError("step cannot be negative")
        if not self.enabled or (step == 0 and not self.include_step_zero):
            return False
        return step % self.interval == 0

    def bind_output_dir(self, output_dir: str | Path) -> None:
        """Bind an artifact directory and restore compatible JSON history."""

        self.output_dir = Path(output_dir).resolve()
        path = self.output_dir / "gradient_cosine.json"
        if self.history or not path.is_file():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != self.schema_version:
            raise ValueError(f"Unsupported gradient monitor schema in {path}")
        reports = payload.get("reports", [])
        if not isinstance(reports, list):
            raise ValueError(f"Gradient monitor reports must be a list in {path}")
        self.history = [_report_from_dict(item) for item in reports]

    def measure(
        self,
        task_losses: Mapping[str, Tensor],
        shared_parameters: Iterable[Tensor],
        *,
        step: int,
        retain_graph: bool = True,
    ) -> GradientCosineReport:
        """Compute a report regardless of the enabled/interval feature flag."""

        return compute_gradient_cosine_report(
            task_losses,
            shared_parameters,
            step=step,
            epsilon=self.epsilon,
            retain_graph=retain_graph,
        )

    def maybe_measure(
        self,
        task_losses: Mapping[str, Tensor],
        shared_parameters: Iterable[Tensor],
        *,
        step: int,
        retain_graph: bool = True,
    ) -> GradientCosineReport | None:
        """Measure, record, and persist only when the explicit flag is due."""

        if not self.should_measure(step):
            return None
        report = self.measure(
            task_losses,
            shared_parameters,
            step=step,
            retain_graph=retain_graph,
        )
        self.record(report)
        return report

    def record(self, report: GradientCosineReport) -> None:
        self.history = [item for item in self.history if item.step != report.step]
        self.history.append(report)
        self.history.sort(key=lambda item: item.step)
        self.persist()

    def persist(self) -> None:
        """Atomically persist all observations as JSON and normalized CSV."""

        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "gradient_cosine.json"
        temporary_json = json_path.with_name(f"{json_path.name}.tmp")
        payload = {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "interval": self.interval,
            "include_step_zero": self.include_step_zero,
            "reports": [report.to_dict() for report in self.history],
        }
        temporary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary_json.replace(json_path)

        csv_path = self.output_dir / "gradient_cosine.csv"
        temporary_csv = csv_path.with_name(f"{csv_path.name}.tmp")
        fieldnames = (
            "step",
            "record_type",
            "task",
            "task_a",
            "task_b",
            "gradient_l2_norm",
            "parameters_with_gradient",
            "nonzero_elements",
            "nonfinite_elements",
            "cosine",
            "dot_product",
            "status",
        )
        with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for report in self.history:
                for task in report.task_stats:
                    writer.writerow(
                        {
                            "step": report.step,
                            "record_type": "task",
                            "task": task.task,
                            "gradient_l2_norm": task.gradient_l2_norm,
                            "parameters_with_gradient": task.parameters_with_gradient,
                            "nonzero_elements": task.nonzero_elements,
                            "nonfinite_elements": task.nonfinite_elements,
                            "status": "ok" if task.finite else "nonfinite_gradient",
                        }
                    )
                for pair in report.pairs:
                    writer.writerow(
                        {
                            "step": report.step,
                            "record_type": "pair",
                            "task_a": pair.task_a,
                            "task_b": pair.task_b,
                            "cosine": pair.cosine,
                            "dot_product": pair.dot_product,
                            "status": pair.status,
                        }
                    )
        temporary_csv.replace(csv_path)


def _report_from_dict(payload: Mapping[str, Any]) -> GradientCosineReport:
    tasks = tuple(
        TaskGradientStats(
            task=str(item["task"]),
            gradient_l2_norm=(
                None
                if item.get("gradient_l2_norm") is None
                else float(item["gradient_l2_norm"])
            ),
            parameter_count=int(item["parameter_count"]),
            parameter_element_count=int(item["parameter_element_count"]),
            parameters_with_gradient=int(item["parameters_with_gradient"]),
            nonzero_elements=int(item["nonzero_elements"]),
            nonfinite_elements=int(item["nonfinite_elements"]),
        )
        for item in payload.get("tasks", [])
    )
    pairs = tuple(
        GradientCosinePair(
            task_a=str(item["task_a"]),
            task_b=str(item["task_b"]),
            cosine=None if item.get("cosine") is None else float(item["cosine"]),
            dot_product=(
                None if item.get("dot_product") is None else float(item["dot_product"])
            ),
            norm_a=None if item.get("norm_a") is None else float(item["norm_a"]),
            norm_b=None if item.get("norm_b") is None else float(item["norm_b"]),
            status=str(item["status"]),
        )
        for item in payload.get("pairs", [])
    )
    return GradientCosineReport(
        step=int(payload["step"]),
        task_stats=tasks,
        pairs=pairs,
        shared_parameter_count=int(payload["shared_parameter_count"]),
        shared_parameter_element_count=int(payload["shared_parameter_element_count"]),
    )
