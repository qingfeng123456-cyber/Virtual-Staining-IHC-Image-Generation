"""Side-effect-free multi-task gradient cosine monitor tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from virtual_staining.engine.gradient_monitor import (
    GradientCosineMonitor,
    compute_gradient_cosine_report,
    compute_task_gradient_vectors,
)


def _pair_map(report: object) -> dict[tuple[str, str], float | None]:
    return {
        (pair.task_a, pair.task_b): pair.cosine
        for pair in report.pairs  # type: ignore[attr-defined]
    }


def test_gradient_cosines_cover_positive_negative_and_orthogonal() -> None:
    shared = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    losses = {
        "positive_a": shared[0] + 2.0 * shared[1],
        "positive_b": 2.0 * shared[0] + 4.0 * shared[1],
        "negative": -shared[0] - 2.0 * shared[1],
        "orthogonal": 2.0 * shared[0] - shared[1],
    }

    report = compute_gradient_cosine_report(losses, [shared], step=5)
    pairs = _pair_map(report)

    assert pairs[("positive_a", "positive_b")] == pytest.approx(1.0)
    assert pairs[("negative", "positive_a")] == pytest.approx(-1.0)
    assert pairs[("orthogonal", "positive_a")] == pytest.approx(0.0, abs=1e-7)
    assert report.summary["negative_pair_count"] == 2
    assert report.summary["defined_pair_count"] == 6


def test_unused_parameter_and_single_task_are_reported_without_failure() -> None:
    shared = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    unrelated = torch.nn.Parameter(torch.tensor(3.0))
    report = compute_gradient_cosine_report(
        {"unrelated": unrelated.square()},
        [shared],
        step=1,
    )

    assert report.pairs == ()
    assert report.summary["pair_count"] == 0
    assert report.summary["zero_gradient_task_count"] == 1
    assert report.task_stats[0].parameters_with_gradient == 0
    assert report.task_stats[0].gradient_l2_norm == 0.0
    assert report.task_stats[0].has_gradient is False


def test_zero_norm_pair_is_explicitly_undefined() -> None:
    shared = torch.nn.Parameter(torch.tensor(2.0))
    unrelated = torch.nn.Parameter(torch.tensor(3.0))
    report = compute_gradient_cosine_report(
        {"active": shared.square(), "unused": unrelated.square()},
        [shared],
        step=2,
    )

    assert len(report.pairs) == 1
    assert report.pairs[0].cosine is None
    assert report.pairs[0].status == "zero_norm"
    assert report.summary["defined_pair_count"] == 0


def test_monitor_never_populates_or_changes_parameter_gradients() -> None:
    shared = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    shared.grad = torch.tensor([7.0, 11.0])
    original_gradient = shared.grad.clone()
    losses = {
        "first": shared.square().sum(),
        "second": (3.0 * shared).sum(),
    }

    vectors = compute_task_gradient_vectors(losses, [shared])

    assert all(vector.dtype == torch.float32 and vector.device.type == "cpu" for vector in vectors.values())
    assert torch.equal(shared.grad, original_gradient)
    sum(losses.values()).backward()
    assert torch.equal(shared.grad, original_gradient + torch.tensor([5.0, 1.0]))


def test_monitor_deduplicates_parameters_and_ignores_frozen_parameters() -> None:
    shared = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    frozen = torch.nn.Parameter(torch.tensor([4.0]), requires_grad=False)

    report = compute_gradient_cosine_report(
        {"task": shared.sum()},
        [shared, shared, frozen],
        step=1,
    )

    assert report.shared_parameter_count == 1
    assert report.shared_parameter_element_count == 2
    assert report.task_stats[0].parameter_count == 1
    assert report.task_stats[0].parameter_element_count == 2


def test_autocast_losses_produce_unscaled_float32_diagnostic_vectors() -> None:
    shared = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    inputs = torch.ones(2, 2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = (shared @ inputs).square().mean()

    vectors = compute_task_gradient_vectors({"task": loss}, [shared])

    assert vectors["task"].dtype == torch.float32
    assert torch.isfinite(vectors["task"]).all()


def test_feature_flag_interval_and_atomic_json_csv_persistence(tmp_path: Path) -> None:
    shared = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    losses = {"a": shared.sum(), "b": shared.square().sum()}
    monitor = GradientCosineMonitor(enabled=True, interval=3)
    monitor.bind_output_dir(tmp_path / "含 空格")

    assert monitor.maybe_measure(losses, [shared], step=0) is None
    assert monitor.maybe_measure(losses, [shared], step=1) is None
    report = monitor.maybe_measure(losses, [shared], step=3)

    assert report is not None
    json_path = tmp_path / "含 空格" / "gradient_cosine.json"
    csv_path = tmp_path / "含 空格" / "gradient_cosine.csv"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["reports"][0]["step"] == 3
    assert payload["reports"][0]["pairs"][0]["status"] == "ok"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["record_type"] for row in rows] == ["task", "task", "pair"]

    restored = GradientCosineMonitor(enabled=True, interval=3)
    restored.bind_output_dir(tmp_path / "含 空格")
    assert len(restored.history) == 1
    assert restored.history[0].to_dict() == report.to_dict()


def test_disabled_monitor_and_invalid_inputs_are_safe() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    monitor = GradientCosineMonitor(enabled=False, interval=2)
    assert monitor.maybe_measure({"task": parameter.square()}, [parameter], step=2) is None

    with pytest.raises(ValueError, match="at least one task loss"):
        compute_gradient_cosine_report({}, [parameter], step=0)
    with pytest.raises(ValueError, match="must be scalar"):
        compute_gradient_cosine_report({"task": parameter.repeat(2)}, [parameter], step=0)
    with pytest.raises(ValueError, match="step cannot be negative"):
        monitor.should_measure(-1)


def test_nonfinite_gradient_remains_json_serializable(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    report = compute_gradient_cosine_report(
        {"bad": parameter * torch.tensor(float("inf"))},
        [parameter],
        step=4,
    )
    monitor = GradientCosineMonitor(enabled=True, interval=1)
    monitor.bind_output_dir(tmp_path)
    monitor.record(report)

    assert report.task_stats[0].nonfinite_elements == 1
    assert report.task_stats[0].gradient_l2_norm is None
    json.loads((tmp_path / "gradient_cosine.json").read_text(encoding="utf-8"))
