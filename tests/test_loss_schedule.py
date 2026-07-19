"""Continuity, finite-gradient, and resume checks for the two-phase loss."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from torch import nn

from virtual_staining.cli import _build_loss
from virtual_staining.engine.checkpoint import load_checkpoint, save_checkpoint
from virtual_staining.engine.multitask_optimizer import FAMOTaskBalancer
from virtual_staining.losses import (
    ScheduledCompositeLoss,
    ScheduledLossWeights,
    TwoPhaseLossSchedule,
    epoch_progress,
    prototype_usage_entropy_loss,
)
from virtual_staining.models import RestorationOutput


def _schedule() -> TwoPhaseLossSchedule:
    return TwoPhaseLossSchedule(
        phase_a=ScheduledLossWeights(
            mse=0.1,
            charbonnier=0.4,
            ssim=0.4,
            ms_ssim=0.0,
            pyramid=0.1,
            gradient=0.0,
            statistics=0.0,
        ),
        phase_b=ScheduledLossWeights(
            mse=0.4,
            charbonnier=0.1,
            ssim=0.4,
            ms_ssim=0.0,
            pyramid=0.1,
            gradient=0.0,
            statistics=0.0,
        ),
        phase_a_ratio=0.6,
        interpolation="cosine",
    )


def test_two_phase_schedule_is_continuous_at_transition_and_reaches_endpoints() -> None:
    schedule = _schedule()
    before = asdict(schedule.weights_at(0.6 - 1e-6))
    boundary = asdict(schedule.weights_at(0.6))
    after = asdict(schedule.weights_at(0.6 + 1e-6))

    assert boundary == asdict(schedule.phase_a)
    endpoint = asdict(schedule.weights_at(1.0))
    for name, value in asdict(schedule.phase_b).items():
        assert endpoint[name] == pytest.approx(value, abs=1e-12)
    for name in boundary:
        assert abs(before[name] - boundary[name]) < 1e-9
        assert abs(after[name] - boundary[name]) < 1e-9

    midpoint = schedule.weights_at(0.8)
    assert schedule.phase_a.mse < midpoint.mse < schedule.phase_b.mse
    assert schedule.phase_b.charbonnier < midpoint.charbonnier < schedule.phase_a.charbonnier


def test_schedule_and_loss_progress_state_round_trip() -> None:
    schedule = _schedule()
    restored_schedule = TwoPhaseLossSchedule.from_state_dict(schedule.state_dict())
    assert restored_schedule.state_dict() == schedule.state_dict()
    assert restored_schedule.weights_at(0.83) == schedule.weights_at(0.83)

    criterion = ScheduledCompositeLoss(schedule, pyramid_levels=2)
    criterion.set_progress(0.83)
    state = criterion.state_dict()
    restored_criterion = ScheduledCompositeLoss(restored_schedule, pyramid_levels=2)
    restored_criterion.load_state_dict(state)

    assert restored_criterion.progress == pytest.approx(0.83)
    assert restored_criterion.current_weights() == criterion.current_weights()


def test_scheduled_composite_loss_is_finite_and_backward() -> None:
    prediction = torch.rand(1, 1, 16, 16, requires_grad=True)
    target = torch.rand_like(prediction)
    criterion = ScheduledCompositeLoss(_schedule(), pyramid_levels=3)
    result = criterion(prediction, target, progress=0.75)

    assert result.total.dtype == torch.float32
    assert torch.isfinite(result.total)
    assert result.components["schedule/progress"] == pytest.approx(
        torch.tensor(0.75), abs=1e-6
    )
    assert "target/mse" in result.components
    assert "target/pyramid" in result.components

    result.total.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0


def test_epoch_progress_is_resume_stable() -> None:
    assert epoch_progress(0, 5) == 0.0
    assert epoch_progress(2, 5) == 0.5
    assert epoch_progress(4, 5) == 1.0
    assert epoch_progress(20, 5) == 1.0


def _mse_only_schedule() -> TwoPhaseLossSchedule:
    weights = ScheduledLossWeights(
        mse=1.0,
        charbonnier=0.0,
        ssim=0.0,
        ms_ssim=0.0,
        pyramid=0.0,
        gradient=0.0,
        statistics=0.0,
    )
    return TwoPhaseLossSchedule(weights, weights)


def test_deep_supervision_changes_task_reconstruction_and_backpropagates() -> None:
    full_prediction = torch.zeros(1, 1, 16, 16, requires_grad=True)
    auxiliary_prediction = torch.ones(1, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(full_prediction)
    criterion = ScheduledCompositeLoss(
        _mse_only_schedule(),
        pyramid_levels=2,
        deep_supervision_weights=(1.0, 0.5),
    )

    full_only = criterion(full_prediction, target)
    with_deep_supervision = criterion(
        RestorationOutput(
            predictions={"CD68": full_prediction},
            deep_supervision={"CD68": {8: auxiliary_prediction}},
        ),
        {"CD68": target},
    )

    assert full_only.total == 0.0
    assert with_deep_supervision.total > full_only.total
    assert "CD68/8x8/reconstruction" in with_deep_supervision.components
    with_deep_supervision.total.backward()
    assert auxiliary_prediction.grad is not None
    assert torch.isfinite(auxiliary_prediction.grad).all()
    assert torch.count_nonzero(auxiliary_prediction.grad) > 0


def test_single_task_fixed_auxiliaries_are_finite_and_backward() -> None:
    prediction = torch.rand(1, 1, 16, 16, requires_grad=True)
    target = torch.rand_like(prediction)
    features = torch.randn(1, 12, 4, requires_grad=True)
    shared_bank = torch.randn(3, 4, requires_grad=True)
    task_bank = torch.randn(2, 4, requires_grad=True)
    usage = torch.softmax(torch.randn(3, requires_grad=True), dim=0)
    output = RestorationOutput(
        predictions={"CD68": prediction},
        prototype_features={"CD68": features},
        prototype_banks={"shared": shared_bank, "CD68": task_bank},
        prototype_usage={"CD68": {"16/shared": usage}},
    )
    criterion = ScheduledCompositeLoss(
        _schedule(),
        pyramid_levels=2,
        frequency=0.05,
        correlation=0.02,
        prototype_activation=0.001,
        prototype_diversity=0.001,
        prototype_usage_entropy=0.001,
        shift_tolerant_enabled=True,
        shift_tolerant_weight=0.1,
        shift_tolerant_max_shift=1,
        shift_tolerant_mode="softmin",
    )
    result = criterion(output, {"CD68": target}, progress=0.4)

    assert torch.isfinite(result.total)
    assert set(result.per_task) == {"CD68"}
    for name in (
        "frequency",
        "correlation",
        "prototype_activation",
        "prototype_diversity",
        "prototype_usage_entropy",
        "shift_tolerant",
        "auxiliary_total",
    ):
        assert name in result.components
        assert torch.isfinite(result.components[name])

    result.total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert shared_bank.grad is not None and torch.isfinite(shared_bank.grad).all()


class _RecordingFAMO(FAMOTaskBalancer):
    def __init__(self, task_names: tuple[str, ...]) -> None:
        super().__init__(task_names)
        self.received_keys: tuple[str, ...] = ()

    def forward(self, losses: dict[str, torch.Tensor]):  # type: ignore[override]
        self.received_keys = tuple(losses)
        return super().forward(losses)


def test_famo_balances_only_multitask_reconstruction_before_auxiliaries() -> None:
    tasks = ("CD68", "Vimentin")
    balancer = _RecordingFAMO(tasks)
    predictions = {
        task: torch.rand(1, 1, 16, 16, requires_grad=True) for task in tasks
    }
    targets = {task: torch.rand_like(prediction) for task, prediction in predictions.items()}
    output = RestorationOutput(
        predictions=predictions,
        prototype_usage={
            task: {"16/shared": torch.tensor([0.7, 0.2, 0.1])} for task in tasks
        },
    )
    criterion = ScheduledCompositeLoss(
        _schedule(),
        pyramid_levels=2,
        task_balancer=balancer,
        frequency=0.1,
        correlation=0.1,
        prototype_usage_entropy=0.01,
    )
    result = criterion(output, targets)

    assert balancer.received_keys == tasks
    assert set(result.per_task) == set(tasks)
    assert balancer.previous_losses.shape == (2,)
    assert torch.allclose(
        result.total,
        result.components["reconstruction_total"]
        + result.components["auxiliary_total"],
    )
    assert "task_balance/CD68" in result.components
    assert "task_balance/Vimentin" in result.components
    result.total.backward()
    assert all(prediction.grad is not None for prediction in predictions.values())


def test_hierarchical_prototype_banks_and_zero_usage_are_finite() -> None:
    prediction = torch.rand(1, 1, 16, 16, requires_grad=True)
    features = torch.randn(1, 8, 4, requires_grad=True)
    output = RestorationOutput(
        predictions={"CD68": prediction},
        prototype_features={"CD68": features},
        prototype_banks={
            "8/shared": torch.randn(2, 4, requires_grad=True),
            "8/marker/CD68": torch.randn(2, 4, requires_grad=True),
            "16/shared": torch.randn(3, 4, requires_grad=True),
            "16/marker/CD68": torch.randn(2, 4, requires_grad=True),
        },
        prototype_usage={"CD68": {"16/shared": torch.zeros(3, requires_grad=True)}},
    )
    criterion = ScheduledCompositeLoss(
        _mse_only_schedule(),
        pyramid_levels=2,
        prototype_activation=0.01,
        prototype_diversity=0.01,
        prototype_usage_entropy=0.01,
    )
    result = criterion(output, {"CD68": torch.rand_like(prediction)})

    assert torch.isfinite(result.total)
    assert torch.isfinite(result.components["prototype_usage_entropy"])
    assert result.components["prototype_usage_entropy"] == 0.0
    result.total.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_prototype_usage_entropy_is_stable_for_sparse_and_uniform_usage() -> None:
    sparse = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
    uniform = torch.full((3,), 1.0 / 3.0, requires_grad=True)
    loss = prototype_usage_entropy_loss(
        {"CD68": {"16/shared": sparse, "16/marker": uniform}}
    )

    assert torch.isfinite(loss)
    assert -torch.log(torch.tensor(3.0)) < loss < 0.0
    loss.backward()
    assert sparse.grad is not None and torch.isfinite(sparse.grad).all()
    assert uniform.grad is not None and torch.isfinite(uniform.grad).all()


def test_loss_builder_wires_fixed_auxiliary_and_shift_flags() -> None:
    criterion = _build_loss(
        {
            "data": {"targets": ["CD68", "Vimentin"]},
            "loss": {
                "schedule": "two_phase",
                "phase_a": asdict(_schedule().phase_a),
                "phase_b": asdict(_schedule().phase_b),
                "phase_a_ratio": 0.6,
                "deep_supervision_weights": [1.0, 0.25],
                "frequency": 0.11,
                "correlation": 0.12,
                "prototype_activation": 0.9,
                "prototype_diversity": 0.8,
                "prototype": {
                    "activation": 0.013,
                    "diversity": 0.014,
                    "usage_entropy": 0.015,
                },
                "shift_tolerant": {
                    "enabled": True,
                    "max_shift": 2,
                    "mode": "softmin",
                    "temperature": 0.2,
                },
                "data_range": 1.0,
            },
            "multitask": {"optimizer": "famo"},
        }
    )

    assert isinstance(criterion, ScheduledCompositeLoss)
    assert isinstance(criterion.task_balancer, FAMOTaskBalancer)
    assert criterion.deep_supervision_weights == (1.0, 0.25)
    assert criterion.auxiliary_weights == {
        "frequency": 0.11,
        "correlation": 0.12,
        "prototype_activation": 0.013,
        "prototype_diversity": 0.014,
        "prototype_usage_entropy": 0.015,
        "shift_tolerant": 1.0,
    }
    assert criterion.shift_tolerant_loss is not None
    assert criterion.shift_tolerant_loss.max_shift == 2
    assert criterion.shift_tolerant_loss.mode == "softmin"
    assert criterion.shift_tolerant_loss.temperature == pytest.approx(0.2)

    legacy = _build_loss(
        {
            "data": {"targets": ["CD68"]},
            "loss": {
                "schedule": "two_phase",
                "phase_a": asdict(_schedule().phase_a),
                "phase_b": asdict(_schedule().phase_b),
                "prototype_activation": 0.021,
                "prototype_diversity": 0.022,
            },
            "multitask": {"optimizer": "equal"},
        }
    )
    assert isinstance(legacy, ScheduledCompositeLoss)
    assert legacy.auxiliary_weights["prototype_activation"] == pytest.approx(0.021)
    assert legacy.auxiliary_weights["prototype_diversity"] == pytest.approx(0.022)
    assert legacy.auxiliary_weights["prototype_usage_entropy"] == 0.0
    assert legacy.auxiliary_weights["shift_tolerant"] == 0.0
    assert legacy.shift_tolerant_loss is None


def test_scheduled_loss_famo_and_progress_resume_through_checkpoint(tmp_path) -> None:
    tasks = ("CD68", "Vimentin")
    criterion = ScheduledCompositeLoss(
        _schedule(),
        pyramid_levels=2,
        task_balancer=FAMOTaskBalancer(tasks, adaptation_rate=0.5),
        frequency=0.05,
    )
    predictions = {task: torch.rand(1, 1, 8, 8) for task in tasks}
    targets = {task: torch.rand_like(prediction) for task, prediction in predictions.items()}
    criterion(RestorationOutput(predictions=predictions), targets, progress=0.82)
    criterion(RestorationOutput(predictions=predictions), targets)
    model = nn.Conv2d(1, 1, kernel_size=1)
    checkpoint = save_checkpoint(
        tmp_path / "loss_resume.ckpt",
        model,
        loss_fn=criterion,
        epoch=2,
        global_step=7,
        git_commit="unit-test",
    )

    restored = ScheduledCompositeLoss(
        _schedule(),
        pyramid_levels=2,
        task_balancer=FAMOTaskBalancer(tasks, adaptation_rate=0.5),
        frequency=0.05,
    )
    load_checkpoint(checkpoint, model, loss_fn=restored)

    assert restored.progress == pytest.approx(0.82)
    assert isinstance(restored.task_balancer, FAMOTaskBalancer)
    assert torch.equal(restored.task_balancer.logits, criterion.task_balancer.logits)
    assert torch.equal(
        restored.task_balancer.previous_losses,
        criterion.task_balancer.previous_losses,
    )
