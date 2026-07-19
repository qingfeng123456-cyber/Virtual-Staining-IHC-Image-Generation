"""Reference and adaptive multi-task loss-balancer tests."""

from __future__ import annotations

import pytest
import torch

from virtual_staining.engine.multitask_optimizer import (
    EqualTaskBalancer,
    FAMOTaskBalancer,
    UncertaintyTaskBalancer,
    build_task_balancer,
)


def _assert_probability_weights(weights: dict[str, torch.Tensor]) -> None:
    values = torch.stack(tuple(weights.values()))
    assert torch.isfinite(values).all()
    assert torch.all(values >= 0.0)
    assert float(values.detach().sum()) == pytest.approx(1.0, abs=1e-6)


def test_equal_balancer_is_the_arithmetic_mean_reference() -> None:
    first = torch.tensor(1.0, requires_grad=True)
    second = torch.tensor(3.0, requires_grad=True)
    output = EqualTaskBalancer()({"CD68": first, "Vimentin": second})

    assert float(output.total.detach()) == pytest.approx(2.0)
    _assert_probability_weights(output.weights)
    output.total.backward()
    assert float(first.grad) == pytest.approx(0.5)
    assert float(second.grad) == pytest.approx(0.5)


def test_uncertainty_balancer_learns_finite_normalized_weights() -> None:
    balancer = UncertaintyTaskBalancer(("CD68", "Vimentin"))
    with torch.no_grad():
        balancer.log_variances.copy_(torch.tensor([-0.5, 0.5]))
    first = torch.tensor(1.0, requires_grad=True)
    second = torch.tensor(2.0, requires_grad=True)
    output = balancer({"CD68": first, "Vimentin": second})

    assert torch.isfinite(output.total)
    _assert_probability_weights(output.weights)
    assert output.weights["CD68"] > output.weights["Vimentin"]

    output.total.backward()
    assert first.grad is not None and torch.isfinite(first.grad)
    assert second.grad is not None and torch.isfinite(second.grad)
    assert balancer.log_variances.grad is not None
    assert torch.isfinite(balancer.log_variances.grad).all()


def test_famo_updates_training_only_state_and_restores_it() -> None:
    balancer = FAMOTaskBalancer(("CD68", "Vimentin"), adaptation_rate=0.5)
    first = balancer(
        {
            "CD68": torch.tensor(4.0, requires_grad=True),
            "Vimentin": torch.tensor(4.0, requires_grad=True),
        }
    )
    _assert_probability_weights(first.weights)
    assert float(first.weights["CD68"]) == pytest.approx(0.5)

    cd68 = torch.tensor(2.0, requires_grad=True)
    vimentin = torch.tensor(3.9, requires_grad=True)
    second = balancer({"CD68": cd68, "Vimentin": vimentin})
    _assert_probability_weights(second.weights)
    assert second.weights["Vimentin"] > second.weights["CD68"]
    second.total.backward()
    assert cd68.grad is not None and torch.isfinite(cd68.grad)
    assert vimentin.grad is not None and torch.isfinite(vimentin.grad)

    restored = FAMOTaskBalancer(("CD68", "Vimentin"), adaptation_rate=0.5)
    restored.load_state_dict(balancer.state_dict())
    assert torch.equal(restored.logits, balancer.logits)
    assert torch.equal(restored.previous_losses, balancer.previous_losses)


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    [
        ("equal", EqualTaskBalancer),
        ("famo", FAMOTaskBalancer),
        ("uncertainty", UncertaintyTaskBalancer),
    ],
)
def test_task_balancer_feature_flag(mode: str, expected_type: type[torch.nn.Module]) -> None:
    assert isinstance(build_task_balancer(mode, ("CD68", "Vimentin")), expected_type)
