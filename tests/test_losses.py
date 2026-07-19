"""Finite-value, behavior, backward, and AMP checks for restoration losses."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from virtual_staining.losses import (
    CharbonnierLoss,
    CompositeRestorationLoss,
    FrequencyAmplitudeLoss,
    GradientLoss,
    MSSSIMLoss,
    SSIMLoss,
    differentiable_ms_ssim,
    differentiable_ssim,
    prototype_activation_loss,
    prototype_diversity_loss,
    structure_weight_map,
    task_correlation_loss,
)
from virtual_staining.metrics.image_metrics import calculate_ssim
from virtual_staining.models import RestorationOutput


def test_charbonnier_structure_weight_is_finite_and_polarity_neutral() -> None:
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 5:11] = 1.0
    prediction = (target * 0.8).requires_grad_()
    weight = structure_weight_map(target, alpha=1.0)
    inverted_weight = structure_weight_map(1.0 - target, alpha=1.0)

    assert weight.shape == (1, 1, 16, 16)
    assert torch.allclose(weight, inverted_weight, atol=1e-5)
    loss = CharbonnierLoss()(prediction, target, weight)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_ssim_and_ms_ssim_identical_and_noise_behavior() -> None:
    torch.manual_seed(2026)
    target = torch.rand(2, 3, 32, 32)
    noisy = (target + 0.15 * torch.randn_like(target)).clamp(0.0, 1.0)

    identical_ssim = differentiable_ssim(target, target)
    noisy_ssim = differentiable_ssim(noisy, target)
    identical_ms_ssim = differentiable_ms_ssim(target, target)
    noisy_ms_ssim = differentiable_ms_ssim(noisy, target)
    assert torch.allclose(identical_ssim, torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(identical_ms_ssim, torch.tensor(1.0), atol=1e-5)
    assert noisy_ssim < identical_ssim
    assert noisy_ms_ssim < identical_ms_ssim


def test_training_ssim_tracks_reference_ssim() -> None:
    generator = torch.Generator().manual_seed(17)
    target = torch.rand((1, 1, 64, 64), generator=generator)
    noisy = (target + 0.08 * torch.randn(target.shape, generator=generator)).clamp(0.0, 1.0)
    training_score = float(differentiable_ssim(noisy, target))
    reference_score = calculate_ssim(noisy[0], target[0])
    assert 0.0 < training_score < 1.0
    assert 0.0 < reference_score < 1.0
    assert abs(training_score - reference_score) < 0.15


def test_ssim_losses_are_small_image_safe_and_differentiable() -> None:
    prediction = torch.rand(1, 1, 2, 3, requires_grad=True)
    target = torch.rand_like(prediction)
    loss = SSIMLoss()(prediction, target) + MSSSIMLoss()(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_gradient_and_frequency_losses_detect_change() -> None:
    target = torch.zeros(1, 1, 16, 16)
    prediction = target.clone()
    changed = target.clone()
    changed[:, :, 4:12, 4:12] = 1.0

    gradient = GradientLoss(mode="l1")
    frequency = FrequencyAmplitudeLoss(mode="l1")
    assert gradient(prediction, target) == 0.0
    assert frequency(prediction, target) == 0.0
    assert gradient(changed, target) > 0.0
    assert frequency(changed, target) > 0.0


def test_prototype_losses_are_finite_and_backpropagate() -> None:
    features = {
        "CD68": torch.randn(2, 32, 8, requires_grad=True),
        "CD45RO": torch.randn(2, 32, 8, requires_grad=True),
    }
    banks = {
        "shared": torch.randn(4, 8, requires_grad=True),
        "CD68": torch.randn(3, 8, requires_grad=True),
        "CD45RO": torch.randn(3, 8, requires_grad=True),
    }
    activation = prototype_activation_loss(features, banks, maximum_tokens=16)
    diversity = prototype_diversity_loss(banks)
    total = activation + diversity
    assert torch.isfinite(total)
    total.backward()
    assert features["CD68"].grad is not None
    assert banks["shared"].grad is not None
    assert torch.isfinite(banks["shared"].grad).all()


def test_task_correlation_matches_equal_prediction_statistics() -> None:
    first = torch.rand(2, 1, 16, 16, requires_grad=True)
    second = torch.rand(2, 1, 16, 16, requires_grad=True)
    predictions = {"CD68": first, "CD45RO": second}
    targets = {"CD68": first.detach().clone(), "CD45RO": second.detach().clone()}
    loss = task_correlation_loss(predictions, targets, pool_size=8)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
    loss.backward()
    assert first.grad is not None
    assert second.grad is not None


def test_composite_multiscale_multitask_loss_is_finite_and_backward() -> None:
    torch.manual_seed(7)
    prediction_a = torch.rand(1, 1, 16, 16, requires_grad=True)
    prediction_b = torch.rand(1, 1, 16, 16, requires_grad=True)
    prototype_features = {
        "CD68": torch.randn(1, 16, 8, requires_grad=True),
        "CD45RO": torch.randn(1, 16, 8, requires_grad=True),
    }
    prototype_banks = {
        "shared": torch.randn(2, 8, requires_grad=True),
        "CD68": torch.randn(2, 8, requires_grad=True),
        "CD45RO": torch.randn(2, 8, requires_grad=True),
    }
    output = RestorationOutput(
        predictions={"CD68": prediction_a, "CD45RO": prediction_b},
        deep_supervision={
            "CD68": {
                4: F.interpolate(prediction_a, size=(4, 4), mode="area"),
                8: F.interpolate(prediction_a, size=(8, 8), mode="area"),
                16: prediction_a,
            },
            "CD45RO": {
                4: F.interpolate(prediction_b, size=(4, 4), mode="area"),
                8: F.interpolate(prediction_b, size=(8, 8), mode="area"),
                16: prediction_b,
            },
        },
        prototype_features=prototype_features,
        prototype_banks=prototype_banks,
    )
    targets = {"CD68": torch.rand_like(prediction_a), "CD45RO": torch.rand_like(prediction_b)}
    criterion = CompositeRestorationLoss()
    result = criterion(output, targets)

    assert torch.isfinite(result.total)
    assert set(result.per_task) == {"CD68", "CD45RO"}
    assert "CD68/16x16/ssim" in result.components
    assert "correlation" in result.components
    assert "prototype_activation" in result.components
    unpacked_total, unpacked_components = result
    assert unpacked_total is result.total
    assert unpacked_components is result.components

    result.total.backward()
    assert prediction_a.grad is not None
    assert prototype_features["CD68"].grad is not None
    assert prototype_banks["shared"].grad is not None
    assert torch.isfinite(prediction_a.grad).all()


def test_composite_loss_under_cpu_autocast_stays_float32_and_finite() -> None:
    prediction = torch.rand(1, 1, 16, 16, requires_grad=True)
    target = torch.rand_like(prediction)
    criterion = CompositeRestorationLoss(correlation=0.0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = criterion(prediction, target)
    assert result.total.dtype == torch.float32
    assert torch.isfinite(result.total)
    result.total.backward()
    assert prediction.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_composite_loss_under_cuda_amp_stays_float32_and_finite() -> None:
    prediction = torch.rand(1, 1, 16, 16, device="cuda", requires_grad=True)
    target = torch.rand_like(prediction)
    criterion = CompositeRestorationLoss(correlation=0.0).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = criterion(prediction, target)
    assert result.total.dtype == torch.float32
    assert torch.isfinite(result.total)
    result.total.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
