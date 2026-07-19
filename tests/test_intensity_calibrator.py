"""Identity initialization, bounded parameters, and gradient checks for calibration."""

from __future__ import annotations

import torch
from torch import nn

from virtual_staining.models.intensity_calibrator import GlobalIntensityCalibrator


def test_intensity_calibrator_is_exact_identity_at_initialization() -> None:
    calibrator = GlobalIntensityCalibrator(
        bottleneck_channels=8,
        embedding_dim=6,
        out_channels=2,
        max_gain_delta=0.2,
        max_bias=0.1,
        hidden_dim=12,
    )
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    bottleneck = torch.rand(2, 8, 2, 2, requires_grad=True)
    marker_embedding = torch.rand(2, 6, requires_grad=True)
    organ_embedding = torch.rand(2, 6, requires_grad=True)

    output = calibrator(logits, bottleneck, marker_embedding, organ_embedding)

    assert torch.equal(output.logits, logits)
    assert torch.equal(output.gain, torch.ones(2, 2, 1, 1))
    assert torch.count_nonzero(output.bias) == 0
    assert torch.isfinite(output.logits).all()

    output.logits.square().mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert bottleneck.grad is not None
    assert marker_embedding.grad is not None
    assert organ_embedding.grad is not None
    assert torch.isfinite(bottleneck.grad).all()
    assert torch.isfinite(marker_embedding.grad).all()
    assert torch.isfinite(organ_embedding.grad).all()
    output_layer = calibrator.predictor[-1]
    assert isinstance(output_layer, nn.Linear)
    assert output_layer.weight.grad is not None
    assert torch.isfinite(output_layer.weight.grad).all()


def test_intensity_calibrator_parameters_stay_within_configured_bounds() -> None:
    calibrator = GlobalIntensityCalibrator(
        bottleneck_channels=4,
        embedding_dim=4,
        out_channels=1,
        max_gain_delta=0.2,
        max_bias=0.1,
        hidden_dim=8,
    )
    output_layer = calibrator.predictor[-1]
    assert isinstance(output_layer, nn.Linear)
    with torch.no_grad():
        output_layer.bias.copy_(torch.tensor([100.0, -100.0]))

    output = calibrator(
        torch.zeros(2, 1, 4, 4),
        torch.rand(2, 4, 2, 2),
        torch.rand(2, 4),
        torch.rand(2, 4),
    )

    assert torch.all(output.gain >= 0.8)
    assert torch.all(output.gain <= 1.2)
    assert torch.all(output.bias >= -0.1)
    assert torch.all(output.bias <= 0.1)
    assert torch.isfinite(output.logits).all()
