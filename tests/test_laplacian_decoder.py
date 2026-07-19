"""Logit reconstruction and gradient checks for the base/detail head."""

from __future__ import annotations

import torch

from virtual_staining.models.laplacian_decoder import LaplacianBaseDetailHead


def test_laplacian_head_reconstructs_final_logits_and_backward_is_finite() -> None:
    head = LaplacianBaseDetailHead(
        low_channels=8,
        full_channels=4,
        out_channels=2,
        max_detail_amplitude=0.4,
    )
    low_resolution = torch.rand(2, 8, 4, 5, requires_grad=True)
    full_resolution = torch.rand(2, 4, 9, 11, requires_grad=True)

    output = head(low_resolution, full_resolution, output_size=(10, 12))

    assert output.base_logits.shape == (2, 2, 10, 12)
    assert output.detail_logits.shape == (2, 2, 10, 12)
    assert torch.equal(output.final_logits, output.base_logits + output.detail_logits)
    assert torch.equal(output.base, torch.sigmoid(output.base_logits))
    assert torch.equal(output.detail, output.detail_logits)
    assert torch.equal(output.prediction, torch.sigmoid(output.final_logits))
    assert float(output.detail_logits.detach().abs().max()) <= 0.4
    assert torch.isfinite(output.prediction).all()

    output.prediction.mean().backward()
    assert low_resolution.grad is not None
    assert full_resolution.grad is not None
    assert torch.isfinite(low_resolution.grad).all()
    assert torch.isfinite(full_resolution.grad).all()
    assert head.low_resolution_head.weight.grad is not None
    assert head.detail_head[-1].weight.grad is not None


def test_zero_detail_amplitude_reduces_to_base_prediction() -> None:
    head = LaplacianBaseDetailHead(
        low_channels=4,
        full_channels=4,
        max_detail_amplitude=0.0,
    )
    output = head(torch.rand(1, 4, 3, 3), torch.rand(1, 4, 6, 6))

    assert torch.count_nonzero(output.detail_logits) == 0
    assert torch.equal(output.final_logits, output.base_logits)
    assert torch.equal(output.prediction, output.base)
