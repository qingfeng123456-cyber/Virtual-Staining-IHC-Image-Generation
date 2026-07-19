"""Mask semantics and differentiability checks for fold-local DAPI MAE."""

from __future__ import annotations

import torch

from virtual_staining.models.dapi_mae import BlockMaskGenerator, DAPIMaskedAutoencoder


def _tiny_mae() -> DAPIMaskedAutoencoder:
    return DAPIMaskedAutoencoder(
        in_channels=1,
        widths=(4, 8, 16, 32),
        encoder_depths=(1, 1, 1, 1),
        decoder_depths=(1, 1, 1),
        block_size=8,
        mask_ratio=0.5,
        use_sobel_input=False,
    )


def test_block_mask_has_exact_count_and_is_seed_reproducible() -> None:
    generator = BlockMaskGenerator(block_size=8, mask_ratio=0.5)
    first = generator(
        2,
        32,
        32,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(2026),
    )
    second = generator(
        2,
        32,
        32,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(2026),
    )

    assert torch.equal(first, second)
    assert first.shape == (2, 1, 32, 32)
    assert set(first.unique().tolist()) == {0.0, 1.0}
    assert torch.equal(first.sum(dim=(1, 2, 3)), torch.tensor([512.0, 512.0]))


def test_dapi_mae_forward_respects_explicit_mask_and_backpropagates() -> None:
    model = _tiny_mae()
    inputs = torch.rand(2, 1, 32, 32)
    mask = torch.zeros_like(inputs)
    mask[:, :, :16, :16] = 1.0
    output = model(inputs, mask=mask)

    assert output.reconstruction.shape == inputs.shape
    assert output.mask.shape == (2, 1, 32, 32)
    assert torch.equal(output.mask, mask)
    assert torch.equal(output.masked_input[mask.bool()], torch.zeros(512))
    assert output.latent.shape == (2, 32)
    assert torch.isfinite(output.reconstruction).all()
    assert torch.all((0.0 <= output.reconstruction) & (output.reconstruction <= 1.0))

    masked_error = ((output.reconstruction - inputs).abs() * output.mask).sum()
    loss = masked_error / output.mask.sum().clamp_min(1.0)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_dapi_mae_can_disable_masking_without_changing_output_contract() -> None:
    model = DAPIMaskedAutoencoder(
        in_channels=1,
        widths=(4, 8, 16, 32),
        encoder_depths=(1, 1, 1, 1),
        decoder_depths=(1, 1, 1),
        masking_enabled=False,
        use_sobel_input=False,
    )
    inputs = torch.rand(1, 1, 32, 32)
    output = model(inputs)

    assert torch.count_nonzero(output.mask) == 0
    assert torch.equal(output.masked_input, inputs)
    assert output.prediction is output.reconstruction
