"""Masking, metadata, and gradient checks for the shared context encoder."""

from __future__ import annotations

import torch

from virtual_staining.models.context_tile_encoder import SharedTinyContextEncoder


def test_context_encoder_masks_invalid_tiles_and_backward_is_finite() -> None:
    encoder = SharedTinyContextEncoder(
        in_channels=1,
        width=4,
        token_dim=8,
        grid_size=3,
        depth=1,
    )
    tiles = torch.rand(2, 9, 1, 16, 16, requires_grad=True)
    valid_mask = torch.tensor(
        [
            [True, True, False, True, True, False, True, True, True],
            [False, False, False, False, False, False, False, False, False],
        ]
    )

    output = encoder(tiles, context_valid_mask=valid_mask)

    assert output.tokens.shape == (2, 9, 8)
    assert torch.equal(output.valid_mask, valid_mask)
    assert torch.count_nonzero(output.tokens[~valid_mask]) == 0
    assert torch.isfinite(output.tokens).all()

    output.tokens.square().mean().backward()
    assert tiles.grad is not None
    assert torch.isfinite(tiles.grad).all()
    assert torch.count_nonzero(tiles.grad[~valid_mask]) == 0
    assert encoder.token_projection.weight.grad is not None
    assert torch.isfinite(encoder.token_projection.weight.grad).all()


def test_context_encoder_accepts_shared_offsets_and_organ_embedding() -> None:
    encoder = SharedTinyContextEncoder(
        in_channels=1,
        width=4,
        token_dim=6,
        grid_size=3,
        depth=1,
    )
    tiles = torch.rand(2, 9, 1, 12, 12)
    offsets = encoder.default_offsets(device=tiles.device, dtype=tiles.dtype)
    organ_embedding = torch.rand(2, 6)

    output = encoder(
        tiles,
        context_offsets=offsets,
        organ_embedding=organ_embedding,
    )

    assert output.tokens.shape == (2, 9, 6)
    assert output.valid_mask.all()
    assert torch.isfinite(output.tokens).all()
