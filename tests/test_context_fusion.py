"""Masked pooling and identity-initialized context fusion checks."""

from __future__ import annotations

import torch

from virtual_staining.models.context_fusion import (
    BottleneckContextCrossAttention,
    MaskedAttentionPool,
    MultiScaleContextFusion,
)


def test_masked_attention_pool_ignores_invalid_tokens() -> None:
    pool = MaskedAttentionPool(token_dim=8)
    tokens = torch.rand(2, 9, 8)
    valid_mask = torch.tensor(
        [
            [True, False, True, False, True, False, True, False, True],
            [False, False, False, False, False, False, False, False, False],
        ]
    )
    altered = tokens.clone()
    altered[~valid_mask] = 1_000.0

    pooled, weights = pool(tokens, valid_mask)
    altered_pooled, altered_weights = pool(altered, valid_mask)

    assert torch.allclose(pooled, altered_pooled)
    assert torch.allclose(weights, altered_weights)
    assert torch.count_nonzero(weights[~valid_mask]) == 0
    assert torch.allclose(weights[0].sum(), torch.tensor(1.0))
    assert torch.count_nonzero(weights[1]) == 0
    assert torch.count_nonzero(pooled[1]) == 0
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(weights).all()


def test_multiscale_context_film_is_exact_identity_and_backward_is_finite() -> None:
    fusion = MultiScaleContextFusion(
        {4: 4, 8: 8, 16: 16},
        token_dim=8,
        fusion_scales=(4, 8, 16),
        context_dropout=0.0,
    )
    features = {
        4: torch.rand(2, 4, 8, 8, requires_grad=True),
        8: torch.rand(2, 8, 4, 4, requires_grad=True),
        16: torch.rand(2, 16, 2, 2, requires_grad=True),
    }
    tokens = torch.rand(2, 9, 8, requires_grad=True)
    valid_mask = torch.tensor(
        [
            [True, True, False, True, True, True, False, True, True],
            [False, False, False, False, True, False, False, False, False],
        ]
    )

    outputs, attention = fusion(features, tokens, valid_mask)

    assert set(attention) == {"4", "8", "16"}
    for scale, original in features.items():
        assert torch.equal(outputs[scale], original)
        assert torch.isfinite(outputs[scale]).all()
        assert torch.count_nonzero(attention[str(scale)][~valid_mask]) == 0

    sum(value.mean() for value in outputs.values()).backward()
    for feature in features.values():
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
    for layer in fusion.layers.values():
        assert layer.affine.weight.grad is not None
        assert torch.isfinite(layer.affine.weight.grad).all()


def test_zero_init_cross_attention_is_identity_for_all_invalid_context() -> None:
    fusion = BottleneckContextCrossAttention(
        local_channels=8,
        token_dim=8,
        heads=2,
        residual_init=0.0,
    )
    local = torch.rand(2, 8, 3, 3, requires_grad=True)
    tokens = torch.rand(2, 9, 8, requires_grad=True)
    valid_mask = torch.zeros(2, 9, dtype=torch.bool)

    output, attention = fusion(local, tokens, valid_mask)

    assert torch.equal(output, local)
    assert attention.shape == (2, 9, 9)
    assert torch.isfinite(attention).all()
    output.mean().backward()
    assert local.grad is not None
    assert torch.isfinite(local.grad).all()
    assert fusion.residual_scale.grad is not None
    assert torch.isfinite(fusion.residual_scale.grad)
