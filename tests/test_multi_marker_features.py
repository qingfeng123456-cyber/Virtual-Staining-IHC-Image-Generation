"""A2/A3 flags and v1 state compatibility for MultiMarkerRestorer."""

from __future__ import annotations

import pytest
import torch

from virtual_staining.models import MultiMarkerRestorer


def _options() -> dict[str, object]:
    return {
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 4,
        "encoder_depths": (1, 1, 1, 1),
        "decoder_depths": (1, 1, 1),
        "target_names": ("CD68",),
        "use_sobel_input": False,
        "use_task_adapters": True,
        "use_prototypes": False,
        "deep_supervision": True,
    }


def test_v1_state_dict_strict_load_and_disabled_output_are_unchanged() -> None:
    torch.manual_seed(17)
    legacy = MultiMarkerRestorer(**_options()).eval()
    legacy_state = legacy.state_dict()
    upgraded = MultiMarkerRestorer(
        **_options(),
        base_detail=False,
        context={
            "enabled": False,
            "fusion_scales": [4, 8, 16],
            "require_verified_grid": True,
        },
    ).eval()

    result = upgraded.load_state_dict(legacy_state, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert tuple(upgraded.state_dict()) == tuple(legacy_state)
    assert not any(key.startswith("base_detail_heads.") for key in upgraded.state_dict())
    assert not any(key.startswith("context_encoder.") for key in upgraded.state_dict())
    assert not any(
        key.startswith("context_cross_attention.") for key in upgraded.state_dict()
    )
    inputs = torch.rand(1, 1, 32, 32)
    ignored_tiles = torch.rand(1, 3, 5, 7, 9)
    with torch.inference_mode():
        expected = legacy(inputs).prediction
        actual = upgraded(
            inputs,
            context_tiles=ignored_tiles,
            context_valid_mask=torch.zeros(1, 3, dtype=torch.bool),
            organ_id="colon",
        ).prediction
    assert torch.equal(actual, expected)
    output = upgraded(inputs)
    assert output.base_predictions == {}
    assert output.detail_predictions == {}
    assert output.logits == {}
    assert output.context_attention == {}


def test_a2_base_detail_shapes_reconstruction_and_backward() -> None:
    model = MultiMarkerRestorer(
        **_options(),
        base_detail=True,
        max_detail_amplitude=0.4,
    )
    inputs = torch.rand(2, 1, 32, 32, requires_grad=True)

    output = model(inputs)

    assert output.prediction.shape == (2, 1, 32, 32)
    assert output.base_prediction.shape == output.prediction.shape
    assert output.detail_prediction.shape == output.prediction.shape
    assert output.logits["CD68"].shape == output.prediction.shape
    reconstructed = torch.sigmoid(
        torch.logit(output.base_prediction) + output.detail_prediction
    )
    assert torch.allclose(output.prediction, reconstructed, atol=1e-6, rtol=1e-5)
    assert torch.equal(output.deep_supervision["CD68"][32], output.prediction)
    assert float(output.detail_prediction.detach().abs().max()) <= 0.4
    assert torch.isfinite(output.prediction).all()

    output.prediction.mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert model.base_detail_heads is not None
    head = model.base_detail_heads["task_0"]
    assert head.low_resolution_head.weight.grad is not None
    assert head.detail_head[-1].weight.grad is not None
    assert torch.isfinite(head.low_resolution_head.weight.grad).all()
    assert torch.isfinite(head.detail_head[-1].weight.grad).all()


def _context_options() -> dict[str, object]:
    return {
        "enabled": True,
        "grid_size": 3,
        "token_dim": 8,
        "encoder_width": 4,
        "encoder_depth": 1,
        "context_dropout": 0.0,
        "fusion_scales": [1, 2, 4, 8],
        "stop_gradient": False,
    }


def test_a3_context_film_is_zero_init_identity_with_mask_and_backward() -> None:
    model = MultiMarkerRestorer(**_options(), context=_context_options()).eval()
    inputs = torch.rand(2, 1, 32, 32, requires_grad=True)
    first_tiles = torch.rand(2, 9, 1, 32, 32, requires_grad=True)
    second_tiles = torch.rand(2, 9, 1, 32, 32)
    valid_mask = torch.tensor(
        [
            [True, True, False, True, True, True, False, True, True],
            [False, False, False, False, True, False, False, False, False],
        ]
    )

    first = model(
        inputs,
        context_tiles=first_tiles,
        context_valid_mask=valid_mask,
        organ_id=("colon", "colon"),
    )
    with torch.no_grad():
        second = model(
            inputs.detach(),
            context_tiles=second_tiles,
            context_valid_mask=valid_mask,
        )

    assert torch.equal(first.prediction.detach(), second.prediction)
    assert set(first.context_attention) == {"1", "2", "4", "8"}
    for attention in first.context_attention.values():
        assert attention.shape == (2, 9)
        assert torch.count_nonzero(attention[~valid_mask]) == 0
        assert torch.isfinite(attention).all()
    assert not any("cross" in name for name, _ in model.named_parameters())

    first.prediction.mean().backward()
    assert inputs.grad is not None
    assert first_tiles.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(first_tiles.grad).all()
    assert model.context_fusion is not None
    for layer in model.context_fusion.layers.values():
        assert layer.affine.weight.grad is not None
        assert torch.isfinite(layer.affine.weight.grad).all()


def test_a3_enabled_context_requires_tiles() -> None:
    model = MultiMarkerRestorer(**_options(), context=_context_options())
    with pytest.raises(ValueError, match="context_tiles are required"):
        model(torch.rand(1, 1, 32, 32))


def test_a4_cross_attention_is_a_zero_init_single_variable_extension() -> None:
    a3_options = _context_options()
    a4_options = {**a3_options, "bottleneck_cross_attention": True}
    torch.manual_seed(29)
    a3_model = MultiMarkerRestorer(**_options(), context=a3_options).eval()
    a4_model = MultiMarkerRestorer(**_options(), context=a4_options).eval()

    incompatible = a4_model.load_state_dict(a3_model.state_dict(), strict=False)
    assert incompatible.missing_keys
    assert all(
        key.startswith("context_cross_attention.")
        for key in incompatible.missing_keys
    )
    assert incompatible.unexpected_keys == []
    assert a3_model.feature_flags["context_cross_attention"] is False
    assert a4_model.feature_flags["context_cross_attention"] is True

    inputs = torch.rand(2, 1, 32, 32, requires_grad=True)
    context_tiles = torch.rand(2, 9, 1, 32, 32, requires_grad=True)
    valid_mask = torch.tensor(
        [
            [True, True, False, True, True, True, False, True, True],
            [False, False, False, False, True, False, False, False, False],
        ]
    )
    with torch.no_grad():
        a3_prediction = a3_model(
            inputs.detach(),
            context_tiles=context_tiles.detach(),
            context_valid_mask=valid_mask,
        ).prediction
    a4_output = a4_model(
        inputs,
        context_tiles=context_tiles,
        context_valid_mask=valid_mask,
    )

    assert torch.equal(a4_output.prediction.detach(), a3_prediction)
    assert set(a4_output.context_attention) == {"1", "2", "4", "8", "cross_8"}
    cross_attention = a4_output.context_attention["cross_8"]
    assert cross_attention.shape == (2, 16, 9)
    for batch_index, mask in enumerate(valid_mask):
        assert torch.count_nonzero(cross_attention[batch_index, :, ~mask]) == 0
    assert torch.isfinite(cross_attention).all()

    a4_output.prediction.mean().backward()
    assert inputs.grad is not None
    assert context_tiles.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(context_tiles.grad).all()
    assert a4_model.context_cross_attention is not None
    residual_scale_grad = a4_model.context_cross_attention.residual_scale.grad
    assert residual_scale_grad is not None
    assert torch.isfinite(residual_scale_grad).all()


def test_a4_cross_attention_requires_enabled_context() -> None:
    with pytest.raises(ValueError, match="requires context.enabled=true"):
        MultiMarkerRestorer(
            **_options(),
            context={"enabled": False, "bottleneck_cross_attention": True},
        )
