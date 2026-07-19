"""Shape, usage, identity, and gradient checks for hierarchical prototypes."""

from __future__ import annotations

from copy import deepcopy

import torch

from virtual_staining.models.hierarchical_prototypes import (
    HierarchicalPrototypeMixer,
    PrototypeScaleMixer,
    prototype_bank_diversity,
    prototype_usage_entropy,
)


def test_zero_init_prototype_scale_mixer_is_identity_with_valid_usage() -> None:
    mixer = PrototypeScaleMixer(
        channels=8,
        target_names=("CD68",),
        organ_names=("colon",),
        shared_count=3,
        marker_count=2,
        organ_count=2,
        prototype_dim=6,
        residual_init=0.0,
    )
    inputs = torch.rand(2, 8, 4, 4)

    output = mixer(inputs, "cd_68", organ_id=("colon", "unseen-organ"))

    assert torch.equal(output.features, inputs)
    assert output.normalized_tokens.shape == (2, 16, 6)
    assert set(output.attention) == {"shared", "marker", "organ"}
    assert set(output.usage) == {"shared", "marker", "organ"}
    for attention in output.attention.values():
        assert torch.allclose(attention.sum(dim=1), torch.ones(2, 4, 4))
        assert torch.isfinite(attention).all()
    for usage in output.usage.values():
        assert torch.allclose(usage.sum(), torch.tensor(1.0))
        assert torch.isfinite(usage).all()


def test_hierarchical_prototypes_forward_backward_and_regularizers_are_finite() -> None:
    mixer = HierarchicalPrototypeMixer(
        channels_by_scale={8: 8, 16: 16},
        target_names=("CD68", "Vimentin"),
        organ_names=("colon", "liver"),
        scales=(8, 16),
        shared_count=2,
        marker_count=3,
        organ_count=2,
        prototype_dim=6,
        residual_init=0.1,
    )
    features = {
        8: torch.rand(2, 8, 4, 4, requires_grad=True),
        16: torch.rand(2, 16, 2, 2, requires_grad=True),
    }

    outputs, diagnostics = mixer(
        features,
        "Vimentin",
        organ_id=("colon", "liver"),
    )
    usage = {
        f"{scale}/{name}": value
        for scale, result in diagnostics.items()
        for name, value in result.usage.items()
    }
    usage_loss = prototype_usage_entropy(usage)
    diversity_loss = prototype_bank_diversity(mixer.banks())
    loss = sum(value.square().mean() for value in outputs.values())
    loss = loss + 0.01 * usage_loss + 0.01 * diversity_loss

    assert set(outputs) == {8, 16}
    assert set(diagnostics) == {8, 16}
    assert torch.isfinite(loss)
    loss.backward()

    for feature in features.values():
        assert feature.grad is not None
        assert torch.isfinite(feature.grad).all()
    for bank in mixer.banks().values():
        assert bank.grad is not None
        assert torch.isfinite(bank.grad).all()


def test_dead_prototype_row_reset_is_deterministic_and_does_not_touch_global_rng() -> None:
    first = HierarchicalPrototypeMixer(
        channels_by_scale={8: 8, 16: 16},
        target_names=("CD68",),
        organ_names=("colon",),
        scales=(8, 16),
        shared_count=3,
        marker_count=3,
        organ_count=2,
        prototype_dim=6,
    )
    second = deepcopy(first)
    original = {
        key: value.detach().clone() for key, value in first.banks().items()
    }
    reset_plan = {"8/shared": [0], "16/marker/CD68": [1]}

    rng_before = torch.get_rng_state().clone()
    first_records = first.reset_prototype_rows(reset_plan, seed=2026, std=0.02)
    rng_after = torch.get_rng_state()
    second_records = second.reset_prototype_rows(reset_plan, seed=2026, std=0.02)

    assert torch.equal(rng_before, rng_after)
    assert first_records == second_records
    assert first.resolve_bank_key("CD68/8/shared") == "8/shared"
    assert first.resolve_bank_key("CD68/16/marker") == "16/marker/CD68"
    assert first.resolve_bank_key("CD68/8/organ") is None
    for bank_key, index in (("8/shared", 0), ("16/marker/CD68", 1)):
        assert torch.equal(
            first.banks()[bank_key][index], second.banks()[bank_key][index]
        )
        assert not torch.equal(first.banks()[bank_key][index], original[bank_key][index])
    assert torch.equal(first.banks()["8/shared"][1], original["8/shared"][1])
