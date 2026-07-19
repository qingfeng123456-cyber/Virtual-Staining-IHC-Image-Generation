"""CAMP-VS v2 local/context, single/multi-task shape and gradient tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from virtual_staining.config import load_config
from virtual_staining.models import CAMPVSv2, RestorationOutput, available_models, build_model


def _tiny_model(
    *,
    context_enabled: bool,
    target_names: Sequence[str],
) -> CAMPVSv2:
    multiple_tasks = len(target_names) > 1
    output_channels = {
        task: (2 if task == "Vimentin" else 1) for task in target_names
    }
    return CAMPVSv2(
        in_channels=1,
        out_channels=output_channels,
        target_names=target_names,
        local_encoder={
            "type": "naf",
            "widths": [4, 8, 16, 32],
            "depths": [1, 1, 1, 1],
            "drop_path": 0.0,
        },
        decoder_depths=(1, 1, 1),
        use_sobel_input=False,
        context={
            "enabled": context_enabled,
            "grid_size": 3,
            "cache_size": 64,
            "token_dim": 16,
            "encoder_width": 4,
            "encoder_depth": 1,
            "context_dropout": 0.0,
            "fusion_scales": [4, 8, 16],
            "bottleneck_cross_attention": context_enabled,
            "cross_attention_heads": 2,
            "residual_init": 0.0,
            "tile_chunk_size": 3,
        },
        global_mixer={
            "enabled": True,
            "type": "restormer_lite",
            "blocks_1_8": 1,
            "blocks_1_16": 1,
            "heads": [1, 2],
            "expansion": 2,
        },
        conditioning={
            "marker_embedding": True,
            "organ_embedding": multiple_tasks,
            "film": True,
            "zero_init": True,
            "embedding_dim": 16,
        },
        adapters={
            "marker": True,
            "organ": multiple_tasks,
            "mixture_of_experts": multiple_tasks,
            "expert_count": 2,
            "reduction": 2,
        },
        prototypes={
            "enabled": multiple_tasks,
            "scales": [8, 16],
            "shared_count": 2,
            "marker_count": 2,
            "organ_count": 2,
            "dim": 8,
            "temperature": 0.1,
            "residual_init": 0.0,
        },
        output={
            "base_detail": True,
            "max_detail_amplitude": 1.0,
            "deep_supervision": True,
            "deep_supervision_scales": [1, 2, 4, 8],
        },
        intensity_calibrator={
            "enabled": True,
            "max_gain_delta": 0.15,
            "max_bias": 0.15,
        },
    )


@pytest.mark.parametrize("context_enabled", [False, True], ids=["local", "context"])
@pytest.mark.parametrize(
    "target_names",
    [("CD68",), ("CD68", "Vimentin")],
    ids=["single", "multitask"],
)
def test_camp_small_forward_backward_matrix(
    context_enabled: bool,
    target_names: tuple[str, ...],
) -> None:
    model = _tiny_model(context_enabled=context_enabled, target_names=target_names)
    center = torch.rand(1, 1, 32, 32, requires_grad=True)
    context_tiles = (
        torch.rand(1, 9, 1, 32, 32, requires_grad=True)
        if context_enabled
        else None
    )
    valid_mask = (
        torch.tensor([[True, True, False, True, True, True, False, True, True]])
        if context_enabled
        else None
    )
    selected_task = "cd_68" if len(target_names) == 1 else None

    output = model(
        center,
        context_tiles=context_tiles,
        context_valid_mask=valid_mask,
        task_name=selected_task,
        organ_id="colon",
    )

    assert isinstance(output, RestorationOutput)
    assert tuple(output.predictions) == target_names
    assert output.predictions["CD68"].shape == (1, 1, 32, 32)
    if "Vimentin" in target_names:
        assert output.predictions["Vimentin"].shape == (1, 2, 32, 32)
    assert all(set(scales) == {4, 8, 16, 32} for scales in output.deep_supervision.values())
    assert all(value.shape[-2:] == (32, 32) for value in output.base_predictions.values())
    assert all(value.shape[-2:] == (32, 32) for value in output.detail_predictions.values())
    assert all(torch.isfinite(value).all() for value in output.predictions.values())
    assert all(
        0.0
        <= float(value.detach().min())
        <= float(value.detach().max())
        <= 1.0
        for value in output.predictions.values()
    )
    assert set(output.calibration_parameters) == set(target_names)
    assert model.feature_flags["context"] is context_enabled

    if context_enabled:
        assert set(output.context_attention) == {"4", "8", "16", "cross_16"}
    else:
        assert output.context_attention == {}
    if len(target_names) > 1:
        assert set(output.prototype_attention) == set(target_names)
        assert set(output.prototype_usage) == set(target_names)
        assert output.prototype_banks
    else:
        assert output.prototype_attention == {}
        assert output.prototype_usage == {}
        assert output.prototype_features == {}

    loss = sum(value.float().mean() for value in output.predictions.values())
    loss.backward()
    assert center.grad is not None
    assert torch.isfinite(center.grad).all()
    if context_tiles is not None:
        assert context_tiles.grad is not None
        assert torch.isfinite(context_tiles.grad).all()


def test_context_feature_flag_requires_context_and_local_rollback_ignores_it() -> None:
    context_model = _tiny_model(context_enabled=True, target_names=("CD68",))
    local_model = _tiny_model(context_enabled=False, target_names=("CD68",))
    center = torch.rand(1, 1, 32, 32)
    tiles = torch.rand(1, 9, 1, 32, 32)

    with pytest.raises(ValueError, match="context_tiles are required"):
        context_model(center)
    without_context = local_model(center).prediction
    with_ignored_context = local_model(center, context_tiles=tiles).prediction
    assert torch.equal(without_context, with_ignored_context)


def test_zero_initialized_context_is_an_identity_at_initialization() -> None:
    model = _tiny_model(context_enabled=True, target_names=("CD68",)).eval()
    center = torch.rand(1, 1, 32, 32)
    first_tiles = torch.rand(1, 9, 1, 32, 32)
    second_tiles = torch.rand(1, 9, 1, 32, 32)
    mask = torch.ones(1, 9, dtype=torch.bool)

    with torch.inference_mode():
        first = model(center, context_tiles=first_tiles, context_valid_mask=mask).prediction
        second = model(center, context_tiles=second_tiles, context_valid_mask=mask).prediction
    assert torch.equal(first, second)


def test_camp_registry_builds_nested_feature_configuration() -> None:
    assert "campvsv2" in available_models()
    model = build_model(
        {
            "name": "camp_vs_v2",
            "local_encoder": {"widths": [4, 8, 16, 32], "depths": [1, 1, 1, 1]},
            "context": {"enabled": False},
            "global_mixer": {"enabled": False},
            "conditioning": {"embedding_dim": 8},
            "adapters": {"marker": False},
            "prototypes": {"enabled": False},
            "output": {"base_detail": False, "deep_supervision": False},
            "intensity_calibrator": {"enabled": False},
            "decoder_depths": [1, 1, 1],
            "use_sobel_input": False,
        },
        target_names=["CD68"],
    )
    assert isinstance(model, CAMPVSv2)
    assert model.feature_flags == {
        "context": False,
        "context_cross_attention": False,
        "global_mixer": False,
        "prototypes": False,
        "marker_adapters": False,
        "organ_adapters": False,
        "mixture_of_experts": False,
        "base_detail": False,
        "intensity_calibrator": False,
    }
    output = model(torch.rand(1, 1, 32, 32))
    assert output.prediction.shape == (1, 1, 32, 32)
    assert output.base_predictions == {}
    assert output.calibration_parameters == {}


def test_camp_accepts_resolved_dataset_only_context_cache_size() -> None:
    config = load_config(
        "configs/performance_v2/camp_context.yaml",
        include_resolved=False,
    )

    model = build_model(config, target_names=config["data"]["targets"])

    assert isinstance(model, CAMPVSv2)
    assert model.context_cache_size == 0
    assert model.context_enabled is True
    assert not any("cache" in key for key in model.state_dict())


@pytest.mark.parametrize("cache_size", [-1, 1.5, True])
def test_camp_rejects_invalid_direct_context_cache_size(cache_size: object) -> None:
    with pytest.raises(ValueError, match="cache_size"):
        CAMPVSv2(
            target_names=("CD68",),
            local_encoder={"widths": [4, 8, 16, 32], "depths": [1, 1, 1, 1]},
            context={"enabled": False, "cache_size": cache_size},
            global_mixer={"enabled": False},
            conditioning={"embedding_dim": 8},
            adapters={"marker": False},
            prototypes={"enabled": False},
            output={"base_detail": False, "deep_supervision": False},
            intensity_calibrator={"enabled": False},
            decoder_depths=(1, 1, 1),
            use_sobel_input=False,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_camp_context_cuda_amp_forward_backward_is_finite() -> None:
    model = _tiny_model(context_enabled=True, target_names=("CD68",)).cuda()
    center = torch.rand(1, 1, 32, 32, device="cuda", requires_grad=True)
    tiles = torch.rand(1, 9, 1, 32, 32, device="cuda", requires_grad=True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        output = model(center, context_tiles=tiles, organ_id="colon")
        loss = output.prediction.float().mean()
    loss.backward()

    assert torch.isfinite(output.prediction).all()
    assert center.grad is not None
    assert torch.isfinite(center.grad).all()
    assert tiles.grad is not None
    assert torch.isfinite(tiles.grad).all()
