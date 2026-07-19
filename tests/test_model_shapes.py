"""Shape, backward, registry, and AMP checks for restoration models."""

from __future__ import annotations

import pytest
import torch

from virtual_staining.models import (
    MultiMarkerRestorer,
    ResidualUNet,
    RestorationOutput,
    available_models,
    build_model,
    count_parameters,
    model_statistics,
)


def test_residual_unet_shape_deep_supervision_and_backward() -> None:
    model = ResidualUNet(
        in_channels=1,
        out_channels=3,
        base_channels=4,
        target_name="CD68",
    )
    inputs = torch.rand(2, 1, 64, 64, requires_grad=True)
    output = model(inputs)

    assert isinstance(output, RestorationOutput)
    assert output.prediction.shape == (2, 3, 64, 64)
    assert set(output.deep_supervision["CD68"]) == {16, 32, 64}
    assert all(0.0 <= prediction.min() <= prediction.max() <= 1.0 for prediction in output.predictions.values())

    output.prediction.mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert model.output_head.weight.grad is not None


def test_residual_unet_preserves_odd_spatial_shape() -> None:
    model = ResidualUNet(base_channels=4, deep_supervision=False)
    output = model(torch.rand(1, 1, 65, 63))
    assert output.prediction.shape == (1, 1, 65, 63)
    assert list(output.deep_supervision["CD68"].values())[0].shape[-2:] == (65, 63)


def test_multi_marker_all_tasks_shapes_attention_and_backward() -> None:
    output_channels = {"HLA-DR": 3, "CD45RO": 1, "Vimentin": 1, "CD68": 1}
    model = MultiMarkerRestorer(
        in_channels=1,
        out_channels=output_channels,
        base_channels=4,
        encoder_depths=(1, 1, 1, 1),
        decoder_depths=(1, 1, 1),
        use_sobel_input=True,
        use_prototypes=True,
        shared_prototypes=2,
        task_prototypes=3,
    )
    inputs = torch.rand(1, 1, 64, 64)
    output = model(inputs)

    assert tuple(output.predictions) == ("HLA-DR", "CD45RO", "Vimentin", "CD68")
    assert output.predictions["HLA-DR"].shape == (1, 3, 64, 64)
    assert output.predictions["CD68"].shape == (1, 1, 64, 64)
    assert all(set(scales) == {16, 32, 64} for scales in output.deep_supervision.values())
    assert output.prototype_attention["CD68"]["shared"].shape == (1, 2, 8, 8)
    assert output.prototype_attention["CD68"]["task"].shape == (1, 3, 8, 8)
    assert output.prototype_features["CD68"].shape == (1, 64, 32)
    assert set(output.prototype_banks) == {"shared", "HLA-DR", "CD45RO", "Vimentin", "CD68"}

    scalar = sum(prediction.mean() for prediction in output.predictions.values())
    scalar.backward()
    assert model.stem.weight.grad is not None
    assert model.prototype_mixer is not None
    assert model.prototype_mixer.shared_bank.grad is not None
    assert torch.isfinite(model.prototype_mixer.shared_bank.grad).all()


def test_multi_marker_single_task_alias_and_separate_decoder() -> None:
    model = MultiMarkerRestorer(
        base_channels=4,
        encoder_depths=(1, 1, 1, 1),
        decoder_depths=(1, 1, 1),
        use_sobel_input=False,
        use_prototypes=False,
        decoder_mode="separate_decoders",
    )
    output = model(torch.rand(1, 1, 32, 32), task_name="hla_dr")
    assert tuple(output.predictions) == ("HLA-DR",)
    assert output.prediction.shape == (1, 1, 32, 32)
    assert output.prototype_attention == {}
    assert output.prototype_banks == {}


def test_registry_and_parameter_statistics() -> None:
    assert "residualunet" in available_models()
    assert "multimarkerrestorer" in available_models()
    model = build_model(
        {
            "name": "baseline_unet",
            "base_channels": 4,
            "deep_supervision": False,
            "encoder_depths": [1, 1, 1, 1],
            "use_sobel_input": False,
            "output_activation": "sigmoid",
        },
        target_names=["CD68"],
    )
    assert isinstance(model, ResidualUNet)
    assert count_parameters(model) > 0
    statistics = model_statistics(model, (1, 1, 32, 32))
    assert statistics["parameters"] == count_parameters(model)
    assert statistics["trainable_parameters"] == statistics["parameters"]
    assert statistics["approximate_macs"] > 0


def test_model_rejects_unknown_task() -> None:
    model = MultiMarkerRestorer(
        base_channels=4,
        encoder_depths=(1, 1, 1, 1),
        decoder_depths=(1, 1, 1),
    )
    with pytest.raises(KeyError, match="Unknown task"):
        model(torch.rand(1, 1, 32, 32), task_name="unknown")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_forward_backward_is_finite() -> None:
    model = ResidualUNet(base_channels=4).cuda()
    inputs = torch.rand(1, 1, 32, 32, device="cuda", requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(inputs)
        loss = output.prediction.float().mean()
    loss.backward()
    assert torch.isfinite(output.prediction).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
