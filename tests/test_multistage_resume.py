"""State, freezing-policy, and checkpoint checks for staged fine-tuning."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from virtual_staining.engine.checkpoint import load_checkpoint, save_checkpoint
from virtual_staining.engine.multistage import MultiStageController, MultiStageState


class _StageAwareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 4)
        self.marker_head = nn.Linear(4, 1)
        self.organ_adapter = nn.Linear(4, 4)
        self.calibrator = nn.Linear(1, 1)
        self.base_detail_head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.decoder(self.encoder(inputs))
        return self.calibrator(self.marker_head(features)) + self.base_detail_head(features)


def test_multistage_state_round_trip_and_monotonic_transition() -> None:
    controller = MultiStageController()
    controller.advance_epoch()
    controller.transition("target_finetune", source_checkpoint="multitask.ckpt")
    controller.advance_epoch()

    restored = MultiStageController()
    restored.load_state_dict(controller.state_dict())
    assert restored.state == controller.state
    assert restored.state.stage == "target_finetune"
    assert restored.state.stage_epoch == 1
    assert restored.state.global_epoch == 2
    assert restored.state.completed_stages == ("multitask_pretrain",)
    assert restored.state.source_checkpoint == "multitask.ckpt"

    with pytest.raises(ValueError, match="backwards"):
        restored.transition("multitask_pretrain")


def test_multistage_freezing_policy_has_explicit_rollback() -> None:
    model = _StageAwareModel()
    controller = MultiStageController()
    all_trainable = controller.configure_trainable_parameters(model)
    assert all_trainable["trainable_parameters"] == all_trainable["total_parameters"]

    controller.transition("target_finetune")
    target_counts = controller.configure_trainable_parameters(model)
    assert target_counts["trainable_parameters"] < target_counts["total_parameters"]
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
    assert all(parameter.requires_grad for parameter in model.marker_head.parameters())

    controller.transition("metric_finetune")
    metric_counts = controller.configure_trainable_parameters(model)
    assert 0 < metric_counts["trainable_parameters"] < metric_counts["total_parameters"]
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.decoder.parameters())
    assert all(parameter.requires_grad for parameter in model.calibrator.parameters())
    assert all(parameter.requires_grad for parameter in model.base_detail_head.parameters())


def test_multistage_state_survives_checkpoint_extra(tmp_path) -> None:
    model = _StageAwareModel()
    controller = MultiStageController(
        MultiStageState(
            stage="organ_finetune",
            stage_index=2,
            stage_epoch=3,
            global_epoch=9,
            completed_stages=("multitask_pretrain", "target_finetune"),
            source_checkpoint="target.ckpt",
        )
    )
    checkpoint = save_checkpoint(
        tmp_path / "multistage.ckpt",
        model,
        epoch=8,
        global_step=31,
        extra={"multistage": controller.state_dict()},
        git_commit="unit-test",
    )

    payload = load_checkpoint(checkpoint)
    assert payload["extra"]["multistage"] == controller.state_dict()
    restored = MultiStageController()
    restored.load_state_dict(payload["extra"]["multistage"])
    assert restored.state == controller.state


def test_finetune_child_preserves_parent_stage_lineage_and_global_epoch() -> None:
    parent = MultiStageController()
    parent.advance_epoch()
    parent.transition("target_finetune", source_checkpoint="multitask.ckpt")
    parent.advance_epoch()
    parent.advance_epoch()

    child = MultiStageController()
    child.load_state_dict(parent.state_dict())
    child.transition("organ_finetune", source_checkpoint="target.ckpt")

    assert child.state.stage == "organ_finetune"
    assert child.state.stage_epoch == 0
    assert child.state.global_epoch == 3
    assert child.state.completed_stages == (
        "multitask_pretrain",
        "target_finetune",
    )
    assert child.state.source_checkpoint == "target.ckpt"
