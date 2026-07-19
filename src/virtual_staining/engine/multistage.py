"""Explicit, checkpointable stage transitions for CAMP-VS v2 fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from torch import nn

STAGE_ORDER = (
    "multitask_pretrain",
    "target_finetune",
    "organ_finetune",
    "metric_finetune",
)


@dataclass(slots=True)
class MultiStageState:
    stage: str = STAGE_ORDER[0]
    stage_index: int = 0
    stage_epoch: int = 0
    global_epoch: int = 0
    completed_stages: tuple[str, ...] = ()
    source_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in STAGE_ORDER:
            raise ValueError(f"Unknown training stage: {self.stage}")
        if self.stage_index != STAGE_ORDER.index(self.stage):
            raise ValueError("stage_index does not match stage")
        if min(self.stage_epoch, self.global_epoch) < 0:
            raise ValueError("Stage epoch counters cannot be negative")
        if any(stage not in STAGE_ORDER for stage in self.completed_stages):
            raise ValueError("completed_stages contains an unknown stage")

    def state_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["completed_stages"] = list(self.completed_stages)
        return payload

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> MultiStageState:
        values = dict(payload)
        values["completed_stages"] = tuple(values.get("completed_stages", ()))
        return cls(**values)


class MultiStageController:
    """Own stage state and apply conservative parameter-freezing policies."""

    def __init__(self, state: MultiStageState | None = None) -> None:
        self.state = state or MultiStageState()

    def state_dict(self) -> dict[str, Any]:
        return self.state.state_dict()

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.state = MultiStageState.from_state_dict(payload)

    def advance_epoch(self) -> None:
        self.state.stage_epoch += 1
        self.state.global_epoch += 1

    def transition(self, stage: str, *, source_checkpoint: str | None = None) -> None:
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown training stage: {stage}")
        target_index = STAGE_ORDER.index(stage)
        if target_index < self.state.stage_index:
            raise ValueError("Training stages cannot move backwards")
        completed = list(self.state.completed_stages)
        if target_index > self.state.stage_index and self.state.stage not in completed:
            completed.append(self.state.stage)
        self.state = MultiStageState(
            stage=stage,
            stage_index=target_index,
            stage_epoch=0 if stage != self.state.stage else self.state.stage_epoch,
            global_epoch=self.state.global_epoch,
            completed_stages=tuple(completed),
            source_checkpoint=source_checkpoint or self.state.source_checkpoint,
        )

    def configure_trainable_parameters(self, model: nn.Module) -> dict[str, int]:
        """Apply the documented stage policy and return parameter counts."""

        stage = self.state.stage
        for parameter in model.parameters():
            parameter.requires_grad_(stage == "multitask_pretrain")
        if stage != "multitask_pretrain":
            stage_tokens = {
                "target_finetune": ("decoder", "head", "marker", "adapter", "calibrator"),
                "organ_finetune": ("organ", "adapter", "calibrator", "head"),
                "metric_finetune": ("head", "calibrator", "base_detail", "detail"),
            }[stage]
            for name, parameter in model.named_parameters():
                if any(token in name.casefold() for token in stage_tokens):
                    parameter.requires_grad_(True)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        if trainable == 0:
            raise ValueError(f"Stage {stage} did not select any trainable parameters")
        return {"total_parameters": total, "trainable_parameters": trainable}
