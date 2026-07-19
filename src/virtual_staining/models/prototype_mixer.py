"""Shared and task-specific cosine prototype mixing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(slots=True)
class PrototypeMixResult:
    """Feature map and interpretable attention produced by a prototype mixer."""

    features: Tensor
    normalized_tokens: Tensor
    shared_attention: Tensor
    task_attention: Tensor


class MultiTaskPrototypeMixer(nn.Module):
    """Fuse shared and task banks into a bottleneck feature map by cosine attention."""

    def __init__(
        self,
        channels: int,
        task_names: tuple[str, ...] | list[str],
        shared_prototypes: int = 8,
        task_prototypes: int = 8,
        temperature: float = 0.1,
        fusion_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if channels < 1 or shared_prototypes < 1 or task_prototypes < 1:
            raise ValueError("Channels and prototype counts must be positive")
        if temperature <= 0.0:
            raise ValueError("Prototype temperature must be positive")
        if not task_names:
            raise ValueError("At least one task is required")

        self.channels = channels
        self.task_names = tuple(task_names)
        self.temperature = float(temperature)
        self.shared_bank = nn.Parameter(torch.empty(shared_prototypes, channels))
        self._task_keys = {task: f"task_{index}" for index, task in enumerate(self.task_names)}
        self.task_banks = nn.ParameterDict(
            {
                key: nn.Parameter(torch.empty(task_prototypes, channels))
                for key in self._task_keys.values()
            }
        )
        self.projection = nn.Linear(channels * 2, channels)
        self.fusion_scale = nn.Parameter(torch.tensor(float(fusion_weight)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.shared_bank, std=0.02)
        for bank in self.task_banks.values():
            nn.init.trunc_normal_(bank, std=0.02)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def task_bank(self, task_name: str) -> Tensor:
        if task_name not in self._task_keys:
            available = ", ".join(self.task_names)
            raise KeyError(f"Unknown prototype task {task_name!r}; available: {available}")
        return self.task_banks[self._task_keys[task_name]]

    def banks(self) -> dict[str, Tensor]:
        """Return raw trainable prototype banks keyed by shared/task name."""

        return {"shared": self.shared_bank, **{task: self.task_bank(task) for task in self.task_names}}

    def forward(self, inputs: Tensor, task_name: str) -> PrototypeMixResult:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"Expected BCHW features with {self.channels} channels, got {tuple(inputs.shape)}"
            )
        batch, channels, height, width = inputs.shape
        tokens = inputs.flatten(2).transpose(1, 2)
        normalized_tokens = F.normalize(tokens.float(), dim=-1, eps=1e-6)
        shared_bank = F.normalize(self.shared_bank.float(), dim=-1, eps=1e-6)
        task_bank = F.normalize(self.task_bank(task_name).float(), dim=-1, eps=1e-6)

        shared_logits = torch.matmul(normalized_tokens, shared_bank.transpose(0, 1))
        task_logits = torch.matmul(normalized_tokens, task_bank.transpose(0, 1))
        shared_attention = torch.softmax(shared_logits / self.temperature, dim=-1)
        task_attention = torch.softmax(task_logits / self.temperature, dim=-1)
        shared_context = torch.matmul(shared_attention, shared_bank)
        task_context = torch.matmul(task_attention, task_bank)
        context = self.projection(torch.cat((shared_context, task_context), dim=-1))
        context = context.transpose(1, 2).reshape(batch, channels, height, width)
        context = context.to(dtype=inputs.dtype)
        mixed = inputs + self.fusion_scale.to(dtype=inputs.dtype) * context

        shared_map = shared_attention.transpose(1, 2).reshape(batch, -1, height, width)
        task_map = task_attention.transpose(1, 2).reshape(batch, -1, height, width)
        return PrototypeMixResult(
            features=mixed,
            normalized_tokens=normalized_tokens,
            shared_attention=shared_map,
            task_attention=task_map,
        )
