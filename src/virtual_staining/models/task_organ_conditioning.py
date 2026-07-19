"""Marker and organ identity conditioning for CAMP-VS v2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .naf_blocks import LayerNorm2d


def normalize_identity(value: str) -> str:
    """Normalize marker and organ spelling without losing deterministic identity."""

    return "".join(character for character in value.casefold() if character.isalnum())


@dataclass(slots=True)
class ConditioningOutput:
    """Separate and combined embeddings for diagnostic and downstream use."""

    marker: Tensor
    organ: Tensor
    combined: Tensor


class TaskOrganConditioner(nn.Module):
    """Resolve marker/organ labels and provide learned identity embeddings."""

    def __init__(
        self,
        target_names: Sequence[str],
        organ_names: Sequence[str] = ("colon", "liver", "stomach"),
        *,
        embedding_dim: int = 64,
        marker_enabled: bool = True,
        organ_enabled: bool = True,
    ) -> None:
        super().__init__()
        targets = tuple(str(name) for name in target_names)
        organs = tuple(str(name) for name in organ_names)
        if not targets or embedding_dim < 1:
            raise ValueError("At least one target and a positive embedding_dim are required")
        if len({normalize_identity(name) for name in targets}) != len(targets):
            raise ValueError("Target identities are ambiguous after normalization")
        known_organs = [name for name in organs if normalize_identity(name) != "unknown"]
        if len({normalize_identity(name) for name in known_organs}) != len(known_organs):
            raise ValueError("Organ identities are ambiguous after normalization")
        self.target_names = targets
        self.organ_names = tuple((*known_organs, "unknown"))
        self.embedding_dim = int(embedding_dim)
        self.marker_enabled = bool(marker_enabled)
        self.organ_enabled = bool(organ_enabled)
        self._target_lookup = {normalize_identity(name): index for index, name in enumerate(targets)}
        self._organ_lookup = {
            normalize_identity(name): index for index, name in enumerate(self.organ_names)
        }
        self.marker_embeddings = nn.Embedding(len(targets), embedding_dim)
        self.organ_embeddings = nn.Embedding(len(self.organ_names), embedding_dim)
        self.combination = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def resolve_target(self, task_name: str) -> int:
        key = normalize_identity(task_name)
        if key not in self._target_lookup:
            available = ", ".join(self.target_names)
            raise KeyError(f"Unknown task {task_name!r}; available: {available}")
        return self._target_lookup[key]

    def resolve_organ(self, organ_name: str | None) -> int:
        key = normalize_identity(organ_name or "unknown")
        return self._organ_lookup.get(key, self._organ_lookup["unknown"])

    @staticmethod
    def _labels(value: str | Sequence[str] | None, batch: int, default: str) -> tuple[str, ...]:
        if value is None:
            return (default,) * batch
        if isinstance(value, str):
            return (value,) * batch
        labels = tuple(str(label) for label in value)
        if len(labels) != batch:
            raise ValueError(f"Expected {batch} labels, got {len(labels)}")
        return labels

    def forward(
        self,
        task_name: str,
        organ_id: str | Sequence[str] | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> ConditioningOutput:
        task_index = self.resolve_target(task_name)
        task_indices = torch.full((batch_size,), task_index, device=device, dtype=torch.long)
        organ_labels = self._labels(organ_id, batch_size, "unknown")
        organ_indices = torch.tensor(
            [self.resolve_organ(label) for label in organ_labels],
            device=device,
            dtype=torch.long,
        )
        marker = self.marker_embeddings(task_indices)
        organ = self.organ_embeddings(organ_indices)
        if not self.marker_enabled:
            marker = torch.zeros_like(marker)
        if not self.organ_enabled:
            organ = torch.zeros_like(organ)
        combined = self.combination(torch.cat((marker, organ), dim=-1))
        return ConditioningOutput(marker=marker, organ=organ, combined=combined)


class IdentityConditioningFiLM(nn.Module):
    """Embedding-derived FiLM initialized to exact identity."""

    def __init__(self, channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.affine = nn.Linear(embedding_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, inputs: Tensor, embedding: Tensor) -> Tensor:
        gamma, beta = self.affine(embedding).chunk(2, dim=-1)
        return (
            inputs * (1.0 + gamma[:, :, None, None].to(dtype=inputs.dtype))
            + beta[:, :, None, None].to(dtype=inputs.dtype)
        )


class ConditionalResidualAdapter(nn.Module):
    """Low-cost marker/organ adapter with zero-initialized residual output."""

    def __init__(self, channels: int, embedding_dim: int, reduction: int = 2) -> None:
        super().__init__()
        if reduction < 1:
            raise ValueError("Adapter reduction must be positive")
        hidden = max(channels // reduction, 2)
        self.film = IdentityConditioningFiLM(channels, embedding_dim)
        self.norm = LayerNorm2d(channels)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.activation = nn.GELU()
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: Tensor, embedding: Tensor) -> Tensor:
        conditioned = self.film(inputs, embedding)
        features = self.reduce(self.norm(conditioned))
        features = self.activation(self.depthwise(features))
        features = self.expand(features)
        return conditioned + features * self.residual_scale


class GatedResidualExperts(nn.Module):
    """Optional two-expert residual branch kept identity at initialization."""

    def __init__(self, channels: int, embedding_dim: int, expert_count: int = 2) -> None:
        super().__init__()
        if expert_count < 2:
            raise ValueError("At least two experts are required")
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1),
            )
            for _ in range(expert_count)
        )
        self.gate = nn.Linear(embedding_dim, expert_count)
        self.residual_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: Tensor, embedding: Tensor) -> Tensor:
        weights = torch.softmax(self.gate(embedding).float(), dim=-1).to(inputs.dtype)
        updates = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        update = (updates * weights[:, :, None, None, None]).sum(dim=1)
        return inputs + update * self.residual_scale
