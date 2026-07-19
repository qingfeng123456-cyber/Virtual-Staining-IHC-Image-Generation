"""Activation/commitment and diversity regularizers for prototype banks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .charbonnier import float32_context


def _flatten_features(features: Tensor) -> Tensor:
    if features.ndim == 4:
        return features.flatten(2).transpose(1, 2).reshape(-1, features.shape[1])
    if features.ndim == 3:
        return features.reshape(-1, features.shape[-1])
    if features.ndim == 2:
        return features
    raise ValueError(f"Prototype features must be 2D, 3D, or 4D, got {features.ndim}D")


def _deterministic_subsample(features: Tensor, maximum_tokens: int) -> Tensor:
    if features.shape[0] <= maximum_tokens:
        return features
    indices = torch.linspace(
        0,
        features.shape[0] - 1,
        steps=maximum_tokens,
        device=features.device,
    ).round().long()
    return features.index_select(0, indices)


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _hierarchical_bank_candidates(
    banks: Mapping[str, Tensor],
    task: str,
) -> tuple[Tensor | None, Tensor | None]:
    """Resolve CAMP scale-prefixed banks without changing legacy bank lookup."""

    shared_by_scale: dict[int, Tensor] = {}
    task_by_scale: dict[int, Tensor] = {}
    normalized_task = _normalized_name(task)
    for name, bank in banks.items():
        parts = str(name).split("/")
        if len(parts) < 2:
            continue
        try:
            scale = int(parts[0])
        except ValueError:
            continue
        if parts[-1].casefold() == "shared":
            shared_by_scale[scale] = bank
        if "marker" in (part.casefold() for part in parts[:-1]) and _normalized_name(
            parts[-1]
        ) == normalized_task:
            task_by_scale[scale] = bank

    common_scales = sorted(set(shared_by_scale).intersection(task_by_scale))
    if common_scales:
        scale = common_scales[-1]
        return shared_by_scale[scale], task_by_scale[scale]
    shared = shared_by_scale[max(shared_by_scale)] if shared_by_scale else None
    task_bank = task_by_scale[max(task_by_scale)] if task_by_scale else None
    return shared, task_bank


def _iter_usage_tensors(values: Mapping[str, Any]) -> Iterator[Tensor]:
    for value in values.values():
        if isinstance(value, Tensor):
            yield value
        elif isinstance(value, Mapping):
            yield from _iter_usage_tensors(value)


def prototype_usage_entropy_loss(
    usage: Mapping[str, Any],
    *,
    epsilon: float = 1e-8,
) -> Tensor:
    """Return stable negative entropy; minimizing encourages broad prototype use.

    Each diagnostic is normalized independently over its last dimension.  Zero
    mass remains a graph-connected zero instead of producing ``log(0)`` or a
    division-by-zero NaN.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    losses: list[Tensor] = []
    for value in _iter_usage_tensors(usage):
        with float32_context(value):
            vector = value.float()
            if vector.ndim == 0:
                vector = vector.reshape(1)
            elif vector.ndim > 1:
                vector = vector.reshape(-1, vector.shape[-1]).mean(dim=0)
            finite = torch.isfinite(vector)
            vector = torch.where(finite, vector, torch.zeros_like(vector)).clamp_min(0.0)
            mass = vector.sum()
            probability = vector / mass.clamp_min(epsilon)
            negative_entropy = (
                probability * probability.clamp_min(epsilon).log()
            ).sum()
            losses.append(
                torch.where(
                    mass > epsilon,
                    negative_entropy,
                    vector.sum() * 0.0,
                )
            )
    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0)


def prototype_activation_loss(
    features: Mapping[str, Tensor],
    banks: Mapping[str, Tensor],
    *,
    maximum_tokens: int = 4096,
) -> Tensor:
    """Encourage each selected feature token to approach at least one prototype."""

    if maximum_tokens < 1:
        raise ValueError("maximum_tokens must be positive")
    if not features:
        reference = next(iter(banks.values()), None)
        return reference.sum() * 0.0 if reference is not None else torch.tensor(0.0)

    losses: list[Tensor] = []
    for task, task_features in features.items():
        task_bank = banks.get(task)
        shared_bank = banks.get("shared")
        if task_bank is None and shared_bank is None:
            shared_bank, task_bank = _hierarchical_bank_candidates(banks, task)
        available = [bank for bank in (shared_bank, task_bank) if bank is not None]
        if not available:
            raise KeyError(f"No shared or task prototype bank found for {task!r}")
        with float32_context(task_features):
            tokens = _deterministic_subsample(_flatten_features(task_features).float(), maximum_tokens)
            combined_bank = torch.cat([bank.float() for bank in available], dim=0)
            if tokens.shape[-1] != combined_bank.shape[-1]:
                raise ValueError(
                    f"Feature/prototype dimensions differ for {task}: "
                    f"{tokens.shape[-1]} vs {combined_bank.shape[-1]}"
                )
            tokens = F.normalize(tokens, dim=-1, eps=1e-6)
            combined_bank = F.normalize(combined_bank, dim=-1, eps=1e-6)
            best_similarity = torch.matmul(tokens, combined_bank.transpose(0, 1)).max(dim=1).values
            losses.append((1.0 - best_similarity).mean())
    return torch.stack(losses).mean()


def prototype_diversity_loss(banks: Mapping[str, Tensor]) -> Tensor:
    """Penalize squared off-diagonal cosine similarity within every bank."""

    if not banks:
        return torch.tensor(0.0)
    losses: list[Tensor] = []
    for bank in banks.values():
        with float32_context(bank):
            normalized = F.normalize(bank.float(), dim=-1, eps=1e-6)
            similarity = torch.matmul(normalized, normalized.transpose(0, 1))
            count = similarity.shape[0]
            if count < 2:
                losses.append(similarity.sum() * 0.0)
            else:
                mask = ~torch.eye(count, dtype=torch.bool, device=similarity.device)
                losses.append(similarity[mask].square().mean())
    return torch.stack(losses).mean()


@dataclass(slots=True)
class PrototypeLossOutput:
    """Separate prototype terms so the composite loss can weight them independently."""

    activation: Tensor
    diversity: Tensor


class PrototypeRegularization(nn.Module):
    """Module wrapper for prototype activation and diversity losses."""

    def __init__(self, maximum_tokens: int = 4096) -> None:
        super().__init__()
        self.maximum_tokens = maximum_tokens

    def forward(
        self,
        features: Mapping[str, Tensor],
        banks: Mapping[str, Tensor],
    ) -> PrototypeLossOutput:
        return PrototypeLossOutput(
            activation=prototype_activation_loss(
                features,
                banks,
                maximum_tokens=self.maximum_tokens,
            ),
            diversity=prototype_diversity_loss(banks),
        )
