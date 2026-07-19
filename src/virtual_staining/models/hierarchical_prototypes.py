"""Multi-scale shared, marker, and organ prototype mixing."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .task_organ_conditioning import normalize_identity


@dataclass(slots=True)
class PrototypeScaleOutput:
    """Result and diagnostics for one feature scale."""

    features: Tensor
    normalized_tokens: Tensor
    attention: dict[str, Tensor]
    usage: dict[str, Tensor]


def prototype_usage_entropy(usage: Mapping[str, Tensor]) -> Tensor:
    """Return the mean negative entropy term; minimizing promotes broad usage."""

    values = [
        (probability.float().clamp_min(1e-8) * probability.float().clamp_min(1e-8).log()).sum()
        for probability in usage.values()
    ]
    if not values:
        return torch.tensor(0.0)
    return torch.stack(values).mean()


def prototype_bank_diversity(banks: Mapping[str, Tensor]) -> Tensor:
    """Mean squared off-diagonal cosine similarity across prototype banks."""

    losses: list[Tensor] = []
    for bank in banks.values():
        normalized = F.normalize(bank.float(), dim=-1, eps=1e-6)
        similarity = normalized @ normalized.transpose(0, 1)
        if bank.shape[0] > 1:
            mask = ~torch.eye(bank.shape[0], device=bank.device, dtype=torch.bool)
            losses.append(similarity[mask].square().mean())
        else:
            losses.append(similarity.sum() * 0.0)
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


class PrototypeScaleMixer(nn.Module):
    """Cosine prototype aggregation for a single spatial scale."""

    def __init__(
        self,
        channels: int,
        target_names: Sequence[str],
        organ_names: Sequence[str],
        *,
        shared_count: int = 8,
        marker_count: int = 8,
        organ_count: int = 4,
        prototype_dim: int = 128,
        temperature: float = 0.1,
        residual_init: float = 0.0,
        organ_enabled: bool = True,
    ) -> None:
        super().__init__()
        if min(channels, shared_count, marker_count, prototype_dim) < 1:
            raise ValueError("Prototype dimensions and counts must be positive")
        if organ_enabled and organ_count < 1:
            raise ValueError("organ_count must be positive when organ prototypes are enabled")
        if temperature <= 0.0:
            raise ValueError("Prototype temperature must be positive")
        self.channels = int(channels)
        self.prototype_dim = int(prototype_dim)
        self.temperature = float(temperature)
        self.organ_enabled = bool(organ_enabled)
        self.target_names = tuple(str(name) for name in target_names)
        known_organs = [name for name in organ_names if normalize_identity(name) != "unknown"]
        self.organ_names = tuple((*known_organs, "unknown"))
        self._target_keys = {
            normalize_identity(name): f"marker_{index}"
            for index, name in enumerate(self.target_names)
        }
        self._organ_keys = {
            normalize_identity(name): f"organ_{index}"
            for index, name in enumerate(self.organ_names)
        }

        self.input_projection = nn.Conv2d(channels, prototype_dim, kernel_size=1)
        self.output_projection = nn.Conv2d(prototype_dim, channels, kernel_size=1)
        self.shared_bank = nn.Parameter(torch.empty(shared_count, prototype_dim))
        self.marker_banks = nn.ParameterDict(
            {
                key: nn.Parameter(torch.empty(marker_count, prototype_dim))
                for key in self._target_keys.values()
            }
        )
        self.organ_banks = nn.ParameterDict(
            {
                key: nn.Parameter(torch.empty(organ_count, prototype_dim))
                for key in self._organ_keys.values()
            }
            if self.organ_enabled
            else {}
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.shared_bank, std=0.02)
        for bank in (*self.marker_banks.values(), *self.organ_banks.values()):
            nn.init.trunc_normal_(bank, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _marker_bank(self, task_name: str) -> Tensor:
        key = normalize_identity(task_name)
        if key not in self._target_keys:
            raise KeyError(f"Unknown marker prototype identity: {task_name!r}")
        return self.marker_banks[self._target_keys[key]]

    def _organ_bank(self, organ_name: str | None) -> Tensor:
        key = normalize_identity(organ_name or "unknown")
        key = key if key in self._organ_keys else "unknown"
        return self.organ_banks[self._organ_keys[key]]

    def banks(self) -> dict[str, Tensor]:
        values = {"shared": self.shared_bank}
        values.update(
            {f"marker/{name}": self._marker_bank(name) for name in self.target_names}
        )
        if self.organ_enabled:
            values.update(
                {f"organ/{name}": self._organ_bank(name) for name in self.organ_names}
            )
        return values

    def _attend(self, tokens: Tensor, bank: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        normalized_bank = F.normalize(bank.float(), dim=-1, eps=1e-6)
        logits = tokens.float() @ normalized_bank.transpose(0, 1)
        attention = torch.softmax(logits / self.temperature, dim=-1)
        context = attention @ normalized_bank
        usage = attention.mean(dim=(0, 1))
        return context.to(dtype=tokens.dtype), attention, usage

    @staticmethod
    def _organ_labels(
        organ_id: str | Sequence[str] | None,
        batch: int,
    ) -> tuple[str, ...]:
        if organ_id is None:
            return ("unknown",) * batch
        if isinstance(organ_id, str):
            return (organ_id,) * batch
        labels = tuple(str(value) for value in organ_id)
        if len(labels) != batch:
            raise ValueError(f"Expected {batch} organ labels, got {len(labels)}")
        return labels

    def forward(
        self,
        inputs: Tensor,
        task_name: str,
        organ_id: str | Sequence[str] | None = None,
    ) -> PrototypeScaleOutput:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"Expected BCHW features with {self.channels} channels, got {tuple(inputs.shape)}"
            )
        batch, _, height, width = inputs.shape
        projected = self.input_projection(inputs)
        tokens = projected.flatten(2).transpose(1, 2)
        normalized_tokens = F.normalize(tokens.float(), dim=-1, eps=1e-6)
        shared_context, shared_attention, shared_usage = self._attend(
            normalized_tokens, self.shared_bank
        )
        marker_context, marker_attention, marker_usage = self._attend(
            normalized_tokens, self._marker_bank(task_name)
        )
        contexts = [shared_context, marker_context]
        attention = {"shared": shared_attention, "marker": marker_attention}
        usage = {"shared": shared_usage, "marker": marker_usage}

        if self.organ_enabled:
            organ_contexts: list[Tensor] = []
            organ_attentions: list[Tensor] = []
            labels = self._organ_labels(organ_id, batch)
            for batch_index, label in enumerate(labels):
                context, weights, _ = self._attend(
                    normalized_tokens[batch_index : batch_index + 1],
                    self._organ_bank(label),
                )
                organ_contexts.append(context)
                organ_attentions.append(weights)
            organ_context = torch.cat(organ_contexts, dim=0)
            organ_attention = torch.cat(organ_attentions, dim=0)
            contexts.append(organ_context)
            attention["organ"] = organ_attention
            usage["organ"] = organ_attention.mean(dim=(0, 1))

        aggregate = torch.stack(contexts, dim=0).mean(dim=0)
        aggregate = aggregate.transpose(1, 2).reshape(
            batch, self.prototype_dim, height, width
        )
        update = self.output_projection(aggregate.to(dtype=inputs.dtype))
        features = inputs + self.residual_scale.to(dtype=inputs.dtype) * update
        attention_maps = {
            name: values.transpose(1, 2).reshape(batch, -1, height, width)
            for name, values in attention.items()
        }
        return PrototypeScaleOutput(
            features=features,
            normalized_tokens=normalized_tokens,
            attention=attention_maps,
            usage=usage,
        )


class HierarchicalPrototypeMixer(nn.Module):
    """Prototype mixers at the 1/8 and 1/16 local scales."""

    def __init__(
        self,
        channels_by_scale: Mapping[int, int],
        target_names: Sequence[str],
        organ_names: Sequence[str] = ("colon", "liver", "stomach"),
        *,
        scales: Sequence[int] = (8, 16),
        shared_count: int = 8,
        marker_count: int = 8,
        organ_count: int = 4,
        prototype_dim: int = 128,
        temperature: float = 0.1,
        residual_init: float = 0.0,
        organ_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.scales = tuple(int(scale) for scale in scales)
        missing = sorted(set(self.scales).difference(channels_by_scale))
        if missing:
            raise ValueError(f"Missing prototype channel definitions for scales: {missing}")
        self.mixers = nn.ModuleDict(
            {
                str(scale): PrototypeScaleMixer(
                    channels_by_scale[scale],
                    target_names,
                    organ_names,
                    shared_count=shared_count,
                    marker_count=marker_count,
                    organ_count=organ_count,
                    prototype_dim=prototype_dim,
                    temperature=temperature,
                    residual_init=residual_init,
                    organ_enabled=organ_enabled,
                )
                for scale in self.scales
            }
        )

    def forward(
        self,
        features: Mapping[int, Tensor],
        task_name: str,
        organ_id: str | Sequence[str] | None = None,
    ) -> tuple[dict[int, Tensor], dict[int, PrototypeScaleOutput]]:
        outputs = dict(features)
        diagnostics: dict[int, PrototypeScaleOutput] = {}
        for scale in self.scales:
            result = self.mixers[str(scale)](outputs[scale], task_name, organ_id)
            outputs[scale] = result.features
            diagnostics[scale] = result
        return outputs, diagnostics

    def banks(self) -> dict[str, Tensor]:
        return {
            f"{scale}/{name}": bank
            for scale in self.scales
            for name, bank in self.mixers[str(scale)].banks().items()
        }

    @staticmethod
    def _normalized_bank_key(value: str) -> str:
        return "".join(character for character in value.casefold() if character.isalnum())

    def resolve_bank_key(self, diagnostic: str) -> str | None:
        """Map an emitted usage diagnostic to one unique trainable bank.

        Shared and marker diagnostics carry enough identity to resolve safely.
        The current aggregate ``organ`` diagnostic does not identify which organ
        bank contributed when a batch contains multiple organs, so it is
        intentionally unresolved instead of guessed.
        """

        requested = str(diagnostic).strip("/")
        banks = self.banks()
        direct = [
            key
            for key in banks
            if self._normalized_bank_key(key)
            == self._normalized_bank_key(requested)
        ]
        if len(direct) == 1:
            return direct[0]

        segments = requested.split("/")
        scale_index = next(
            (
                index
                for index, segment in enumerate(segments)
                if segment in self.mixers
            ),
            None,
        )
        if scale_index is None or scale_index + 1 >= len(segments):
            return None
        scale = segments[scale_index]
        bank_kind = segments[scale_index + 1].casefold()
        task = "/".join(segments[:scale_index])
        suffix = segments[scale_index + 2 :]
        if bank_kind == "shared" and not suffix:
            expected = f"{scale}/shared"
        elif bank_kind in {"marker", "task"} and task and not suffix:
            expected = f"{scale}/marker/{task}"
        elif bank_kind == "organ" and suffix:
            expected = f"{scale}/organ/{'/'.join(suffix)}"
        else:
            return None
        matches = [
            key
            for key in banks
            if self._normalized_bank_key(key)
            == self._normalized_bank_key(expected)
        ]
        return matches[0] if len(matches) == 1 else None

    def bank_parameters(self) -> dict[str, nn.Parameter]:
        """Return prototype banks as parameters for optimizer/EMA row repair."""

        return {
            key: value
            for key, value in self.banks().items()
            if isinstance(value, nn.Parameter)
        }

    @torch.no_grad()
    def reset_prototype_rows(
        self,
        rows_by_bank: Mapping[str, Sequence[int]],
        *,
        seed: int,
        std: float = 0.02,
    ) -> list[dict[str, int | str | float]]:
        """Deterministically reinitialize selected prototype rows.

        A private CPU generator derived from the run seed, bank key, and row
        index avoids perturbing global training RNG state.
        """

        if isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("Prototype reset seed must be a nonnegative integer")
        if not torch.isfinite(torch.tensor(float(std))) or float(std) <= 0.0:
            raise ValueError("Prototype reset std must be finite and positive")
        banks = self.bank_parameters()
        unknown = sorted(set(rows_by_bank).difference(banks))
        if unknown:
            raise KeyError(f"Unknown prototype reset banks: {', '.join(unknown)}")
        records: list[dict[str, int | str | float]] = []
        for bank_key in sorted(rows_by_bank):
            bank = banks[bank_key]
            indices = sorted({int(index) for index in rows_by_bank[bank_key]})
            if any(index < 0 or index >= bank.shape[0] for index in indices):
                raise IndexError(
                    f"Prototype reset index is outside bank {bank_key} with "
                    f"{bank.shape[0]} rows"
                )
            for index in indices:
                digest = hashlib.sha256(
                    f"{int(seed)}:{bank_key}:{index}".encode()
                ).digest()
                row_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(row_seed)
                replacement = torch.empty(
                    bank.shape[1], device="cpu", dtype=torch.float32
                ).normal_(mean=0.0, std=float(std), generator=generator)
                bank[index].copy_(replacement.to(device=bank.device, dtype=bank.dtype))
                records.append(
                    {
                        "bank_key": bank_key,
                        "prototype_index": index,
                        "row_seed": row_seed,
                        "std": float(std),
                    }
                )
        return records
