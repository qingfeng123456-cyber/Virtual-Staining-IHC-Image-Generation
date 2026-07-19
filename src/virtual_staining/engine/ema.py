"""Exponential moving average weights for stable validation and inference."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn


class ExponentialMovingAverage:
    """Maintain an EMA copy of all model parameters and persistent buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow: OrderedDict[str, Tensor] = OrderedDict(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        )
        self._backup: OrderedDict[str, Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights from the current model state."""
        state = model.state_dict()
        if set(state) != set(self.shadow):
            missing = sorted(set(self.shadow).symmetric_difference(state))
            raise KeyError(f"Model state changed after EMA construction: {missing}")
        self.num_updates += 1
        for name, current in state.items():
            shadow = self.shadow[name]
            current_value = current.detach().to(device=shadow.device)
            if torch.is_floating_point(shadow) or torch.is_complex(shadow):
                shadow.mul_(self.decay).add_(current_value, alpha=1.0 - self.decay)
            else:
                shadow.copy_(current_value)

    @torch.no_grad()
    def sync_from(self, model: nn.Module, *, reset_num_updates: bool = True) -> None:
        """Synchronize the complete shadow state from a newly initialized model.

        Fine-tuning constructs the trainer (and therefore its EMA object) before
        loading an initial checkpoint.  Calling this method immediately after
        that load prevents the EMA shadow from retaining the model's random
        pre-checkpoint initialization.  Resume should continue to use
        :meth:`load_state_dict` so the historical EMA and update count are
        restored instead.
        """

        if self._backup is not None:
            raise RuntimeError("Cannot synchronize EMA while model parameters are stored")
        self.shadow = OrderedDict(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        )
        if reset_num_updates:
            self.num_updates = 0

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA state into a model."""
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Back up a model before temporarily applying EMA weights."""
        if self._backup is not None:
            raise RuntimeError("EMA model state is already stored")
        self._backup = OrderedDict(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        )

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore model weights saved by :meth:`store`."""
        if self._backup is None:
            raise RuntimeError("EMA restore requested before store")
        model.load_state_dict(self._backup, strict=True)
        self._backup = None

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily evaluate a model using EMA parameters."""
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable EMA state."""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": OrderedDict((key, value.detach().clone()) for key, value in self.shadow.items()),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore EMA state, accepting both wrapped and raw state dictionaries."""
        if "shadow" in state_dict:
            shadow = state_dict["shadow"]
            self.decay = float(state_dict.get("decay", self.decay))
            self.num_updates = int(state_dict.get("num_updates", 0))
        else:
            shadow = state_dict
        if not isinstance(shadow, dict):
            raise TypeError("EMA shadow state must be a mapping")
        self.shadow = OrderedDict(
            (str(key), value.detach().clone()) for key, value in shadow.items() if isinstance(value, Tensor)
        )

    def to(self, device: torch.device | str) -> ExponentialMovingAverage:
        """Move shadow weights to a device and return self."""
        self.shadow = OrderedDict((key, value.to(device)) for key, value in self.shadow.items())
        return self
