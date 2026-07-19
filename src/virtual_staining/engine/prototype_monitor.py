"""Epoch-level prototype usage and dead-prototype diagnostics."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _usage_mapping(output: Any) -> Mapping[str, Any]:
    usage = getattr(output, "prototype_usage", None)
    if isinstance(usage, Mapping):
        return usage
    if isinstance(output, Mapping):
        candidate = output.get("prototype_usage")
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _iter_usage_tensors(
    values: Mapping[str, Any],
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[str, Tensor]]:
    for key in sorted(values, key=str):
        value = values[key]
        path = (*prefix, str(key))
        if isinstance(value, Tensor):
            yield "/".join(path), value
        elif isinstance(value, Mapping):
            yield from _iter_usage_tensors(value, path)


class PrototypeUsageMonitor:
    """Aggregate detached prototype usage without influencing optimization."""

    schema_version = 1

    def __init__(self, *, dead_threshold: float = 1e-4) -> None:
        if dead_threshold < 0.0:
            raise ValueError("dead_threshold cannot be negative")
        self.dead_threshold = float(dead_threshold)
        self.output_dir: Path | None = None
        self.history: list[dict[str, Any]] = []
        self.reset_events: list[dict[str, Any]] = []
        self._dead_streaks: dict[str, int] = {}
        self._active_epoch: int | None = None
        self._usage_sums: dict[str, Tensor] = {}
        self._observation_counts: dict[str, int] = {}
        self._observed_outputs = 0
        self._nonfinite_values = 0

    def bind_output_dir(self, output_dir: str | Path) -> None:
        """Select the run artifact directory and restore prior epoch records."""

        self.output_dir = Path(output_dir).resolve()
        json_path = self.output_dir / "prototype_usage.json"
        if self.history or not json_path.is_file():
            return
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != self.schema_version:
            raise ValueError(f"Unsupported prototype monitor schema in {json_path}")
        epochs = payload.get("epochs", [])
        if not isinstance(epochs, list):
            raise ValueError(f"Prototype monitor epochs must be a list in {json_path}")
        self.history = [dict(item) for item in epochs if isinstance(item, Mapping)]
        reset_events = payload.get("reset_events", [])
        if not isinstance(reset_events, list):
            raise ValueError(
                f"Prototype monitor reset_events must be a list in {json_path}"
            )
        self.reset_events = [
            dict(item) for item in reset_events if isinstance(item, Mapping)
        ]
        saved_streaks = payload.get("dead_streaks")
        if saved_streaks is None:
            self._dead_streaks = self._derive_dead_streaks(self.history)
        elif not isinstance(saved_streaks, Mapping):
            raise ValueError(
                f"Prototype monitor dead_streaks must be a mapping in {json_path}"
            )
        else:
            self._dead_streaks = self._validated_dead_streaks(saved_streaks)

    def start_epoch(self, epoch: int) -> None:
        """Reset accumulators for a new epoch."""

        self._active_epoch = int(epoch)
        self._usage_sums = {}
        self._observation_counts = {}
        self._observed_outputs = 0
        self._nonfinite_values = 0

    @staticmethod
    def _prototype_vector(value: Tensor) -> Tensor:
        detached = value.detach().float().cpu()
        if detached.ndim == 0:
            return detached.reshape(1)
        if detached.ndim == 1:
            return detached
        return detached.reshape(-1, detached.shape[-1]).mean(dim=0)

    def observe(self, output: Any) -> bool:
        """Observe one forward output; return whether any usage was present."""

        if self._active_epoch is None:
            raise RuntimeError("start_epoch must be called before observing prototype usage")
        found = False
        for diagnostic, value in _iter_usage_tensors(_usage_mapping(output)):
            vector = self._prototype_vector(value)
            finite = torch.isfinite(vector)
            self._nonfinite_values += int((~finite).sum().item())
            vector = torch.where(finite, vector, torch.zeros_like(vector))
            previous = self._usage_sums.get(diagnostic)
            if previous is not None and previous.shape != vector.shape:
                raise ValueError(
                    f"Prototype usage shape changed for {diagnostic}: "
                    f"{tuple(previous.shape)} versus {tuple(vector.shape)}"
                )
            self._usage_sums[diagnostic] = vector if previous is None else previous + vector
            self._observation_counts[diagnostic] = (
                self._observation_counts.get(diagnostic, 0) + 1
            )
            found = True
        if found:
            self._observed_outputs += 1
        return found

    @staticmethod
    def _entropy(usage: Tensor) -> float:
        nonnegative = usage.clamp_min(0.0)
        total = nonnegative.sum()
        if float(total) <= 0.0:
            return 0.0
        probabilities = nonnegative / total
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        return float(entropy)

    def finalize_epoch(self, epoch: int | None = None) -> dict[str, int | float]:
        """Finalize an epoch, persist its rows, and return a compact summary."""

        if self._active_epoch is None:
            raise RuntimeError("No active prototype monitoring epoch")
        resolved_epoch = self._active_epoch if epoch is None else int(epoch)
        if resolved_epoch != self._active_epoch:
            raise ValueError(
                f"Cannot finalize epoch {resolved_epoch}; active epoch is {self._active_epoch}"
            )

        rows: list[dict[str, Any]] = []
        entropies: list[float] = []
        dead_count = 0
        total_count = 0
        for diagnostic in sorted(self._usage_sums):
            observations = self._observation_counts[diagnostic]
            mean_usage = self._usage_sums[diagnostic] / float(observations)
            entropies.append(self._entropy(mean_usage))
            for index, value in enumerate(mean_usage.tolist()):
                usage = float(value)
                dead = usage <= self.dead_threshold
                dead_count += int(dead)
                total_count += 1
                rows.append(
                    {
                        "epoch": resolved_epoch,
                        "diagnostic": diagnostic,
                        "prototype_index": index,
                        "mean_usage": usage,
                        "dead": dead,
                        "observations": observations,
                    }
                )

        summary: dict[str, int | float] = {
            "epoch": resolved_epoch,
            "observed_outputs": self._observed_outputs,
            "diagnostic_count": len(self._usage_sums),
            "total_prototypes": total_count,
            "dead_prototypes": dead_count,
            "dead_fraction": dead_count / total_count if total_count else 0.0,
            "mean_entropy": float(sum(entropies) / len(entropies)) if entropies else 0.0,
            "nonfinite_values": self._nonfinite_values,
        }
        record: dict[str, Any] = {**summary, "rows": rows}
        self.history = [
            item for item in self.history if int(item.get("epoch", -1)) != resolved_epoch
        ]
        self.history.append(record)
        self.history.sort(key=lambda item: int(item.get("epoch", -1)))
        observed_keys: set[str] = set()
        for row in rows:
            streak_key = self._streak_key(
                str(row["diagnostic"]), int(row["prototype_index"])
            )
            observed_keys.add(streak_key)
            self._dead_streaks[streak_key] = (
                self._dead_streaks.get(streak_key, 0) + 1
                if bool(row["dead"])
                else 0
            )
        for stale_key in set(self._dead_streaks).difference(observed_keys):
            self._dead_streaks[stale_key] = 0
        self._active_epoch = None
        self.persist()
        return summary

    @staticmethod
    def _streak_key(diagnostic: str, prototype_index: int) -> str:
        return f"{diagnostic}\u241f{int(prototype_index)}"

    @staticmethod
    def _validated_dead_streaks(values: Mapping[str, Any]) -> dict[str, int]:
        result: dict[str, int] = {}
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Prototype dead streaks must be nonnegative integers")
            result[str(key)] = int(value)
        return result

    @classmethod
    def _derive_dead_streaks(
        cls, history: list[dict[str, Any]]
    ) -> dict[str, int]:
        streaks: dict[str, int] = {}
        for epoch_record in sorted(
            history, key=lambda item: int(item.get("epoch", -1))
        ):
            rows = epoch_record.get("rows", [])
            if not isinstance(rows, list):
                continue
            observed: set[str] = set()
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                key = cls._streak_key(
                    str(row.get("diagnostic", "")),
                    int(row.get("prototype_index", -1)),
                )
                observed.add(key)
                streaks[key] = streaks.get(key, 0) + 1 if row.get("dead") else 0
            for stale_key in set(streaks).difference(observed):
                streaks[stale_key] = 0
        return streaks

    def latest_rows_with_streaks(self) -> list[dict[str, Any]]:
        """Return the last completed epoch rows with consecutive-dead counts."""

        if not self.history:
            return []
        rows = self.history[-1].get("rows", [])
        if not isinstance(rows, list):
            return []
        result: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            item = dict(row)
            item["dead_streak"] = self._dead_streaks.get(
                self._streak_key(
                    str(item.get("diagnostic", "")),
                    int(item.get("prototype_index", -1)),
                ),
                0,
            )
            result.append(item)
        return result

    def record_reset_event(
        self,
        *,
        epoch: int,
        seed: int,
        reset_rows: list[Mapping[str, Any]],
        skipped_rows: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Persist a deterministic training-only reset and clear reset streaks."""

        normalized_resets = [dict(row) for row in reset_rows]
        normalized_skips = [dict(row) for row in skipped_rows]
        for row in normalized_resets:
            diagnostics = row.get("diagnostics", [])
            index = int(row["prototype_index"])
            for diagnostic in diagnostics:
                self._dead_streaks[self._streak_key(str(diagnostic), index)] = 0
        event = {
            "epoch": int(epoch),
            "seed": int(seed),
            "reset_rows": normalized_resets,
            "skipped_rows": normalized_skips,
        }
        self.reset_events = [
            item
            for item in self.reset_events
            if int(item.get("epoch", -1)) != int(epoch)
        ]
        self.reset_events.append(event)
        self.reset_events.sort(key=lambda item: int(item.get("epoch", -1)))
        self.persist()
        return event

    def persist(self) -> None:
        """Atomically write JSON history and normalized CSV rows when bound."""

        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "prototype_usage.json"
        json_temporary = json_path.with_name(f"{json_path.name}.tmp")
        payload = {
            "schema_version": self.schema_version,
            "dead_threshold": self.dead_threshold,
            "epochs": self.history,
            "dead_streaks": self._dead_streaks,
            "reset_events": self.reset_events,
        }
        json_temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        json_temporary.replace(json_path)

        csv_path = self.output_dir / "prototype_usage.csv"
        csv_temporary = csv_path.with_name(f"{csv_path.name}.tmp")
        fieldnames = (
            "epoch",
            "diagnostic",
            "prototype_index",
            "mean_usage",
            "dead",
            "observations",
        )
        with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for epoch_record in self.history:
                for row in epoch_record.get("rows", []):
                    writer.writerow({key: row[key] for key in fieldnames})
        csv_temporary.replace(csv_path)

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-compatible history for checkpoint provenance."""

        return {
            "schema_version": self.schema_version,
            "dead_threshold": self.dead_threshold,
            "history": self.history,
            "dead_streaks": self._dead_streaks,
            "reset_events": self.reset_events,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore checkpointed history and consecutive-dead state exactly."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("Prototype monitor state must be a mapping")
        if int(state_dict.get("schema_version", -1)) != self.schema_version:
            raise ValueError("Unsupported prototype monitor checkpoint schema")
        saved_threshold = float(state_dict.get("dead_threshold", self.dead_threshold))
        if saved_threshold != self.dead_threshold:
            raise ValueError(
                "Prototype monitor dead_threshold differs from the checkpoint"
            )
        history = state_dict.get("history", [])
        reset_events = state_dict.get("reset_events", [])
        if not isinstance(history, list) or not isinstance(reset_events, list):
            raise ValueError("Prototype monitor history and reset_events must be lists")
        self.history = [dict(item) for item in history if isinstance(item, Mapping)]
        self.reset_events = [
            dict(item) for item in reset_events if isinstance(item, Mapping)
        ]
        streaks = state_dict.get("dead_streaks")
        self._dead_streaks = (
            self._derive_dead_streaks(self.history)
            if streaks is None
            else self._validated_dead_streaks(streaks)
        )
        self._active_epoch = None
        self._usage_sums = {}
        self._observation_counts = {}
        self._observed_outputs = 0
        self._nonfinite_values = 0
        self.persist()
