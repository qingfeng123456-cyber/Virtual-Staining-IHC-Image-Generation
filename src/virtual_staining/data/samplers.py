"""Deterministic samplers for reproducible grouped training."""

from __future__ import annotations

import bisect
import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from torch.utils.data import Sampler

_TRAIN_SPLITS = {"train", "official_train", "final_train"}


class EpochShuffleSampler(Sampler[int]):
    """Stable random permutation whose order can be advanced by epoch."""

    def __init__(self, data_source: Sequence[object], seed: int = 2026) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.data_source)))
        random.Random(self.seed + self.epoch).shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.data_source)


class GroupInterleavedSampler(Sampler[int]):
    """Shuffle groups and interleave their members without losing reproducibility."""

    def __init__(self, group_ids: Sequence[str], seed: int = 2026) -> None:
        self.group_ids = tuple(str(group) for group in group_ids)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(self.group_ids):
            grouped[group].append(index)
        rng = random.Random(self.seed + self.epoch)
        group_order = sorted(grouped, key=str.casefold)
        rng.shuffle(group_order)
        for group in group_order:
            members = grouped[group]
            rng.shuffle(members)
        remaining = True
        offset = 0
        while remaining:
            remaining = False
            for group in group_order:
                members = grouped[group]
                if offset < len(members):
                    remaining = True
                    yield members[offset]
            offset += 1

    def __len__(self) -> int:
        return len(self.group_ids)


class ActivityStratifiedSampler(Sampler[int]):
    """Quantile-balanced sampling from precomputed training-only activity values."""

    _STATE_VERSION = 1

    def __init__(
        self,
        activities: Sequence[float],
        *,
        splits: Sequence[str],
        sample_indices: Sequence[int] | None = None,
        num_bins: int = 4,
        seed: int = 2026,
        num_samples: int | None = None,
    ) -> None:
        if len(activities) == 0:
            raise ValueError("activities cannot be empty")
        if len(activities) != len(splits):
            raise ValueError("activities and splits must have equal length")
        if sample_indices is None:
            resolved_indices = tuple(range(len(activities)))
        else:
            if len(sample_indices) != len(activities):
                raise ValueError(
                    "sample_indices must have the same length as activities"
                )
            resolved_indices = tuple(sample_indices)
            if any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in resolved_indices
            ):
                raise ValueError("sample_indices must contain nonnegative integers")
            if len(set(resolved_indices)) != len(resolved_indices):
                raise ValueError("sample_indices must be unique")
        normalized_splits = tuple(str(value).strip().casefold() for value in splits)
        invalid_splits = sorted(set(normalized_splits).difference(_TRAIN_SPLITS))
        if invalid_splits:
            raise ValueError(
                "Activity sampling accepts training rows only; rejected split(s): "
                + ", ".join(invalid_splits)
            )
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 1:
            raise ValueError("num_bins must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if num_samples is not None and (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples < 1
        ):
            raise ValueError("num_samples must be a positive integer when provided")
        values = tuple(float(value) for value in activities)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("activities must contain finite numeric values")

        self.activities = values
        self.splits = normalized_splits
        self.sample_indices = resolved_indices
        self.num_bins = num_bins
        self.seed = seed
        self.num_samples = num_samples or len(values)
        sorted_values = sorted(values)
        boundaries: list[float] = []
        for quantile_index in range(1, num_bins):
            position = (len(sorted_values) - 1) * quantile_index / num_bins
            lower = math.floor(position)
            upper = math.ceil(position)
            fraction = position - lower
            boundary = (
                sorted_values[lower] * (1.0 - fraction)
                + sorted_values[upper] * fraction
            )
            if not boundaries or boundary > boundaries[-1]:
                boundaries.append(boundary)
        self.quantile_boundaries = tuple(boundaries)
        raw_assignments = tuple(
            bisect.bisect_left(self.quantile_boundaries, value) for value in values
        )
        populated_ids = sorted(set(raw_assignments))
        compact_ids = {raw_id: index for index, raw_id in enumerate(populated_ids)}
        assignments = tuple(compact_ids[raw_id] for raw_id in raw_assignments)
        bins: dict[int, list[int]] = defaultdict(list)
        for sample_index, bin_id in zip(resolved_indices, assignments, strict=True):
            bins[bin_id].append(sample_index)
        self._bins = tuple(tuple(bins[key]) for key in sorted(bins))
        self.bin_assignments = assignments
        self.bin_by_sample_index = dict(zip(resolved_indices, assignments, strict=True))
        self.epoch = 0
        self._position = 0
        fingerprint_payload = "|".join(
            [
                str(num_bins),
                str(self.num_samples),
                *(str(index) for index in resolved_indices),
                *normalized_splits,
                *(repr(v) for v in values),
            ]
        )
        self._fingerprint = hashlib.sha256(fingerprint_payload.encode()).hexdigest()

    @classmethod
    def from_manifest(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        activity_key: str = "dapi_activity",
        split_key: str = "split",
        sample_indices: Sequence[int] | None = None,
        num_bins: int = 4,
        seed: int = 2026,
        num_samples: int | None = None,
    ) -> ActivityStratifiedSampler:
        """Read precomputed scalar fields only; this method performs no image I/O."""

        missing = [index for index, row in enumerate(rows) if activity_key not in row]
        if missing:
            raise ValueError(
                f"Manifest rows lack precomputed {activity_key!r}: {missing[:10]}"
            )
        return cls(
            [float(row[activity_key]) for row in rows],
            splits=[str(row.get(split_key, "")) for row in rows],
            sample_indices=sample_indices,
            num_bins=num_bins,
            seed=seed,
            num_samples=num_samples,
        )

    @property
    def bin_counts(self) -> tuple[int, ...]:
        return tuple(len(values) for values in self._bins)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch
        self._position = 0

    def _epoch_order(self) -> list[int]:
        rng = random.Random(self.seed + self.epoch)
        pools = [list(values) for values in self._bins]
        for pool in pools:
            rng.shuffle(pool)
        bin_order = list(range(len(pools)))
        rng.shuffle(bin_order)
        cursors = [0] * len(pools)
        order: list[int] = []
        for sample_index in range(self.num_samples):
            bin_index = bin_order[sample_index % len(bin_order)]
            pool = pools[bin_index]
            cursor = cursors[bin_index]
            if cursor >= len(pool):
                rng.shuffle(pool)
                cursor = 0
            order.append(pool[cursor])
            cursors[bin_index] = cursor + 1
        return order

    def __iter__(self) -> Iterator[int]:
        order = self._epoch_order()
        while self._position < len(order):
            index = order[self._position]
            self._position += 1
            yield index

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "fingerprint": self._fingerprint,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self._position,
            "num_samples": self.num_samples,
            "num_bins": self.num_bins,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", -1)) != self._STATE_VERSION:
            raise ValueError("Unsupported activity sampler state version")
        if str(state.get("fingerprint", "")) != self._fingerprint:
            raise ValueError("Activity sampler state does not match current training data")
        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("Activity sampler state seed does not match")
        if int(state.get("num_samples", -1)) != self.num_samples:
            raise ValueError("Activity sampler state num_samples does not match")
        if int(state.get("num_bins", -1)) != self.num_bins:
            raise ValueError("Activity sampler state num_bins does not match")
        epoch = int(state.get("epoch", -1))
        position = int(state.get("position", -1))
        if epoch < 0 or position < 0 or position > self.num_samples:
            raise ValueError("Activity sampler state has invalid epoch or position")
        self.epoch = epoch
        self._position = position
