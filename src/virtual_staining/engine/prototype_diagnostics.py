"""Read-only aggregation and atomic persistence for prototype diagnostics."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor


@dataclass(slots=True)
class _UsageAccumulator:
    total: Tensor
    weight: int
    observations: int
    source: str
    nonfinite_values: int = 0


@dataclass(slots=True)
class _AttentionVisualSample:
    selection_digest: str
    canonical_key: str
    stem: str
    split: str
    organ: str
    attention: dict[tuple[str, str], Tensor]


def _mapping_field(output: Any, name: str) -> Mapping[str, Any]:
    value = getattr(output, name, None)
    if isinstance(value, Mapping):
        return value
    if isinstance(output, Mapping):
        candidate = output.get(name)
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _iter_tensor_leaves(
    values: Mapping[str, Any],
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], Tensor]]:
    for key in sorted(values, key=str):
        value = values[key]
        path = (*prefix, str(key))
        if isinstance(value, Tensor):
            yield path, value
        elif isinstance(value, Mapping):
            yield from _iter_tensor_leaves(value, path)


def _task_and_diagnostic(path: tuple[str, ...]) -> tuple[str, str]:
    if not path:
        raise ValueError("Prototype diagnostic paths cannot be empty")
    if len(path) == 1:
        return "", path[0]
    return path[0], "/".join(path[1:])


def _normalized(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metadata_item(metadata: Mapping[str, Any], key: str, index: int) -> Any:
    value = metadata.get(key)
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return value.item()
        item = value[index]
        return item.item() if item.numel() == 1 else item
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _portable_component(value: str, *, maximum_length: int = 48) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "-", str(value).strip())
    cleaned = re.sub(r"\s+", "-", cleaned).strip(" .-")
    return (cleaned or "unnamed")[:maximum_length]


def _png_bytes(values: Tensor, size: int) -> bytes:
    """Render one attention plane with a fixed, dependency-free heat colormap."""

    array = values.detach().to(device="cpu", dtype=torch.float32).numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    grayscale = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(grayscale, mode="L").resize(
        (size, size), resample=Image.Resampling.BILINEAR
    )
    scaled = np.asarray(image, dtype=np.float32) / 255.0
    # A fixed blue -> cyan -> yellow -> red map. Fixed [0, 1] scaling preserves
    # comparability across samples and prototype indices.
    red = np.clip(1.5 * scaled, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(3.0 * scaled - 1.5), 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - scaled), 0.0, 1.0)
    rgb = np.rint(np.stack((red, green, blue), axis=-1) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return buffer.getvalue()


class PrototypeDiagnosticsAggregator:
    """Collect validation/inference prototype diagnostics without model mutation.

    Attention maps are preferred over already-reduced usage vectors because they
    retain the number of contributing pixels. Prototype banks are copied to CPU
    and checked for stability so outputs from different model states cannot be
    combined silently. The class never resets or otherwise edits a prototype bank.
    """

    schema_version = 2
    usage_filename = "prototype_usage.csv"
    dead_filename = "dead_prototypes.csv"
    similarity_filename = "prototype_similarity.npy"
    metadata_filename = "prototype_diagnostics.json"

    attention_visuals_dirname = "prototype_attention_visuals"
    attention_visuals_manifest = "manifest.json"

    def __init__(
        self,
        *,
        dead_threshold: float = 1e-4,
        attention_visuals_enabled: bool = False,
        attention_visual_count: int = 4,
        attention_visual_seed: int = 2026,
        attention_visual_size: int = 256,
    ) -> None:
        if not np.isfinite(dead_threshold) or dead_threshold < 0.0:
            raise ValueError("dead_threshold must be finite and nonnegative")
        if (
            isinstance(attention_visual_count, bool)
            or not isinstance(attention_visual_count, int)
            or attention_visual_count < 1
        ):
            raise ValueError("attention_visual_count must be a positive integer")
        if (
            isinstance(attention_visual_seed, bool)
            or not isinstance(attention_visual_seed, int)
            or attention_visual_seed < 0
        ):
            raise ValueError("attention_visual_seed must be a nonnegative integer")
        if (
            isinstance(attention_visual_size, bool)
            or not isinstance(attention_visual_size, int)
            or attention_visual_size < 8
            or attention_visual_size > 2048
        ):
            raise ValueError("attention_visual_size must be an integer in [8, 2048]")
        self.dead_threshold = float(dead_threshold)
        self.attention_visuals_enabled = bool(attention_visuals_enabled)
        self.attention_visual_count = int(attention_visual_count)
        self.attention_visual_seed = int(attention_visual_seed)
        self.attention_visual_size = int(attention_visual_size)
        self._usage: dict[tuple[str, str], _UsageAccumulator] = {}
        self._banks: dict[str, Tensor] = {}
        self._observed_outputs = 0
        self._attention_visual_samples: dict[str, _AttentionVisualSample] = {}
        self._attention_visual_seen: set[str] = set()

    @staticmethod
    def _reported_usage_vector(value: Tensor) -> tuple[Tensor, int]:
        detached = value.detach().to(device="cpu", dtype=torch.float64)
        if detached.numel() == 0:
            raise ValueError("Prototype usage tensors cannot be empty")
        if detached.ndim == 0:
            return detached.reshape(1), 1
        if detached.ndim == 1:
            return detached, 1
        vectors = detached.reshape(-1, detached.shape[-1])
        return vectors.sum(dim=0), vectors.shape[0]

    @staticmethod
    def _attention_usage_vector(
        value: Tensor,
        expected_prototypes: int | None,
    ) -> tuple[Tensor, int]:
        detached = value.detach().to(device="cpu", dtype=torch.float64)
        if detached.numel() == 0 or detached.ndim < 1:
            raise ValueError("Prototype attention tensors must be nonempty")
        if detached.ndim == 1:
            prototype_axis = 0
        elif expected_prototypes is not None and detached.shape[1] == expected_prototypes:
            prototype_axis = 1
        elif expected_prototypes is not None and detached.shape[-1] == expected_prototypes:
            prototype_axis = detached.ndim - 1
        else:
            # Both current model families expose BCHW attention, with prototypes
            # on axis one. Requiring a vector or rank >= 2 keeps this fallback
            # explicit instead of guessing from arbitrary equal-sized axes.
            prototype_axis = 1
        prototype_first = detached.movedim(prototype_axis, 0)
        vectors = prototype_first.reshape(prototype_first.shape[0], -1)
        return vectors.sum(dim=1), vectors.shape[1]

    def _accumulate(
        self,
        key: tuple[str, str],
        total: Tensor,
        weight: int,
        source: str,
    ) -> None:
        if weight < 1:
            raise ValueError("Prototype diagnostic weights must be positive")
        finite = torch.isfinite(total)
        nonfinite_values = int((~finite).sum().item())
        sanitized = torch.where(finite, total, torch.zeros_like(total))
        existing = self._usage.get(key)
        if existing is None:
            self._usage[key] = _UsageAccumulator(
                total=sanitized,
                weight=weight,
                observations=1,
                source=source,
                nonfinite_values=nonfinite_values,
            )
            return
        if existing.source != source:
            raise ValueError(
                f"Prototype diagnostic source changed for {key}: "
                f"{existing.source} versus {source}"
            )
        if existing.total.shape != sanitized.shape:
            raise ValueError(
                f"Prototype count changed for {key}: "
                f"{tuple(existing.total.shape)} versus {tuple(sanitized.shape)}"
            )
        existing.total += sanitized
        existing.weight += weight
        existing.observations += 1
        existing.nonfinite_values += nonfinite_values

    def _observe_banks(self, values: Mapping[str, Any]) -> None:
        for path, value in _iter_tensor_leaves(values):
            bank_key = "/".join(path)
            bank = value.detach().to(device="cpu", dtype=torch.float32).clone()
            if bank.ndim != 2 or min(bank.shape) < 1:
                raise ValueError(
                    f"Prototype bank {bank_key!r} must be a nonempty matrix, "
                    f"got {tuple(bank.shape)}"
                )
            if not torch.isfinite(bank).all():
                raise ValueError(f"Prototype bank {bank_key!r} contains nonfinite values")
            existing = self._banks.get(bank_key)
            if existing is None:
                self._banks[bank_key] = bank
            elif existing.shape != bank.shape or not torch.allclose(
                existing, bank, rtol=1e-5, atol=1e-7
            ):
                raise ValueError(
                    f"Prototype bank {bank_key!r} changed across observed outputs"
                )

    @staticmethod
    def _is_test_split(split: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", split.casefold()).strip("_")
        return normalized == "test" or normalized.endswith("_test") or normalized.startswith(
            "test_"
        )

    def _observe_attention_visuals(
        self,
        attention_values: Mapping[tuple[str, str], Tensor],
        metadata: Mapping[str, Any] | None,
    ) -> None:
        if not attention_values:
            return
        if metadata is None:
            raise ValueError(
                "Prototype attention visualization requires validation metadata"
            )
        for value in attention_values.values():
            if value.ndim != 4:
                raise ValueError(
                    "Prototype attention visualization requires BCHW attention tensors"
                )
        batch_sizes = {int(value.shape[0]) for value in attention_values.values()}
        if len(batch_sizes) != 1:
            raise ValueError("Prototype attention tensors have inconsistent batch sizes")
        batch_size = next(iter(batch_sizes))
        for index in range(batch_size):
            canonical_value = _metadata_item(metadata, "canonical_key", index)
            split_value = _metadata_item(metadata, "split", index)
            if canonical_value is None or not str(canonical_value).strip():
                raise ValueError(
                    "Prototype attention visualization requires canonical_key metadata"
                )
            if split_value is None or not str(split_value).strip():
                raise ValueError("Prototype attention visualization requires split metadata")
            canonical_key = str(canonical_value)
            split = str(split_value)
            if self._is_test_split(split):
                raise RuntimeError(
                    "Prototype attention visuals cannot observe test-split samples"
                )
            if canonical_key in self._attention_visual_seen:
                continue
            self._attention_visual_seen.add(canonical_key)
            digest = _sha256(
                f"{self.attention_visual_seed}\n{canonical_key}".encode()
            )
            candidate = _AttentionVisualSample(
                selection_digest=digest,
                canonical_key=canonical_key,
                stem=str(_metadata_item(metadata, "stem", index) or ""),
                split=split,
                organ=str(_metadata_item(metadata, "organ", index) or "unknown"),
                attention={
                    key: value[index].detach().to(device="cpu", dtype=torch.float32).clone()
                    for key, value in attention_values.items()
                },
            )
            self._attention_visual_samples[canonical_key] = candidate
            if len(self._attention_visual_samples) > self.attention_visual_count:
                worst_key = max(
                    self._attention_visual_samples,
                    key=lambda key: (
                        self._attention_visual_samples[key].selection_digest,
                        key,
                    ),
                )
                del self._attention_visual_samples[worst_key]

    def observe(
        self,
        output: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        capture_visuals: bool = True,
    ) -> bool:
        """Collect one output and return whether it contained any diagnostics."""

        usage_values = {
            _task_and_diagnostic(path): value
            for path, value in _iter_tensor_leaves(
                _mapping_field(output, "prototype_usage")
            )
        }
        attention_values = {
            _task_and_diagnostic(path): value
            for path, value in _iter_tensor_leaves(
                _mapping_field(output, "prototype_attention")
            )
        }
        if self.attention_visuals_enabled and capture_visuals:
            self._observe_attention_visuals(attention_values, metadata)
        banks = _mapping_field(output, "prototype_banks")
        self._observe_banks(banks)

        for key, attention in attention_values.items():
            reported = usage_values.get(key)
            expected = None
            if reported is not None:
                reported_vector, _ = self._reported_usage_vector(reported)
                expected = int(reported_vector.numel())
            total, weight = self._attention_usage_vector(attention, expected)
            self._accumulate(key, total, weight, "attention")
        for key, usage in usage_values.items():
            if key in attention_values:
                continue
            total, weight = self._reported_usage_vector(usage)
            self._accumulate(key, total, weight, "reported_usage")

        found = bool(usage_values or attention_values or banks)
        if found:
            self._observed_outputs += 1
        return found

    def _resolve_bank_key(self, task: str, diagnostic: str) -> str:
        if diagnostic in self._banks:
            return diagnostic
        candidates: list[str] = []
        diagnostic_segments = diagnostic.split("/")
        bank_kind = diagnostic_segments[-1]
        prefix = "/".join(diagnostic_segments[:-1])
        if bank_kind in {"marker", "task"} and task:
            requested = f"{prefix + '/' if prefix else ''}{bank_kind}/{task}"
            candidates.extend(
                key for key in self._banks if _normalized(key) == _normalized(requested)
            )
            candidates.extend(
                key for key in self._banks if _normalized(key) == _normalized(task)
            )
        if bank_kind == "shared":
            requested = f"{prefix + '/' if prefix else ''}shared"
            candidates.extend(
                key for key in self._banks if _normalized(key) == _normalized(requested)
            )
        unique = sorted(set(candidates))
        return unique[0] if len(unique) == 1 else ""

    def _usage_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for task, diagnostic in sorted(self._usage):
            accumulator = self._usage[(task, diagnostic)]
            mean = accumulator.total / float(accumulator.weight)
            bank_key = self._resolve_bank_key(task, diagnostic)
            for index, value in enumerate(mean.tolist()):
                usage = float(value)
                rows.append(
                    {
                        "task": task,
                        "diagnostic": diagnostic,
                        "bank_key": bank_key,
                        "prototype_index": index,
                        "mean_usage": usage,
                        "dead": usage <= self.dead_threshold,
                        "observations": accumulator.observations,
                        "attention_elements": accumulator.weight,
                        "source": accumulator.source,
                        "nonfinite_values": accumulator.nonfinite_values,
                    }
                )
        return rows

    def _similarity(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        bank_records: list[dict[str, Any]] = []
        all_vectors: list[Tensor] = []
        offset = 0
        for bank_key in sorted(self._banks):
            bank = self._banks[bank_key]
            count, dimension = (int(bank.shape[0]), int(bank.shape[1]))
            norms = torch.linalg.vector_norm(bank.float(), dim=1)
            bank_records.append(
                {
                    "bank_key": bank_key,
                    "offset": offset,
                    "count": count,
                    "dimension": dimension,
                    "zero_norm_prototypes": int((norms <= 1e-12).sum().item()),
                }
            )
            all_vectors.append(bank)
            offset += count

        similarity = np.full((offset, offset), np.nan, dtype=np.float32)
        dimensions = sorted({int(bank.shape[1]) for bank in all_vectors})
        for dimension in dimensions:
            indices: list[int] = []
            vectors: list[Tensor] = []
            for record, bank in zip(bank_records, all_vectors, strict=True):
                if int(record["dimension"]) != dimension:
                    continue
                start = int(record["offset"])
                count = int(record["count"])
                indices.extend(range(start, start + count))
                vectors.append(bank)
            normalized = torch.nn.functional.normalize(
                torch.cat(vectors, dim=0).float(), dim=1, eps=1e-12
            )
            block = (normalized @ normalized.transpose(0, 1)).cpu().numpy()
            similarity[np.ix_(indices, indices)] = block.astype(np.float32, copy=False)
        return similarity, bank_records

    @staticmethod
    def _csv_bytes(rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> bytes:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})
        return buffer.getvalue().encode("utf-8")

    @staticmethod
    def _npy_bytes(values: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        return buffer.getvalue()

    @staticmethod
    def _write_atomically(output_dir: Path, payloads: Mapping[str, bytes]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_paths: dict[str, Path] = {}
        try:
            for filename, payload in payloads.items():
                relative = Path(filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Artifact path must remain relative: {filename}")
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(
                    f".{destination.name}.{uuid.uuid4().hex[:8]}.tmp"
                )
                temporary.write_bytes(payload)
                temporary_paths[filename] = temporary
            for filename in payloads:
                temporary_paths[filename].replace(output_dir / filename)
        finally:
            for temporary in temporary_paths.values():
                temporary.unlink(missing_ok=True)

    def _attention_visual_payloads(self) -> tuple[dict[str, bytes], dict[str, Any]]:
        if not self.attention_visuals_enabled:
            return {}, {
                "enabled": False,
                "status": "disabled",
                "png_count": 0,
            }

        payloads: dict[str, bytes] = {}
        image_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        selected = sorted(
            self._attention_visual_samples.values(),
            key=lambda value: (value.selection_digest, value.canonical_key),
        )
        for sample_index, sample in enumerate(selected):
            sample_dir = f"sample_{sample_index:03d}_{sample.selection_digest[:12]}"
            sample_rows.append(
                {
                    "sample_index": sample_index,
                    "selection_digest": sample.selection_digest,
                    "canonical_key": sample.canonical_key,
                    "stem": sample.stem,
                    "split": sample.split,
                    "organ": sample.organ,
                    "directory": sample_dir,
                }
            )
            for (task, diagnostic), attention in sorted(sample.attention.items()):
                for prototype_index in range(int(attention.shape[0])):
                    filename = (
                        f"task_{_portable_component(task, maximum_length=20)}__"
                        f"diag_{_portable_component(diagnostic, maximum_length=24)}__"
                        f"prototype_{prototype_index:03d}.png"
                    )
                    relative = Path(self.attention_visuals_dirname) / sample_dir / filename
                    payload = _png_bytes(
                        attention[prototype_index], self.attention_visual_size
                    )
                    portable_path = relative.as_posix()
                    payloads[portable_path] = payload
                    image_rows.append(
                        {
                            "sample_index": sample_index,
                            "task": task,
                            "diagnostic": diagnostic,
                            "prototype_index": prototype_index,
                            "relative_path": portable_path,
                            "width": self.attention_visual_size,
                            "height": self.attention_visual_size,
                            "sha256": _sha256(payload),
                        }
                    )
        status = "complete" if image_rows else "no_attention"
        manifest = {
            "schema_version": 1,
            "status": status,
            "selection": {
                "method": "smallest_sha256_of_seed_and_canonical_key",
                "seed": self.attention_visual_seed,
                "maximum_samples": self.attention_visual_count,
                "candidate_samples": len(self._attention_visual_seen),
                "selected_samples": len(selected),
            },
            "split_policy": "validator_input_only_test_splits_rejected",
            "rendering": {
                "size": self.attention_visual_size,
                "value_range": [0.0, 1.0],
                "format": "RGB PNG",
                "colormap": "fixed_blue_cyan_yellow_red_v1",
            },
            "samples": sample_rows,
            "images": image_rows,
        }
        manifest_payload = json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        manifest_path = (
            Path(self.attention_visuals_dirname) / self.attention_visuals_manifest
        ).as_posix()
        payloads[manifest_path] = manifest_payload
        return payloads, {
            "enabled": True,
            "status": status,
            "directory": self.attention_visuals_dirname,
            "manifest": manifest_path,
            "manifest_sha256": _sha256(manifest_payload),
            "candidate_samples": len(self._attention_visual_seen),
            "selected_samples": len(selected),
            "png_count": len(image_rows),
        }

    def write(
        self,
        output_dir: str | Path,
        *,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        """Write normalized diagnostics, raising on empty input by default."""

        has_data = bool(self._usage or self._banks)
        if not has_data and not allow_empty:
            raise RuntimeError(
                "No prototype attention, usage, or banks were observed; "
                "set allow_empty=True to record an explicit no-data result"
            )

        usage_rows = self._usage_rows()
        dead_rows = [row for row in usage_rows if bool(row["dead"])]
        similarity, bank_records = self._similarity()
        usage_fields = (
            "task",
            "diagnostic",
            "bank_key",
            "prototype_index",
            "mean_usage",
            "dead",
            "observations",
            "attention_elements",
            "source",
            "nonfinite_values",
        )
        usage_payload = self._csv_bytes(usage_rows, usage_fields)
        dead_payload = self._csv_bytes(dead_rows, usage_fields)
        similarity_payload = self._npy_bytes(similarity)
        visual_payloads, visual_metadata = self._attention_visual_payloads()
        warnings: list[str] = []
        if not self._usage:
            warnings.append("no_usage_or_attention_observed")
        if not self._banks:
            warnings.append("no_prototype_banks_observed")
        unresolved = sum(1 for row in usage_rows if not row["bank_key"])
        if unresolved:
            warnings.append("some_usage_rows_have_no_unique_bank_mapping")
        finite_similarity = similarity[np.isfinite(similarity)]
        metadata: dict[str, Any] = {
            "schema_version": self.schema_version,
            "status": "complete" if self._usage and self._banks else ("partial" if has_data else "no_data"),
            "dead_threshold": self.dead_threshold,
            "observed_outputs": self._observed_outputs,
            "usage_diagnostics": len(self._usage),
            "usage_rows": len(usage_rows),
            "dead_prototypes": len(dead_rows),
            "unresolved_bank_rows": unresolved,
            "prototype_banks": bank_records,
            "prototype_count": int(similarity.shape[0]),
            "similarity_shape": list(similarity.shape),
            "finite_similarity_values": int(finite_similarity.size),
            "nonfinite_similarity_values": int(np.isnan(similarity).sum()),
            "reset_performed": False,
            "attention_visuals": visual_metadata,
            "warnings": warnings,
            "artifacts": {
                self.usage_filename: {"sha256": _sha256(usage_payload)},
                self.dead_filename: {"sha256": _sha256(dead_payload)},
                self.similarity_filename: {"sha256": _sha256(similarity_payload)},
            },
        }
        if visual_metadata["enabled"]:
            metadata["artifacts"][str(visual_metadata["manifest"])] = {
                "sha256": str(visual_metadata["manifest_sha256"])
            }
        metadata_payload = json.dumps(
            metadata, ensure_ascii=False, indent=2, sort_keys=True
        ).encode("utf-8")
        self._write_atomically(
            Path(output_dir).resolve(),
            {
                self.usage_filename: usage_payload,
                self.dead_filename: dead_payload,
                self.similarity_filename: similarity_payload,
                self.metadata_filename: metadata_payload,
                **visual_payloads,
            },
        )
        return metadata


def write_prototype_diagnostics(
    outputs: Iterable[Any],
    output_dir: str | Path,
    *,
    dead_threshold: float = 1e-4,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Aggregate an output iterable and persist its read-only diagnostics."""

    aggregator = PrototypeDiagnosticsAggregator(dead_threshold=dead_threshold)
    for output in outputs:
        aggregator.observe(output)
    return aggregator.write(output_dir, allow_empty=allow_empty)


__all__ = ["PrototypeDiagnosticsAggregator", "write_prototype_diagnostics"]
