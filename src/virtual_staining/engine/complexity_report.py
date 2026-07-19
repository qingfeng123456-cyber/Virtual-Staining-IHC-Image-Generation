"""Evidence-only complexity and runtime reporting for Performance V2 runs."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(frozen=True, slots=True)
class ComplexityRunInput:
    """One named run directory and an optional explicit checkpoint."""

    stage: str
    run_dir: str | Path
    checkpoint: str | Path | None = None


def _measurement(
    value: Any,
    *,
    source: str | None,
    reason: str | None = None,
    unit: str | None = None,
    derivation: str | None = None,
) -> dict[str, Any]:
    if value is None and not reason:
        raise ValueError("A null measurement requires a reason")
    result: dict[str, Any] = {
        "value": value,
        "source": source,
        "reason": reason if value is None else None,
    }
    if unit is not None:
        result["unit"] = unit
    if derivation is not None:
        result["derivation"] = derivation
    return result


def _numeric(value: Any, *, minimum: float | None = None) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        return None
    return int(value) if isinstance(value, int) else number


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Provenance:
    def __init__(self) -> None:
        self._records: dict[Path, dict[str, Any]] = {}

    def source(self, path: Path, role: str) -> dict[str, Any]:
        resolved = path.resolve()
        if not resolved.is_file():
            return {
                "path": str(resolved),
                "role": role,
                "exists": False,
                "size_bytes": None,
                "sha256": None,
                "reason": "file_not_found",
            }
        existing = self._records.get(resolved)
        if existing is None:
            existing = {
                "path": str(resolved),
                "role": role,
                "exists": True,
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
                "reason": None,
            }
            self._records[resolved] = existing
        return dict(existing)

    def records(self) -> list[dict[str, Any]]:
        return [dict(self._records[path]) for path in sorted(self._records, key=str)]

    def digest(self) -> str | None:
        if not self._records:
            return None
        digest = hashlib.sha256()
        for record in self.records():
            digest.update(str(record["path"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(record["size_bytes"]).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(record["sha256"]).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _nested_value(values: Mapping[str, Any], *path: str) -> Any:
    current: Any = values
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _contains_runtime_key(value: Any, keys: set[str]) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in keys:
                return True
            if _contains_runtime_key(child, keys):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_runtime_key(child, keys) for child in value)
    return False


def _configured_features(config: Mapping[str, Any] | None, source: str | None) -> dict[str, Any]:
    if config is None:
        missing = _measurement(
            None,
            source=source,
            reason="effective configuration unavailable",
        )
        return {"context": dict(missing), "ensemble": dict(missing)}

    context_value = _nested_value(config, "model", "context", "enabled")
    if not isinstance(context_value, bool):
        context_value = _nested_value(config, "context", "enabled")
    context = (
        _measurement(context_value, source=source)
        if isinstance(context_value, bool)
        else _measurement(
            None,
            source=source,
            reason="context feature flag absent from effective configuration",
        )
    )

    ensemble_enabled = _nested_value(config, "ensemble", "enabled")
    if not isinstance(ensemble_enabled, bool):
        checkpoints = _nested_value(config, "inference", "ensemble_checkpoints")
        weights = _nested_value(config, "ensemble", "weights")
        if isinstance(checkpoints, Sequence) and not isinstance(checkpoints, (str, bytes)):
            ensemble_enabled = bool(checkpoints)
        elif isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
            ensemble_enabled = bool(weights)
    ensemble = (
        _measurement(ensemble_enabled, source=source)
        if isinstance(ensemble_enabled, bool)
        else _measurement(
            None,
            source=source,
            reason="ensemble feature flag or member list absent from effective configuration",
        )
    )
    return {"context": context, "ensemble": ensemble}


def _runtime_feature_evidence(
    configured: Mapping[str, Any],
    metrics: Any,
    metrics_source: str | None,
) -> dict[str, Any]:
    definitions = {
        "context": {
            "context_measured",
            "context_available",
            "context_valid_fraction",
            "context_runtime_seconds",
        },
        "ensemble": {
            "ensemble_measured",
            "ensemble_member_count",
            "ensemble_size",
            "ensemble_runtime_seconds",
        },
    }
    result: dict[str, Any] = {}
    for feature, keys in definitions.items():
        configured_value = configured[feature]["value"]
        if configured_value is False:
            result[feature] = _measurement(
                False,
                source=configured[feature]["source"],
                derivation="feature explicitly disabled; no feature-on runtime was measured",
            )
        elif metrics is not None and _contains_runtime_key(metrics, keys):
            result[feature] = _measurement(
                True,
                source=metrics_source,
                derivation="explicit runtime evidence field present",
            )
        else:
            result[feature] = _measurement(
                None,
                source=metrics_source,
                reason=f"no explicit {feature} runtime evidence field found",
            )
    return result


def _model_summary(payload: Any, source: str | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        reason = "model_stats.json unavailable"
        return {
            "parameters": _measurement(None, source=source, reason=reason, unit="parameters"),
            "trainable_parameters": _measurement(
                None, source=source, reason=reason, unit="parameters"
            ),
            "approximate_macs": _measurement(None, source=source, reason=reason, unit="MACs"),
        }

    fields = {
        "parameters": ("parameters", False),
        "trainable_parameters": ("parameters", False),
        "approximate_macs": ("MACs", True),
    }
    summary: dict[str, Any] = {}
    for key, (unit, approximate) in fields.items():
        value = _numeric(payload.get(key), minimum=0.0)
        summary[key] = (
            _measurement(
                value,
                source=source,
                unit=unit,
                derivation="model profiler output" if approximate else None,
            )
            if value is not None
            else _measurement(
                None,
                source=source,
                reason=f"{key} missing or invalid in model_stats.json",
                unit=unit,
            )
        )
    return summary


def _metric_sections(payload: Any) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if payload is None:
        return [], []
    entries: list[Mapping[str, Any]]
    if isinstance(payload, list):
        entries = [item for item in payload if isinstance(item, Mapping)]
    elif isinstance(payload, Mapping):
        entries = [payload]
    else:
        raise ValueError("metrics.json must contain an object or a list of objects")

    training: list[Mapping[str, Any]] = []
    validation: list[Mapping[str, Any]] = []
    for entry in entries:
        train = entry.get("train")
        if isinstance(train, Mapping):
            training.append(train)
        val = entry.get("validation")
        if isinstance(val, Mapping):
            validation.append(val)
        elif any(key in entry for key in ("domains", "tta", "duration_seconds", "records")):
            validation.append(entry)
    return training, validation


def _valid_numbers(
    values: Sequence[Mapping[str, Any]], key: str, *, minimum: float = 0.0
) -> list[float]:
    numbers: list[float] = []
    for values_for_epoch in values:
        number = _numeric(values_for_epoch.get(key), minimum=minimum)
        if number is not None:
            numbers.append(float(number))
    return numbers


def _training_summary(training: list[Mapping[str, Any]], source: str | None) -> dict[str, Any]:
    if not training:
        reason = "no training metric records found"
        return {
            "epochs_observed": _measurement(None, source=source, reason=reason, unit="epochs"),
            "duration_seconds": _measurement(None, source=source, reason=reason, unit="seconds"),
            "seen_samples": _measurement(None, source=source, reason=reason, unit="images"),
            "images_per_second": _measurement(
                None, source=source, reason=reason, unit="images/second"
            ),
            "last_images_per_second": _measurement(
                None, source=source, reason=reason, unit="images/second"
            ),
            "peak_vram_bytes": _measurement(None, source=source, reason=reason, unit="bytes"),
            "oom_retries_sum": _measurement(None, source=source, reason=reason, unit="retries"),
            "oom_encountered": _measurement(None, source=source, reason=reason),
        }

    durations = _valid_numbers(training, "duration_seconds")
    seen_samples = _valid_numbers(training, "seen_samples")
    throughputs = _valid_numbers(training, "images_per_second")
    peaks = _valid_numbers(training, "peak_vram_bytes")
    oom_retries = _valid_numbers(training, "oom_retries")
    paired = [
        (float(seen), float(duration))
        for record in training
        if (seen := _numeric(record.get("seen_samples"), minimum=0.0)) is not None
        and (duration := _numeric(record.get("duration_seconds"), minimum=1e-12)) is not None
    ]
    total_seen = sum(seen for seen, _ in paired)
    total_paired_duration = sum(duration for _, duration in paired)
    aggregate_throughput = (
        total_seen / total_paired_duration if paired and total_paired_duration > 0.0 else None
    )

    def measured_or_missing(
        value: Any,
        reason: str,
        *,
        unit: str | None = None,
        derivation: str | None = None,
    ) -> dict[str, Any]:
        return (
            _measurement(value, source=source, unit=unit, derivation=derivation)
            if value is not None
            else _measurement(None, source=source, reason=reason, unit=unit)
        )

    return {
        "epochs_observed": _measurement(len(training), source=source, unit="epochs"),
        "duration_seconds": measured_or_missing(
            sum(durations) if durations else None,
            "duration_seconds absent from training records",
            unit="seconds",
            derivation="sum of reported epoch durations",
        ),
        "seen_samples": measured_or_missing(
            int(sum(seen_samples)) if seen_samples else None,
            "seen_samples absent from training records",
            unit="images",
            derivation="sum of reported epoch sample counts",
        ),
        "images_per_second": measured_or_missing(
            aggregate_throughput,
            "paired seen_samples and positive duration_seconds unavailable",
            unit="images/second",
            derivation="sum(seen_samples) / sum(duration_seconds) for paired records",
        ),
        "last_images_per_second": measured_or_missing(
            throughputs[-1] if throughputs else None,
            "images_per_second absent from training records",
            unit="images/second",
            derivation="last reported epoch throughput",
        ),
        "peak_vram_bytes": measured_or_missing(
            int(max(peaks)) if peaks else None,
            "peak_vram_bytes absent from training records",
            unit="bytes",
            derivation="maximum reported epoch peak",
        ),
        "oom_retries_sum": measured_or_missing(
            int(sum(oom_retries)) if oom_retries else None,
            "oom_retries absent from training records",
            unit="retries",
            derivation="sum of reported epoch retry counts",
        ),
        "oom_encountered": measured_or_missing(
            any(value > 0.0 for value in oom_retries) if oom_retries else None,
            "oom_retries absent from training records",
            derivation="whether any reported epoch retry count exceeded zero",
        ),
    }


def _tta_runtime_multiplier(
    validation: Sequence[Mapping[str, Any]], source: str | None
) -> dict[str, Any]:
    seconds_per_image: dict[str, list[float]] = {}
    for record in validation:
        mode = record.get("tta")
        duration = _numeric(record.get("duration_seconds"), minimum=1e-12)
        count = _numeric(record.get("count"), minimum=1.0)
        if isinstance(mode, str) and duration is not None and count is not None:
            seconds_per_image.setdefault(mode.casefold(), []).append(
                float(duration) / float(count)
            )
    if not seconds_per_image.get("none") or not seconds_per_image.get("d4"):
        return _measurement(
            None,
            source=source,
            reason="comparable none and d4 validation timings were not both measured",
            unit="x runtime",
        )
    none_value = statistics.median(seconds_per_image["none"])
    d4_value = statistics.median(seconds_per_image["d4"])
    return _measurement(
        d4_value / none_value,
        source=source,
        unit="x runtime",
        derivation="median D4 seconds/image divided by median none seconds/image",
    )


def _validation_summary(
    validation: list[Mapping[str, Any]], source: str | None
) -> dict[str, Any]:
    if not validation:
        reason = "no validation or inference metric records found"
        return {
            "evaluations_observed": _measurement(None, source=source, reason=reason),
            "tta_modes": _measurement(None, source=source, reason=reason),
            "last_duration_seconds": _measurement(
                None, source=source, reason=reason, unit="seconds"
            ),
            "last_images_per_second": _measurement(
                None, source=source, reason=reason, unit="images/second"
            ),
            "tta_runtime_multiplier": _measurement(
                None, source=source, reason=reason, unit="x runtime"
            ),
        }
    modes = sorted(
        {str(record["tta"]) for record in validation if isinstance(record.get("tta"), str)}
    )
    last = validation[-1]
    duration = _numeric(last.get("duration_seconds"), minimum=1e-12)
    count = _numeric(last.get("count"), minimum=1.0)
    throughput = float(count) / float(duration) if count is not None and duration is not None else None
    return {
        "evaluations_observed": _measurement(len(validation), source=source),
        "tta_modes": (
            _measurement(modes, source=source)
            if modes
            else _measurement(None, source=source, reason="tta field absent from metric records")
        ),
        "last_duration_seconds": (
            _measurement(duration, source=source, unit="seconds")
            if duration is not None
            else _measurement(
                None,
                source=source,
                reason="duration_seconds absent from last validation record",
                unit="seconds",
            )
        ),
        "last_images_per_second": (
            _measurement(
                throughput,
                source=source,
                unit="images/second",
                derivation="last validation count / duration_seconds",
            )
            if throughput is not None
            else _measurement(
                None,
                source=source,
                reason="count and positive duration_seconds not both present in last validation",
                unit="images/second",
            )
        ),
        "tta_runtime_multiplier": _tta_runtime_multiplier(validation, source),
    }


def _checkpoint_summary(path: Path, source: str | None) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception as error:
        raise ValueError(f"Unable to read checkpoint metadata from {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    extra = payload.get("extra") if isinstance(payload.get("extra"), Mapping) else {}
    summary = {
        "checkpoint_version": payload.get("checkpoint_version"),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "seed": payload.get("seed"),
        "targets": list(payload.get("targets", []))
        if isinstance(payload.get("targets"), Sequence)
        and not isinstance(payload.get("targets"), (str, bytes))
        else None,
        "manifest_hash": payload.get("manifest_hash"),
        "git_commit": payload.get("git_commit"),
        "ema_available": payload.get("ema") is not None,
        "selected_weight_source": extra.get("selected_weight_source"),
    }
    return _measurement(summary, source=source)


def _missing_checkpoint(source: str | None, reason: str) -> dict[str, Any]:
    return _measurement(None, source=source, reason=reason)


def _stage_summary(
    item: ComplexityRunInput,
    provenance: _Provenance,
    *,
    include_checkpoint_metadata: bool,
) -> dict[str, Any]:
    stage = item.stage.strip()
    if not stage:
        raise ValueError("Complexity stage names cannot be empty")
    run_dir = Path(item.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    stats_path = run_dir / "model_stats.json"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "effective_config.yaml"
    checkpoint_path = (
        Path(item.checkpoint).expanduser().resolve()
        if item.checkpoint is not None
        else run_dir / "checkpoints" / "last.ckpt"
    )
    stats_source = provenance.source(stats_path, f"{stage}:model_stats")
    metrics_source = provenance.source(metrics_path, f"{stage}:metrics")
    config_source = provenance.source(config_path, f"{stage}:effective_config")
    checkpoint_source = provenance.source(checkpoint_path, f"{stage}:checkpoint")

    stats_payload = _read_json(stats_path) if stats_source["exists"] else None
    metrics_payload = _read_json(metrics_path) if metrics_source["exists"] else None
    config_payload = _read_yaml_mapping(config_path) if config_source["exists"] else None
    training, validation = _metric_sections(metrics_payload)
    configured = _configured_features(
        config_payload,
        str(config_path) if config_source["exists"] else None,
    )
    runtime_features = _runtime_feature_evidence(
        configured,
        metrics_payload,
        str(metrics_path) if metrics_source["exists"] else None,
    )
    if not include_checkpoint_metadata:
        checkpoint = _missing_checkpoint(
            str(checkpoint_path) if checkpoint_source["exists"] else None,
            "checkpoint metadata reading disabled by caller",
        )
    elif checkpoint_source["exists"]:
        checkpoint = _checkpoint_summary(checkpoint_path, str(checkpoint_path))
    else:
        checkpoint = _missing_checkpoint(None, "last checkpoint not found")

    core_present = bool(stats_source["exists"] and metrics_source["exists"])
    return {
        "stage": stage,
        "run_dir": str(run_dir),
        "status": "complete" if core_present else "partial",
        "sources": {
            "model_stats": stats_source,
            "metrics": metrics_source,
            "effective_config": config_source,
            "checkpoint": checkpoint_source,
        },
        "model": _model_summary(
            stats_payload,
            str(stats_path) if stats_source["exists"] else None,
        ),
        "training_runtime": _training_summary(
            training,
            str(metrics_path) if metrics_source["exists"] else None,
        ),
        "validation_runtime": _validation_summary(
            validation,
            str(metrics_path) if metrics_source["exists"] else None,
        ),
        "features": {
            "configured": configured,
            "runtime_measured": runtime_features,
        },
        "checkpoint_metadata": checkpoint,
    }


def _baseline_summary(path: Path, provenance: _Provenance) -> dict[str, Any]:
    source = provenance.source(path, "baseline_benchmark")
    if not source["exists"]:
        return _measurement(None, source=None, reason="baseline benchmark file not found")
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Baseline benchmark must be a mapping: {path}")
    combinations = payload.get("combinations")
    if not isinstance(combinations, Mapping):
        return _measurement(
            {
                "combinations": {},
                "tta_pairs": [],
                "warning": "combinations mapping missing",
            },
            source=str(path),
        )
    default_count = _numeric(payload.get("validation_count"), minimum=1.0)
    summarized: dict[str, Any] = {}
    timings: dict[str, tuple[str, float, float]] = {}
    for name in sorted(combinations, key=str):
        values = combinations[name]
        if not isinstance(values, Mapping):
            summarized[str(name)] = {
                "duration_seconds": None,
                "count": None,
                "tta": None,
                "reason": "combination is not a mapping",
            }
            continue
        duration = _numeric(values.get("duration_seconds"), minimum=1e-12)
        count = _numeric(values.get("count"), minimum=1.0) or default_count
        tta = values.get("tta") if isinstance(values.get("tta"), str) else None
        summarized[str(name)] = {
            "duration_seconds": duration,
            "count": count,
            "tta": tta,
            "reason": None
            if duration is not None and count is not None and tta is not None
            else "duration, count, or explicit tta mode missing",
        }
        if duration is not None and count is not None and tta is not None:
            timings[str(name)] = (tta.casefold(), float(duration), float(count))

    pairs: list[dict[str, Any]] = []
    for none_name, (none_mode, none_duration, none_count) in sorted(timings.items()):
        if none_mode != "none" or not none_name.endswith("_none"):
            continue
        prefix = none_name[: -len("_none")]
        d4_name = f"{prefix}_d4"
        d4 = timings.get(d4_name)
        if d4 is None or d4[0] != "d4":
            continue
        _, d4_duration, d4_count = d4
        if d4_count != none_count:
            pairs.append(
                {
                    "pair": prefix,
                    "runtime_multiplier": None,
                    "throughput_multiplier": None,
                    "reason": "none and d4 sample counts differ",
                }
            )
            continue
        runtime_multiplier = (d4_duration / d4_count) / (none_duration / none_count)
        pairs.append(
            {
                "pair": prefix,
                "none_combination": none_name,
                "d4_combination": d4_name,
                "sample_count": int(none_count),
                "runtime_multiplier": runtime_multiplier,
                "throughput_multiplier": 1.0 / runtime_multiplier,
                "reason": None,
                "derivation": "measured D4 seconds/image divided by measured none seconds/image",
            }
        )
    return _measurement(
        {"combinations": summarized, "tta_pairs": pairs}, source=str(path)
    )


def _normalize_inputs(
    stages: Mapping[str, str | Path] | Sequence[ComplexityRunInput],
) -> list[ComplexityRunInput]:
    if isinstance(stages, Mapping):
        items = [
            ComplexityRunInput(stage=str(stage), run_dir=run_dir)
            for stage, run_dir in stages.items()
        ]
    else:
        items = list(stages)
        if not all(isinstance(item, ComplexityRunInput) for item in items):
            raise TypeError("Stage sequences must contain ComplexityRunInput values")
    if not items:
        raise ValueError("At least one complexity run is required")
    names = [item.stage.strip() for item in items]
    if len(set(names)) != len(names):
        raise ValueError("Complexity stage names must be unique")
    return items


def generate_complexity_report(
    stages: Mapping[str, str | Path] | Sequence[ComplexityRunInput],
    output_path: str | Path,
    *,
    baseline_benchmark: str | Path | None = None,
    include_checkpoint_metadata: bool = True,
) -> dict[str, Any]:
    """Generate and atomically write a measured-only multi-stage report."""

    inputs = _normalize_inputs(stages)
    provenance = _Provenance()
    stage_reports = [
        _stage_summary(
            item,
            provenance,
            include_checkpoint_metadata=include_checkpoint_metadata,
        )
        for item in inputs
    ]
    baseline = (
        _baseline_summary(Path(baseline_benchmark).expanduser().resolve(), provenance)
        if baseline_benchmark is not None
        else _measurement(None, source=None, reason="baseline benchmark not provided")
    )
    input_digest = provenance.digest()
    report = {
        "schema_version": 1,
        "report_type": "performance_v2_complexity_runtime",
        "evidence_policy": "measured_artifacts_only",
        "status": "complete"
        if all(stage["status"] == "complete" for stage in stage_reports)
        else "partial",
        "stages": stage_reports,
        "baseline_benchmark": baseline,
        "provenance": {
            "inputs": provenance.records(),
            "input_set_sha256": input_digest,
            "input_set_reason": None if input_digest is not None else "no readable inputs",
        },
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return report


__all__ = ["ComplexityRunInput", "generate_complexity_report"]
