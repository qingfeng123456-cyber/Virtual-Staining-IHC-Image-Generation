"""Console logging and portable experiment-log bundles.

The project keeps heavy checkpoints and predictions in outputs. This module
creates a separate, intentionally small log bundle under log/<run-id> so a
remote training run can be downloaded and reviewed without copying model weights.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import shutil
import time
import traceback
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .device import environment_report

_ACTIVE_SESSION: ContextVar[ExperimentLogSession | None] = ContextVar(
    "virtual_staining_active_log_session", default=None
)
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _compact_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_name(value: object, *, fallback: str) -> str:
    normalized = _SAFE_NAME.sub("_", str(value).strip()).strip("._")
    return normalized or fallback


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _to_jsonable(value: Any) -> Any:
    """Convert argparse, config, and metric values to conservative JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _without_per_image_records(value: Any) -> Any:
    """Remove repeated validator rows from the portable lightweight log bundle."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_per_image_records(item)
            for key, item in value.items()
            if str(key) != "records"
        }
    if isinstance(value, (list, tuple)):
        return [_without_per_image_records(item) for item in value]
    return value


def _invocation_from_args(args: Any) -> dict[str, Any]:
    values = vars(args) if hasattr(args, "__dict__") else {}
    return {
        str(key): _to_jsonable(value)
        for key, value in values.items()
        if key != "handler" and not callable(value)
    }


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flat.update(_flatten_mapping(item, name))
        elif isinstance(item, (list, tuple, set)):
            flat[name] = json.dumps(_to_jsonable(item), ensure_ascii=False)
        else:
            flat[name] = _to_jsonable(item)
    return flat


def _module_inventory(model: Any) -> dict[str, Any]:
    """Summarize model components without profiling hooks or training overhead."""

    modules: list[dict[str, Any]] = []
    for name, module in model.named_children():
        parameters = list(module.parameters())
        modules.append(
            {
                "name": name,
                "class_name": type(module).__name__,
                "parameters": int(sum(parameter.numel() for parameter in parameters)),
                "trainable_parameters": int(
                    sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
                ),
                "buffers": int(sum(buffer.numel() for buffer in module.buffers())),
            }
        )
    all_parameters = list(model.parameters())
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in all_parameters)),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in all_parameters if parameter.requires_grad)
        ),
        "feature_flags": _to_jsonable(getattr(model, "feature_flags", {})),
        "top_level_modules": modules,
        "note": (
            "Parameter counts are structural module sizes. Runtime performance is "
            "recorded per epoch in epoch_metrics.jsonl and performance_summary.json."
        ),
    }


def configure_logging(log_file: str | Path | None = None, verbose: bool = False) -> logging.Logger:
    """Configure an idempotent project logger for console and an optional file."""

    logger = logging.getLogger("virtual_staining")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@dataclass
class ExperimentLogSession:
    """Own all lightweight logs generated by one CLI invocation."""

    root: Path
    command: str
    run_key: str
    invocation: Mapping[str, Any]
    started_at: str = field(default_factory=_utc_timestamp)
    started_monotonic: float = field(default_factory=time.perf_counter)
    run_dir: Path | None = None
    _epoch_entries: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()
        self.command = _safe_name(self.command, fallback="command")
        self.run_key = _safe_name(self.run_key, fallback=f"{self.command}_{_compact_timestamp()}")
        self.directory = self.root / self.run_key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.console_path = self.directory / f"{self.command}.log"
        self.epoch_path = self.directory / "epoch_metrics.jsonl"
        _write_json(self.directory / "invocation.json", dict(self.invocation))
        _write_json(
            self.directory / "environment.json",
            {
                **environment_report(),
                "started_at_utc": self.started_at,
                "working_directory": str(Path.cwd().resolve()),
            },
        )
        _write_json(
            self.directory / "status.json",
            {
                "status": "running",
                "command": self.command,
                "run_key": self.run_key,
                "started_at_utc": self.started_at,
            },
        )
        (self.directory / "README.txt").write_text(
            "This folder is a lightweight experiment-review package. It intentionally "
            "does not contain checkpoints or prediction images. Download this folder or "
            "the generated log bundle after training.\n",
            encoding="utf-8",
        )

    @classmethod
    def from_args(cls, args: Any) -> ExperimentLogSession:
        command = str(getattr(args, "command", "command"))
        run_id = getattr(args, "run_id", None)
        checkpoint = getattr(args, "checkpoint", None)
        if run_id:
            key = str(run_id)
        elif checkpoint:
            checkpoint_path = Path(str(checkpoint))
            key = checkpoint_path.parent.parent.name or checkpoint_path.stem
        else:
            key = f"{command}_{_compact_timestamp()}"
        return cls(
            root=Path(str(getattr(args, "log_root", "log"))),
            command=command,
            run_key=key,
            invocation=_invocation_from_args(args),
        )

    def _write_status(self, status: str, **extra: Any) -> None:
        _write_json(
            self.directory / "status.json",
            {
                "status": status,
                "command": self.command,
                "run_key": self.run_key,
                "started_at_utc": self.started_at,
                "finished_at_utc": _utc_timestamp() if status != "running" else None,
                "elapsed_seconds": round(time.perf_counter() - self.started_monotonic, 6),
                **_to_jsonable(extra),
            },
        )

    def bind_training_run(
        self,
        run_dir: str | Path,
        *,
        config: Mapping[str, Any],
        model: Any,
        model_stats: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach a training run and record static model/configuration metadata."""

        self.run_dir = Path(run_dir).expanduser().resolve()
        _write_json(
            self.directory / "run_binding.json",
            {
                "run_dir": str(self.run_dir),
                "configured_run_id": _to_jsonable(config.get("project", {}).get("run_id"))
                if isinstance(config.get("project"), Mapping)
                else None,
            },
        )
        _write_json(self.directory / "effective_config.json", _to_jsonable(dict(config)))
        _write_json(self.directory / "module_inventory.json", _module_inventory(model))
        if model_stats is not None:
            _write_json(self.directory / "model_stats.json", _to_jsonable(dict(model_stats)))

    def record_epoch(self, entry: Mapping[str, Any]) -> None:
        """Persist a completed epoch immediately, including partial-run evidence."""

        normalized = _to_jsonable(_without_per_image_records(dict(entry)))
        if not isinstance(normalized, dict):
            raise TypeError("epoch entry must serialize to a mapping")
        self._epoch_entries.append(normalized)
        with self.epoch_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(normalized, ensure_ascii=False, default=_json_default))
            handle.write("\n")
        _write_json(self.directory / "latest_epoch.json", normalized)
        train = normalized.get("train", {})
        validation = normalized.get("validation", {})
        macro = validation.get("macro", {}) if isinstance(validation, Mapping) else {}
        logging.getLogger("virtual_staining").info(
            "Epoch %s | loss=%s | %.3f img/s | peak_vram_mib=%.1f | val_ssim=%s | val_psnr=%s",
            train.get("epoch", "?") if isinstance(train, Mapping) else "?",
            train.get("loss", "?") if isinstance(train, Mapping) else "?",
            float(train.get("images_per_second", 0.0)) if isinstance(train, Mapping) else 0.0,
            float(train.get("peak_vram_bytes", 0.0)) / 1024**2
            if isinstance(train, Mapping)
            else 0.0,
            macro.get("mean_ssim", "?") if isinstance(macro, Mapping) else "?",
            macro.get("mean_psnr", "?") if isinstance(macro, Mapping) else "?",
        )

    def _copy_run_artifacts(self, run_dir: Path) -> None:
        files = {
            "effective_config.yaml": "effective_config.yaml",
            "model_stats.json": "model_stats_from_run.json",
            "metrics.json": "metrics_history.json",
            "pretrain_metrics.json": "pretrain_metrics.json",
            "pipeline_report.json": "pipeline_report.json",
            "inference_report.json": "inference_report.json",
            "validation/metrics.json": "validation_metrics.json",
            "validation/per_image.csv": "validation_per_image.csv",
            "validation/per_image_tta_d4.csv": "validation_per_image_tta_d4.csv",
        }
        copied: dict[str, str] = {}
        for relative, destination_name in files.items():
            source = run_dir / relative
            if source.is_file():
                destination = self.directory / destination_name
                if relative == "metrics.json":
                    _write_json(
                        destination,
                        _without_per_image_records(_read_json(source)),
                    )
                else:
                    shutil.copy2(source, destination)
                copied[relative] = destination_name
        _write_json(
            self.directory / "copied_artifacts.json",
            {"source_run_dir": str(run_dir), "files": copied},
        )

    def _history_from_run(self) -> list[dict[str, Any]]:
        metrics_path = self.directory / "metrics_history.json"
        payload = _read_json(metrics_path)
        if not isinstance(payload, list):
            return list(self._epoch_entries)
        return [
            dict(_without_per_image_records(item))
            for item in payload
            if isinstance(item, Mapping)
        ]

    def _write_epoch_csv(self, history: list[dict[str, Any]]) -> None:
        if not history:
            return
        rows = [_flatten_mapping(entry) for entry in history]
        fields = sorted({key for row in rows for key in row})
        with (self.directory / "epoch_metrics.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _write_performance_summary(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        train_rows = [
            dict(entry["train"])
            for entry in history
            if isinstance(entry.get("train"), Mapping)
        ]
        durations = [
            float(row.get("duration_seconds", 0.0))
            for row in train_rows
            if float(row.get("duration_seconds", 0.0)) >= 0.0
        ]
        samples = [int(row.get("seen_samples", 0)) for row in train_rows]
        total_duration = float(sum(durations))
        total_samples = int(sum(samples))
        latest_validation = (
            history[-1].get("validation", {})
            if history and isinstance(history[-1], Mapping)
            else {}
        )
        best_entry: Mapping[str, Any] | None = None
        best_ssim = -math.inf
        for entry in history:
            macro = entry.get("validation", {}).get("macro", {}) if isinstance(entry, Mapping) else {}
            if isinstance(macro, Mapping):
                value = macro.get("mean_ssim")
                if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > best_ssim:
                    best_ssim = float(value)
                    best_entry = entry
        peak_allocated = max(
            (int(row.get("peak_vram_bytes", 0)) for row in train_rows), default=0
        )
        peak_reserved = max(
            (int(row.get("peak_vram_reserved_bytes", 0)) for row in train_rows),
            default=0,
        )
        command_elapsed = float(time.perf_counter() - self.started_monotonic)
        summary = {
            "run_key": self.run_key,
            "command": self.command,
            "run_dir": str(self.run_dir) if self.run_dir is not None else None,
            "training": {
                "epochs_recorded": len(train_rows),
                "command_elapsed_seconds": command_elapsed,
                "total_duration_seconds": total_duration,
                "non_train_seconds": max(0.0, command_elapsed - total_duration),
                "total_seen_samples": total_samples,
                "mean_epoch_duration_seconds": total_duration / len(durations) if durations else None,
                "end_to_end_images_per_second": total_samples / total_duration
                if total_duration > 0.0
                else None,
                "peak_vram_allocated_bytes": peak_allocated,
                "peak_vram_reserved_bytes": peak_reserved,
                "peak_vram_allocated_mib": peak_allocated / 1024**2,
                "peak_vram_reserved_mib": peak_reserved / 1024**2,
                "max_oom_retries": max(
                    (int(row.get("oom_retries", 0)) for row in train_rows), default=0
                ),
            },
            "latest_validation": _to_jsonable(latest_validation),
            "best_validation_by_primary_ssim": _to_jsonable(best_entry)
            if best_entry is not None
            else None,
        }
        _write_json(self.directory / "performance_summary.json", summary)
        return summary

    def _write_summary_markdown(self, summary: Mapping[str, Any], status: str) -> None:
        training = summary.get("training", {})
        lines = [
            "# Experiment log summary",
            "",
            f"- Status: {status}",
            f"- Command: {self.command}",
            f"- Run key: {self.run_key}",
            f"- Run directory: {summary.get('run_dir')}",
            f"- Recorded epochs: {training.get('epochs_recorded')}",
            f"- Full command seconds: {training.get('command_elapsed_seconds')}",
            f"- Total training seconds: {training.get('total_duration_seconds')}",
            f"- End-to-end images/second: {training.get('end_to_end_images_per_second')}",
            f"- Peak allocated VRAM MiB: {training.get('peak_vram_allocated_mib')}",
            f"- Peak reserved VRAM MiB: {training.get('peak_vram_reserved_mib')}",
            "",
            "For review, share this folder or the generated log bundle together with",
            "performance_summary.json, epoch_metrics.csv, environment.json,",
            "effective_config.yaml, and validation_metrics.json when available.",
        ]
        (self.directory / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _bundle(self) -> Path:
        bundle = self.directory / f"{self.command}_log_bundle.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self.directory.rglob("*")):
                if path.is_file() and path != bundle:
                    archive.write(path, path.relative_to(self.directory))
        return bundle

    def complete(self, result: Mapping[str, Any]) -> dict[str, str]:
        run_dir = result.get("run_dir")
        if run_dir:
            self.run_dir = Path(str(run_dir)).expanduser().resolve()
        if self.run_dir is not None and self.run_dir.is_dir():
            self._copy_run_artifacts(self.run_dir)
        _write_json(self.directory / "command_result.json", _to_jsonable(dict(result)))
        history = self._history_from_run()
        self._write_epoch_csv(history)
        summary = self._write_performance_summary(history)
        self._write_summary_markdown(summary, "completed")
        self._write_status("completed", run_dir=str(self.run_dir) if self.run_dir else None)
        bundle = self._bundle()
        logging.getLogger("virtual_staining").info("Experiment log bundle: %s", bundle)
        return {"log_dir": str(self.directory), "log_bundle": str(bundle)}

    def fail(self, error: BaseException) -> dict[str, str]:
        _write_json(
            self.directory / "failure.json",
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        self._write_status("failed", error_type=type(error).__name__, message=str(error))
        bundle = self._bundle()
        return {"log_dir": str(self.directory), "log_bundle": str(bundle)}


@contextmanager
def activate_experiment_log(session: ExperimentLogSession) -> Iterator[ExperimentLogSession]:
    """Make a session available to training code without changing public APIs."""

    token = _ACTIVE_SESSION.set(session)
    try:
        yield session
    finally:
        _ACTIVE_SESSION.reset(token)


def active_experiment_log() -> ExperimentLogSession | None:
    """Return the active CLI log session, if this call originated from the CLI."""

    return _ACTIVE_SESSION.get()
