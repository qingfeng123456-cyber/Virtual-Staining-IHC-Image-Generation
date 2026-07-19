"""Append-only provenance registry for performance-v2 experiments."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPERIMENT_FIELDS = (
    "run_id",
    "parent_run",
    "git_commit",
    "config_hash",
    "manifest_hash",
    "model",
    "target",
    "organ",
    "fold",
    "seed",
    "context",
    "pretrain",
    "prototype",
    "task_optimizer",
    "loss_schedule",
    "params",
    "flops",
    "peak_vram",
    "train_time",
    "float_ssim",
    "float_psnr",
    "uint8_ssim",
    "uint8_psnr",
    "jpg_ssim",
    "jpg_psnr",
    "weighted_score_proxy",
    "checkpoint",
    "status",
    "failure_reason",
)

_VALID_STATUSES = {
    "planned",
    "running",
    "completed",
    "completed_pretrain",
    "failed",
    "blocked_unverified_grid",
    "blocked_requires_oof_ensemble",
    "blocked_insufficient_ensemble_members",
    "blocked_requires_explicit_validation_ensemble",
    "not_promotable",
    "promoted",
}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class ExperimentRegistry:
    """A deterministic CSV registry with safe run-id upserts.

    Re-running a failed or interrupted command updates that run's single row;
    unrelated runs retain their original insertion order.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def read(self) -> list[dict[str, str]]:
        if not self.path.is_file():
            return []
        with self.path.open("r", newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def upsert(self, record: Mapping[str, Any]) -> dict[str, str]:
        unknown = sorted(set(record).difference(EXPERIMENT_FIELDS))
        if unknown:
            raise KeyError(f"Unknown experiment registry fields: {', '.join(unknown)}")
        run_id = _stringify(record.get("run_id")).strip()
        if not run_id:
            raise ValueError("Experiment record requires a non-empty run_id")
        status = _stringify(record.get("status", "planned"))
        if status not in _VALID_STATUSES:
            raise ValueError(f"Unsupported experiment status: {status}")
        normalized = {field: _stringify(record.get(field)) for field in EXPERIMENT_FIELDS}
        normalized["run_id"] = run_id
        normalized["status"] = status
        rows = self.read()
        replaced = False
        updated: list[dict[str, str]] = []
        for row in rows:
            if row.get("run_id") == run_id:
                updated.append(normalized)
                replaced = True
            else:
                updated.append({field: row.get(field, "") for field in EXPERIMENT_FIELDS})
        if not replaced:
            updated.append(normalized)
        self._write(updated)
        return normalized

    def _write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _stringify(row.get(field)) for field in EXPERIMENT_FIELDS})
        temporary.replace(self.path)

    def select(self, run_ids: Sequence[str] | None = None) -> list[dict[str, str]]:
        rows = self.read()
        if not run_ids:
            return rows
        wanted = set(run_ids)
        selected = [row for row in rows if row.get("run_id") in wanted]
        missing = sorted(wanted.difference(row.get("run_id", "") for row in selected))
        if missing:
            raise KeyError(f"Unknown experiment run(s): {', '.join(missing)}")
        return selected


def compare_experiments(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str = "jpg_ssim",
) -> dict[str, Any]:
    """Rank completed records without treating absent metrics as zero."""

    if primary_metric not in EXPERIMENT_FIELDS:
        raise KeyError(f"Unknown comparison metric: {primary_metric}")
    scored: list[dict[str, Any]] = []
    unscored: list[dict[str, str]] = []
    for row in rows:
        value = str(row.get(primary_metric, "")).strip()
        try:
            metric = float(value)
        except ValueError:
            unscored.append(dict(row))
            continue
        item = dict(row)
        item[primary_metric] = metric
        scored.append(item)
    scored.sort(key=lambda item: (-float(item[primary_metric]), str(item.get("run_id", ""))))
    return {
        "primary_metric": primary_metric,
        "ranked": scored,
        "unscored": unscored,
        "best_run": scored[0].get("run_id") if scored else None,
    }
