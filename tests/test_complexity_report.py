"""Evidence-only complexity/runtime report tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from virtual_staining.engine.complexity_report import (
    ComplexityRunInput,
    generate_complexity_report,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _complete_run(path: Path) -> None:
    _write_json(
        path / "model_stats.json",
        {
            "parameters": 1000,
            "trainable_parameters": 900,
            "approximate_macs": 123456,
        },
    )
    _write_json(
        path / "metrics.json",
        [
            {
                "train": {
                    "epoch": 0,
                    "duration_seconds": 10.0,
                    "seen_samples": 50,
                    "images_per_second": 5.0,
                    "peak_vram_bytes": 1024,
                    "oom_retries": 0,
                },
                "validation": {
                    "count": 20,
                    "duration_seconds": 2.0,
                    "tta": "none",
                },
            },
            {
                "train": {
                    "epoch": 1,
                    "duration_seconds": 20.0,
                    "seen_samples": 50,
                    "images_per_second": 2.5,
                    "peak_vram_bytes": 2048,
                    "oom_retries": 1,
                },
                "validation": {
                    "count": 20,
                    "duration_seconds": 6.0,
                    "tta": "d4",
                },
            },
        ],
    )
    config = {
        "model": {"context": {"enabled": False}},
        "inference": {"ensemble_checkpoints": []},
        "ensemble": {"weights": []},
    }
    (path / "effective_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    checkpoint = path / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": 2,
            "model": {"weight": torch.ones(1)},
            "ema": {"shadow": {}},
            "epoch": 1,
            "global_step": 10,
            "seed": 2026,
            "targets": ["CD68"],
            "manifest_hash": "manifest-hash",
            "git_commit": None,
            "extra": {"selected_weight_source": "raw"},
        },
        checkpoint,
    )


def test_report_summarizes_multiple_stages_with_real_provenance(tmp_path: Path) -> None:
    complete = tmp_path / "运行 A0"
    _complete_run(complete)
    partial = tmp_path / "A3 blocked"
    partial.mkdir()
    _write_json(
        partial / "metrics.json",
        {"count": 4, "duration_seconds": 1.0, "tta": "none"},
    )
    destination = tmp_path / "报告" / "complexity.json"

    report = generate_complexity_report(
        [
            ComplexityRunInput("A0", complete),
            ComplexityRunInput("A3", partial),
        ],
        destination,
    )

    assert report["status"] == "partial"
    a0, a3 = report["stages"]
    assert a0["status"] == "complete"
    assert a0["model"]["parameters"]["value"] == 1000
    assert a0["training_runtime"]["duration_seconds"]["value"] == 30.0
    assert a0["training_runtime"]["images_per_second"]["value"] == pytest.approx(
        100 / 30
    )
    assert a0["training_runtime"]["peak_vram_bytes"]["value"] == 2048
    assert a0["training_runtime"]["oom_retries_sum"]["value"] == 1
    assert a0["training_runtime"]["oom_encountered"]["value"] is True
    assert a0["validation_runtime"]["tta_runtime_multiplier"]["value"] == pytest.approx(3.0)
    assert a0["features"]["runtime_measured"]["context"]["value"] is False
    assert a0["checkpoint_metadata"]["value"]["global_step"] == 10
    assert a0["checkpoint_metadata"]["value"]["ema_available"] is True
    assert a3["model"]["parameters"]["value"] is None
    assert a3["model"]["parameters"]["reason"]
    assert a3["checkpoint_metadata"]["value"] is None
    assert a3["checkpoint_metadata"]["reason"] == "last checkpoint not found"
    assert report["provenance"]["input_set_sha256"]
    assert len(report["provenance"]["inputs"]) == 5
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    assert not list(destination.parent.glob(".*.tmp"))


def test_baseline_tta_multiplier_requires_measured_comparable_counts(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _complete_run(run)
    baseline = tmp_path / "baseline.json"
    _write_json(
        baseline,
        {
            "validation_count": 100,
            "combinations": {
                "raw_none": {"duration_seconds": 5.0, "count": 100, "tta": "none"},
                "raw_d4": {"duration_seconds": 12.5, "count": 100, "tta": "d4"},
                "ema_none": {"duration_seconds": 4.0, "count": 100, "tta": "none"},
                "ema_d4": {"duration_seconds": 8.0, "count": 80, "tta": "d4"},
            },
        },
    )

    report = generate_complexity_report(
        {"A0": run},
        tmp_path / "report.json",
        baseline_benchmark=baseline,
        include_checkpoint_metadata=False,
    )

    baseline_value = report["baseline_benchmark"]["value"]
    pairs = {item["pair"]: item for item in baseline_value["tta_pairs"]}
    assert pairs["raw"]["runtime_multiplier"] == 2.5
    assert pairs["raw"]["throughput_multiplier"] == 0.4
    assert pairs["ema"]["runtime_multiplier"] is None
    assert pairs["ema"]["reason"] == "none and d4 sample counts differ"
    checkpoint = report["stages"][0]["checkpoint_metadata"]
    assert checkpoint["value"] is None
    assert checkpoint["reason"] == "checkpoint metadata reading disabled by caller"


def test_missing_evidence_is_null_with_reason_and_never_inferred(tmp_path: Path) -> None:
    run = tmp_path / "empty_run"
    run.mkdir()

    report = generate_complexity_report(
        {"A3": run}, tmp_path / "report.json"
    )

    stage = report["stages"][0]
    for section, field in (
        ("model", "parameters"),
        ("training_runtime", "peak_vram_bytes"),
        ("validation_runtime", "tta_runtime_multiplier"),
    ):
        assert stage[section][field]["value"] is None
        assert stage[section][field]["reason"]
    assert stage["features"]["configured"]["context"]["value"] is None
    assert stage["features"]["runtime_measured"]["context"]["value"] is None
    assert report["baseline_benchmark"]["value"] is None
    assert report["baseline_benchmark"]["reason"] == "baseline benchmark not provided"
    assert report["provenance"]["inputs"] == []
    assert report["provenance"]["input_set_sha256"] is None
    assert report["provenance"]["input_set_reason"] == "no readable inputs"


def test_context_runtime_requires_explicit_metric_evidence(tmp_path: Path) -> None:
    run = tmp_path / "context"
    run.mkdir()
    (run / "effective_config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"context": {"enabled": True}},
                "inference": {"ensemble_checkpoints": ["one.ckpt", "two.ckpt"]},
            }
        ),
        encoding="utf-8",
    )
    _write_json(
        run / "metrics.json",
        {
            "count": 2,
            "duration_seconds": 1.0,
            "tta": "none",
            "records": [{"context_valid_fraction": 0.75}],
            "ensemble_member_count": 2,
        },
    )

    report = generate_complexity_report(
        {"A3": run}, tmp_path / "report.json"
    )

    features = report["stages"][0]["features"]
    assert features["configured"]["context"]["value"] is True
    assert features["configured"]["ensemble"]["value"] is True
    assert features["runtime_measured"]["context"]["value"] is True
    assert features["runtime_measured"]["ensemble"]["value"] is True


def test_invalid_inputs_fail_without_replacing_existing_report(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "model_stats.json").write_text("not-json", encoding="utf-8")
    destination = tmp_path / "report.json"
    destination.write_text("stable", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        generate_complexity_report({"A0": run}, destination)
    assert destination.read_text(encoding="utf-8") == "stable"
    with pytest.raises(ValueError, match="At least one"):
        generate_complexity_report({}, destination)
    with pytest.raises(ValueError, match="unique"):
        generate_complexity_report(
            [
                ComplexityRunInput("A0", run),
                ComplexityRunInput("A0", run),
            ],
            destination,
        )


def test_input_set_digest_is_bound_to_file_hashes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _complete_run(run)
    report = generate_complexity_report(
        {"A0": run}, tmp_path / "report.json"
    )
    digest = hashlib.sha256()
    for source in report["provenance"]["inputs"]:
        digest.update(source["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(source["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(source["sha256"].encode("ascii"))
        digest.update(b"\n")
    assert digest.hexdigest() == report["provenance"]["input_set_sha256"]
