from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

import virtual_staining.cli as cli_module
from virtual_staining.cli import build_parser
from virtual_staining.data.roi_index import audit_roi_grid
from virtual_staining.engine.checkpoint import save_checkpoint
from virtual_staining.engine.experiment_registry import (
    ExperimentRegistry,
    compare_experiments,
)
from virtual_staining.engine.model_soup import (
    build_initialization_provenance,
    checkpoint_file_sha256,
    state_dict_value_sha256,
)

V2_COMMANDS = (
    "benchmark-baseline",
    "audit-roi-grid",
    "pretrain-dapi",
    "train-v2",
    "finetune-target",
    "finetune-organ",
    "finetune-metric",
    "run-ablation",
    "compare-runs",
    "build-model-soup",
    "optimize-ensemble",
    "predict-v2",
)


def _subcommand_choices(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return subparsers.choices


def test_parser_exposes_all_twelve_performance_v2_commands() -> None:
    choices = _subcommand_choices(build_parser())

    assert len(V2_COMMANDS) == 12
    assert set(V2_COMMANDS).issubset(choices)
    for command in V2_COMMANDS:
        assert choices[command].description is not None or choices[command].format_help()


def test_model_soup_parser_is_lineage_safe_by_default() -> None:
    args = build_parser().parse_args(
        ["build-model-soup", "--checkpoints", "one.ckpt", "two.ckpt"]
    )

    assert args.allow_unsafe_lineage_mismatch is False
    assert args.allow_unsafe_engineering_validation is False
    assert args.target_marker is None
    assert not hasattr(args, "lineage")
    assert not hasattr(args, "require_matching_lineage")


def test_unsafe_model_soup_output_is_forcibly_named(tmp_path: Path) -> None:
    safe = cli_module._model_soup_output_path(
        tmp_path / "soup.ckpt", unsafe_lineage_override=False
    )
    unsafe = cli_module._model_soup_output_path(
        tmp_path / "soup.ckpt", unsafe_lineage_override=True
    )
    unsafe_validation = cli_module._model_soup_output_path(
        tmp_path / "soup.ckpt",
        unsafe_lineage_override=False,
        unsafe_engineering_validation=True,
    )

    assert safe.name == "soup.ckpt"
    assert unsafe.name == "soup_UNSAFE_LINEAGE_OVERRIDE.ckpt"
    assert (
        unsafe_validation.name
        == "soup_UNSAFE_ENGINEERING_VALIDATION.ckpt"
    )


def test_optimize_ensemble_parser_requires_external_audit_contract(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "optimize-ensemble",
                "--predictions",
                "one.npy",
                "--target-array",
                "target.npy",
                "--source",
                "validation",
            ]
        )
    args = parser.parse_args(
        [
            "optimize-ensemble",
            "--predictions",
            "one.npy",
            "--target-array",
            "target.npy",
            "--source",
            "validation",
            "--manifest",
            "val.csv",
            "--target-marker",
            "CD68",
            "--roi-audit",
            "roi.json",
            "--audited-manifests",
            "train.csv",
            "val.csv",
        ]
    )
    assert args.metric_domain == "jpg_roundtrip"
    assert args.audited_manifests == ["train.csv", "val.csv"]
    assert args.allow_unsafe_engineering_manifest is False
    unsafe = cli_module._ensemble_output_path(
        tmp_path / "weights.json",
        unsafe_engineering_manifest=True,
    )
    assert unsafe.name == "weights_UNSAFE_ENGINEERING_MANIFEST.json"


@pytest.mark.parametrize("command", V2_COMMANDS)
def test_each_performance_v2_command_parses_help(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args([command, "--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert f"virtual_staining {command}" in help_text
    assert "--help" in help_text


def test_experiment_registry_upsert_and_comparison_are_deterministic(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "含 空格" / "experiments.csv")
    registry.upsert(
        {
            "run_id": "A0",
            "status": "running",
            "jpg_ssim": 0.71,
            "jpg_psnr": 20.0,
        }
    )
    registry.upsert(
        {
            "run_id": "A1",
            "status": "completed",
            "jpg_ssim": 0.73,
            "jpg_psnr": 20.5,
        }
    )
    registry.upsert(
        {
            "run_id": "A0",
            "status": "completed",
            "jpg_ssim": 0.72,
            "jpg_psnr": 20.2,
        }
    )
    registry.upsert({"run_id": "A2", "status": "blocked_unverified_grid"})

    rows = registry.read()
    comparison = compare_experiments(rows, primary_metric="jpg_ssim")

    assert [row["run_id"] for row in rows] == ["A0", "A1", "A2"]
    assert rows[0]["jpg_ssim"] == "0.72"
    assert comparison["best_run"] == "A1"
    assert [row["run_id"] for row in comparison["ranked"]] == ["A1", "A0"]
    assert [row["run_id"] for row in comparison["unscored"]] == ["A2"]


def _write_ablation_suite(path: Path) -> None:
    path.write_text(
        """suite: isolated_p0
order: [A0, A1, A2, A3]
stages:
  A0: {config: a0.yaml}
  A1: {parent: A0, config: a1.yaml}
  A2: {parent: A1, config: a2.yaml}
  A3: {parent: A2, config: a3.yaml, requires_verified_grid: true}
""",
        encoding="utf-8",
    )


def _ablation_config() -> dict[str, Any]:
    return {
        "project": {"seed": 2026},
        "data": {"targets": ["CD68"]},
        "model": {
            "name": "camp_vs_v2",
            "context": {"enabled": False},
            "prototypes": {"enabled": False},
        },
        "train": {"epochs": 1},
        "loss": {"schedule": "none"},
        "pretrain": {"enabled": False},
        "multitask": {"optimizer": "equal"},
        "budget": {"smoke": {"epochs": 1}},
    }


def _verified_context_audit() -> dict[str, Any]:
    return {
        "total_rows": 4,
        "parsed_rows": 4,
        "filename_grid_verified": True,
        "duplicate_coordinates": [],
        "train_val_shared_rois": [],
        "cross_split_adjacent_pairs": [],
        "boundary": {
            "direction_verified": True,
            "continuity_verified": True,
        },
        "context_gate_reasons": [],
        "context_enabled": True,
    }


def _ablation_args(
    suite: Path,
    registry: Path,
    stage: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        suite=str(suite),
        budget="smoke",
        from_stage=stage,
        through_stage=stage,
        data_root=None,
        set=[],
        max_epochs=1,
        max_train_samples=2,
        max_val_samples=2,
        registry=str(registry),
    )


@pytest.mark.parametrize(
    ("stage", "expected_parent"),
    (
        ("A2", "isolated_p0_A1_smoke_seed2026"),
        ("A3", "isolated_p0_A2_smoke_seed2026"),
    ),
)
def test_run_ablation_preserves_declared_lineage_when_started_mid_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    expected_parent: str,
) -> None:
    suite = tmp_path / "suite.yaml"
    registry_path = tmp_path / "registry.csv"
    _write_ablation_suite(suite)
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda *_args, **_kwargs: _ablation_config(),
    )
    monkeypatch.setattr(
        cli_module,
        "_context_audit_for_config",
        lambda *_args, **_kwargs: _verified_context_audit(),
    )

    def fake_run_train(
        config: dict[str, Any],
        data_root: str | Path | None,
        run_id: str,
    ) -> dict[str, str]:
        del config, data_root
        return {"run_dir": str(tmp_path / "outputs" / run_id)}

    monkeypatch.setattr(cli_module, "_run_train", fake_run_train)

    cli_module.command_run_ablation(_ablation_args(suite, registry_path, stage))

    rows = ExperimentRegistry(registry_path).read()
    assert len(rows) == 1
    assert rows[0]["run_id"] == f"isolated_p0_{stage}_smoke_seed2026"
    assert rows[0]["parent_run"] == expected_parent
    assert rows[0]["status"] == "completed"


def test_run_ablation_does_not_promote_filename_only_roi_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite.yaml"
    registry_path = tmp_path / "registry.csv"
    _write_ablation_suite(suite)
    filename_only_audit = _verified_context_audit()
    filename_only_audit["context_enabled"] = False
    filename_only_audit["boundary"] = {
        "direction_verified": False,
        "continuity_verified": True,
    }
    filename_only_audit["context_gate_reasons"] = [
        "coordinate_direction_not_verified"
    ]
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda *_args, **_kwargs: _ablation_config(),
    )
    monkeypatch.setattr(
        cli_module,
        "_context_audit_for_config",
        lambda *_args, **_kwargs: filename_only_audit,
    )

    def fake_run_train(
        config: dict[str, Any],
        data_root: str | Path | None,
        run_id: str,
    ) -> dict[str, str]:
        del config, data_root
        return {"run_dir": str(tmp_path / "outputs" / run_id)}

    monkeypatch.setattr(cli_module, "_run_train", fake_run_train)

    result = cli_module.command_run_ablation(
        _ablation_args(suite, registry_path, "A2")
    )

    row = ExperimentRegistry(registry_path).read()[0]
    assert result["stages"]["A2"]["status"] == "not_promotable"
    assert row["status"] == "not_promotable"
    assert "coordinate_direction_not_verified" in row["failure_reason"]
    assert "context_gate_disabled" in row["failure_reason"]


def test_run_ablation_confirm_expands_two_independent_fold_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "confirm_suite.yaml"
    suite.write_text(
        """suite: confirm_p0
order: [A0]
stages:
  A0: {config: a0.yaml}
""",
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.csv"
    config = _ablation_config()
    config["budget"]["confirm"] = {
        "epochs": 80,
        "folds": [0, 1],
        "seeds": [2026],
    }
    observed: list[tuple[str, int, int, bool]] = []
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        cli_module,
        "_context_audit_for_config",
        lambda *_args, **_kwargs: _verified_context_audit(),
    )

    def fake_run_train(
        effective: dict[str, Any],
        data_root: str | Path | None,
        run_id: str,
    ) -> dict[str, str]:
        del data_root
        observed.append(
            (
                run_id,
                int(effective["project"]["fold"]),
                int(effective["train"]["epochs"]),
                bool(effective["data"]["grouped_inner_folds"]["enabled"]),
            )
        )
        return {"run_dir": str(tmp_path / "outputs" / run_id)}

    monkeypatch.setattr(cli_module, "_run_train", fake_run_train)
    args = _ablation_args(suite, registry_path, "A0")
    args.budget = "confirm"
    args.max_epochs = None

    result = cli_module.command_run_ablation(args)

    assert observed == [
        ("confirm_p0_A0_confirm_fold0_seed2026", 0, 80, True),
        ("confirm_p0_A0_confirm_fold1_seed2026", 1, 80, True),
    ]
    assert result["stages"]["A0"]["budget_plan"]["evidence_count"] == 2
    assert len(ExperimentRegistry(registry_path).read()) == 2


def test_run_ablation_confirm_hard_blocks_unverified_grid_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "confirm_suite.yaml"
    suite.write_text(
        """suite: confirm_p0
order: [A0]
stages:
  A0: {config: a0.yaml}
""",
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.csv"
    config = _ablation_config()
    config["budget"]["confirm"] = {
        "epochs": 80,
        "folds": [0, 1],
        "seeds": [2026],
    }
    audit = _verified_context_audit()
    audit["filename_grid_verified"] = False
    audit["context_enabled"] = False
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        cli_module,
        "_context_audit_for_config",
        lambda *_args, **_kwargs: audit,
    )

    def fail_if_training_starts(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("confirm training must not start on an unverified grid")

    monkeypatch.setattr(cli_module, "_run_train", fail_if_training_starts)
    args = _ablation_args(suite, registry_path, "A0")
    args.budget = "confirm"
    args.max_epochs = None

    result = cli_module.command_run_ablation(args)

    assert result["stages"]["A0"]["status"] == "blocked_unverified_grid"
    assert "confirm_requires_verified_grouped_roi_grid" in result["stages"]["A0"]["reasons"]
    assert all(
        row["status"] == "blocked_unverified_grid"
        for row in ExperimentRegistry(registry_path).read()
    )


def test_run_ablation_report_is_evidence_bound_and_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """suite: immutable_report
order: [A0]
stages:
  A0: {config: a0.yaml}
""",
        encoding="utf-8",
    )
    registry_path = tmp_path / "custom_registry.csv"
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda *_args, **_kwargs: _ablation_config(),
    )
    monkeypatch.setattr(
        cli_module,
        "_context_audit_for_config",
        lambda *_args, **_kwargs: _verified_context_audit(),
    )
    calls = 0

    def fake_run_train(
        config: dict[str, Any],
        data_root: str | Path | None,
        run_id: str,
    ) -> dict[str, str]:
        nonlocal calls
        del config, data_root
        calls += 1
        return {"run_dir": str(tmp_path / "outputs" / run_id)}

    monkeypatch.setattr(cli_module, "_run_train", fake_run_train)
    args = _ablation_args(suite, registry_path, "A0")

    result = cli_module.command_run_ablation(args)

    report = Path(result["report"])
    assert report.name == (
        "custom_registry_immutable_report_smoke_fold0_seed2026_report.json"
    )
    assert report.is_file()
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        cli_module.command_run_ablation(args)
    assert calls == 1


def test_roi_audit_hard_blocks_numeric_stems_even_when_images_exist(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for index, split in enumerate(("train", "val")):
        path = tmp_path / "样例 数据" / "colon" / "DAPI" / f"{index:05d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((6, 7), 20 + index, dtype=np.uint8), mode="L").save(path)
        rows.append(
            {
                "organ": "colon",
                "split": split,
                "stem": f"{index:05d}",
                "canonical_key": f"colon/{index:05d}",
                "dapi_path": path.relative_to(tmp_path).as_posix(),
                "roi_id": "surrogate_00000",
                "roi_id_source": "surrogate_numeric_block",
            }
        )

    result = audit_roi_grid(rows, tmp_path, border_width=1)

    assert result.total_rows == 2
    assert result.parsed_rows == 0
    assert result.filename_grid_verified is False
    assert result.context_enabled is False
    assert "unverified_filename_coordinates" in result.context_gate_reasons
    assert "coordinate_direction_not_verified" in result.context_gate_reasons
    assert "boundary_continuity_not_verified" in result.context_gate_reasons


def test_checkpoint_runtime_config_enables_required_context_loader() -> None:
    config = {
        "data": {"targets": ["CD68"]},
        "model": {"name": "camp_vs_v2", "context": {"enabled": False}},
        "inference": {"context": False},
    }
    payload = {
        "config": {
            "data": {"targets": ["CD68"], "input_channels": 1},
            "model": {
                "name": "camp_vs_v2",
                "context": {"enabled": True, "require_verified_grid": True},
            },
        }
    }

    runtime = cli_module._checkpoint_runtime_config(config, payload)

    assert runtime["model"]["context"]["enabled"] is True
    assert runtime["inference"]["context"] is True
    assert runtime["data"]["input_channels"] == 1


def test_evaluation_weight_source_uses_checkpoint_selection_and_fallback() -> None:
    config = {"inference": {"weight_source": "best_jpg", "use_ema": True}}
    raw_payload = {"extra": {"selected_weight_source": "raw"}, "ema": {"shadow": {}}}
    assert cli_module._evaluation_weight_source(config, raw_payload) == "raw"

    missing_ema = {"extra": {"selected_weight_source": "ema"}}
    assert cli_module._evaluation_weight_source(config, missing_ema) == "raw"
    with pytest.raises(ValueError, match="no EMA"):
        cli_module._evaluation_weight_source(
            {"inference": {"weight_source": "ema"}}, missing_ema
        )


def test_averaged_restorer_forwards_context_kwargs() -> None:
    class ContextModel(torch.nn.Module):
        def forward(
            self,
            inputs: torch.Tensor,
            *,
            context_tiles: torch.Tensor,
            context_valid_mask: torch.Tensor,
            context_offsets: torch.Tensor,
            task_name: str | None = None,
        ) -> dict[str, torch.Tensor]:
            del context_valid_mask, context_offsets
            assert task_name == "CD68"
            return {"CD68": inputs + context_tiles[:, 0]}

    averaged = cli_module._AveragedRestorer(
        [ContextModel(), ContextModel()], [0.25, 0.75]
    )
    inputs = torch.ones(1, 1, 4, 4)
    output = averaged(
        inputs,
        task_name="CD68",
        context_tiles=torch.ones(1, 9, 1, 4, 4),
        context_valid_mask=torch.ones(1, 9),
        context_offsets=torch.zeros(1, 9, 2, dtype=torch.int64),
    )
    assert torch.equal(output.predictions["CD68"], torch.full_like(inputs, 2.0))


def test_build_model_soup_persists_hashes_and_common_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = torch.nn.Linear(2, 2)
    parent_checkpoint_sha256 = "a" * 64
    provenance = build_initialization_provenance(
        initial.state_dict(),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )
    config = {
        "project": {"initialization_provenance": provenance},
        "data": {"targets": ["CD68"], "max_val_samples": None},
        "validation": {"primary_domain": "jpg"},
        "ensemble": {
            "allow_unsafe_model_soup_lineage": False,
            "allow_unsafe_model_soup_validation": False,
        },
    }
    checkpoints: list[Path] = []
    for index, value in enumerate((0.1, 0.3)):
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.fill_(value)
            model.bias.fill_(value)
        checkpoints.append(
            save_checkpoint(
                tmp_path / f"member_{index}.ckpt",
                model,
                config=config,
            )
        )
    fields = (
        "canonical_key",
        "organ",
        "split",
        "source_split",
        "stem",
        "roi_id",
        "roi_id_source",
    )
    training_manifest = tmp_path / "train.csv"
    validation_manifest = tmp_path / "val.csv"
    with training_manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "canonical_key": "colon/ROI000_00_00",
                "organ": "colon",
                "split": "train",
                "source_split": "train",
                "stem": "ROI000_00_00",
                "roi_id": "ROI000",
                "roi_id_source": "filename_regex",
            }
        )
    with validation_manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "canonical_key": "colon/ROI001_00_00",
                "organ": "colon",
                "split": "val",
                "source_split": "val",
                "stem": "ROI001_00_00",
                "roi_id": "ROI001",
                "roi_id_source": "filename_regex",
            }
        )
    monkeypatch.setattr(cli_module, "_load_effective", lambda _args: config)
    monkeypatch.setattr(cli_module, "_apply_hardware_defaults", lambda _config: None)
    monkeypatch.setattr(
        cli_module,
        "_resolve_discovery",
        lambda *_args, **_kwargs: SimpleNamespace(selected_root=tmp_path),
    )
    monkeypatch.setattr(
        cli_module,
        "_ensure_manifests",
        lambda *_args, **_kwargs: SimpleNamespace(
            train_manifest=training_manifest,
            val_manifest=validation_manifest,
        ),
    )
    def fake_roi_audit(
        _rows: Any,
        _root: Any,
        output_dir: str | Path,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_rows": 2,
            "parsed_rows": 2,
            "filename_grid_verified": True,
            "duplicate_coordinates": [],
            "train_val_shared_rois": [],
            "cross_split_adjacent_pairs": [],
            "boundary": {
                "direction_verified": True,
                "continuity_verified": True,
            },
            "context_gate_reasons": [],
            "context_enabled": True,
        }
        audit_path = destination / "roi_grid_audit.json"
        audit_path.write_text(json.dumps(payload), encoding="utf-8")
        return {**payload, "audit_path": str(audit_path)}

    monkeypatch.setattr(cli_module, "write_roi_grid_audit", fake_roi_audit)
    first_payload = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    monkeypatch.setattr(
        cli_module,
        "_load_model_for_evaluation",
        lambda *_args, **_kwargs: (torch.nn.Linear(2, 2), first_payload),
    )
    monkeypatch.setattr(
        cli_module,
        "_checkpoint_runtime_config",
        lambda runtime, _payload: runtime,
    )
    monkeypatch.setattr(
        cli_module,
        "_make_loader",
        lambda *_args, **_kwargs: SimpleNamespace(dataset=[0]),
    )

    class FakeValidator:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.initialized = True

        def evaluate(
            self, *, records_path: str | Path | None = None
        ) -> dict[str, Any]:
            records = [
                {
                    "target": "CD68",
                    "canonical_key": "colon/ROI001_00_00",
                    "jpg_ssim": 0.5,
                    "jpg_psnr": 20.0,
                }
            ]
            if records_path is not None:
                destination = Path(records_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open(
                    "w", newline="", encoding="utf-8-sig"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)
            return {
                "domains": {
                    "jpg": {
                        "per_target": {
                            "CD68": {"local_proxy_score": 0.5}
                        },
                        "macro": {"local_proxy_score": 0.5},
                    }
                },
                "records": records,
            }

    monkeypatch.setattr(cli_module, "Validator", FakeValidator)
    output = tmp_path / "safe_soup.ckpt"
    report_path = tmp_path / "safe_soup_provenance.json"
    args = argparse.Namespace(
        checkpoints=[str(path) for path in checkpoints],
        validation_scores=[0.8, 0.7],
        weight_source="raw",
        allow_unsafe_lineage_mismatch=False,
        allow_unsafe_engineering_validation=False,
        target_marker="CD68",
        metric="local_proxy_score",
        min_improvement=0.0,
        output=str(output),
        report=str(report_path),
        data_root=None,
    )

    result = cli_module.command_build_model_soup(args)

    payload = torch.load(output, map_location="cpu", weights_only=False)
    stored = payload["extra"]["model_soup_provenance"]
    assert result["checkpoint_sha256"] == checkpoint_file_sha256(output)
    assert stored["safety_mode"] == "strict"
    assert stored["common_initialization_lineage"] == provenance[
        "initialization_lineage"
    ]
    assert stored["common_parent_checkpoint_sha256"] == parent_checkpoint_sha256
    assert stored["common_pretrain_checkpoint_sha256"] is None
    assert stored["validation_contract"]["authoritative"] is True
    assert stored["validation_contract"]["metric_domain"] == "jpg_roundtrip"
    assert [
        item["checkpoint_sha256"] for item in stored["input_checkpoints"]
    ] == [checkpoint_file_sha256(path) for path in checkpoints]
    assert report_path.is_file()


def test_run_train_records_exact_initial_state_lineage_before_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config: dict[str, Any] = {
        "project": {
            "seed": 2026,
            "output_root": str(tmp_path / "outputs"),
            "artifact_root": str(tmp_path / "artifacts"),
        },
        "data": {"targets": ["CD68"], "input_channels": 1, "image_size": 8},
        "model": {"context": {"enabled": False}},
        "train": {"epochs": 1},
        "inference": {},
    }
    model = torch.nn.Conv2d(1, 1, 1)

    class FakeTrainer:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.stage_controller = None
            self.training_manifest_hash = "train-hash"
            self.validation_manifest_hash = "val-hash"
            self.dapi_pretrain_manifest_hash = "dapi-hash"
            self.fold_provenance = None
            self.activity_report = None

        def fit(self, **_kwargs: Any) -> list[Any]:
            recorded = config["project"]["initialization_provenance"]
            assert recorded["initial_state_sha256"] == state_dict_value_sha256(
                model.state_dict()
            )
            return []

    trainer = FakeTrainer()
    manifests = SimpleNamespace(val_manifest=tmp_path / "val.csv")
    monkeypatch.setattr(cli_module, "_apply_hardware_defaults", lambda _config: None)
    monkeypatch.setattr(
        cli_module,
        "_resolve_discovery",
        lambda *_args, **_kwargs: SimpleNamespace(selected_root=tmp_path),
    )
    monkeypatch.setattr(
        cli_module, "_ensure_manifests", lambda *_args, **_kwargs: manifests
    )
    monkeypatch.setattr(
        cli_module,
        "_prepare_training",
        lambda *_args, **_kwargs: (trainer, object(), model, object(), object()),
    )
    monkeypatch.setattr(
        cli_module,
        "model_statistics",
        lambda *_args, **_kwargs: {"parameters": 2, "approximate_macs": 1},
    )
    monkeypatch.setattr(cli_module, "_append_experiment", lambda *_args: None)

    cli_module._run_train(config, tmp_path, "lineage_contract")

    recorded = config["project"]["initialization_provenance"]
    assert recorded["initialization_lineage"] == recorded["initial_state_sha256"]
    assert recorded["parent_checkpoint_sha256"] is None
    assert recorded["pretrain_checkpoint_sha256"] is None
