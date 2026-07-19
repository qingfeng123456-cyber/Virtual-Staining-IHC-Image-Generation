"""Read-only prototype diagnostic aggregation and persistence tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from virtual_staining.engine.prototype_diagnostics import (
    PrototypeDiagnosticsAggregator,
    write_prototype_diagnostics,
)
from virtual_staining.engine.validator import Validator
from virtual_staining.models import RestorationOutput


class _DiagnosticRestorer(nn.Module):
    def forward(
        self, inputs: torch.Tensor, *, task_name: str | None = None
    ) -> RestorationOutput:
        assert task_name == "CD68"
        batch = inputs.shape[0]
        return RestorationOutput(
            predictions={"CD68": inputs.clamp(0.0, 1.0)},
            prototype_usage={
                "CD68": {
                    "8/shared": inputs.new_tensor([[0.75, 0.25]]).repeat(
                        batch, 1
                    )
                }
            },
            prototype_banks={
                "8/shared": inputs.new_tensor([[1.0, 0.0], [0.0, 1.0]])
            },
        )


class _AttentionDiagnosticRestorer(nn.Module):
    def forward(
        self, inputs: torch.Tensor, *, task_name: str | None = None
    ) -> RestorationOutput:
        assert task_name == "CD68"
        attention = torch.stack((inputs, 1.0 - inputs), dim=1).mean(dim=2)
        return RestorationOutput(
            predictions={"CD68": inputs.clamp(0.0, 1.0)},
            prototype_attention={"CD68": {"8/shared": attention}},
            prototype_banks={"8/shared": inputs.new_tensor([[1.0], [-1.0]])},
        )


def _camp_output(shared_bank: torch.Tensor, attention: torch.Tensor) -> RestorationOutput:
    usage = attention.mean(dim=(0, 2, 3))
    return RestorationOutput(
        predictions={"CD68": torch.zeros(attention.shape[0], 1, 2, 2)},
        prototype_attention={"CD68": {"8/shared": attention}},
        prototype_usage={"CD68": {"8/shared": usage}},
        prototype_banks={"8/shared": shared_bank},
    )


def test_prototype_diagnostics_use_attention_weighting_and_write_all_files(
    tmp_path: Path,
) -> None:
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    first_attention = torch.tensor(
        [[[[0.8]], [[0.2]], [[0.0]]]],
        dtype=torch.float32,
    )
    second_attention = torch.tensor(
        [
            [[[0.1]], [[0.7]], [[0.2]]],
            [[[0.1]], [[0.7]], [[0.2]]],
        ],
        dtype=torch.float32,
    )
    aggregator = PrototypeDiagnosticsAggregator(dead_threshold=0.15)
    assert aggregator.observe(_camp_output(bank, first_attention))
    assert aggregator.observe(_camp_output(bank, second_attention))

    metadata = aggregator.write(tmp_path)

    expected_files = {
        "prototype_usage.csv",
        "dead_prototypes.csv",
        "prototype_similarity.npy",
        "prototype_diagnostics.json",
    }
    assert expected_files == {path.name for path in tmp_path.iterdir()}
    with (tmp_path / "prototype_usage.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["bank_key"] == "8/shared"
    assert rows[0]["source"] == "attention"
    assert rows[0]["attention_elements"] == "3"
    assert float(rows[0]["mean_usage"]) == pytest.approx(1.0 / 3.0)
    assert float(rows[2]["mean_usage"]) == pytest.approx(2.0 / 15.0)
    with (tmp_path / "dead_prototypes.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        dead_rows = list(csv.DictReader(handle))
    assert [int(row["prototype_index"]) for row in dead_rows] == [2]

    similarity = np.load(tmp_path / "prototype_similarity.npy", allow_pickle=False)
    assert similarity.shape == (3, 3)
    assert similarity[0, 1] == pytest.approx(0.0)
    assert similarity[0, 2] == pytest.approx(2**-0.5)
    persisted = json.loads(
        (tmp_path / "prototype_diagnostics.json").read_text(encoding="utf-8")
    )
    assert persisted == metadata
    assert metadata["reset_performed"] is False
    assert metadata["dead_prototypes"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_attention_without_usage_resolves_task_bank(tmp_path: Path) -> None:
    output = RestorationOutput(
        predictions={"CD68": torch.zeros(1, 1, 1, 1)},
        prototype_attention={
            "CD68": {
                "shared": torch.tensor([[[[0.75]], [[0.25]]]]),
                "task": torch.tensor([[[[0.4]], [[0.6]]]]),
            }
        },
        prototype_banks={
            "shared": torch.eye(2),
            "CD68": torch.tensor([[1.0, 1.0], [1.0, -1.0]]),
        },
    )

    metadata = write_prototype_diagnostics([output], tmp_path)

    assert metadata["status"] == "complete"
    with (tmp_path / "prototype_usage.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    bank_keys = {(row["diagnostic"], row["bank_key"]) for row in rows}
    assert bank_keys == {("shared", "shared"), ("task", "CD68")}


def test_empty_collection_is_explicit_error_or_no_data_record(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No prototype"):
        write_prototype_diagnostics([], tmp_path / "error")
    assert not (tmp_path / "error").exists()

    metadata = write_prototype_diagnostics(
        [], tmp_path / "recorded", allow_empty=True
    )

    assert metadata["status"] == "no_data"
    assert metadata["warnings"] == [
        "no_usage_or_attention_observed",
        "no_prototype_banks_observed",
    ]
    similarity = np.load(
        tmp_path / "recorded" / "prototype_similarity.npy", allow_pickle=False
    )
    assert similarity.shape == (0, 0)


def test_diagnostics_reject_mixed_model_states_and_leave_banks_unchanged() -> None:
    bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    original = bank.clone()
    attention = torch.tensor([[[[0.5]], [[0.5]]]])
    aggregator = PrototypeDiagnosticsAggregator()
    aggregator.observe(_camp_output(bank, attention))
    assert torch.equal(bank, original)

    changed_bank = bank.clone()
    changed_bank[0, 0] = 0.5
    with pytest.raises(ValueError, match="changed across observed outputs"):
        aggregator.observe(_camp_output(changed_bank, attention))
    assert torch.equal(bank, original)


def test_similarity_marks_cross_dimension_values_as_unavailable(tmp_path: Path) -> None:
    output = RestorationOutput(
        predictions={"CD68": torch.zeros(1, 1, 1, 1)},
        prototype_banks={
            "small": torch.eye(2),
            "wide": torch.eye(3),
        },
    )

    metadata = write_prototype_diagnostics([output], tmp_path)

    similarity = np.load(tmp_path / "prototype_similarity.npy", allow_pickle=False)
    assert similarity.shape == (5, 5)
    assert np.isnan(similarity[:2, 2:]).all()
    assert metadata["status"] == "partial"
    assert metadata["nonfinite_similarity_values"] == 12


def test_validator_writes_feature_flagged_prototype_diagnostics(
    tmp_path: Path,
) -> None:
    image = torch.rand(1, 1, 16, 16)
    loader = [
        {
            "input": image,
            "target": image.clone(),
            "stem": ["ROI000_00_00"],
            "canonical_key": ["colon/ROI000_00_00"],
            "roi_id": ["ROI000"],
            "organ": ["colon"],
        }
    ]
    output_dir = tmp_path / "原型 诊断"
    validator = Validator(
        _DiagnosticRestorer(),
        loader,
        device="cpu",
        task_name="CD68",
        config={
            "train": {
                "amp": False,
                "prototype_monitor": {
                    "enabled": True,
                    "dead_threshold": 1e-4,
                },
            },
            "validation": {"primary_domain": "jpg"},
            "inference": {"jpeg_quality": 100, "jpeg_subsampling": 0},
        },
        prototype_diagnostics_dir=output_dir,
    )
    validator.prototype_diagnostics_epoch = 3
    validator.prototype_diagnostics_weight_source = "raw"

    report = validator.evaluate()
    destination = output_dir / "epoch_0003" / "weight_raw"

    assert report["prototype_diagnostics"]["status"] == "complete"
    assert (destination / "prototype_usage.csv").is_file()
    assert (destination / "dead_prototypes.csv").is_file()
    assert (destination / "prototype_similarity.npy").is_file()
    assert (destination / "prototype_diagnostics.json").is_file()


def _visual_metadata(canonical_key: str, *, split: str = "val") -> dict[str, list[str]]:
    return {
        "canonical_key": [canonical_key],
        "stem": [canonical_key.rsplit("/", 1)[-1]],
        "split": [split],
        "organ": [canonical_key.split("/", 1)[0]],
    }


def _visual_output(value: float) -> RestorationOutput:
    attention = torch.tensor(
        [
            [
                [[value, 1.0 - value], [0.25, 0.75]],
                [[1.0 - value, value], [0.75, 0.25]],
            ]
        ],
        dtype=torch.float32,
    )
    return _camp_output(torch.eye(2), attention)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_attention_visuals_select_fixed_samples_and_are_order_deterministic(
    tmp_path: Path,
) -> None:
    samples = (
        ("colon/ROI002_00_00", 0.2),
        ("colon/ROI000_00_00", 0.4),
        ("colon/ROI001_00_00", 0.6),
    )

    def produce(destination: Path, values: tuple[tuple[str, float], ...]) -> dict:
        aggregator = PrototypeDiagnosticsAggregator(
            attention_visuals_enabled=True,
            attention_visual_count=2,
            attention_visual_seed=91,
            attention_visual_size=32,
        )
        for canonical_key, value in values:
            aggregator.observe(
                _visual_output(value), metadata=_visual_metadata(canonical_key)
            )
        return aggregator.write(destination)

    first = tmp_path / "first"
    second = tmp_path / "second"
    first_report = produce(first, samples)
    second_report = produce(second, tuple(reversed(samples)))

    first_visuals = first / "prototype_attention_visuals"
    second_visuals = second / "prototype_attention_visuals"
    assert _tree_bytes(first_visuals) == _tree_bytes(second_visuals)
    assert first_report["attention_visuals"] == second_report["attention_visuals"]
    assert first_report["attention_visuals"]["candidate_samples"] == 3
    assert first_report["attention_visuals"]["selected_samples"] == 2
    assert first_report["attention_visuals"]["png_count"] == 4

    manifest = json.loads(
        (first_visuals / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["selection"]["method"] == (
        "smallest_sha256_of_seed_and_canonical_key"
    )
    assert all(sample["split"] == "val" for sample in manifest["samples"])
    pngs = sorted(first_visuals.rglob("*.png"))
    assert len(pngs) == 4
    assert all("__prototype_" in path.name for path in pngs)
    assert all(path.name.startswith("task_CD68__diag_8-shared") for path in pngs)
    with Image.open(pngs[0]) as image:
        assert image.mode == "RGB"
        assert image.size == (32, 32)


def test_attention_visuals_reject_test_split_before_writing(tmp_path: Path) -> None:
    aggregator = PrototypeDiagnosticsAggregator(
        attention_visuals_enabled=True,
        attention_visual_count=1,
    )

    with pytest.raises(RuntimeError, match="cannot observe test-split"):
        aggregator.observe(
            _visual_output(0.5),
            metadata=_visual_metadata("colon/ROI000_00_00", split="official_test"),
        )

    assert not list(tmp_path.iterdir())


def test_validator_integrates_attention_visual_feature_flag(tmp_path: Path) -> None:
    image = torch.linspace(0.0, 1.0, steps=64).reshape(1, 1, 8, 8)
    loader = [
        {
            "input": image,
            "target": image.clone(),
            "stem": ["ROI000_00_00"],
            "canonical_key": ["colon/ROI000_00_00"],
            "roi_id": ["ROI000"],
            "organ": ["colon"],
            "split": ["val"],
        }
    ]
    destination = tmp_path / "prototype_validation"
    validator = Validator(
        _AttentionDiagnosticRestorer(),
        loader,
        device="cpu",
        task_name="CD68",
        config={
            "project": {"seed": 33},
            "train": {
                "amp": False,
                "prototype_monitor": {
                    "enabled": True,
                    "dead_threshold": 1e-4,
                    "attention_visuals_enabled": True,
                    "attention_visual_count": 1,
                    "attention_visual_seed": 33,
                    "attention_visual_size": 24,
                },
            },
            "validation": {"primary_domain": "jpg"},
            "inference": {"jpeg_quality": 100, "jpeg_subsampling": 0},
        },
        prototype_diagnostics_dir=destination,
    )

    report = validator.evaluate()

    visual_report = report["prototype_diagnostics"]["attention_visuals"]
    assert visual_report["status"] == "complete"
    assert visual_report["selected_samples"] == 1
    assert visual_report["png_count"] == 2
    assert (destination / "prototype_attention_visuals" / "manifest.json").is_file()
