"""Prototype usage aggregation, persistence, and Trainer compatibility tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from virtual_staining.engine.prototype_monitor import PrototypeUsageMonitor
from virtual_staining.engine.trainer import Trainer
from virtual_staining.models import RestorationOutput


class _UsageModel(torch.nn.Module):
    def __init__(self, *, diagnostics: bool = True) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.diagnostics = diagnostics

    def forward(self, inputs: torch.Tensor, task_name: str | None = None) -> RestorationOutput:
        prediction = torch.sigmoid(inputs * self.scale)
        usage = (
            {
                "output": {
                    "8/shared": torch.tensor(
                        [0.7, 0.29995, 0.00005],
                        device=inputs.device,
                    )
                }
            }
            if self.diagnostics
            else {}
        )
        return RestorationOutput(
            predictions={"output": prediction},
            prototype_usage=usage,
        )


class _ResetUsageModel(_UsageModel):
    def __init__(self) -> None:
        super().__init__()
        self.prototype_bank = torch.nn.Parameter(
            torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
        )

    def forward(self, inputs: torch.Tensor, task_name: str | None = None) -> RestorationOutput:
        output = super().forward(inputs, task_name)
        output.predictions["output"] = (
            output.predictions["output"] + self.prototype_bank.mean() * 0.0
        )
        return output

    @staticmethod
    def resolve_prototype_bank_key(diagnostic: str) -> str | None:
        return "shared" if diagnostic == "output/8/shared" else None

    def prototype_bank_parameters(self) -> dict[str, torch.nn.Parameter]:
        return {"shared": self.prototype_bank}

    @torch.no_grad()
    def reset_prototype_rows(
        self,
        rows_by_bank: dict[str, list[int]],
        *,
        seed: int,
        std: float,
    ) -> list[dict[str, int | float | str]]:
        if set(rows_by_bank) != {"shared"}:
            raise KeyError("Unexpected prototype bank")
        records: list[dict[str, int | float | str]] = []
        for index in sorted(set(rows_by_bank["shared"])):
            generator = torch.Generator(device="cpu").manual_seed(seed + index)
            replacement = torch.empty(1).normal_(
                mean=0.0,
                std=std,
                generator=generator,
            )
            self.prototype_bank[index].copy_(replacement)
            records.append(
                {
                    "bank_key": "shared",
                    "prototype_index": index,
                    "row_seed": seed + index,
                    "std": std,
                }
            )
        return records


def test_prototype_monitor_aggregates_dead_usage_and_persists(tmp_path: Path) -> None:
    monitor = PrototypeUsageMonitor(dead_threshold=1e-4)
    monitor.bind_output_dir(tmp_path)
    monitor.start_epoch(3)
    first = RestorationOutput(
        predictions={"CD68": torch.zeros(1, 1, 2, 2)},
        prototype_usage={
            "CD68": {"8/shared": torch.tensor([0.7, 0.29995, 0.00005])}
        },
    )
    second = RestorationOutput(
        predictions={"CD68": torch.zeros(1, 1, 2, 2)},
        prototype_usage={
            "CD68": {"8/shared": torch.tensor([0.6, 0.39995, 0.00005])}
        },
    )

    assert monitor.observe(first)
    assert monitor.observe(second)
    summary = monitor.finalize_epoch()

    assert summary["observed_outputs"] == 2
    assert summary["diagnostic_count"] == 1
    assert summary["total_prototypes"] == 3
    assert summary["dead_prototypes"] == 1
    assert summary["dead_fraction"] == 1 / 3
    assert summary["nonfinite_values"] == 0

    payload = json.loads((tmp_path / "prototype_usage.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["epochs"][0]["epoch"] == 3
    with (tmp_path / "prototype_usage.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[-1]["diagnostic"] == "CD68/8/shared"
    assert rows[-1]["dead"] == "True"
    assert rows[-1]["observations"] == "2"


def _loader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    inputs = torch.linspace(0.0, 1.0, 32).reshape(2, 1, 4, 4)
    targets = inputs * 0.8 + 0.1
    return DataLoader(TensorDataset(inputs, targets), batch_size=1, num_workers=0)


def test_trainer_writes_monitor_artifacts_only_when_enabled(tmp_path: Path) -> None:
    config = {
        "train": {
            "amp": False,
            "ema": False,
            "gradient_accumulation": 1,
            "prototype_monitor": {"enabled": True, "dead_threshold": 1e-4},
        }
    }
    run_dir = tmp_path / "enabled_run"
    trainer = Trainer(
        _UsageModel(),
        _loader(),
        torch.nn.MSELoss(),
        device="cpu",
        config=config,
    )

    history = trainer.fit(epochs=1, checkpoint_dir=run_dir / "checkpoints")

    assert history[0]["train"]["prototype/dead_prototypes"] == 1
    assert history[0]["train"]["prototype/observed_outputs"] == 2
    assert (run_dir / "artifacts" / "prototype_usage.json").is_file()
    assert (run_dir / "artifacts" / "prototype_usage.csv").is_file()

    disabled_run = tmp_path / "disabled_run"
    disabled = Trainer(
        _UsageModel(),
        _loader(),
        torch.nn.MSELoss(),
        device="cpu",
        config={"train": {"amp": False, "ema": False}},
    )
    disabled.fit(epochs=1, checkpoint_dir=disabled_run / "checkpoints")
    assert not (disabled_run / "artifacts").exists()


def test_enabled_monitor_does_not_require_prototype_diagnostics(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy_run"
    trainer = Trainer(
        _UsageModel(diagnostics=False),
        _loader(),
        torch.nn.MSELoss(),
        device="cpu",
        config={
            "train": {
                "amp": False,
                "ema": False,
                "prototype_monitor": {"enabled": True},
            }
        },
    )

    history = trainer.fit(epochs=1, checkpoint_dir=run_dir / "checkpoints")

    assert not any(key.startswith("prototype/") for key in history[0]["train"])
    payload = json.loads(
        (run_dir / "artifacts" / "prototype_usage.json").read_text(encoding="utf-8")
    )
    assert payload["epochs"][0]["total_prototypes"] == 0
    assert payload["epochs"][0]["rows"] == []


def test_prototype_monitor_resume_preserves_consecutive_dead_streaks() -> None:
    monitor = PrototypeUsageMonitor(dead_threshold=1e-4)
    for epoch in range(2):
        monitor.start_epoch(epoch)
        monitor.observe(
            RestorationOutput(
                predictions={"CD68": torch.zeros(1, 1, 2, 2)},
                prototype_usage={
                    "CD68": {"8/shared": torch.tensor([0.8, 0.19995, 0.00005])}
                },
            )
        )
        monitor.finalize_epoch(epoch)

    restored = PrototypeUsageMonitor(dead_threshold=1e-4)
    restored.load_state_dict(monitor.state_dict())

    rows = restored.latest_rows_with_streaks()
    assert rows[0]["dead_streak"] == 0
    assert rows[2]["dead_streak"] == 2
    assert restored.state_dict() == monitor.state_dict()


def test_trainer_resets_only_consecutively_dead_rows_when_explicitly_enabled(
    tmp_path: Path,
) -> None:
    model = _ResetUsageModel()
    original = model.prototype_bank.detach().clone()
    trainer = Trainer(
        model,
        _loader(),
        torch.nn.MSELoss(),
        device="cpu",
        config={
            "project": {"seed": 2026},
            "model": {
                "prototypes": {
                    "reset_dead": True,
                    "reset_patience": 2,
                    "reset_seed": 2026,
                    "reset_std": 0.02,
                }
            },
            "train": {
                "amp": False,
                "ema": True,
                "weight_decay": 0.0,
                "gradient_accumulation": 1,
                "prototype_monitor": {
                    "enabled": True,
                    "dead_threshold": 1e-4,
                },
            },
        },
    )

    history = trainer.fit(
        epochs=2,
        checkpoint_dir=tmp_path / "reset_run" / "checkpoints",
    )

    assert history[0]["train"].get("prototype/reset_rows", 0) == 0
    assert history[1]["train"]["prototype/reset_rows"] == 1
    assert torch.equal(model.prototype_bank[:2], original[:2])
    assert not torch.equal(model.prototype_bank[2], original[2])
    assert trainer.prototype_monitor is not None
    assert trainer.prototype_monitor.reset_events[0]["seed"] == 2027
    last_checkpoint = torch.load(
        tmp_path / "reset_run" / "checkpoints" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert last_checkpoint["extra"]["prototype_reset"] == {
        "enabled": True,
        "patience": 2,
        "seed": 2026,
        "std": 0.02,
    }
