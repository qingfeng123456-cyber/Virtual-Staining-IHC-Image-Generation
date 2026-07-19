"""Focused regressions for conditioning, initialization, and learned loss state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from virtual_staining.engine.checkpoint import load_checkpoint, save_checkpoint
from virtual_staining.engine.common import call_model, model_kwargs_from_metadata
from virtual_staining.engine.ema import ExponentialMovingAverage
from virtual_staining.engine.experiment_registry import ExperimentRegistry
from virtual_staining.engine.multitask_optimizer import UncertaintyTaskBalancer
from virtual_staining.engine.pretrainer import (
    DAPIPretrainer,
    transfer_local_encoder_from_checkpoint,
)
from virtual_staining.engine.trainer import Trainer
from virtual_staining.losses.composite import LossOutput
from virtual_staining.losses.scheduled_composite import ScheduledCompositeLoss
from virtual_staining.models.dapi_mae import DAPIMaskedAutoencoder
from virtual_staining.models.naf_local_encoder import NAFLocalEncoder


class _OrganAwareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_organ_id: Any = None
        self.seen_context: torch.Tensor | None = None

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        organ_id: str | list[str] | None = None,
        context_tiles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.seen_organ_id = organ_id
        self.seen_context = context_tiles
        return inputs


class _LegacyModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + 1.0


class _EncoderOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local_encoder = NAFLocalEncoder(
            in_channels=1,
            widths=(4, 8, 16, 32),
            depths=(1, 1, 1, 1),
            use_sobel_input=False,
        )


class _WeightSourceValidator:
    def __init__(self, ema: ExponentialMovingAverage) -> None:
        self.ema = ema
        self.use_ema = False
        self.calls: list[str] = []

    def evaluate(self) -> dict[str, Any]:
        source = "ema" if self.use_ema else "raw"
        self.calls.append(source)
        score = 0.8 if source == "ema" else 0.7
        macro = {
            "mean_ssim": score,
            "mean_psnr": 20.0 + score,
            "local_proxy_score": score,
        }
        return {
            "macro": macro,
            "domains": {"jpg": {"macro": dict(macro)}},
            "records": [{"canonical_key": "colon/ROI000_00_00"}],
        }


class _TwoTaskModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local_encoder = nn.Conv2d(1, 2, kernel_size=1)
        self.first = nn.Conv2d(2, 1, kernel_size=1)
        self.second = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        shared = self.local_encoder(inputs)
        return {
            "predictions": {
                "CD68": self.first(shared),
                "Vimentin": self.second(shared),
            }
        }


class _TwoTaskLoss:
    def __call__(
        self,
        output: dict[str, dict[str, torch.Tensor]],
        targets: dict[str, torch.Tensor],
    ) -> LossOutput:
        predictions = output["predictions"]
        per_task = {
            task: torch.mean((predictions[task] - targets[task]) ** 2)
            for task in sorted(targets)
        }
        total = torch.stack(tuple(per_task.values())).mean()
        return LossOutput(total=total, components={"total": total}, per_task=per_task)


def _tiny_dapi_mae() -> DAPIMaskedAutoencoder:
    return DAPIMaskedAutoencoder(
        in_channels=1,
        widths=(4, 8, 16, 32),
        encoder_depths=(1, 1, 1, 1),
        decoder_depths=(1, 1, 1),
        use_sobel_input=False,
    )


def _trainer_config(*, ema: bool) -> dict[str, Any]:
    return {
        "train": {
            "device": "cpu",
            "ema": ema,
            "amp": False,
            "scheduler": "none",
            "lr": 1e-2,
            "weight_decay": 0.0,
        }
    }


def test_ordinary_organ_metadata_is_forwarded_with_context_and_legacy_filtering() -> None:
    inputs = torch.rand(2, 1, 8, 8)
    context = torch.rand(2, 9, 1, 8, 8)
    kwargs = model_kwargs_from_metadata(
        {"organ": ["colon", "liver"], "context_tiles": context}
    )
    conditioned = _OrganAwareModel()

    assert kwargs["organ_id"] == ["colon", "liver"]
    assert kwargs["context_tiles"] is context
    assert torch.equal(call_model(conditioned, inputs, model_kwargs=kwargs), inputs)
    assert conditioned.seen_organ_id == ["colon", "liver"]
    assert conditioned.seen_context is context

    legacy = _LegacyModel()
    assert torch.equal(call_model(legacy, inputs, model_kwargs=kwargs), inputs + 1.0)


def test_explicit_organ_id_takes_precedence_over_ordinary_organ_metadata() -> None:
    kwargs = model_kwargs_from_metadata(
        {"organ": ["colon"], "organ_id": ["stomach"]}
    )

    assert kwargs == {"organ_id": ["stomach"]}


def test_trainer_can_explicitly_sync_ema_after_initial_weight_loading() -> None:
    model = nn.Linear(3, 2)
    trainer = Trainer(
        model,
        [],
        nn.MSELoss(),
        config=_trainer_config(ema=True),
    )
    assert trainer.ema is not None
    initial_shadow = {
        name: value.clone() for name, value in trainer.ema.shadow.items()
    }
    trainer.ema.num_updates = 7
    with torch.no_grad():
        for value in trainer.model.state_dict().values():
            value.fill_(0.375)

    assert any(
        not torch.equal(initial_shadow[name], value)
        for name, value in trainer.model.state_dict().items()
    )
    assert trainer.sync_ema_from_model()
    assert trainer.ema.num_updates == 0
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(trainer.ema.shadow[name], value)


def test_ema_sync_rejects_an_active_temporary_average() -> None:
    model = nn.Linear(2, 1)
    ema = ExponentialMovingAverage(model)

    with ema.average_parameters(model), pytest.raises(RuntimeError, match="stored"):
        ema.sync_from(model)


@pytest.mark.parametrize(
    "status",
    [
        "completed_pretrain",
        "blocked_requires_oof_ensemble",
        "blocked_insufficient_ensemble_members",
        "blocked_requires_explicit_validation_ensemble",
    ],
)
def test_registry_accepts_explicit_terminal_statuses(
    tmp_path: Path,
    status: str,
) -> None:
    registry = ExperimentRegistry(tmp_path / "实验 注册表" / "experiments.csv")

    normalized = registry.upsert(
        {"run_id": "terminal_run", "pretrain": True, "status": status}
    )

    assert normalized["status"] == status
    assert registry.read()[0]["status"] == status
    with pytest.raises(ValueError, match="Unsupported experiment status"):
        registry.upsert({"run_id": "invalid", "status": "pretrain_done"})


@pytest.mark.parametrize("explicit_optimizer", [False, True])
def test_uncertainty_balancer_moves_with_loss_and_joins_optimizer(
    explicit_optimizer: bool,
) -> None:
    model = nn.Conv2d(1, 1, kernel_size=1)
    balancer = UncertaintyTaskBalancer(("CD68", "Vimentin"))
    criterion = ScheduledCompositeLoss(
        task_balancer=balancer,
        pyramid_levels=1,
    )
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=1e-2)
        if explicit_optimizer
        else None
    )
    trainer = Trainer(
        model,
        [],
        criterion,
        optimizer=optimizer,
        config=_trainer_config(ema=False),
    )
    learned = criterion.optimizer_parameters()
    optimizer_parameters = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }

    assert len(learned) == 1 and learned[0] is balancer.log_variances
    assert balancer.log_variances.device == trainer.device
    assert id(balancer.log_variances) in optimizer_parameters
    before = balancer.log_variances.detach().clone()
    trainer.optimizer.zero_grad(set_to_none=True)
    balancer.log_variances.sum().backward()
    trainer.optimizer.step()
    assert not torch.equal(balancer.log_variances.detach(), before)


def test_checkpoint_local_encoder_transfer_is_provenance_checked(tmp_path: Path) -> None:
    source = _tiny_dapi_mae()
    destination = _EncoderOwner()
    with torch.no_grad():
        for index, value in enumerate(source.local_encoder.state_dict().values()):
            value.fill_(0.01 * (index + 1))
    checkpoint = save_checkpoint(
        tmp_path / "预训练 权重" / "dapi_mae.ckpt",
        source,
        manifest_hash="official-train-fold-0",
        extra={
            "pretrain_state_version": 1,
            "training_type": "fold_local_dapi_mae",
            "uses_target_labels": False,
            "transfer_scope": ["local_encoder"],
        },
    )

    report = transfer_local_encoder_from_checkpoint(
        checkpoint,
        destination,
        expected_manifest_hash="official-train-fold-0",
    )

    assert report["transfer_type"] == "dapi_mae_local_encoder"
    assert report["source_manifest_hash"] == "official-train-fold-0"
    assert report["manifest_hash_verified"] is True
    assert report["uses_target_labels"] is False
    assert len(report["source_checkpoint_sha256"]) == 64
    assert report["transfer_scope"] == ["local_encoder"]
    assert report["transferred_tensors"] == len(source.local_encoder.state_dict())
    assert report["destination_missing_keys"] == []
    for name, value in destination.local_encoder.state_dict().items():
        assert torch.equal(value, source.local_encoder.state_dict()[name])

    with pytest.raises(ValueError, match="destination fold"):
        transfer_local_encoder_from_checkpoint(
            checkpoint,
            _EncoderOwner(),
            expected_manifest_hash="different-fold",
        )


def test_dapi_pretrainer_checkpoint_records_transfer_safety_state(tmp_path: Path) -> None:
    model = _tiny_dapi_mae()
    pretrainer = DAPIPretrainer(
        model,
        [{"input": torch.rand(1, 1, 32, 32), "split": ["train"]}],
        device="cpu",
        amp=False,
    )
    checkpoint = tmp_path / "预训练" / "last.ckpt"

    history = pretrainer.fit(
        1,
        checkpoint,
        config={"project": {"seed": 2026}},
        manifest_hash="official-fold-train",
    )
    payload = load_checkpoint(checkpoint)

    assert len(history) == 1
    assert payload["manifest_hash"] == "official-fold-train"
    assert payload["extra"] == {
        "pretrain_state_version": 1,
        "training_type": "fold_local_dapi_mae",
        "uses_target_labels": False,
        "allowed_splits": ["final_train", "official_train", "train"],
        "transfer_scope": ["local_encoder"],
    }


def test_checkpoint_local_encoder_transfer_rejects_unsafe_provenance(
    tmp_path: Path,
) -> None:
    checkpoint = save_checkpoint(
        tmp_path / "unsafe.ckpt",
        _tiny_dapi_mae(),
        manifest_hash="fold-0",
        extra={
            "training_type": "fold_local_dapi_mae",
            "uses_target_labels": True,
        },
    )

    with pytest.raises(ValueError, match="uses_target_labels=false"):
        transfer_local_encoder_from_checkpoint(checkpoint, _EncoderOwner())


def test_trainer_evaluates_raw_and_ema_and_persists_jpg_selection(
    tmp_path: Path,
) -> None:
    model = nn.Conv2d(1, 1, kernel_size=1)
    config = _trainer_config(ema=True)
    config["train"]["evaluate_weight_sources"] = ["raw", "ema"]
    trainer = Trainer(
        model,
        [{"input": torch.rand(1, 1, 4, 4), "target": torch.rand(1, 1, 4, 4)}],
        nn.MSELoss(),
        config=config,
    )
    assert trainer.ema is not None
    validator = _WeightSourceValidator(trainer.ema)

    history = trainer.fit(
        epochs=1,
        validator=validator,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    payload = load_checkpoint(tmp_path / "checkpoints" / "last.ckpt")

    assert validator.calls == ["raw", "ema"]
    assert history[0]["selected_weight_source"] == "ema"
    assert set(history[0]["validation_weight_sources"]) == {"raw", "ema"}
    assert history[0]["validation"]["macro"]["mean_ssim"] == 0.8
    assert payload["extra"]["selected_weight_source"] == "ema"


def test_trainer_rejects_ema_evaluation_without_ema_state() -> None:
    config = _trainer_config(ema=False)
    config["train"]["evaluate_weight_sources"] = ["raw", "ema"]
    trainer = Trainer(nn.Linear(2, 1), [], nn.MSELoss(), config=config)

    with pytest.raises(ValueError, match="EMA validation"):
        trainer._validation_weight_sources(object())


def test_trainer_feature_flag_writes_multitask_gradient_cosines(tmp_path: Path) -> None:
    config = _trainer_config(ema=False)
    config["multitask"] = {
        "gradient_cosine_enabled": True,
        "log_gradient_cosine_every": 1,
    }
    batches = [
        {
            "input": torch.rand(1, 1, 4, 4),
            "targets": {
                "CD68": torch.rand(1, 1, 4, 4),
                "Vimentin": torch.rand(1, 1, 4, 4),
            },
        }
        for _ in range(2)
    ]
    trainer = Trainer(_TwoTaskModel(), batches, _TwoTaskLoss(), config=config)

    history = trainer.fit(
        epochs=1,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    payload = load_checkpoint(tmp_path / "checkpoints" / "last.ckpt")

    assert (tmp_path / "artifacts" / "gradient_cosine.json").is_file()
    assert (tmp_path / "artifacts" / "gradient_cosine.csv").is_file()
    assert len(trainer.gradient_monitor.history) == 1
    assert trainer.gradient_monitor.history[0].step == 1
    assert "loss/gradient_cosine/pair_count" in history[0]["train"]
    assert payload["extra"]["gradient_cosine_monitor"]["report_count"] == 1
