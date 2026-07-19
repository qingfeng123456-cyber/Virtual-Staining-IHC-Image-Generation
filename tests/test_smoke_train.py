from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from virtual_staining.engine.checkpoint import resume_from_checkpoint
from virtual_staining.engine.trainer import Trainer
from virtual_staining.engine.validator import Validator


class _PairedData(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(index)
        value = torch.rand((1, 8, 8), generator=generator)
        return {
            "input": value,
            "targets": {"CD68": value * 0.8 + 0.1},
            "canonical_key": f"colon/{index}",
            "stem": f"{index:05d}",
            "roi_id": f"group_{index // 2}",
            "organ": "colon",
        }


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor, task_name: str | None = None) -> torch.Tensor:
        return torch.sigmoid(self.layer(inputs))


def test_cpu_amp_smoke_train_validate_checkpoint_resume(tmp_path) -> None:
    loader = DataLoader(_PairedData(), batch_size=2, shuffle=False, num_workers=0)
    config = {
        "train": {
            "amp": True,
            "amp_dtype": "bfloat16",
            "lr": 1e-2,
            "gradient_accumulation": 2,
            "grad_clip": 1.0,
            "ema": True,
            "ema_decay": 0.9,
        },
        "validation": {"psnr_norm_min": 10.0, "psnr_norm_max": 40.0},
        "inference": {"use_ema": True},
    }
    model = _TinyModel()
    trainer = Trainer(model, loader, torch.nn.MSELoss(), device="cpu", config=config)
    stats = trainer.train_epoch(0)
    assert stats["loss"] > 0
    assert stats["global_step"] == 1
    validator = Validator(
        model,
        loader,
        device="cpu",
        config=config,
        ema=trainer.ema,
        use_ema=True,
    )
    validation = validator.evaluate(records_path=tmp_path / "metrics.csv")
    assert validation["macro"]["mean_ssim"] <= 1.0
    assert validation["primary_domain"] == "float"
    assert validation["domains"]["float"]["stratified"]["organ"]["colon"]["count"] == 4
    assert validation["records"][0]["border_class"] == "unknown"
    assert validation["records"][0]["context_availability"] == "local_only"
    assert validation["domains"]["jpg"]["raw_ssim"] <= 1.0
    history = trainer.fit(
        epochs=1,
        validator=validator,
        checkpoint_dir=tmp_path / "checkpoints",
        manifest_hash="smoke-hash",
        image_spec={"width": 8, "height": 8},
        targets=["CD68"],
    )
    assert len(history) == 1
    assert (tmp_path / "checkpoints" / "last.ckpt").is_file()
    assert (tmp_path / "checkpoints" / "best_ssim.ckpt").is_file()

    restored = _TinyModel()
    restored_trainer = Trainer(restored, loader, torch.nn.MSELoss(), device="cpu", config=config)
    resume = resume_from_checkpoint(
        tmp_path / "checkpoints" / "last.ckpt",
        restored,
        ema=restored_trainer.ema,
        optimizer=restored_trainer.optimizer,
        scaler=restored_trainer.scaler,
        restore_rng=False,
    )
    assert resume["start_epoch"] == 1
    assert resume["manifest_hash"] == "smoke-hash"
