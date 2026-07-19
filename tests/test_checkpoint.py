from __future__ import annotations

import random

import numpy as np
import torch

from virtual_staining.engine.checkpoint import (
    load_checkpoint,
    resume_from_checkpoint,
    save_checkpoint,
)
from virtual_staining.engine.ema import ExponentialMovingAverage


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.SiLU(), torch.nn.Conv2d(2, 1, 1))


def test_ema_update_and_temporary_application() -> None:
    model = _model()
    ema = ExponentialMovingAverage(model, decay=0.5)
    before = {key: value.clone() for key, value in ema.shadow.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    live = {key: value.clone() for key, value in model.state_dict().items()}
    ema.update(model)
    assert ema.num_updates == 1
    floating_key = next(key for key, value in before.items() if torch.is_floating_point(value))
    assert torch.allclose(ema.shadow[floating_key], (before[floating_key] + live[floating_key]) / 2)
    with ema.average_parameters(model):
        assert torch.equal(model.state_dict()[floating_key], ema.shadow[floating_key])
    assert torch.equal(model.state_dict()[floating_key], live[floating_key])


def test_checkpoint_full_state_and_resume(tmp_path) -> None:
    torch.manual_seed(3)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = ExponentialMovingAverage(model, decay=0.9)
    loss = model(torch.rand(2, 1, 8, 8)).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    ema.update(model)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    loader_generator = torch.Generator().manual_seed(2026)
    torch.rand(4, generator=loader_generator)
    loader_state = loader_generator.get_state()
    checkpoint = save_checkpoint(
        tmp_path / "checkpoints" / "last.ckpt",
        model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=4,
        global_step=17,
        config={"train": {"epochs": 10}},
        manifest_hash="abc123",
        image_spec={"width": 256, "height": 256},
        targets=["CD68"],
        metric_history=[{"ssim": 0.5}],
        dataloader_generator_state=loader_state,
        git_commit="unit-test",
    )
    random_value = random.random()
    numpy_value = np.random.random()

    restored_model = _model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=9e-2)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_ema = ExponentialMovingAverage(restored_model)
    restored_loader_generator = torch.Generator().manual_seed(1)
    resume = resume_from_checkpoint(
        checkpoint,
        restored_model,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        dataloader_generator=restored_loader_generator,
        restore_rng=True,
    )
    assert resume["start_epoch"] == 5
    assert resume["global_step"] == 17
    assert resume["manifest_hash"] == "abc123"
    assert resume["targets"] == ["CD68"]
    assert torch.equal(restored_loader_generator.get_state(), loader_state)
    for key, value in restored_model.state_dict().items():
        assert torch.equal(value, expected[key])
    assert random.random() == random_value
    assert np.random.random() == numpy_value

    ema_loaded_model = _model()
    payload = load_checkpoint(checkpoint, ema_loaded_model, use_ema_as_model=True)
    assert payload["git_commit"] == "unit-test"
    for key, value in ema_loaded_model.state_dict().items():
        assert torch.equal(value, ema.shadow[key])
