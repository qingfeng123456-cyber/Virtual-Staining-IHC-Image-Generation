from __future__ import annotations

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from virtual_staining.engine.ema import ExponentialMovingAverage
from virtual_staining.engine.inferencer import Inferencer


class _InferenceData(Dataset[dict[str, object]]):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "input": torch.full((1, 12, 10), 0.2 + index * 0.1),
            "stem": f"sample_{index:02d}",
            "dapi_path": f"unicode_路径/sample_{index:02d}.jpg",
        }


class _EmptyInferenceData(_InferenceData):
    def __len__(self) -> int:
        return 0


class _SingleTaskModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor, task_name: str | None = None) -> dict[str, torch.Tensor]:
        return {task_name or "CD68": inputs * self.scale}


def test_inference_saves_stable_original_size_jpegs(tmp_path) -> None:
    loader = DataLoader(_InferenceData(), batch_size=2, shuffle=False, num_workers=0)
    model = _SingleTaskModel()
    ema = ExponentialMovingAverage(model)
    inferencer = Inferencer(
        model,
        device="cpu",
        config={"train": {"amp": False}, "inference": {"jpeg_quality": 100}},
        ema=ema,
        tta=False,
    )
    report = inferencer.predict_loader(loader, output_dir=tmp_path / "predictions", target="CD68")
    assert report["count"] == 3
    assert report["peak_vram_bytes"] == 0
    expected = [tmp_path / "predictions" / "CD68" / f"sample_{index:02d}.jpg" for index in range(3)]
    assert [path.exists() for path in expected] == [True, True, True]
    with Image.open(expected[0]) as image:
        assert image.size == (10, 12)
        assert image.mode == "L"
        assert image.format == "JPEG"


def test_inference_rejects_shuffle(tmp_path) -> None:
    loader = DataLoader(_InferenceData(), batch_size=1, shuffle=True, num_workers=0)
    inferencer = Inferencer(_SingleTaskModel(), device="cpu", config={"train": {"amp": False}})
    try:
        inferencer.predict_loader(loader, output_dir=tmp_path)
    except ValueError as error:
        assert "must not shuffle" in str(error)
    else:
        raise AssertionError("A shuffled inference loader was accepted")


def test_inference_uses_configured_cpu_and_restores_rgb_storage(tmp_path) -> None:
    loader = DataLoader(_InferenceData(), batch_size=1, shuffle=False, num_workers=0)
    inferencer = Inferencer(
        _SingleTaskModel(),
        config={"train": {"amp": False}, "inference": {"device": "cpu"}},
        image_spec={"CD68": {"logical_channels": 1, "storage_channels": 3}},
    )
    assert inferencer.device.type == "cpu"
    inferencer.predict_loader(loader, output_dir=tmp_path, target="CD68")
    with Image.open(tmp_path / "CD68" / "sample_00.jpg") as image:
        assert image.mode == "RGB"


def test_inference_rejects_zero_inputs(tmp_path) -> None:
    loader = DataLoader(_EmptyInferenceData(), batch_size=1, shuffle=False, num_workers=0)
    inferencer = Inferencer(
        _SingleTaskModel(), device="cpu", config={"train": {"amp": False}}
    )
    try:
        inferencer.predict_loader(loader, output_dir=tmp_path, target="CD68")
    except ValueError as error:
        assert "zero inputs" in str(error)
    else:
        raise AssertionError("An empty inference dataloader was accepted")
