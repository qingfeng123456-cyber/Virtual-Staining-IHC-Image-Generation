from pathlib import Path

import numpy as np
import torch
from PIL import Image

from virtual_staining.utils.image_io import ImageSpec, load_image_tensor, save_image_tensor


def test_grayscale_and_rgb_round_trip_in_chinese_path(tmp_path: Path) -> None:
    root = tmp_path / "中文 路径"
    gray = torch.linspace(0, 1, 64).reshape(1, 8, 8)
    gray_path = save_image_tensor(gray, root / "灰度.png")
    loaded_gray = load_image_tensor(gray_path)
    assert loaded_gray.shape == (1, 8, 8)
    assert torch.max(torch.abs(gray - loaded_gray)) <= 1.0 / 255.0

    rgb = torch.zeros(3, 8, 8)
    rgb[0] = 1.0
    rgb_path = save_image_tensor(rgb, root / "红色.png")
    loaded_rgb = load_image_tensor(rgb_path)
    assert loaded_rgb.shape == (3, 8, 8)
    assert loaded_rgb[0].mean() == 1
    assert loaded_rgb[1:].max() == 0


def test_logical_grayscale_and_rgb_storage_mode(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    pixels = np.full((8, 8, 3), 77, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(source)
    logical = load_image_tensor(source, logical_channels=1)
    assert logical.shape == (1, 8, 8)
    spec = ImageSpec(8, 8, 3, 1, "RGB", save_format="JPEG", grayscale_rgb=True)
    destination = save_image_tensor(logical, tmp_path / "prediction.jpg", spec)
    with Image.open(destination) as image:
        assert image.mode == "RGB"
        assert image.size == (8, 8)


def test_values_are_scaled_once(tmp_path: Path) -> None:
    tensor = torch.full((1, 16, 16), 128.0 / 255.0)
    path = save_image_tensor(tensor, tmp_path / "value.png")
    loaded = load_image_tensor(path)
    assert loaded.min() >= 0
    assert loaded.max() <= 1
    assert abs(float(loaded.mean()) - 128.0 / 255.0) < 1e-6

