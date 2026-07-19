from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from virtual_staining.metrics.domain_metrics import (
    attach_rank_proxy,
    calculate_domain_metric_record,
    calculate_rank_proxy,
    calculate_three_domain_metrics,
    jpeg_round_trip,
    roi_bootstrap_difference,
)
from virtual_staining.utils.image_io import (
    ImageSpec,
    read_image_array,
    save_image_tensor,
)


def _textured_gray(size: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return ((xx * 17 + yy * 29 + (xx * yy) % 31) % 256).astype(np.float32) / 255.0


def test_jpeg_round_trip_matches_competition_image_writer(tmp_path: Path) -> None:
    prediction = _textured_gray()
    spec = ImageSpec(16, 16, 3, 1, "RGB", grayscale_rgb=True)
    expected_path = tmp_path / "中文 路径" / "prediction.jpg"
    save_image_tensor(prediction, expected_path, spec, quality=100, subsampling=0)
    expected, mode = read_image_array(expected_path)
    expected_logical = np.rint(expected.astype(np.float32).mean(axis=-1)).astype(np.uint8)

    actual = jpeg_round_trip(
        prediction,
        image_spec=spec,
        reference_shape=prediction.shape,
    )

    assert mode == "RGB"
    np.testing.assert_array_equal(np.rint(actual * 255.0).astype(np.uint8), expected_logical)


def test_three_domain_metrics_use_quantized_target_and_prediction_only_jpeg() -> None:
    target = _textured_gray()
    prediction = np.clip(target * 0.94 + 0.015, 0.0, 1.0)
    metrics = calculate_three_domain_metrics(prediction, target)
    flat = calculate_domain_metric_record(prediction, target)

    assert set(metrics) == {"float", "uint8", "jpg"}
    assert set(flat) == {
        "float_ssim",
        "float_psnr",
        "uint8_ssim",
        "uint8_psnr",
        "jpg_ssim",
        "jpg_psnr",
    }
    assert all(np.isfinite(value) for value in flat.values())
    assert flat["jpg_ssim"] == pytest.approx(metrics["jpg"]["ssim"])
    assert metrics["float"]["ssim"] != pytest.approx(metrics["uint8"]["ssim"], abs=1e-10)
    assert metrics["jpg"]["ssim"] != pytest.approx(metrics["uint8"]["ssim"], abs=1e-10)


def test_rgb_and_grayscale_storage_modes_restore_logical_shape() -> None:
    gray = _textured_gray(12)
    rgb = np.stack((gray, np.sqrt(gray), gray**2), axis=-1)
    rgb_spec = ImageSpec(12, 12, 3, 1, "RGB", grayscale_rgb=True)
    gray_spec = ImageSpec(12, 12, 1, 1, "L")

    logical_gray = jpeg_round_trip(gray, image_spec=rgb_spec, reference_shape=gray.shape)
    logical_rgb = jpeg_round_trip(rgb, image_spec=gray_spec, reference_shape=rgb.shape)

    assert logical_gray.shape == gray.shape
    assert logical_rgb.shape == rgb.shape
    np.testing.assert_array_equal(logical_rgb[..., 0], logical_rgb[..., 1])
    np.testing.assert_array_equal(logical_rgb[..., 1], logical_rgb[..., 2])


def test_rank_proxy_uses_average_tie_percentile_ranks() -> None:
    proxy = calculate_rank_proxy([0.2, 0.8, 0.8], [30.0, 20.0, 40.0])
    np.testing.assert_allclose(proxy, [0.15, 0.525, 0.825])

    attached = attach_rank_proxy(
        [
            {"id": "a", "jpg_ssim": 0.2, "jpg_psnr": 30.0},
            {"id": "b", "jpg_ssim": 0.8, "jpg_psnr": 20.0},
            {"id": "c", "jpg_ssim": 0.8, "jpg_psnr": 40.0},
        ]
    )
    np.testing.assert_allclose([row["rank_proxy"] for row in attached], proxy)
    assert all("rank_proxy" not in source for source in [{"id": "a"}, {"id": "b"}])


def test_roi_bootstrap_is_paired_deterministic_and_counts_outcomes() -> None:
    baseline = [
        {"roi_id": "r1", "jpg_ssim": 0.50},
        {"roi_id": "r1", "jpg_ssim": 0.70},
        {"roi_id": "r2", "jpg_ssim": 0.40},
        {"roi_id": "r3", "jpg_ssim": 0.80},
    ]
    candidate = [
        {"roi_id": "r1", "jpg_ssim": 0.70},
        {"roi_id": "r1", "jpg_ssim": 0.70},
        {"roi_id": "r2", "jpg_ssim": 0.40},
        {"roi_id": "r3", "jpg_ssim": 0.75},
    ]

    first = roi_bootstrap_difference(
        candidate, baseline, bootstrap_samples=2000, seed=77
    )
    second = roi_bootstrap_difference(
        candidate, baseline, bootstrap_samples=2000, seed=77
    )

    assert first == second
    assert first["difference"] == pytest.approx((0.1 + 0.0 - 0.05) / 3)
    assert first["win_tie_loss"] == {"win": 1, "tie": 1, "loss": 1}
    assert first["ci_low"] <= first["difference"] <= first["ci_high"]


def test_roi_bootstrap_rejects_unpaired_rois_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="ROI sets differ"):
        roi_bootstrap_difference(
            [{"roi_id": "candidate", "jpg_ssim": 0.5}],
            [{"roi_id": "baseline", "jpg_ssim": 0.5}],
        )
    with pytest.raises(ValueError, match="finite"):
        roi_bootstrap_difference(
            [{"roi_id": "r", "jpg_ssim": float("inf")}],
            [{"roi_id": "r", "jpg_ssim": 0.5}],
        )
