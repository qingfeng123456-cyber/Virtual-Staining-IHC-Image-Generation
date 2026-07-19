from __future__ import annotations

import numpy as np
import pytest

from virtual_staining.metrics.aggregation import (
    aggregate_metric_records,
    aggregate_stratified_metrics,
    aggregate_target_metrics,
    weighted_organ_metrics,
)
from virtual_staining.metrics.image_metrics import (
    calculate_image_metrics,
    calculate_psnr,
    calculate_ssim,
)


def test_identical_and_noisy_grayscale_metrics() -> None:
    reference = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    identical = calculate_image_metrics(reference, reference)
    noisy = np.clip(reference + np.random.default_rng(7).normal(0, 0.05, reference.shape), 0, 1)
    degraded = calculate_image_metrics(noisy, reference)
    assert identical["ssim"] == pytest.approx(1.0)
    assert np.isinf(identical["psnr"])
    assert degraded["ssim"] < identical["ssim"]
    assert degraded["psnr"] < identical["psnr"]


def test_rgb_chw_and_hwc_metrics_agree() -> None:
    rng = np.random.default_rng(9)
    reference = rng.random((24, 25, 3), dtype=np.float32)
    prediction = np.clip(reference * 0.97, 0, 1)
    hwc_ssim = calculate_ssim(prediction, reference)
    chw_ssim = calculate_ssim(prediction.transpose(2, 0, 1), reference.transpose(2, 0, 1))
    assert hwc_ssim == pytest.approx(chw_ssim)
    assert calculate_psnr(prediction, reference) > 20


def test_metric_validation_and_psnr_cap() -> None:
    image = np.ones((8, 8), dtype=np.float32)
    assert calculate_psnr(image, image, cap=80.0) == 80.0
    with pytest.raises(ValueError, match="shapes differ"):
        calculate_ssim(image, np.ones((7, 8), dtype=np.float32))
    invalid = image.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        calculate_psnr(invalid, image)


def test_per_target_roi_macro_aggregation() -> None:
    rows = [
        {"target": "CD68", "roi_id": "a", "ssim": 0.8, "psnr": 25.0},
        {"target": "CD68", "roi_id": "a", "ssim": 0.9, "psnr": 30.0},
        {"target": "CD68", "roi_id": "b", "ssim": 0.7, "psnr": 20.0},
        {"target": "CD45RO", "roi_id": "a", "ssim": 0.6, "psnr": 18.0},
    ]
    aggregate = aggregate_target_metrics(rows, psnr_norm_min=15, psnr_norm_max=40)
    assert aggregate["per_target"]["CD68"]["count"] == 3
    assert aggregate["per_target"]["CD68"]["group_count"] == 2
    assert aggregate["per_target"]["CD68"]["mean_ssim"] == pytest.approx(0.8)
    assert aggregate["macro"]["mean_ssim"] == pytest.approx(0.7)
    direct = aggregate_metric_records(rows[:3])
    assert 0 <= direct["local_proxy_score"] <= 1


def test_stratified_and_weighted_organ_aggregation() -> None:
    rows = []
    for organ, weight_offset in (("colon", 0.0), ("liver", 0.1), ("stomach", 0.2)):
        rows.append(
            {
                "target": "CD68",
                "organ": organ,
                "roi_id": f"{organ}-roi",
                "activity_bin": "high",
                "context_availability": "full",
                "image_mean_bin": "mid",
                "border_class": "border",
                "ssim": 0.6 + weight_offset,
                "psnr": 20.0 + 10.0 * weight_offset,
            }
        )
    stratified = aggregate_stratified_metrics(rows)
    assert set(stratified["organ"]) == {"colon", "liver", "stomach"}
    assert stratified["border_class"]["border"]["count"] == 3

    weighted = weighted_organ_metrics(
        rows, weights={"colon": 0.1, "liver": 0.2, "stomach": 0.7}
    )
    assert weighted is not None
    assert weighted["weighted"]["mean_ssim"] == pytest.approx(0.76)
    assert weighted["weights"] == {"colon": 0.1, "liver": 0.2, "stomach": 0.7}


def test_weighted_organ_metrics_requires_all_configured_organs() -> None:
    rows = [{"target": "CD68", "organ": "colon", "roi_id": "r", "ssim": 0.8, "psnr": 25.0}]
    assert (
        weighted_organ_metrics(
            rows, weights={"colon": 0.1, "liver": 0.2, "stomach": 0.7}
        )
        is None
    )
