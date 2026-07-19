from __future__ import annotations

import torch

from virtual_staining.data.transforms import PairedTransform, apply_d4, invert_d4


def _no_intensity(**kwargs: object) -> PairedTransform:
    return PairedTransform(
        gamma_probability=0.0,
        brightness_probability=0.0,
        contrast_probability=0.0,
        noise_probability=0.0,
        blur_probability=0.0,
        **kwargs,
    )


def test_all_d4_transforms_are_exactly_invertible_for_non_square_tensor() -> None:
    tensor = torch.arange(3 * 5, dtype=torch.float32).reshape(1, 3, 5)
    for transform_id in range(8):
        recovered = invert_d4(apply_d4(tensor, transform_id), transform_id)
        assert torch.equal(recovered, tensor), transform_id


def test_paired_geometry_is_synchronized_for_all_targets() -> None:
    image = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6) / 24
    targets = {"CD68": image.clone(), "HLA-DR": image.clone()}
    transform = _no_intensity(
        horizontal_flip_probability=1.0,
        vertical_flip_probability=1.0,
        rotate_probability=1.0,
    )

    transformed_image, transformed_targets = transform(
        image, targets, generator=torch.Generator().manual_seed(8)
    )

    assert torch.equal(transformed_image, transformed_targets["CD68"])
    assert torch.equal(transformed_image, transformed_targets["HLA-DR"])


def test_intensity_augmentation_changes_only_dapi() -> None:
    image = torch.full((1, 8, 8), 0.25)
    target = torch.full((1, 8, 8), 0.25)
    transform = PairedTransform(
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        rotate_probability=0.0,
        gamma_range=(0.9, 0.9),
        gamma_probability=1.0,
        brightness_probability=0.0,
        contrast_probability=0.0,
        noise_probability=0.0,
        blur_probability=0.0,
    )

    transformed_image, transformed_targets = transform(image, {"CD68": target})

    assert not torch.equal(transformed_image, image)
    assert torch.equal(transformed_targets["CD68"], target)
    assert 0.0 <= float(transformed_image.min()) <= float(transformed_image.max()) <= 1.0


def test_optional_integer_translation_is_paired_and_shape_preserving() -> None:
    image = torch.zeros((1, 7, 9), dtype=torch.float32)
    image[:, 2, 3] = 1.0
    transform = _no_intensity(
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        rotate_probability=0.0,
        max_translation=2,
    )

    transformed_image, transformed_targets = transform(
        image, {"CD68": image.clone()}, generator=torch.Generator().manual_seed(42)
    )

    assert transformed_image.shape == image.shape
    assert torch.equal(transformed_image, transformed_targets["CD68"])
