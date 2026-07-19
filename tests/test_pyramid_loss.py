"""Finite-value and gradient checks for fixed image-pyramid losses."""

from __future__ import annotations

import torch

from virtual_staining.losses import (
    GaussianPyramidLoss,
    LaplacianPyramidLoss,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
)


def test_pyramid_shapes_support_odd_images_and_preserve_channels() -> None:
    image = torch.rand(2, 3, 17, 19)

    gaussian = build_gaussian_pyramid(image, levels=4)
    laplacian = build_laplacian_pyramid(image, levels=4)

    assert [level.shape for level in gaussian] == [
        (2, 3, 17, 19),
        (2, 3, 9, 10),
        (2, 3, 5, 5),
        (2, 3, 3, 3),
    ]
    assert [level.shape for level in laplacian] == [level.shape for level in gaussian]
    assert all(torch.isfinite(level).all() for level in (*gaussian, *laplacian))


def test_gaussian_and_laplacian_losses_are_finite_and_backward() -> None:
    generator = torch.Generator().manual_seed(2026)
    prediction = torch.rand((2, 1, 17, 19), generator=generator, requires_grad=True)
    target = torch.rand((2, 1, 17, 19), generator=generator)
    gaussian_loss = GaussianPyramidLoss(levels=4)(prediction, target)
    laplacian_loss = LaplacianPyramidLoss(levels=4)(prediction, target)
    total = gaussian_loss + laplacian_loss

    assert gaussian_loss.dtype == torch.float32
    assert laplacian_loss.dtype == torch.float32
    assert torch.isfinite(total)
    assert gaussian_loss > 0.0
    assert laplacian_loss > 0.0

    total.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) > 0


def test_pyramid_losses_are_zero_for_identical_images() -> None:
    image = torch.rand(1, 1, 16, 16)

    gaussian = GaussianPyramidLoss(levels=3)(image, image)
    laplacian = LaplacianPyramidLoss(levels=3)(image, image)

    assert torch.equal(gaussian, torch.zeros_like(gaussian))
    assert torch.equal(laplacian, torch.zeros_like(laplacian))
