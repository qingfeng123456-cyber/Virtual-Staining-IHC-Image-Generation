from __future__ import annotations

import pytest
import torch

from virtual_staining.engine.ensemble import (
    ensemble_tensors,
    normalize_nonnegative_weights,
    validation_score_weights,
)
from virtual_staining.engine.inferencer import D4_TRANSFORMS, Inferencer, apply_d4, invert_d4


@pytest.mark.parametrize("transform", D4_TRANSFORMS)
def test_d4_is_strictly_invertible_for_rectangles(transform: str) -> None:
    tensor = torch.arange(2 * 3 * 5).reshape(1, 2, 3, 5)
    assert torch.equal(invert_d4(apply_d4(tensor, transform), transform), tensor)


def test_d4_tta_identity_model_is_identity() -> None:
    model = torch.nn.Identity()
    inferencer = Inferencer(model, device="cpu", config={"train": {"amp": False}}, tta="d4")
    tensor = torch.rand(2, 1, 11, 13)
    result = inferencer.predict_tensor(tensor)["output"]
    assert torch.allclose(result, tensor)


def test_nonnegative_weighted_ensemble() -> None:
    low = torch.zeros(1, 1, 2, 2)
    high = torch.ones(1, 1, 2, 2)
    result = ensemble_tensors([low, high], [1, 3])
    assert torch.allclose(result, torch.full_like(result, 0.75))
    assert sum(validation_score_weights([0.7, 0.8])) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        normalize_nonnegative_weights([1, -1])
