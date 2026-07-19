from __future__ import annotations

import pytest
import torch

from virtual_staining.data.neighborhood import context_offsets
from virtual_staining.data.transforms import apply_context_d4


def _context_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tiles = torch.stack(
        [
            torch.tensor(
                [[[slot * 10 + row * 3 + col for col in range(3)] for row in range(2)]],
                dtype=torch.float32,
            )
            for slot in range(9)
        ]
    )
    mask = torch.tensor([True, False, True, False, True, True, False, True, False])
    return tiles, mask, context_offsets(3)


@pytest.mark.parametrize(
    ("transform_id", "source_slot_at_destination"),
    (
        (4, [2, 1, 0, 5, 4, 3, 8, 7, 6]),  # horizontal flip
        (6, [6, 7, 8, 3, 4, 5, 0, 1, 2]),  # vertical flip
        (1, [2, 5, 8, 1, 4, 7, 0, 3, 6]),  # 90 degrees counter-clockwise
    ),
)
def test_context_geometry_reorders_slots_mask_and_keeps_canonical_offsets(
    transform_id: int,
    source_slot_at_destination: list[int],
) -> None:
    tiles, mask, offsets = _context_fixture()

    transformed_tiles, transformed_mask, transformed_offsets = apply_context_d4(
        tiles,
        mask,
        offsets,
        transform_id,
    )

    expected_tiles = torch.stack(
        [
            torch.rot90(
                torch.flip(tiles[source], dims=(-1,)) if transform_id >= 4 else tiles[source],
                k=transform_id % 4,
                dims=(-2, -1),
            )
            for source in source_slot_at_destination
        ]
    )
    assert torch.equal(transformed_tiles, expected_tiles)
    assert transformed_mask.tolist() == [mask[source].item() for source in source_slot_at_destination]
    assert torch.equal(transformed_offsets, offsets)


@pytest.mark.parametrize("transform_id", range(8))
def test_context_d4_is_exactly_invertible_for_pixels_slots_mask_and_offsets(
    transform_id: int,
) -> None:
    tiles, mask, offsets = _context_fixture()
    transformed = apply_context_d4(tiles, mask, offsets, transform_id)
    inverse_id = (-transform_id) % 4 if transform_id < 4 else transform_id

    recovered_tiles, recovered_mask, recovered_offsets = apply_context_d4(
        *transformed,
        inverse_id,
    )

    assert torch.equal(recovered_tiles, tiles)
    assert torch.equal(recovered_mask, mask)
    assert torch.equal(recovered_offsets, offsets)
