from __future__ import annotations

from collections import Counter

import pytest

from virtual_staining.data.samplers import ActivityStratifiedSampler


def test_activity_sampler_is_seeded_and_balances_populated_quantile_bins() -> None:
    activities = [0.0] * 8 + [1.0] * 2
    splits = ["train"] * len(activities)
    first = ActivityStratifiedSampler(
        activities,
        splits=splits,
        num_bins=4,
        seed=31,
    )
    second = ActivityStratifiedSampler(
        activities,
        splits=splits,
        num_bins=4,
        seed=31,
    )

    first_order = list(first)
    second_order = list(second)
    sampled_bins = Counter(first.bin_assignments[index] for index in first_order)

    assert first_order == second_order
    assert len(sampled_bins) == 2
    assert max(sampled_bins.values()) - min(sampled_bins.values()) <= 1
    assert len(first_order) == len(activities)


def test_activity_sampler_resume_reproduces_exact_remaining_order() -> None:
    activities = [float(index) for index in range(12)]
    splits = ["official_train"] * len(activities)
    sampler = ActivityStratifiedSampler(
        activities,
        splits=splits,
        num_bins=3,
        seed=2026,
        num_samples=18,
    )
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(7)]
    state = sampler.state_dict()
    expected_remaining = list(iterator)

    resumed = ActivityStratifiedSampler(
        activities,
        splits=splits,
        num_bins=3,
        seed=2026,
        num_samples=18,
    )
    resumed.load_state_dict(state)

    assert list(resumed) == expected_remaining
    complete = ActivityStratifiedSampler(
        activities,
        splits=splits,
        num_bins=3,
        seed=2026,
        num_samples=18,
    )
    assert consumed + expected_remaining == list(complete)


@pytest.mark.parametrize("forbidden_split", ("val", "test", "official_test", "smoke"))
def test_activity_sampler_rejects_any_non_train_split(forbidden_split: str) -> None:
    with pytest.raises(ValueError, match="training rows only"):
        ActivityStratifiedSampler(
            [0.1, 0.9],
            splits=["train", forbidden_split],
        )


def test_activity_sampler_manifest_factory_uses_precomputed_values_without_image_io() -> None:
    rows = [
        {
            "split": "train",
            "dapi_activity": value,
            "dapi_path": "this/file/does/not/exist.jpg",
        }
        for value in (0.1, 0.2, 0.8, 0.9)
    ]

    sampler = ActivityStratifiedSampler.from_manifest(rows, num_bins=2, seed=9)

    assert sorted(list(sampler)) == [0, 1, 2, 3]


def test_activity_sampler_rejects_resume_state_from_other_training_data() -> None:
    source = ActivityStratifiedSampler([0.1, 0.9], splits=["train", "train"])
    incompatible = ActivityStratifiedSampler(
        [0.2, 0.8],
        splits=["train", "train"],
    )

    with pytest.raises(ValueError, match="does not match current training data"):
        incompatible.load_state_dict(source.state_dict())


def test_activity_sampler_noncontiguous_indices_resume_exactly() -> None:
    indices = [3, 8, 13, 21]
    sampler = ActivityStratifiedSampler(
        [0.1, 0.2, 0.8, 0.9],
        splits=["final_train"] * 4,
        sample_indices=indices,
        num_bins=2,
        seed=77,
        num_samples=10,
    )
    sampler.set_epoch(4)
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(3)]
    state = sampler.state_dict()
    remaining = list(iterator)

    resumed = ActivityStratifiedSampler(
        [0.1, 0.2, 0.8, 0.9],
        splits=["final_train"] * 4,
        sample_indices=indices,
        num_bins=2,
        seed=77,
        num_samples=10,
    )
    resumed.load_state_dict(state)

    assert list(resumed) == remaining
    assert set(consumed + remaining).issubset(indices)


def test_activity_sampler_rejects_resume_from_other_source_indices() -> None:
    source = ActivityStratifiedSampler(
        [0.1, 0.9],
        splits=["train", "train"],
        sample_indices=[2, 7],
    )
    incompatible = ActivityStratifiedSampler(
        [0.1, 0.9],
        splits=["train", "train"],
        sample_indices=[2, 8],
    )

    with pytest.raises(ValueError, match="does not match current training data"):
        incompatible.load_state_dict(source.state_dict())
