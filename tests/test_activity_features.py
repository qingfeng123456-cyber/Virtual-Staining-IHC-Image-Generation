from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from virtual_staining.data.activity import compute_training_activity


def _row(root: Path, index: int, split: str = "official_train") -> dict[str, str]:
    path = root / "colon" / "CD68" / f"ROI000_00_{index:02d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(64, dtype=np.uint8).reshape(8, 8) + index
    Image.fromarray(values, mode="L").save(path)
    return {
        "canonical_key": f"colon/ROI000_00_{index:02d}",
        "split": split,
        "cd68_path": path.relative_to(root).as_posix(),
    }


def test_training_activity_is_deterministic_and_writes_audit(tmp_path: Path) -> None:
    rows = [_row(tmp_path, 0), _row(tmp_path, 1)]
    output = tmp_path / "artifacts" / "训练 activity.csv"

    first, report = compute_training_activity(
        rows,
        tmp_path,
        target="CD68",
        activity_key="activity",
        output_csv=output,
    )
    second, _ = compute_training_activity(
        rows, tmp_path, target="CD68", activity_key="activity"
    )

    assert [row["activity"] for row in first] == [row["activity"] for row in second]
    assert report["uses_target_labels"] is True
    assert report["count"] == 2
    assert output.is_file()
    assert all(float(row["activity"]) > 0.0 for row in first)


def test_training_activity_rejects_validation_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="training rows only"):
        compute_training_activity([_row(tmp_path, 0, "val")], tmp_path, target="CD68")
