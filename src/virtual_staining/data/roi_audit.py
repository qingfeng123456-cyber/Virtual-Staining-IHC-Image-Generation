"""Artifact writer for the strict ROI-grid and boundary audit."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from virtual_staining.utils.image_io import read_image_array

from .roi_index import ROIGridAudit, audit_roi_grid, parse_patch_coordinate


def _resolved_dapi(row: Mapping[str, Any], data_root: Path) -> Path:
    value = str(row.get("dapi_path", "")).strip()
    if not value:
        raise ValueError("A manifest row has no dapi_path")
    path = Path(value)
    return (path if path.is_absolute() else data_root / path).resolve()


def _write_missing(path: Path, audit: ROIGridAudit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("organ", "split", "roi_id", "row", "col")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for grid in audit.grids:
            for row, col in grid.missing_coordinates:
                writer.writerow(
                    {
                        "organ": grid.organ,
                        "split": grid.split,
                        "roi_id": grid.roi_id,
                        "row": row,
                        "col": col,
                    }
                )


def _write_mosaics(
    rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    output_dir: Path,
    *,
    maximum: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str], list[tuple[int, int, Mapping[str, Any]]]] = {}
    for row in rows:
        coordinate = parse_patch_coordinate(
            str(row.get("stem") or row.get("patch_id") or row.get("dapi_path") or "")
        )
        if coordinate is None:
            continue
        key = (
            str(row.get("organ", "unknown")),
            str(row.get("split", "unknown")),
            coordinate.roi_id,
        )
        grouped.setdefault(key, []).append((coordinate.row, coordinate.col, row))
    written: list[str] = []
    for (organ, split, roi_id), entries in sorted(grouped.items())[:maximum]:
        row_min = min(item[0] for item in entries)
        row_max = max(item[0] for item in entries)
        col_min = min(item[1] for item in entries)
        col_max = max(item[1] for item in entries)
        first, _ = read_image_array(_resolved_dapi(entries[0][2], data_root))
        if first.ndim == 2:
            first = np.repeat(first[..., None], 3, axis=-1)
        tile_height, tile_width = first.shape[:2]
        canvas = np.zeros(
            (
                (row_max - row_min + 1) * tile_height,
                (col_max - col_min + 1) * tile_width,
                3,
            ),
            dtype=np.uint8,
        )
        for grid_row, grid_col, manifest_row in entries:
            image, _ = read_image_array(_resolved_dapi(manifest_row, data_root))
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            top = (grid_row - row_min) * tile_height
            left = (grid_col - col_min) * tile_width
            canvas[top : top + tile_height, left : left + tile_width] = image
        rendered = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(rendered)
        for row_number in range(1, row_max - row_min + 1):
            y = row_number * tile_height
            draw.line((0, y, rendered.width, y), fill=(255, 0, 255), width=1)
        for col_number in range(1, col_max - col_min + 1):
            x = col_number * tile_width
            draw.line((x, 0, x, rendered.height), fill=(255, 0, 255), width=1)
        destination = output_dir / f"{organ}_{split}_{roi_id}.png"
        rendered.save(destination, format="PNG")
        written.append(str(destination.resolve()))
    return written


def write_roi_grid_audit(
    rows: Sequence[Mapping[str, Any]],
    data_root: str | Path,
    output_dir: str | Path,
    *,
    mosaic_count: int = 8,
    border_width: int = 8,
) -> dict[str, Any]:
    """Run the audit and atomically materialize its machine-readable artifacts."""

    root = Path(data_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    audit = audit_roi_grid(rows, root, border_width=border_width)
    missing_path = destination / "roi_grid_missing.csv"
    _write_missing(missing_path, audit)
    mosaics = _write_mosaics(
        rows,
        root,
        destination / "figures" / "roi_mosaics",
        maximum=max(0, int(mosaic_count)),
    )
    payload = audit.to_dict()
    payload["data_root"] = str(root)
    payload["missing_csv"] = str(missing_path)
    payload["mosaics"] = mosaics
    output_path = destination / "roi_grid_audit.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    payload["audit_path"] = str(output_path)
    return payload
