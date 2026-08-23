"""Build reproducible report figures from the read-only official training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MARKERS = ("DAPI", "HLA-DR", "CD45RO", "Vimentin", "CD68")
PANEL_STEM = "ROI007_09_11"
ACTIVITY_STEMS = (
    ("low (P10)", "ROI011_13_08"),
    ("median (P50)", "ROI000_01_13"),
    ("high (P90)", "ROI014_13_09"),
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_gray(root: Path, marker: str, stem: str) -> np.ndarray:
    path = root / marker / f"{stem}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _gray_rgb(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.repeat(array[..., None], 3, axis=2), mode="RGB")


def _overlay(dapi: np.ndarray, marker: np.ndarray) -> Image.Image:
    rgb = np.zeros((*dapi.shape, 3), dtype=np.uint8)
    rgb[..., 0] = marker
    rgb[..., 1] = np.minimum(marker // 5, 45)
    rgb[..., 2] = dapi
    return Image.fromarray(rgb, mode="RGB")


def _paste_tile(
    canvas: Image.Image,
    image: Image.Image,
    x: int,
    y: int,
    title: str,
    *,
    tile_size: int = 256,
) -> None:
    draw = ImageDraw.Draw(canvas)
    canvas.paste(image.resize((tile_size, tile_size), Image.Resampling.LANCZOS), (x, y + 34))
    draw.rectangle((x, y + 34, x + tile_size - 1, y + 34 + tile_size - 1), outline=(80, 80, 80), width=1)
    draw.text((x + 4, y + 5), title, fill=(25, 25, 25), font=_font(20, bold=True))


def build_marker_panel(data_root: Path, output_dir: Path) -> dict[str, object]:
    arrays = {marker: _load_gray(data_root, marker, PANEL_STEM) for marker in MARKERS}
    gap = 18
    margin = 22
    tile = 256
    cell_w = tile + gap
    cell_h = tile + 54
    canvas = Image.new("RGB", (margin * 2 + cell_w * 3 - gap, margin * 2 + cell_h * 2), "white")
    panels = [
        ("DAPI (nuclei)", _gray_rgb(arrays["DAPI"])),
        ("HLA-DR", _gray_rgb(arrays["HLA-DR"])),
        ("CD45RO", _gray_rgb(arrays["CD45RO"])),
        ("Vimentin", _gray_rgb(arrays["Vimentin"])),
        ("CD68", _gray_rgb(arrays["CD68"])),
        ("Overlay: DAPI + CD68", _overlay(arrays["DAPI"], arrays["CD68"])),
    ]
    for index, (title, image) in enumerate(panels):
        row, col = divmod(index, 3)
        _paste_tile(canvas, image, margin + col * cell_w, margin + row * cell_h, title, tile_size=tile)
    output = output_dir / "marker_channels_roi007_09_11.png"
    canvas.save(output, optimize=True)
    return {"path": output.as_posix(), "stem": PANEL_STEM, "markers": list(MARKERS)}


def build_activity_panel(data_root: Path, output_dir: Path) -> dict[str, object]:
    margin = 22
    label_w = 190
    gap = 16
    tile = 256
    cell_w = tile + gap
    cell_h = tile + 48
    canvas = Image.new("RGB", (margin * 2 + label_w + cell_w * 3 - gap, margin * 2 + cell_h * 3), "white")
    draw = ImageDraw.Draw(canvas)
    details: list[dict[str, object]] = []
    for row, (level, stem) in enumerate(ACTIVITY_STEMS):
        dapi = _load_gray(data_root, "DAPI", stem)
        cd68 = _load_gray(data_root, "CD68", stem)
        mean = float(cd68.mean() / 255.0)
        active_fraction = float(((cd68.astype(np.float32) / 255.0) > 0.25).mean())
        y = margin + row * cell_h
        draw.text((margin, y + 72), level, fill=(25, 25, 25), font=_font(22, bold=True))
        draw.text((margin, y + 108), stem, fill=(55, 55, 55), font=_font(17))
        draw.text((margin, y + 142), f"mean={mean:.3f}", fill=(55, 55, 55), font=_font(17))
        draw.text((margin, y + 170), f">0.25={active_fraction:.1%}", fill=(55, 55, 55), font=_font(17))
        x0 = margin + label_w
        _paste_tile(canvas, _gray_rgb(dapi), x0, y, "DAPI", tile_size=tile)
        _paste_tile(canvas, _gray_rgb(cd68), x0 + cell_w, y, "CD68 target", tile_size=tile)
        _paste_tile(canvas, _overlay(dapi, cd68), x0 + cell_w * 2, y, "Overlay", tile_size=tile)
        details.append({"level": level, "stem": stem, "cd68_mean": mean, "fraction_gt_0.25": active_fraction})
    output = output_dir / "cd68_activity_quantiles.png"
    canvas.save(output, optimize=True)
    return {"path": output.as_posix(), "selection_rule": "CD68 mean-intensity quantiles over 6296 pairs", "samples": details}


def _brightest_crop(array: np.ndarray, crop_size: int = 96) -> tuple[int, int]:
    values = array.astype(np.float64)
    integral = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[crop_size:, crop_size:] - integral[:-crop_size, crop_size:] - integral[crop_size:, :-crop_size] + integral[:-crop_size, :-crop_size]
    row, col = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(row), int(col)


def build_zoom_panel(data_root: Path, output_dir: Path) -> dict[str, object]:
    dapi = _load_gray(data_root, "DAPI", PANEL_STEM)
    cd68 = _load_gray(data_root, "CD68", PANEL_STEM)
    row, col = _brightest_crop(cd68)
    crop = (slice(row, row + 96), slice(col, col + 96))
    margin = 22
    gap = 18
    tile = 256
    cell_w = tile + gap
    cell_h = tile + 54
    canvas = Image.new("RGB", (margin * 2 + cell_w * 3 - gap, margin * 2 + cell_h * 2), "white")
    panels = [
        ("Full DAPI", _gray_rgb(dapi)),
        ("Full CD68", _gray_rgb(cd68)),
        ("Full overlay", _overlay(dapi, cd68)),
        ("DAPI crop (96 x 96)", _gray_rgb(dapi[crop])),
        ("CD68 crop (96 x 96)", _gray_rgb(cd68[crop])),
        ("Crop overlay", _overlay(dapi[crop], cd68[crop])),
    ]
    for index, (title, image) in enumerate(panels):
        grid_row, grid_col = divmod(index, 3)
        _paste_tile(canvas, image, margin + grid_col * cell_w, margin + grid_row * cell_h, title, tile_size=tile)
    draw = ImageDraw.Draw(canvas)
    x0 = margin + col
    y0 = margin + 34 + row
    draw.rectangle((x0, y0, x0 + 95, y0 + 95), outline=(255, 70, 70), width=3)
    output = output_dir / "dapi_cd68_cell_zoom.png"
    canvas.save(output, optimize=True)
    return {"path": output.as_posix(), "stem": PANEL_STEM, "crop_row": row, "crop_col": col, "crop_size": 96}


def _draw_box(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    text: str,
    *,
    fill: tuple[int, int, int],
    font_size: int = 23,
) -> None:
    draw.rounded_rectangle(bounds, radius=12, fill=fill, outline=(65, 75, 88), width=2)
    left, top, right, bottom = bounds
    font = _font(font_size, bold=True)
    lines = text.split("\n")
    line_gap = 8
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    heights = [box[3] - box[1] for box in boxes]
    total_height = sum(heights) + line_gap * (len(lines) - 1)
    y = top + (bottom - top - total_height) // 2
    for line, box, height in zip(lines, boxes, heights, strict=True):
        width = box[2] - box[0]
        draw.text((left + (right - left - width) // 2, y), line, fill=(20, 28, 38), font=font)
        y += height + line_gap


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int] = (35, 92, 157),
    width: int = 5,
) -> None:
    draw.line((start, end), fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1.0, float(np.hypot(dx, dy)))
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = end
    base = (end[0] - ux * 18, end[1] - uy * 18)
    left = (base[0] + px * 9, base[1] + py * 9)
    right = (base[0] - px * 9, base[1] - py * 9)
    draw.polygon((tip, left, right), fill=color)


def build_code_pipeline_panel(data_root: Path, output_dir: Path) -> dict[str, object]:
    dapi = _gray_rgb(_load_gray(data_root, "DAPI", PANEL_STEM)).resize((210, 210))
    cd68 = _gray_rgb(_load_gray(data_root, "CD68", PANEL_STEM)).resize((155, 155))
    canvas = Image.new("RGB", (1800, 680), (249, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 18), "One real paired patch through the code", fill=(20, 28, 38), font=_font(31, bold=True))
    draw.text((35, 60), f"{PANEL_STEM}.jpg  |  training path above, inference/export path below", fill=(70, 78, 88), font=_font(20))

    canvas.paste(dapi, (35, 150))
    draw.rectangle((35, 150, 244, 359), outline=(55, 70, 88), width=2)
    draw.text((48, 112), "DAPI JPEG", fill=(20, 28, 38), font=_font(23, bold=True))
    boxes = [
        ((300, 175, 540, 335), "image_io.py\nload_image_tensor", (225, 237, 250)),
        ((590, 175, 830, 335), "dataset.py\n__getitem__", (226, 244, 235)),
        ((880, 175, 1135, 335), "multi_marker_\nrestorer.py\nforward", (255, 238, 218)),
        ((1185, 175, 1450, 335), "RestorationOutput\npredictions['CD68']", (237, 228, 249)),
    ]
    for bounds, text, fill in boxes:
        _draw_box(draw, bounds, text, fill=fill, font_size=22)
    for x1, x2 in ((244, 300), (540, 590), (830, 880), (1135, 1185)):
        _draw_arrow(draw, (x1, 255), (x2, 255))
    draw.text((310, 350), "RGB uint8 -> 1-channel float32 [0,1]", fill=(55, 65, 76), font=_font(18))
    draw.text((605, 350), "batch['input'] + batch['targets']", fill=(55, 65, 76), font=_font(18))
    draw.text((895, 350), "B x 1 x 256 x 256", fill=(55, 65, 76), font=_font(18))

    canvas.paste(cd68, (930, 475))
    draw.rectangle((930, 475, 1084, 629), outline=(55, 70, 88), width=2)
    draw.text((910, 440), "CD68 target (training only)", fill=(20, 28, 38), font=_font(20, bold=True))
    _draw_box(draw, (1160, 465, 1430, 630), "composite.py\nloss(prediction, target)", fill=(255, 226, 226), font_size=20)
    _draw_arrow(draw, (1084, 552), (1160, 552), color=(179, 53, 63))
    _draw_arrow(draw, (1315, 335), (1315, 465), color=(179, 53, 63))

    _draw_box(draw, (1490, 175, 1760, 335), "inferencer.py\nsave_prediction_jpeg", fill=(222, 241, 243), font_size=20)
    _draw_arrow(draw, (1450, 255), (1490, 255))
    _draw_box(draw, (1490, 440, 1760, 615), "submission/writer.py\n<stem>_fake.jpg\n-> submission ZIP", fill=(226, 244, 235), font_size=20)
    _draw_arrow(draw, (1625, 335), (1625, 440))
    draw.text((1497, 350), "round -> uint8 -> JPEG quality=100", fill=(55, 65, 76), font=_font(17))

    output = output_dir / "real_patch_code_pipeline.png"
    canvas.save(output, optimize=True)
    return {
        "path": output.as_posix(),
        "stem": PANEL_STEM,
        "note": "CD68 tile is the training target, not a model prediction",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset/official/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/essay/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        build_marker_panel(args.data_root, args.output_dir),
        build_activity_panel(args.data_root, args.output_dir),
        build_zoom_panel(args.data_root, args.output_dir),
        build_code_pipeline_panel(args.data_root, args.output_dir),
    ]
    provenance = {
        "data_root": args.data_root.as_posix(),
        "display": "raw uint8 grayscale; overlays map DAPI to blue and CD68 to red without per-image normalization",
        "figures": records,
    }
    (args.output_dir / "report_figure_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
