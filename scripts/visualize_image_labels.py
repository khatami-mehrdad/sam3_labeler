from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SAM 3 image labels.")
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to annotations.jsonl, one annotation JSON file, or an annotations/ directory.",
    )
    parser.add_argument(
        "--input-root",
        help="Root used to resolve image keys when --annotations is a directory.",
    )
    parser.add_argument("--image", required=True, help="Image path to visualize.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    record = _find_record(Path(args.annotations), image_path, args.input_root)
    if record is None:
        raise SystemExit(f"No annotation record found for {image_path}")

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    masks = _load_masks(record.get("mask_file"))
    overlay = np.asarray(image).copy()

    mask_index = 0
    for index, ann in enumerate(record["annotations"]):
        if ann["score"] < args.score_threshold:
            if ann.get("has_mask"):
                mask_index += 1
            continue

        color = _color_for_index(index)

        if ann.get("has_mask") and masks is not None and mask_index < len(masks):
            overlay = _blend_mask(overlay, masks[mask_index].astype(bool), color)
            mask_index += 1

    rendered = Image.fromarray(overlay)
    draw = ImageDraw.Draw(rendered)
    for index, ann in enumerate(record["annotations"]):
        if ann["score"] < args.score_threshold:
            continue
        color = _color_for_index(index)
        label = f"{ann['class_name']} {ann['score']:.2f}"
        _draw_bbox(draw, ann["bbox_xyxy"], color=color, text=label)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)
    print(f"Wrote visualization to {output_path}")


def _find_record(
    annotations_path: Path,
    image_path: Path,
    input_root: str | None,
) -> dict | None:
    if annotations_path.is_dir():
        image_key = _image_key(image_path, Path(input_root) if input_root else image_path.parent)
        annotation_path = annotations_path / f"{image_key}.json"
        if not annotation_path.exists():
            return None
        return json.loads(annotation_path.read_text(encoding="utf-8"))

    if annotations_path.suffix.lower() == ".json":
        return json.loads(annotations_path.read_text(encoding="utf-8"))

    target = str(image_path)
    with annotations_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("image") == target:
                return record
    return None


def _image_key(image_path: Path, input_root: Path) -> str:
    try:
        relative = image_path.resolve().relative_to(input_root.resolve())
    except ValueError:
        relative = Path(image_path.name)
    return relative.with_suffix("").as_posix()


def _load_masks(mask_file: str | None) -> np.ndarray | None:
    if not mask_file:
        return None
    path = Path(mask_file)
    if not path.exists():
        return None
    return np.load(path)["masks"]


def _draw_bbox(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    color: tuple[int, int, int],
    text: str | None = None,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    if text is not None:
        text_y = max(0, y1 - 14)
        draw.rectangle((x1, text_y, x1 + max(70, len(text) * 7), text_y + 13), fill="white")
        draw.text((x1 + 2, text_y), text, fill=color)


def _blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    image[mask] = (image[mask] * (1.0 - alpha) + np.asarray(color) * alpha).astype(
        np.uint8
    )
    return image


def _color_for_index(index: int) -> tuple[int, int, int]:
    palette = (
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
    )
    return palette[index % len(palette)]


if __name__ == "__main__":
    main()
