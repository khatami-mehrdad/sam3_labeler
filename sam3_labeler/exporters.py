from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sam3_labeler.sam3_image import Detection


def write_classes(output_dir: Path, class_names: list[str]) -> None:
    (output_dir / "classes.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )


def build_record(
    image_path: Path,
    image_size: tuple[int, int],
    detections: list[Detection],
    mask_file: str | None,
    image_key: str | None = None,
) -> dict:
    return {
        "image": str(image_path),
        "image_key": image_key,
        "width": image_size[0],
        "height": image_size[1],
        "mask_file": mask_file,
        "annotations": [
            {
                "class_id": det.class_id,
                "class_name": det.class_name,
                "prompt": det.prompt,
                "score": det.score,
                "bbox_xyxy": det.box_xyxy,
                "has_mask": det.mask is not None,
            }
            for det in detections
        ],
    }


def write_annotation_json(
    annotation_dir: Path,
    image_key: str,
    image_path: Path,
    image_size: tuple[int, int],
    detections: list[Detection],
    mask_file: str | None,
) -> Path:
    annotation_path = annotation_dir / f"{image_key}.json"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(
        json.dumps(
            build_record(
                image_path=image_path,
                image_size=image_size,
                detections=detections,
                mask_file=mask_file,
                image_key=image_key,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return annotation_path


def append_jsonl(
    jsonl_path: Path,
    image_path: Path,
    image_size: tuple[int, int],
    detections: list[Detection],
    mask_file: str | None,
    image_key: str | None = None,
) -> None:
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                build_record(
                    image_path=image_path,
                    image_size=image_size,
                    detections=detections,
                    mask_file=mask_file,
                    image_key=image_key,
                )
            )
            + "\n"
        )


def save_masks(mask_dir: Path, image_key: str, detections: list[Detection]) -> str | None:
    masks = [det.mask.astype(np.uint8) for det in detections if det.mask is not None]
    if not masks:
        return None

    mask_path = mask_dir / f"{image_key}.npz"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_path, masks=np.stack(masks, axis=0))
    return str(mask_path)


def write_yolo(
    yolo_dir: Path,
    image_key: str,
    image_size: tuple[int, int],
    detections: list[Detection],
) -> None:
    width, height = image_size
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        box_w = (x2 - x1) / width
        box_h = (y2 - y1) / height
        lines.append(
            f"{det.class_id} {cx:.8f} {cy:.8f} {box_w:.8f} {box_h:.8f} {det.score:.6f}"
        )

    yolo_path = yolo_dir / f"{image_key}.txt"
    yolo_path.parent.mkdir(parents=True, exist_ok=True)
    yolo_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
