from __future__ import annotations

import json
from pathlib import Path

import yaml

from app.core.export_runner import export_sam3_to_yolo


def _write_annotation(
    path: Path,
    *,
    image_path: Path,
    class_id: int,
    class_name: str,
    score: float,
) -> None:
    path.write_text(
        json.dumps({
            "image": str(image_path),
            "image_key": path.stem,
            "width": 100,
            "height": 50,
            "annotations": [
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "prompt": class_name.replace("_", " "),
                    "score": score,
                    "bbox_xyxy": [10, 5, 50, 25],
                    "has_mask": False,
                }
            ],
        }),
        encoding="utf-8",
    )


def test_export_sam3_to_yolo_writes_standard_labels_and_stratified_split(tmp_path: Path) -> None:
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    for idx in range(10):
        image = images_dir / f"apple_{idx}.jpg"
        image.write_bytes(b"apple")
        _write_annotation(
            annotations_dir / f"apple_{idx:02d}.json",
            image_path=image,
            class_id=0,
            class_name="apple",
            score=0.75,
        )
    for idx in range(10):
        image = images_dir / f"banana_{idx}.jpg"
        image.write_bytes(b"banana")
        _write_annotation(
            annotations_dir / f"banana_{idx:02d}.json",
            image_path=image,
            class_id=1,
            class_name="banana",
            score=0.55,
        )

    output_dir = tmp_path / "export"
    summary = export_sam3_to_yolo(
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        train_pct=90,
        val_pct=10,
    )

    assert summary["train_samples"] == 18
    assert summary["val_samples"] == 2
    assert summary["labels"] == 20

    dataset = yaml.safe_load((output_dir / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset["train"] == "images/train"
    assert dataset["val"] == "images/val"
    assert dataset["names"] == ["apple", "banana"]

    val_class_counts = {0: 0, 1: 0}
    for label_path in (output_dir / "labels" / "val").glob("*.txt"):
        parts = label_path.read_text(encoding="utf-8").strip().split()
        assert len(parts) == 5
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:5]]
        assert all(0.0 <= value <= 1.0 for value in coords)
        val_class_counts[class_id] += 1
    assert val_class_counts == {0: 1, 1: 1}

    for image_link in (output_dir / "images" / "train").glob("*.jpg"):
        assert image_link.is_symlink()


def test_export_sam3_to_yolo_skips_missing_source_images(tmp_path: Path) -> None:
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    existing = images_dir / "existing.jpg"
    existing.write_bytes(b"image")
    missing = images_dir / "missing.jpg"

    _write_annotation(
        annotations_dir / "existing.json",
        image_path=existing,
        class_id=0,
        class_name="apple",
        score=0.75,
    )
    _write_annotation(
        annotations_dir / "missing.json",
        image_path=missing,
        class_id=0,
        class_name="apple",
        score=0.75,
    )

    output_dir = tmp_path / "export"
    summary = export_sam3_to_yolo(
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        train_pct=100,
        val_pct=0,
    )

    assert summary["samples"] == 1
    assert summary["missing_images"] == 1
    assert (output_dir / "images" / "train" / "existing.jpg").is_symlink()
    assert not (output_dir / "images" / "train" / "missing.jpg").is_symlink()
    assert (output_dir / "labels" / "train" / "existing.txt").exists()
    assert not (output_dir / "labels" / "train" / "missing.txt").exists()


def test_export_sam3_to_yolo_optionally_filters_small_boxes(tmp_path: Path) -> None:
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image = images_dir / "crumbs.jpg"
    image.write_bytes(b"image")

    (annotations_dir / "crumbs.json").write_text(
        json.dumps({
            "image": str(image),
            "image_key": "crumbs",
            "width": 100,
            "height": 100,
            "annotations": [
                {"class_id": 0, "class_name": "bread", "score": 0.9, "bbox_xyxy": [0, 0, 80, 80]},
                {"class_id": 0, "class_name": "bread", "score": 0.8, "bbox_xyxy": [0, 0, 10, 10]},
                {"class_id": 0, "class_name": "bread", "score": 0.7, "bbox_xyxy": [20, 20, 50, 50]},
            ],
        }),
        encoding="utf-8",
    )

    output_dir = tmp_path / "export"
    summary = export_sam3_to_yolo(
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        train_pct=100,
        val_pct=0,
        filter_small_boxes=True,
        small_box_area_factor=10,
    )

    lines = (output_dir / "labels" / "train" / "crumbs.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert summary["labels"] == 2
    assert summary["filtered_small_boxes"] == 1
    assert summary["small_box_area_factor"] == 10
