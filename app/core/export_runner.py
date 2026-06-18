"""Background runner for exporting SAM3 annotations to YOLO datasets."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.core.config import MAX_CONCURRENT_JOBS


_QUEUE: asyncio.Queue | None = None
_WORKERS: list[asyncio.Task] = []


@dataclass(frozen=True)
class ExportSample:
    annotation_path: Path
    image_path: Path
    width: float
    height: float
    class_names: frozenset[str]
    annotations: tuple[dict, ...]


@dataclass(frozen=True)
class ScanResult:
    samples: list[ExportSample]
    class_to_id: dict[str, int]
    names: list[str]
    missing_images: int


def get_queue() -> asyncio.Queue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = asyncio.Queue()
    return _QUEUE


async def enqueue(job_id: str) -> None:
    await get_queue().put(job_id)


def start_workers() -> None:
    global _WORKERS
    if _WORKERS:
        return
    for _ in range(MAX_CONCURRENT_JOBS):
        _WORKERS.append(asyncio.create_task(_worker()))


async def _worker() -> None:
    while True:
        try:
            job_id = await get_queue().get()
        except asyncio.CancelledError:
            return
        try:
            await _run_job(job_id)
        except Exception:
            traceback.print_exc()
        finally:
            get_queue().task_done()


async def _run_job(job_id: str) -> None:
    from app.core import db

    rec = await db.get_export_job(job_id)
    if rec is None:
        return

    output_dir = Path(rec["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    await db.update_export_job(
        job_id,
        status="running",
        started_at=db.now_utc_iso(),
        current_stage="starting",
        progress_pct=0.0,
        error=None,
    )

    try:
        loop = asyncio.get_running_loop()
        summary = await asyncio.to_thread(
            export_sam3_to_yolo,
            annotations_dir=Path(rec["source_annotations_dir"]),
            output_dir=output_dir,
            train_pct=int(rec["train_pct"]),
            val_pct=int(rec["val_pct"]),
            filter_small_boxes=bool(rec.get("filter_small_boxes")),
            small_box_area_factor=float(rec.get("small_box_area_factor") or 10),
            progress_callback=lambda stage, pct: _set_progress_threadsafe(loop, job_id, stage, pct),
        )
        (output_dir / "export_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        await db.update_export_job(
            job_id,
            status="done",
            current_stage=None,
            progress_pct=1.0,
            finished_at=db.now_utc_iso(),
        )
    except Exception as exc:
        tb = traceback.format_exc()
        (output_dir / "ERROR.log").write_text(tb, encoding="utf-8")
        await db.update_export_job(
            job_id,
            status="failed",
            current_stage="failed",
            finished_at=db.now_utc_iso(),
            error=f"{type(exc).__name__}: {exc}",
        )


def _set_progress_threadsafe(loop: asyncio.AbstractEventLoop, job_id: str, stage: str, pct: float) -> None:
    from app.core import db

    loop.call_soon_threadsafe(
        lambda: loop.create_task(db.update_export_job(job_id, current_stage=stage, progress_pct=pct))
    )


def export_sam3_to_yolo(
    *,
    annotations_dir: Path,
    output_dir: Path,
    train_pct: int = 90,
    val_pct: int = 10,
    filter_small_boxes: bool = False,
    small_box_area_factor: float = 10,
    progress_callback=None,
) -> dict:
    if train_pct < 0 or val_pct < 0 or train_pct + val_pct != 100:
        raise ValueError("train_pct and val_pct must be non-negative and sum to 100")
    if not annotations_dir.exists() or not annotations_dir.is_dir():
        raise ValueError(f"annotations_dir is not a directory: {annotations_dir}")
    if small_box_area_factor <= 0:
        raise ValueError("small_box_area_factor must be greater than 0")

    _progress(progress_callback, "scan_annotations", 0.05)
    scan = _scan_samples(annotations_dir)
    samples = scan.samples
    class_to_id = scan.class_to_id
    names = scan.names
    if not samples:
        raise ValueError(f"no exportable annotations found in {annotations_dir}")

    _progress(progress_callback, "split_dataset", 0.20)
    val_samples = _stratified_val_samples(samples, val_pct=val_pct, train_pct=train_pct)

    _prepare_output_dirs(output_dir)
    _progress(progress_callback, "write_dataset", 0.30)
    counts = _write_yolo_dataset(
        samples=samples,
        val_samples=val_samples,
        class_to_id=class_to_id,
        output_dir=output_dir,
        filter_small_boxes=filter_small_boxes,
        small_box_area_factor=small_box_area_factor,
        progress_callback=progress_callback,
    )
    _write_dataset_yaml(output_dir, names)

    summary = {
        "annotations_dir": str(annotations_dir),
        "output_dir": str(output_dir),
        "train_pct": train_pct,
        "val_pct": val_pct,
        "filter_small_boxes": filter_small_boxes,
        "small_box_area_factor": small_box_area_factor,
        "class_count": len(names),
        "names": names,
        "samples": len(samples),
        "missing_images": scan.missing_images,
        **counts,
    }
    _progress(progress_callback, "done", 1.0)
    return summary


def _scan_samples(annotations_dir: Path) -> ScanResult:
    json_files = sorted(annotations_dir.glob("*.json"))
    raw_class_order: dict[str, int] = {}
    samples: list[ExportSample] = []
    missing_images = 0

    for annotation_path in json_files:
        record = json.loads(annotation_path.read_text(encoding="utf-8"))
        image = record.get("image")
        annotations = record.get("annotations") or []
        width = float(record.get("width") or 0)
        height = float(record.get("height") or 0)
        if not image or not annotations or width <= 0 or height <= 0:
            continue
        image_path = Path(str(image))
        if not image_path.exists():
            missing_images += 1
            continue

        class_names: set[str] = set()
        normalized_annotations: list[dict] = []
        for ann in annotations:
            class_name = str(ann.get("class_name") or "").strip()
            bbox = ann.get("bbox_xyxy")
            if not class_name or not isinstance(bbox, list) or len(bbox) < 4:
                continue
            class_names.add(class_name)
            raw_class_id = ann.get("class_id")
            if isinstance(raw_class_id, int):
                raw_class_order[class_name] = min(raw_class_order.get(class_name, raw_class_id), raw_class_id)
            else:
                raw_class_order.setdefault(class_name, 10**9)
            normalized_annotations.append(ann)

        if not class_names or not normalized_annotations:
            continue
        samples.append(ExportSample(
            annotation_path=annotation_path,
            image_path=image_path,
            width=width,
            height=height,
            class_names=frozenset(class_names),
            annotations=tuple(normalized_annotations),
        ))

    names = sorted(raw_class_order, key=lambda name: (raw_class_order[name], name))
    class_to_id = {name: idx for idx, name in enumerate(names)}
    return ScanResult(
        samples=samples,
        class_to_id=class_to_id,
        names=names,
        missing_images=missing_images,
    )


def _stratified_val_samples(
    samples: list[ExportSample],
    *,
    val_pct: int,
    train_pct: int,
) -> set[Path]:
    if val_pct <= 0:
        return set()
    if train_pct <= 0:
        return {sample.annotation_path for sample in samples}

    by_class: dict[str, list[ExportSample]] = {}
    for sample in samples:
        for class_name in sample.class_names:
            by_class.setdefault(class_name, []).append(sample)

    selected: set[Path] = set()
    for class_name in sorted(by_class):
        candidates = sorted(
            by_class[class_name],
            key=lambda sample: _stable_sort_key(class_name, sample.annotation_path),
        )
        n = len(candidates)
        target = round(n * (val_pct / 100.0))
        if n > 1 and target == 0:
            target = 1
        if n > 1:
            target = min(target, n - 1)

        current = sum(1 for sample in candidates if sample.annotation_path in selected)
        for sample in candidates:
            if current >= target:
                break
            if sample.annotation_path in selected:
                continue
            selected.add(sample.annotation_path)
            current += 1
    return selected


def _stable_sort_key(class_name: str, annotation_path: Path) -> str:
    payload = f"{class_name}:{annotation_path.name}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prepare_output_dirs(output_dir: Path) -> None:
    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / rel).mkdir(parents=True, exist_ok=True)


def _write_yolo_dataset(
    *,
    samples: list[ExportSample],
    val_samples: set[Path],
    class_to_id: dict[str, int],
    output_dir: Path,
    filter_small_boxes: bool,
    small_box_area_factor: float,
    progress_callback,
) -> dict:
    train_count = 0
    val_count = 0
    label_count = 0
    filtered_small_boxes = 0

    total = len(samples)
    for idx, sample in enumerate(samples, start=1):
        split = "val" if sample.annotation_path in val_samples else "train"
        image_target = output_dir / "images" / split / f"{sample.annotation_path.stem}{sample.image_path.suffix}"
        label_target = output_dir / "labels" / split / f"{sample.annotation_path.stem}.txt"

        _symlink_or_replace(sample.image_path, image_target)
        lines = []
        annotations = list(sample.annotations)
        if filter_small_boxes:
            annotations, n_filtered = _filter_small_boxes(
                annotations,
                sample.width,
                sample.height,
                small_box_area_factor,
            )
            filtered_small_boxes += n_filtered

        for ann in annotations:
            class_name = str(ann.get("class_name") or "").strip()
            yolo = _bbox_xyxy_to_yolo(ann.get("bbox_xyxy"), sample.width, sample.height)
            if class_name not in class_to_id or yolo is None:
                continue
            lines.append(
                f"{class_to_id[class_name]} "
                f"{yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}"
            )

        label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        label_count += len(lines)
        if split == "val":
            val_count += 1
        else:
            train_count += 1

        if idx == total or idx % 50 == 0:
            _progress(progress_callback, "write_dataset", 0.30 + 0.65 * (idx / total))

    return {
        "train_samples": train_count,
        "val_samples": val_count,
        "labels": label_count,
        "filtered_small_boxes": filtered_small_boxes,
    }


def _filter_small_boxes(
    annotations: list[dict],
    width: float,
    height: float,
    area_factor: float,
) -> tuple[list[dict], int]:
    areas = [_bbox_area(ann.get("bbox_xyxy"), width, height) for ann in annotations]
    max_area = max(areas, default=0.0)
    if max_area <= 0:
        return annotations, 0
    min_area = max_area / area_factor
    kept = [
        ann
        for ann, area in zip(annotations, areas)
        if area >= min_area
    ]
    return kept, len(annotations) - len(kept)


def _bbox_area(bbox, width: float, height: float) -> float:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    x1 = min(max(x1, 0.0), width)
    x2 = min(max(x2, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    y2 = min(max(y2, 0.0), height)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_xyxy_to_yolo(bbox, width: float, height: float) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    x1 = min(max(x1, 0.0), width)
    x2 = min(max(x2, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    y2 = min(max(y2, 0.0), height)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    cx = (x1 + box_w / 2.0) / width
    cy = (y1 + box_h / 2.0) / height
    return (cx, cy, box_w / width, box_h / height)


def _symlink_or_replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    os.symlink(source, target)


def _write_dataset_yaml(output_dir: Path, names: list[str]) -> None:
    payload = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    (output_dir / "dataset.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _progress(callback, stage: str, pct: float) -> None:
    if callback is not None:
        callback(stage, min(max(float(pct), 0.0), 1.0))
