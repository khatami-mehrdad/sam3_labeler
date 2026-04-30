from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from sam3_labeler.exporters import (
    append_jsonl,
    save_masks,
    write_annotation_json,
    write_classes,
    write_yolo,
)
from sam3_labeler.files import DEFAULT_EXTENSIONS, iter_images
from sam3_labeler.ontology import class_names, load_ontology
from sam3_labeler.sam3_image import Sam3ImageLabeler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label images with SAM 3 text prompts.")
    parser.add_argument("--input", required=True, help="Image file or directory of images.")
    parser.add_argument(
        "--input-list",
        help="Optional text file with one image path per line. Overrides recursive input discovery.",
    )
    parser.add_argument(
        "--input-root",
        help="Root used to build collision-safe relative output keys. Defaults to --input.",
    )
    parser.add_argument("--output", required=True, help="Directory for labels.")
    parser.add_argument("--ontology", required=True, help="YAML mapping of prompt: class_name.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam3.1/sam3.1_multiplex.pt",
        help="Path to local SAM 3 / SAM 3.1 checkpoint.",
    )
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        help="CUDA autocast dtype for SAM 3 image inference.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.0,
        help="Optional class-level IoU threshold for dropping duplicate boxes. Disabled by default.",
    )
    parser.add_argument("--extensions", nargs="*", default=list(DEFAULT_EXTENSIONS))
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-yolo", action="store_true")
    parser.add_argument(
        "--save-jsonl",
        action="store_true",
        help="Also append records to a top-level annotations.jsonl index.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that already have annotations/<image_key>.json.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of deterministic shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = output_dir / "masks"
    yolo_dir = output_dir / "yolo"
    annotation_dir = output_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_yolo:
        yolo_dir.mkdir(parents=True, exist_ok=True)

    ontology = load_ontology(args.ontology)
    write_classes(output_dir, class_names(ontology))

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"SAM checkpoint not found: {checkpoint_path}")

    images = _load_images(args.input, args.input_list, tuple(args.extensions))
    if not images:
        raise SystemExit(f"No images found in {args.input}")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    input_root = Path(args.input_root or args.input)
    if input_root.is_file():
        input_root = input_root.parent

    images = [
        image_path
        for index, image_path in enumerate(images)
        if index % args.num_shards == args.shard_index
    ]
    if not images:
        raise SystemExit(f"No images selected for shard {args.shard_index}/{args.num_shards}")

    jsonl_path = output_dir / "annotations.jsonl"
    if args.save_jsonl and not args.resume and jsonl_path.exists():
        jsonl_path.unlink()

    labeler = Sam3ImageLabeler(
        checkpoint_path=checkpoint_path,
        device=args.device,
        dtype=args.dtype,
    )
    labeler.cache_text_prompts(ontology)
    for image_path in tqdm(images, desc="Labeling images"):
        image_key = _image_key(image_path, input_root)
        annotation_path = annotation_dir / f"{image_key}.json"
        if args.resume and annotation_path.exists():
            continue

        image_size, detections = labeler.label_image(
            image_path=image_path,
            ontology=ontology,
            score_threshold=args.score_threshold,
            nms_iou_threshold=args.nms_iou_threshold or None,
        )

        mask_file = None
        if args.save_masks:
            mask_file = save_masks(mask_dir, image_key, detections)
        if args.save_yolo:
            write_yolo(yolo_dir, image_key, image_size, detections)

        write_annotation_json(
            annotation_dir=annotation_dir,
            image_key=image_key,
            image_path=image_path,
            image_size=image_size,
            detections=detections,
            mask_file=mask_file,
        )
        if args.save_jsonl:
            append_jsonl(
                jsonl_path=jsonl_path,
                image_path=image_path,
                image_size=image_size,
                detections=detections,
                mask_file=mask_file,
                image_key=image_key,
            )

    print(f"Wrote labels to {output_dir}")


def _load_images(
    input_path: str | Path,
    input_list: str | None,
    extensions: tuple[str, ...],
) -> list[Path]:
    if input_list is None:
        return list(iter_images(input_path, extensions))

    paths = []
    with Path(input_list).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            paths.append(Path(stripped))
    return paths


def _image_key(image_path: Path, input_root: Path) -> str:
    try:
        relative = image_path.resolve().relative_to(input_root.resolve())
    except ValueError:
        relative = Path(image_path.name)
    return relative.with_suffix("").as_posix()


if __name__ == "__main__":
    main()
