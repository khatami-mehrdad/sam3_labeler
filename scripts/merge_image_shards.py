from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded SAM 3 image-label outputs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard output directories.")
    parser.add_argument("--output", required=True, help="Merged output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dirs = [Path(path) for path in args.inputs]
    _merge_classes(input_dirs, output_dir / "classes.txt")

    merged_jsonl = output_dir / "annotations.jsonl"
    seen = set()
    with merged_jsonl.open("w", encoding="utf-8") as out_handle:
        for input_dir in input_dirs:
            jsonl_path = input_dir / "annotations.jsonl"
            if not jsonl_path.exists():
                continue
            with jsonl_path.open("r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    image_key = _image_key_from_line(line)
                    if image_key in seen:
                        continue
                    seen.add(image_key)
                    out_handle.write(line)

    print(f"Merged {len(seen)} annotation records into {merged_jsonl}")


def _merge_classes(input_dirs: list[Path], output_path: Path) -> None:
    class_text = None
    for input_dir in input_dirs:
        classes_path = input_dir / "classes.txt"
        if not classes_path.exists():
            continue
        text = classes_path.read_text(encoding="utf-8")
        if class_text is None:
            class_text = text
        elif text != class_text:
            raise SystemExit(f"Class list mismatch in {classes_path}")

    if class_text is not None:
        output_path.write_text(class_text, encoding="utf-8")


def _image_key_from_line(line: str) -> str:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return line
    image_key = record.get("image_key")
    if image_key:
        return str(image_key)
    return line


if __name__ == "__main__":
    main()
