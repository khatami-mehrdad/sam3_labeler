from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch one SAM 3 image-labeling worker per explicit GPU."
    )
    parser.add_argument("--gpus", nargs="+", required=True, help="GPU ids to use, e.g. 0 2 4 6.")
    parser.add_argument("--input", required=True, help="Image file or directory of images.")
    parser.add_argument("--input-root", required=True, help="Root for relative output keys.")
    parser.add_argument("--output", required=True, help="Shared output directory.")
    parser.add_argument("--ontology", required=True, help="YAML mapping of prompt: class_name.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam3.1/sam3.1_multiplex.pt",
        help="Path to local SAM 3 / SAM 3.1 checkpoint.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--save-yolo", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true", help="Launch workers and return.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    log_dir = output_dir / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    workers = []
    num_workers = len(args.gpus)
    for shard_index, gpu in enumerate(args.gpus):
        command = _worker_command(args, num_workers, shard_index)
        log_path = log_dir / f"gpu_{gpu}_worker_{shard_index:02d}.log"
        handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=args.detach,
        )
        handle.close()
        (log_dir / f"gpu_{gpu}_worker_{shard_index:02d}.pid").write_text(
            f"{process.pid}\n",
            encoding="utf-8",
        )
        workers.append(
            {
                "gpu": gpu,
                "worker_index": shard_index,
                "pid": process.pid,
                "log_path": str(log_path),
                "command": " ".join(command),
                "process": process,
            }
        )

    summary = {
        "launched_at": datetime.now().astimezone().isoformat(),
        "num_workers": num_workers,
        "workers": [
            {key: value for key, value in worker.items() if key != "process"}
            for worker in workers
        ],
    }
    summary_path = log_dir / "launch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Launched {num_workers} workers. Summary: {summary_path}")

    if args.detach:
        return

    exit_codes = []
    for worker in workers:
        exit_codes.append(worker["process"].wait())
    if any(code != 0 for code in exit_codes):
        raise SystemExit(max(exit_codes))


def _worker_command(
    args: argparse.Namespace,
    num_workers: int,
    worker_index: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/label_images.py",
        "--input",
        args.input,
        "--input-root",
        args.input_root,
        "--output",
        args.output,
        "--ontology",
        args.ontology,
        "--checkpoint",
        args.checkpoint,
        "--score-threshold",
        str(args.score_threshold),
        "--dtype",
        args.dtype,
        "--device",
        "cuda",
        "--num-shards",
        str(num_workers),
        "--shard-index",
        str(worker_index),
    ]
    if args.resume:
        command.append("--resume")
    if args.save_masks:
        command.append("--save-masks")
    if args.save_yolo:
        command.append("--save-yolo")
    return command


if __name__ == "__main__":
    main()
