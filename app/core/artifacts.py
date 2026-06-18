"""Artifact path helpers for labeling runs."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SAFE_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_key(value: str) -> str:
    """Return a path-safe key while preserving enough source identity."""
    cleaned = SAFE_PART.sub("_", value).strip("._")
    return cleaned or "frame"


def image_key(frame_id: str, image_path: str | Path) -> str:
    suffix = Path(image_path).suffix
    stem = safe_key(frame_id)
    return f"{stem}{suffix}" if suffix else stem


def annotations_dir(job_dir: Path) -> Path:
    return job_dir / "sam3" / "annotations"


def masks_dir(job_dir: Path) -> Path:
    return job_dir / "sam3" / "masks"


def logs_dir(job_dir: Path) -> Path:
    return job_dir / "sam3" / "logs"


def frame_db_path(job_dir: Path) -> Path:
    return job_dir / "frames.db"


def run_metadata_path(job_dir: Path) -> Path:
    return job_dir / "sam3" / "run.json"


def write_run_metadata(job_dir: Path, metadata: dict[str, Any]) -> Path:
    path = run_metadata_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
