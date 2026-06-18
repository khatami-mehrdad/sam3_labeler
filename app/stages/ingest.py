"""Ingest stage: turn whatever the user supplied into a flat list of frames on disk.

Inputs supported:
  - local image directory  → list image files recursively
  - local video file       → extract frames (with optional scene-detect)

Output: writes paths into job_dir/frames/, returns inventory parquet.
"""
import subprocess
from pathlib import Path
from typing import Iterator

import pandas as pd
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def _walk_images(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _extract_video_frames(video_path: Path, out_dir: Path,
                          fps: float = 2.0) -> list[Path]:
    """Extract frames at given fps using ffmpeg. Returns paths to extracted PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{video_path.stem}_%06d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path), "-vf", f"fps={fps}", "-q:v", "3", pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
    return sorted(out_dir.glob(f"{video_path.stem}_*.jpg"))


def _extract_video_keyframes(video_path: Path, out_dir: Path,
                             scene_detect: bool,
                             scene_threshold: float,
                             video_fps: float) -> list[Path]:
    if scene_detect:
        from app.stages.scene_detect import extract_scene_keyframes
        return extract_scene_keyframes(video_path, out_dir, threshold=scene_threshold)
    return _extract_video_frames(video_path, out_dir, fps=video_fps)


def ingest(sources: list[str], job_dir: Path,
           video_fps: float = 2.0,
           scene_detect: bool = True,
           scene_threshold: float = 27.0) -> pd.DataFrame:
    """
    sources: list of image directories and/or video file paths.
    Returns a DataFrame with columns: id, path, source_kind, origin
    """
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    def add(path: Path, source_kind: str, origin: str):
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            return
        rows.append({
            "id": f"{source_kind}_{len(rows):08d}",
            "path": str(path),
            "source_kind": source_kind,
            "origin": origin,
            "width": w, "height": h,
        })

    for src in sources:
        src = src.strip()
        if not src:
            continue

        if src.lower().startswith(("http://", "https://")):
            raise ValueError(f"unsupported source URL: {src}")

        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(f"source path does not exist: {src}")

        if p.is_dir():
            for img in _walk_images(p):
                add(img, "image", str(p))
        elif p.suffix.lower() in VID_EXTS:
            origin = str(p)
            for kf in _extract_video_keyframes(
                p, frames_dir, scene_detect, scene_threshold, video_fps,
            ):
                add(kf, "video", origin)
        else:
            raise ValueError(f"unsupported source; use an image folder or video file: {src}")

    df = pd.DataFrame(rows)
    df.to_parquet(job_dir / "manifest_ingest.parquet", index=False)
    return df
