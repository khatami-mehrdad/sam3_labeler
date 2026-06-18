"""Static throughput lookup table → time estimates for jobs.

Numbers seeded from the weapon project (S2/S4/S6 measurements).
GPU = single Quadro RTX 8000 reference.
"""
from typing import Optional

# imgs/sec/GPU
THROUGHPUT = {
    # Source classifiers
    "siglip2-large":         5.0,    # I/O bound (NFS) — actual measured
    "siglip2-giant":         3.5,
    # VLMs
    "qwen3-vl-8b":           0.7,    # text generation per image
    # Detectors
    "groundingdino-base":    1.0,    # fp32, single image
    "groundingdino-tiny":    3.0,
    "owlv2-large":           2.0,    # fp16 per image
    # Refiners
    "sam2-large":            2.5,
    "sam3-image":            1.0,    # placeholder until local SAM3.1 throughput is measured
    # CPU stages (no GPU)
    "phash-dedup":           500.0,
    "scene-detect":          50.0,   # scenes per second of video processed
    "ingest-images":         200.0,
    "ingest-video-extract":  10.0,   # extracted frames per sec wall clock
    "emit-yolo":             1000.0,
}


def estimate_seconds(stage_name: str, n_images: int, n_gpus: int = 1) -> float:
    rate = THROUGHPUT.get(stage_name)
    if rate is None or n_images == 0:
        return 0.0
    parallel = max(1, n_gpus)
    return n_images / (rate * parallel)


def estimate_pipeline_seconds(stages: list[dict], n_images: int,
                              available_gpus: int = 1) -> dict:
    """Estimate time for a pipeline.

    stages: list of {"name": str, "model": str, "uses_gpu": bool, ...}
    Returns {"total_sec": ..., "per_stage_sec": {...}}
    """
    out = {}
    total = 0.0
    for s in stages:
        gpus = available_gpus if s.get("uses_gpu", True) else 1
        sec = estimate_seconds(s["model"], n_images, gpus)
        # Bias for ambiguous-only stages (VLM review): assume ~15% trigger
        if s.get("ambiguous_only"):
            sec *= 0.15
        out[s["name"]] = sec
        total += sec
    return {"total_sec": total, "per_stage_sec": out}


def humanize(sec: float) -> str:
    if sec < 60:
        return f"{int(sec)}s"
    if sec < 3600:
        return f"{int(sec/60)}m"
    if sec < 86400:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        return f"{h}h {m}m"
    d = int(sec // 86400)
    h = int((sec % 86400) // 3600)
    return f"{d}d {h}h"
