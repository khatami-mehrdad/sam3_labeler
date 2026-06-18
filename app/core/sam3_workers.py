"""Launch and monitor SAM3 frame-labeling workers."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core import artifacts
from app.core.config import ALLOWED_GPUS, MODEL_PYTHON, ROOT, SAM3_CHECKPOINT, SAM3_REPO

MIN_FREE_MEMORY_MIB = 15 * 1024


@dataclass
class WorkerSpec:
    worker_id: str
    gpu: int
    command: list[str]
    log_path: Path


@dataclass
class WorkerHandle:
    spec: WorkerSpec
    process: asyncio.subprocess.Process


@dataclass(frozen=True)
class GpuMemory:
    index: int
    free_mib: int
    total_mib: int

    @property
    def required_free_mib(self) -> int:
        return max(MIN_FREE_MEMORY_MIB, self.total_mib // 2)

    @property
    def has_enough_free_memory(self) -> bool:
        return self.free_mib >= self.required_free_mib


def selected_gpus(configured: list[int] | None = None, count: int | None = None) -> list[int]:
    memory = _detect_gpu_memory()
    if configured:
        if not memory:
            return configured
        allowed = set(configured)
        return [
            gpu.index
            for gpu in memory
            if gpu.index in allowed and gpu.has_enough_free_memory
        ]

    def limit(gpus: list[int]) -> list[int]:
        if count is None:
            return gpus
        return gpus[:max(0, int(count))]

    if ALLOWED_GPUS is not None:
        allowed = set(ALLOWED_GPUS)
        if not memory:
            return limit(sorted(allowed))
        eligible = [
            gpu
            for gpu in memory
            if gpu.index in allowed and gpu.has_enough_free_memory
        ]
        return limit([gpu.index for gpu in sorted(eligible, key=lambda gpu: gpu.free_mib, reverse=True)])
    if memory:
        eligible = [gpu for gpu in memory if gpu.has_enough_free_memory]
        return limit([gpu.index for gpu in sorted(eligible, key=lambda gpu: gpu.free_mib, reverse=True)])
    detected = _detect_gpus()
    return limit(detected) or [0]


def _detect_gpu_memory() -> list[GpuMemory]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except Exception:
        return []

    gpus: list[GpuMemory] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append(GpuMemory(index=int(parts[0]), free_mib=int(parts[1]), total_mib=int(parts[2])))
        except ValueError:
            continue
    return gpus


def _detect_gpus() -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    return [int(line.strip()) for line in out.splitlines() if line.strip()]


def build_worker_specs(job_dir: Path, cfg: dict, producer_done_file: Path) -> list[WorkerSpec]:
    gpus = selected_gpus(cfg.get("allowed_gpus"), cfg.get("gpu_count"))
    if not gpus:
        raise RuntimeError(
            "No configured GPU has enough free memory for SAM3 workers "
            f"(requires at least {MIN_FREE_MEMORY_MIB // 1024}GB or half of GPU memory)."
        )
    model_python = cfg.get("model_python") or MODEL_PYTHON or sys.executable
    sam3_repo = cfg.get("sam3_repo") or SAM3_REPO
    checkpoint = SAM3_CHECKPOINT

    worker_script = ROOT / "app" / "tools" / "sam3_frame_worker.py"
    frames_db = artifacts.frame_db_path(job_dir)
    ontology_json = job_dir / "sam3" / "ontology.json"
    log_dir = artifacts.logs_dir(job_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    specs: list[WorkerSpec] = []
    for idx, gpu in enumerate(gpus):
        worker_id = f"gpu_{gpu}_worker_{idx:02d}"
        command = [
            model_python,
            str(worker_script),
            "--frames-db",
            str(frames_db),
            "--job-dir",
            str(job_dir),
            "--ontology-json",
            str(ontology_json),
            "--producer-done-file",
            str(producer_done_file),
            "--worker-id",
            worker_id,
            "--device",
            "cuda",
            "--dtype",
            "bf16",
            "--score-threshold",
            str(cfg.get("score_threshold", 0.35)),
        ]
        if sam3_repo:
            command.extend(["--sam3-repo", str(sam3_repo)])
        if checkpoint:
            command.extend(["--checkpoint", str(checkpoint)])
        if cfg.get("save_masks", True):
            command.append("--save-masks")
        specs.append(
            WorkerSpec(
                worker_id=worker_id,
                gpu=int(gpu),
                command=command,
                log_path=log_dir / f"{worker_id}.log",
            )
        )
    return specs


async def launch_workers(specs: list[WorkerSpec]) -> list[WorkerHandle]:
    handles: list[WorkerHandle] = []
    for spec in specs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        log_handle = spec.log_path.open("a", encoding="utf-8")
        process = await asyncio.create_subprocess_exec(
            *spec.command,
            cwd=str(ROOT),
            env=env,
            stdout=log_handle,
            stderr=asyncio.subprocess.STDOUT,
        )
        log_handle.close()
        handles.append(WorkerHandle(spec=spec, process=process))
    return handles


async def wait_workers(handles: list[WorkerHandle]) -> list[int]:
    return [await handle.process.wait() for handle in handles]
