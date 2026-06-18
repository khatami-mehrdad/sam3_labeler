"""Job runner for SAM3 raw-annotation labeling jobs."""
import asyncio
import json
import shlex
import traceback
from pathlib import Path

import yaml

from app.core import artifacts, frame_queue, sam3_workers
from app.core.config import MAX_CONCURRENT_JOBS
from app.stages import ingest as ingest_stage


_QUEUE: asyncio.Queue | None = None
_WORKERS: list[asyncio.Task] = []


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


async def _set_stage(job_id: str, stage: str, pct: float = 0.0) -> None:
    from app.core import db

    await db.update_job(job_id, current_stage=stage, progress_pct=pct)


def _log_path(job_dir: Path, stage: str) -> Path:
    p = job_dir / "logs" / f"{stage}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _append_log(job_dir: Path, stage: str, message: str) -> None:
    with _log_path(job_dir, stage).open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


async def _run_job(job_id: str) -> None:
    from app.core import db

    rec = await db.get_job(job_id)
    if rec is None:
        return
    cfg = yaml.safe_load(rec["config_yaml"])
    job_dir = Path(rec["output_path"])
    job_dir.mkdir(parents=True, exist_ok=True)

    await db.update_job(
        job_id,
        status="running",
        started_at=db.now_utc_iso(),
        current_stage="starting",
        progress_pct=0.0,
        error=None,
    )

    try:
        frame_db = artifacts.frame_db_path(job_dir)
        frame_queue.init_db(frame_db)

        ontology = _write_ontology(job_dir, cfg)
        producer_done = job_dir / "sam3" / "frame_prep.done"
        if producer_done.exists():
            producer_done.unlink()

        worker_specs = sam3_workers.build_worker_specs(job_dir, cfg, producer_done)
        gpu_mode = "explicit_ids" if cfg.get("allowed_gpus") else "auto_count"
        _append_log(
            job_dir,
            "sam3_labeling",
            f"gpu_selection mode={gpu_mode} requested={cfg.get('allowed_gpus') or cfg.get('gpu_count') or 'all'} "
            f"selected={[spec.gpu for spec in worker_specs]}",
        )
        for spec in worker_specs:
            _append_log(
                job_dir,
                "sam3_labeling",
                f"worker_spec worker_id={spec.worker_id} gpu={spec.gpu} log={spec.log_path} "
                f"command={shlex.join(spec.command)}",
            )
        artifacts.write_run_metadata(job_dir, {
            "job_id": job_id,
            "name": rec["name"],
            "created_at": db.now_utc_iso(),
            "ontology": ontology,
            "config": cfg,
            "workers": [
                {
                    "worker_id": spec.worker_id,
                    "gpu": spec.gpu,
                    "log_path": str(spec.log_path),
                    "command": spec.command,
                }
                for spec in worker_specs
            ],
        })

        await _set_stage(job_id, "launch_sam3_workers", 0.05)
        worker_handles = await sam3_workers.launch_workers(worker_specs)
        _append_log(job_dir, "sam3_labeling", f"launched_workers count={len(worker_handles)}")

        try:
            await _set_stage(job_id, "prepare_frames", 0.10)
            n_ready = await asyncio.to_thread(_prepare_frames, cfg, job_dir, frame_db)
            _log_path(job_dir, "prepare_frames").write_text(
                f"sources={len(cfg.get('sources') or [])} prepared={n_ready} "
                f"phash_dedup={cfg.get('phash_dedup', True)}\n"
            )
        finally:
            producer_done.parent.mkdir(parents=True, exist_ok=True)
            producer_done.write_text(db.now_utc_iso() + "\n")

        await _set_stage(job_id, "sam3_labeling", 0.50)
        exit_codes = await _wait_workers_with_progress(job_id, frame_db, worker_handles)
        _append_log(job_dir, "sam3_labeling", f"worker_exit_codes {exit_codes}")
        if any(code != 0 for code in exit_codes):
            raise RuntimeError(f"SAM3 workers failed with exit codes {exit_codes}")

        counts = frame_queue.counts(frame_db)
        distribution = _worker_distribution(frame_db)
        _append_log(job_dir, "sam3_labeling", f"frame_counts {json.dumps(counts, sort_keys=True)}")
        _append_log(job_dir, "sam3_labeling", f"worker_distribution {json.dumps(distribution, sort_keys=True)}")
        artifacts.write_run_metadata(job_dir, {
            "job_id": job_id,
            "name": rec["name"],
            "finished_at": db.now_utc_iso(),
            "ontology": ontology,
            "config": cfg,
            "frame_counts": counts,
        })

        await db.update_job(
            job_id,
            status="done",
            current_stage=None,
            progress_pct=1.0,
            finished_at=db.now_utc_iso(),
        )
    except Exception as e:
        tb = traceback.format_exc()
        _log_path(job_dir, "ERROR").write_text(tb)
        await db.update_job(
            job_id,
            status="failed",
            finished_at=db.now_utc_iso(),
            error=f"{type(e).__name__}: {e}",
        )


def _prepare_frames(cfg: dict, job_dir: Path, frame_db: Path) -> int:
    df = ingest_stage.ingest(
        sources=cfg["sources"],
        job_dir=job_dir,
        video_fps=cfg.get("video_fps", 2.0),
        scene_detect=cfg.get("scene_detect", True),
        scene_threshold=cfg.get("scene_threshold", 27.0),
    )
    if cfg.get("phash_dedup", True) and len(df) > 0:
        from app.stages import frame_dedup as frame_dedup_stage

        ingested_df = df
        df = frame_dedup_stage.phash_dedup(df)
        _remove_unused_extracted_frames(ingested_df, df, job_dir)
        df.to_parquet(job_dir / "manifest_dedup.parquet", index=False)
    _validate_frame_paths_exist(df)
    rows = df.to_dict("records")
    frame_queue.upsert_frames(frame_db, rows, status="ready")
    return len(rows)


def _remove_unused_extracted_frames(
    ingested_df,
    queued_df,
    job_dir: Path,
) -> int:
    """Delete deduped-out video frames extracted into this job's frames dir."""
    frames_dir = (job_dir / "frames").resolve()
    queued_paths = {str(Path(path).resolve()) for path in queued_df["path"].tolist()}
    removed = 0

    for row in ingested_df.to_dict("records"):
        if row.get("source_kind") != "video":
            continue

        path = Path(str(row["path"]))
        try:
            resolved = path.resolve()
        except OSError:
            continue

        if str(resolved) in queued_paths:
            continue
        if not resolved.is_relative_to(frames_dir):
            continue
        if not path.is_file():
            continue

        path.unlink()
        removed += 1

    return removed


def _validate_frame_paths_exist(df) -> None:
    missing = [str(path) for path in df["path"].tolist() if not Path(str(path)).is_file()]
    if missing:
        raise FileNotFoundError(f"prepared frame path does not exist: {missing[0]}")


def _worker_distribution(frame_db: Path) -> dict[str, dict[str, int]]:
    with frame_queue.connect(frame_db) as conn:
        rows = conn.execute(
            """
            SELECT COALESCE(worker_id, '(none)') AS worker_id, status, COUNT(*) AS n
            FROM frames
            GROUP BY worker_id, status
            ORDER BY worker_id, status
            """
        ).fetchall()
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        out.setdefault(str(row["worker_id"]), {})[str(row["status"])] = int(row["n"])
    return out


async def _wait_workers_with_progress(job_id: str, frame_db: Path, worker_handles) -> list[int]:
    tasks = [asyncio.create_task(handle.process.wait()) for handle in worker_handles]
    while True:
        if all(task.done() for task in tasks):
            return [int(task.result()) for task in tasks]
        counts = await asyncio.to_thread(frame_queue.counts, frame_db)
        total = sum(counts.values())
        if total:
            finished = counts.get("labeled", 0) + counts.get("failed", 0)
            progress = 0.50 + 0.49 * (finished / total)
            await _set_stage(job_id, "sam3_labeling", min(progress, 0.99))
        await asyncio.sleep(5)


def _write_ontology(job_dir: Path, cfg: dict) -> list[dict]:
    ontology = _normalize_ontology(cfg)
    sam3_dir = job_dir / "sam3"
    sam3_dir.mkdir(parents=True, exist_ok=True)
    ontology_path = sam3_dir / "ontology.json"
    ontology_path.write_text(json.dumps(ontology, indent=2) + "\n", encoding="utf-8")

    class_names = []
    for item in sorted(ontology, key=lambda row: row["class_id"]):
        if item["class_name"] not in class_names:
            class_names.append(item["class_name"])
    (sam3_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    return ontology


def _normalize_ontology(cfg: dict) -> list[dict]:
    raw = cfg.get("ontology")
    if isinstance(raw, dict):
        return [
            {"class_id": i, "class_name": str(class_name), "prompt": str(prompt)}
            for i, (prompt, class_name) in enumerate(raw.items())
        ]
    if isinstance(raw, list):
        out = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                out.append({"class_id": i, "class_name": item, "prompt": item})
            else:
                prompt = item.get("prompt")
                if not prompt:
                    raise ValueError(f"ontology item {i} is missing prompt")
                out.append({
                    "class_id": int(item.get("class_id", i)),
                    "class_name": str(item.get("class_name") or item.get("name") or prompt),
                    "prompt": str(prompt),
                })
        return out

    prompts = cfg.get("prompts") or cfg.get("detection_prompts") or []
    class_name = cfg.get("output_class", "object")
    return [
        {"class_id": 0, "class_name": str(class_name), "prompt": str(prompt)}
        for prompt in prompts
    ]
