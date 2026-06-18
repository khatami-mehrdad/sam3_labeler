from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yaml

from app.core import artifacts, db, frame_queue, job_runner


class FakeWorkerProcess:
    def __init__(self, frame_db: Path, job_dir: Path, done_file: Path, return_code: int = 0) -> None:
        self.frame_db = frame_db
        self.job_dir = job_dir
        self.done_file = done_file
        self.return_code = return_code

    async def wait(self) -> int:
        while not self.done_file.exists() or frame_queue.has_open_work(self.frame_db):
            row = frame_queue.claim_ready_frame(self.frame_db, "fake_sam3")
            if row is None:
                await asyncio.sleep(0)
                continue
            annotation_path = artifacts.annotations_dir(self.job_dir) / f"{row['frame_id']}.json"
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(
                json.dumps(
                    {
                        "frame_id": row["frame_id"],
                        "image_path": row["path"],
                        "annotations": [],
                        "model": "fake-sam3",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            frame_queue.mark_labeled(self.frame_db, row["frame_id"], str(annotation_path), None)
        return self.return_code


class FakeWorkerHandle:
    def __init__(self, process: FakeWorkerProcess) -> None:
        self.process = process


async def _run_fake_job(
    tmp_path: Path,
    monkeypatch,
    cafe_camera_dirs: list[Path],
    worker_return_code: int = 0,
) -> tuple[str, Path]:
    db.DB_PATH = tmp_path / "jobs.db"
    await db.init_db()
    real_sleep = asyncio.sleep

    async def fake_launch_workers(specs):
        return [
            FakeWorkerHandle(
                FakeWorkerProcess(
                    artifacts.frame_db_path(tmp_path / "job_output"),
                    tmp_path / "job_output",
                    tmp_path / "job_output" / "sam3" / "frame_prep.done",
                    return_code=worker_return_code,
                )
            )
            for _ in specs
        ]

    async def fast_sleep(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(job_runner.sam3_workers, "launch_workers", fake_launch_workers)
    monkeypatch.setattr(job_runner.asyncio, "sleep", fast_sleep)

    job_id = "job-functional"
    output_path = tmp_path / "job_output"
    await db.insert_job(
        {
            "id": job_id,
            "name": "functional",
            "description": "",
            "config_yaml": yaml.safe_dump(
                {
                    "sources": [str(path) for path in cafe_camera_dirs],
                    "ontology": [{"class_id": 0, "class_name": "person", "prompt": "person"}],
                    "allowed_gpus": [0],
                    "phash_dedup": False,
                    "scene_detect": False,
                    "save_masks": False,
                }
            ),
            "status": "queued",
            "created_at": db.now_utc_iso(),
            "output_path": str(output_path),
        }
    )

    await job_runner._run_job(job_id)
    return job_id, output_path


def test_run_job_finishes_with_fake_sam3_workers(
    tmp_path: Path,
    monkeypatch,
    cafe_camera_dirs: list[Path],
) -> None:
    job_id, output_path = asyncio.run(_run_fake_job(tmp_path, monkeypatch, cafe_camera_dirs))

    rec = asyncio.run(db.get_job(job_id))
    assert rec is not None
    assert rec["status"] == "done"
    assert rec["current_stage"] is None
    assert rec["progress_pct"] == 1.0
    assert frame_queue.counts(artifacts.frame_db_path(output_path)) == {"labeled": 20}
    assert (output_path / "sam3" / "run.json").exists()
    assert (output_path / "logs" / "prepare_frames.log").read_text() == (
        "sources=2 prepared=20 phash_dedup=False\n"
    )
    sam3_log = (output_path / "logs" / "sam3_labeling.log").read_text()
    assert "gpu_selection mode=explicit_ids requested=[0] selected=[0]" in sam3_log
    assert "worker_exit_codes [0]" in sam3_log
    assert 'worker_distribution {"fake_sam3": {"labeled": 20}}' in sam3_log


def test_run_job_fails_when_fake_sam3_worker_fails(
    tmp_path: Path,
    monkeypatch,
    cafe_camera_dirs: list[Path],
) -> None:
    job_id, output_path = asyncio.run(
        _run_fake_job(tmp_path, monkeypatch, cafe_camera_dirs, worker_return_code=1)
    )

    rec = asyncio.run(db.get_job(job_id))
    assert rec is not None
    assert rec["status"] == "failed"
    assert "SAM3 workers failed" in rec["error"]
    assert (output_path / "logs" / "ERROR.log").exists()
    sam3_log = (output_path / "logs" / "sam3_labeling.log").read_text()
    assert "worker_exit_codes [1]" in sam3_log
