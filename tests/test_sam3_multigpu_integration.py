from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from app.core import artifacts, frame_queue, sam3_workers
from app.core.job_runner import _write_ontology


pytestmark = pytest.mark.sam3


def test_real_sam3_workers_can_label_on_multiple_available_gpus(
    tmp_path: Path,
    cafe_camera_dirs: list[Path],
) -> None:
    if os.environ.get("DGL_RUN_SAM3_GPU_TESTS") != "1":
        pytest.skip("set DGL_RUN_SAM3_GPU_TESTS=1 to run real SAM3 GPU integration tests")
    if not os.environ.get("DGL_MODEL_PYTHON"):
        pytest.skip("DGL_MODEL_PYTHON must point at the SAM3 Python environment")
    if not os.environ.get("DGL_SAM3_REPO"):
        pytest.skip("DGL_SAM3_REPO must point at the upstream SAM3 checkout")

    gpus = sam3_workers.selected_gpus()
    if len(gpus) < 2:
        pytest.skip("at least two GPUs with enough free memory are required")

    job_dir = tmp_path / "sam3_multigpu"
    frame_db = artifacts.frame_db_path(job_dir)
    frame_queue.init_db(frame_db)
    rows = []
    for idx, path in enumerate(
        sorted(cafe_camera_dirs[0].glob("*.jpg"))[:2]
        + sorted(cafe_camera_dirs[1].glob("*.jpg"))[:2]
    ):
        rows.append(
            {
                "id": f"frame_{idx}",
                "path": str(path),
                "origin": str(path.parent),
                "source_kind": "image",
                "width": 0,
                "height": 0,
            }
        )
    frame_queue.upsert_frames(frame_db, rows)
    _write_ontology(
        job_dir,
        {"ontology": [{"class_id": 0, "class_name": "person", "prompt": "person"}]},
    )
    done_file = job_dir / "sam3" / "frame_prep.done"
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.write_text("done\n", encoding="utf-8")

    specs = sam3_workers.build_worker_specs(
        job_dir,
        {
            "allowed_gpus": gpus[:2],
            "save_masks": False,
        },
        done_file,
    )

    async def run_workers() -> list[int]:
        handles = await sam3_workers.launch_workers(specs)
        return await sam3_workers.wait_workers(handles)

    exit_codes = asyncio.run(run_workers())

    assert exit_codes == [0, 0]
    assert frame_queue.counts(frame_db) == {"labeled": 4}
