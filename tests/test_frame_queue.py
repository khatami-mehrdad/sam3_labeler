from __future__ import annotations

from pathlib import Path
from threading import Thread

from app.core import frame_queue


def test_frame_queue_claims_one_ready_frame_at_a_time(tmp_path: Path) -> None:
    db_path = tmp_path / "frames.db"
    frame_queue.init_db(db_path)
    frame_queue.upsert_frames(
        db_path,
        [
            {
                "id": "frame_1",
                "path": "/tmp/frame_1.jpg",
                "origin": "fixture",
                "source_kind": "image",
                "width": 10,
                "height": 20,
            },
            {
                "id": "frame_2",
                "path": "/tmp/frame_2.jpg",
                "origin": "fixture",
                "source_kind": "image",
                "width": 30,
                "height": 40,
            },
        ],
    )

    first = frame_queue.claim_ready_frame(db_path, "gpu_0")
    second = frame_queue.claim_ready_frame(db_path, "gpu_1")
    assert first is not None
    assert second is not None
    assert {first["frame_id"], second["frame_id"]} == {"frame_1", "frame_2"}
    assert frame_queue.claim_ready_frame(db_path, "gpu_2") is None

    frame_queue.mark_labeled(db_path, first["frame_id"], "/tmp/frame_1.json", None)
    frame_queue.mark_failed(db_path, second["frame_id"], "boom")

    assert frame_queue.counts(db_path) == {"failed": 1, "labeled": 1}
    assert not frame_queue.has_open_work(db_path)


def test_frame_queue_concurrent_claims_do_not_duplicate_frames(tmp_path: Path) -> None:
    db_path = tmp_path / "frames.db"
    frame_queue.init_db(db_path)
    frame_queue.upsert_frames(
        db_path,
        [
            {
                "id": f"frame_{idx}",
                "path": f"/tmp/frame_{idx}.jpg",
                "origin": "fixture",
                "source_kind": "image",
                "width": 10,
                "height": 20,
            }
            for idx in range(50)
        ],
    )

    claimed: list[str] = []
    errors: list[BaseException] = []

    def worker(worker_id: str) -> None:
        try:
            while row := frame_queue.claim_ready_frame(db_path, worker_id):
                claimed.append(row["frame_id"])
                frame_queue.mark_labeled(db_path, row["frame_id"], "/tmp/ann.json", None)
        except BaseException as exc:
            errors.append(exc)

    threads = [Thread(target=worker, args=(f"worker_{idx}",)) for idx in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(claimed) == 50
    assert len(set(claimed)) == 50
    assert frame_queue.counts(db_path) == {"labeled": 50}
