from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from app.core import artifacts, frame_queue
from app.core.job_runner import _prepare_frames
from app.stages import ingest as ingest_stage


def test_cafe_fixture_contains_consecutive_frames_from_two_cameras(
    cafe_camera_dirs: list[Path],
    cafe_frame_paths: list[Path],
) -> None:
    assert [path.name for path in cafe_camera_dirs] == ["camera_02", "camera_17"]
    assert len(cafe_frame_paths) > 0

    by_camera = {
        camera_dir.name: [path.name for path in sorted(camera_dir.glob("*.jpg"))]
        for camera_dir in cafe_camera_dirs
    }
    assert by_camera["camera_02"][:10] == [
        "frame_0000.jpg",
        "frame_0006.jpg",
        "frame_0012.jpg",
        "frame_0018.jpg",
        "frame_0024.jpg",
        "frame_0030.jpg",
        "frame_0036.jpg",
        "frame_0042.jpg",
        "frame_0048.jpg",
        "frame_0054.jpg",
    ]
    assert by_camera["camera_17"][:10] == by_camera["camera_02"][:10]


def test_preprocessing_prepares_cafe_frames_and_fake_sam3_finishes_job(
    tmp_path: Path,
    cafe_camera_dirs: list[Path],
    pytestconfig: pytest.Config,
) -> None:
    per_camera = pytestconfig.getoption("--cafe-frames")
    expected_per_camera = min(
        per_camera,
        min(len(list(camera_dir.glob("*.jpg"))) for camera_dir in cafe_camera_dirs),
    )
    frame_db = artifacts.frame_db_path(tmp_path)
    frame_queue.init_db(frame_db)

    prepared = _prepare_frames(
        {
            "sources": [str(path) for path in cafe_camera_dirs],
            "phash_dedup": False,
            "scene_detect": False,
        },
        tmp_path,
        frame_db,
    )

    assert prepared == expected_per_camera * len(cafe_camera_dirs)
    assert frame_queue.counts(frame_db) == {"ready": prepared}

    manifest = pd.read_parquet(tmp_path / "manifest_ingest.parquet")
    assert set(manifest["origin"]) == {str(path) for path in cafe_camera_dirs}
    assert set(manifest["source_kind"]) == {"image"}

    annotations_dir = artifacts.annotations_dir(tmp_path)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    labeled = 0
    while row := frame_queue.claim_ready_frame(frame_db, "fake_sam3"):
        annotation_path = annotations_dir / f"{row['frame_id']}.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "frame_id": row["frame_id"],
                    "image_path": row["path"],
                    "detections": [],
                    "model": "fake-sam3",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        frame_queue.mark_labeled(frame_db, row["frame_id"], str(annotation_path), None)
        labeled += 1

    assert labeled == prepared
    assert frame_queue.counts(frame_db) == {"labeled": prepared}
    assert not frame_queue.has_open_work(frame_db)


def test_phash_dedup_collapses_consecutive_cafe_frames(
    cafe_frame_paths: list[Path],
) -> None:
    from app.stages import frame_dedup

    rows = [
        {
            "id": f"image_{idx:08d}",
            "path": str(path),
            "source_kind": "image",
            "origin": str(path.parent),
            "width": 0,
            "height": 0,
        }
        for idx, path in enumerate(cafe_frame_paths)
    ]

    deduped = frame_dedup.phash_dedup(pd.DataFrame(rows))

    assert len(rows) == 20
    assert len(deduped) == 4
    by_camera = {
        Path(origin).name: count
        for origin, count in deduped.groupby("origin").size().to_dict().items()
    }
    assert by_camera == {
        "camera_02": 1,
        "camera_17": 3,
    }
    assert set(deduped["origin"]).issubset({str(path.parent) for path in cafe_frame_paths})
    assert "cluster_id" in deduped.columns


def test_preprocessing_with_dedup_queues_only_representative_cafe_frames(
    tmp_path: Path,
    cafe_camera_dirs: list[Path],
) -> None:
    frame_db = artifacts.frame_db_path(tmp_path)
    frame_queue.init_db(frame_db)

    prepared = _prepare_frames(
        {
            "sources": [str(path) for path in cafe_camera_dirs],
            "phash_dedup": True,
            "scene_detect": False,
        },
        tmp_path,
        frame_db,
    )

    assert prepared == 4
    assert frame_queue.counts(frame_db) == {"ready": 4}
    assert len(pd.read_parquet(tmp_path / "manifest_ingest.parquet")) == 20
    assert len(pd.read_parquet(tmp_path / "manifest_dedup.parquet")) == 4


def test_scene_detect_video_path_extracts_keyframes_from_cafe_frames(
    tmp_path: Path,
    cafe_camera_dirs: list[Path],
) -> None:
    pytest.importorskip("scenedetect")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for video ingest tests")

    image_dir = tmp_path / "video_frames"
    image_dir.mkdir()
    for idx, image_path in enumerate(sorted(cafe_camera_dirs[0].glob("*.jpg"))[:10]):
        shutil.copy2(image_path, image_dir / f"frame_{idx:04d}.jpg")

    video_path = tmp_path / "camera_02.mp4"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "5",
            "-i",
            str(image_dir / "frame_%04d.jpg"),
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg could not create test video: {proc.stderr.strip()}")

    df = ingest_stage.ingest(
        [str(video_path)],
        tmp_path / "scene_job",
        video_fps=2.0,
        scene_detect=True,
        scene_threshold=27.0,
    )

    assert len(df) >= 1
    assert set(df["source_kind"]) == {"video"}
    assert set(df["origin"]) == {str(video_path)}


def test_fixed_fps_video_preprocessing_extracts_expected_frame_count(
    tmp_path: Path,
    cafe_camera_dirs: list[Path],
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for video ingest tests")

    image_dir = tmp_path / "video_frames"
    image_dir.mkdir()
    for idx, image_path in enumerate(sorted(cafe_camera_dirs[0].glob("*.jpg"))[:10]):
        shutil.copy2(image_path, image_dir / f"frame_{idx:04d}.jpg")

    video_path = tmp_path / "camera_02.mp4"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "5",
            "-i",
            str(image_dir / "frame_%04d.jpg"),
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg could not create test video: {proc.stderr.strip()}")

    df = ingest_stage.ingest(
        [str(video_path)],
        tmp_path / "fixed_fps_job",
        video_fps=2.0,
        scene_detect=False,
        scene_threshold=27.0,
    )

    assert len(df) == 4
    assert set(df["source_kind"]) == {"video"}
    assert set(df["origin"]) == {str(video_path)}
