from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import pandas as pd

from app.core import artifacts, frame_queue
from app.core.job_runner import (
    _normalize_ontology,
    _prepare_frames,
    _remove_unused_extracted_frames,
    _write_ontology,
)


def test_prepare_frames_writes_ready_queue_from_cafe_subset(
    tmp_path: Path,
    cafe_frame_paths: list[Path],
) -> None:
    images = cafe_frame_paths[:4]
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for image in images:
        shutil.copy2(image, image_dir / image.name)

    frame_db = artifacts.frame_db_path(tmp_path)
    frame_queue.init_db(frame_db)

    count = _prepare_frames(
        {
            "sources": [str(image_dir)],
            "phash_dedup": False,
            "scene_detect": False,
        },
        tmp_path,
        frame_db,
    )

    assert count == len(images)
    assert frame_queue.counts(frame_db) == {"ready": len(images)}
    assert (tmp_path / "manifest_ingest.parquet").exists()


def test_remove_unused_extracted_frames_only_deletes_video_frames_in_job_dir(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    kept = frames_dir / "kept.jpg"
    removed = frames_dir / "removed.jpg"
    source_image = tmp_path / "source.jpg"
    for path in (kept, removed, source_image):
        path.write_bytes(b"fake image bytes")

    ingested = pd.DataFrame(
        [
            {"path": str(kept), "source_kind": "video"},
            {"path": str(removed), "source_kind": "video"},
            {"path": str(source_image), "source_kind": "image"},
        ]
    )
    queued = pd.DataFrame(
        [
            {"path": str(kept), "source_kind": "video"},
        ]
    )

    assert _remove_unused_extracted_frames(ingested, queued, tmp_path) == 1
    assert kept.exists()
    assert not removed.exists()
    assert source_image.exists()


def test_write_ontology_accepts_prompt_class_mapping(tmp_path: Path) -> None:
    ontology = _write_ontology(
        tmp_path,
        {
            "ontology": [
                {"class_id": 0, "class_name": "person", "prompt": "person"},
                {"class_id": 1, "class_name": "cup", "prompt": "coffee cup"},
            ]
        },
    )

    assert ontology == [
        {"class_id": 0, "class_name": "person", "prompt": "person"},
        {"class_id": 1, "class_name": "cup", "prompt": "coffee cup"},
    ]
    assert (tmp_path / "sam3" / "ontology.json").exists()
    assert (tmp_path / "sam3" / "classes.txt").read_text() == "person\ncup\n"


def test_normalize_ontology_keeps_legacy_prompts_as_single_class() -> None:
    assert _normalize_ontology(
        {"detection_prompts": ["dog", "cat"], "output_class": "animal"}
    ) == [
        {"class_id": 0, "class_name": "animal", "prompt": "dog"},
        {"class_id": 0, "class_name": "animal", "prompt": "cat"},
    ]


def test_normalize_ontology_accepts_dict_and_string_list_forms() -> None:
    assert _normalize_ontology({"ontology": {"coffee cup": "cup"}}) == [
        {"class_id": 0, "class_name": "cup", "prompt": "coffee cup"}
    ]
    assert _normalize_ontology({"ontology": ["person", "chair"]}) == [
        {"class_id": 0, "class_name": "person", "prompt": "person"},
        {"class_id": 1, "class_name": "chair", "prompt": "chair"},
    ]


def test_normalize_ontology_rejects_items_without_prompt() -> None:
    with pytest.raises(ValueError, match="missing prompt"):
        _normalize_ontology({"ontology": [{"class_name": "person"}]})
