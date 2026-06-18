from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core import db
from app.routes import jobs as jobs_routes
from app.routes import ui as ui_routes


def test_home_page_renders(tmp_path: Path) -> None:
    db.DB_PATH = tmp_path / "jobs.db"
    asyncio.run(db.init_db())

    app = FastAPI()
    app.include_router(ui_routes.router)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Recent labeling runs" in response.text


def test_browse_lists_sources_under_configured_root(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "clip.mp4").write_text("video")
    (tmp_path / "clip.mov").write_text("video")
    (tmp_path / "notes.txt").write_text("ignored")
    monkeypatch.setattr(jobs_routes, "BROWSE_ROOT", tmp_path)

    app = FastAPI()
    app.include_router(jobs_routes.router)

    with TestClient(app) as client:
        response = client.get("/browse")
        escape = client.get("/browse", params={"path": str(tmp_path.parent)})

    assert response.status_code == 200
    payload = response.json()
    entries = {(entry["name"], entry["type"]) for entry in payload["entries"]}
    assert entries == {("images", "dir"), ("clip.mp4", "video"), ("clip.mov", "video")}
    assert escape.status_code == 400


def test_elfinder_connector_opens_configured_root(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (tmp_path / ".cursor").mkdir()
    (tmp_path / "clip.webm").write_text("video")
    (tmp_path / "notes.txt").write_text("ignored")
    monkeypatch.setattr(jobs_routes, "BROWSE_ROOT", tmp_path)

    app = FastAPI()
    app.include_router(jobs_routes.router)

    with TestClient(app) as client:
        init_response = client.get("/elfinder", params={"cmd": "open", "init": "1"})
        child_response = client.get(
            "/elfinder",
            params={"cmd": "open", "target": jobs_routes._hash_path(image_dir)},
        )

    assert init_response.status_code == 200
    payload = init_response.json()
    assert payload["api"] == "2.1"
    assert payload["cwd"]["path"] == str(tmp_path)
    assert payload["cwd"]["phash"] == ""
    assert payload["options"]["path"] == str(tmp_path)
    files = {entry["name"]: entry for entry in payload["files"]}
    assert "images" in files
    assert "clip.webm" in files
    assert ".cursor" not in files
    assert "notes.txt" not in files

    assert child_response.status_code == 200
    child_payload = child_response.json()
    assert child_payload["cwd"]["path"] == str(image_dir)
    assert child_payload["options"]["path"] == str(image_dir)
    assert child_payload["files"][0]["path"] == str(tmp_path)


def test_create_job_api_rejects_url_sources(tmp_path: Path, monkeypatch) -> None:
    db.DB_PATH = tmp_path / "jobs.db"
    asyncio.run(db.init_db())
    monkeypatch.setattr(jobs_routes, "JOBS_DIR", tmp_path / "jobs")

    app = FastAPI()
    app.include_router(jobs_routes.router)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "name": "url source",
                "recipe_overrides": {
                    "sources": ["https://www.youtube.com/watch?v=abc123"],
                    "ontology": [{"class_id": 0, "class_name": "person", "prompt": "person"}],
                },
            },
        )

    assert response.status_code == 400
    assert "not URLs" in response.text


def test_create_job_api_accepts_yaml_mapping_ontology(
    tmp_path: Path,
    monkeypatch,
    cafe_camera_dirs: list[Path],
) -> None:
    db.DB_PATH = tmp_path / "jobs.db"
    asyncio.run(db.init_db())
    monkeypatch.setattr(jobs_routes, "JOBS_DIR", tmp_path / "jobs")

    enqueued: list[str] = []

    async def fake_enqueue(job_id: str) -> None:
        enqueued.append(job_id)

    monkeypatch.setattr(jobs_routes.job_runner, "enqueue", fake_enqueue)

    app = FastAPI()
    app.include_router(jobs_routes.router)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "name": "mapping ontology",
                "recipe_overrides": {
                    "sources": [str(cafe_camera_dirs[0])],
                    "ontology": {"person": "person", "coffee cup": "cup"},
                },
            },
        )

    assert response.status_code == 200
    rec = asyncio.run(db.get_job(response.json()["job_id"]))
    assert rec is not None
    cfg = yaml.safe_load(rec["config_yaml"])
    assert cfg["ontology"] == {"person": "person", "coffee cup": "cup"}
    assert enqueued == [response.json()["job_id"]]


def test_create_job_api_persists_config_and_enqueues(
    tmp_path: Path,
    monkeypatch,
    cafe_camera_dirs: list[Path],
) -> None:
    db.DB_PATH = tmp_path / "jobs.db"
    asyncio.run(db.init_db())
    monkeypatch.setattr(jobs_routes, "JOBS_DIR", tmp_path / "jobs")

    enqueued: list[str] = []

    async def fake_enqueue(job_id: str) -> None:
        enqueued.append(job_id)

    monkeypatch.setattr(jobs_routes.job_runner, "enqueue", fake_enqueue)

    app = FastAPI()
    app.include_router(jobs_routes.router)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "name": "api smoke",
                "description": "test",
                "recipe_overrides": {
                    "sources": [str(cafe_camera_dirs[0])],
                    "gpu_count": 2,
                    "ontology": [
                        {"class_id": 0, "class_name": "person", "prompt": "person"}
                    ],
                    "phash_dedup": True,
                    "scene_detect": False,
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert enqueued == [payload["job_id"]]
    rec = asyncio.run(db.get_job(payload["job_id"]))
    assert rec is not None
    assert rec["status"] == "queued"
    cfg = yaml.safe_load(rec["config_yaml"])
    assert cfg["sources"] == [str(cafe_camera_dirs[0])]
    assert cfg["gpu_count"] == 2
    assert cfg["ontology"][0]["prompt"] == "person"


def test_create_job_api_rejects_missing_ontology(tmp_path: Path, monkeypatch) -> None:
    db.DB_PATH = tmp_path / "jobs.db"
    asyncio.run(db.init_db())
    monkeypatch.setattr(jobs_routes, "JOBS_DIR", tmp_path / "jobs")

    app = FastAPI()
    app.include_router(jobs_routes.router)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={
                "name": "bad",
                "recipe_overrides": {"sources": ["/tmp/images"]},
            },
        )

    assert response.status_code == 400
    assert "prompt/ontology" in response.text


def test_annotation_browser_and_preview_endpoints(tmp_path: Path, monkeypatch) -> None:
    jobs_dir = tmp_path / "jobs"
    ann_dir = jobs_dir / "run1" / "sam3" / "annotations"
    ann_dir.mkdir(parents=True)
    mask_dir = jobs_dir / "run1" / "sam3" / "masks"
    mask_dir.mkdir(parents=True)
    image_path = tmp_path / "ml_data" / "img.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake-image")

    mask_path = mask_dir / "image_00000000.npz"
    np.savez_compressed(mask_path, masks=np.ones((1, 2, 3), dtype=np.uint8))

    annotation_path = ann_dir / "image_00000000.json"
    annotation_path.write_text(json.dumps({
        "image": str(image_path),
        "mask_file": str(mask_path),
        "width": 3,
        "height": 2,
        "annotations": [
            {"class_name": "food_item", "score": 0.9, "bbox_xyxy": [0, 0, 2, 1], "has_mask": True},
        ],
    }))

    monkeypatch.setattr(jobs_routes, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(jobs_routes, "BROWSE_ROOT", tmp_path / "ml_data")

    app = FastAPI()
    app.include_router(jobs_routes.router)

    with TestClient(app) as client:
        browse = client.get("/browse/annotations")
        preview = client.get("/sam3/annotation-preview", params={"path": str(annotation_path)})
        overlay = client.get(
            "/sam3/mask-overlay",
            params={"annotation_path": str(annotation_path), "annotation_index": 0},
        )

    assert browse.status_code == 200
    browse_names = {entry["name"] for entry in browse.json()["entries"]}
    assert "run1" in browse_names
    assert preview.status_code == 200
    assert preview.json()["image"] == str(image_path)
    assert overlay.status_code == 200
    assert overlay.headers["content-type"] == "image/png"


def test_create_export_job_api_persists_and_enqueues(tmp_path: Path, monkeypatch) -> None:
    db.DB_PATH = tmp_path / "jobs.db"
    asyncio.run(db.init_db())

    jobs_dir = tmp_path / "jobs"
    ann_dir = jobs_dir / "run1" / "sam3" / "annotations"
    ann_dir.mkdir(parents=True)
    image_path = tmp_path / "ml_data" / "img.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake-image")
    (ann_dir / "image_00000000.json").write_text(json.dumps({
        "image": str(image_path),
        "width": 3,
        "height": 2,
        "annotations": [
            {"class_id": 0, "class_name": "food_item", "score": 0.9, "bbox_xyxy": [0, 0, 2, 1]},
        ],
    }))

    export_root = tmp_path / "exports"
    monkeypatch.setattr(jobs_routes, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(jobs_routes, "EXPORTS_DIR", export_root)

    enqueued: list[str] = []

    async def fake_enqueue(job_id: str) -> None:
        enqueued.append(job_id)

    monkeypatch.setattr(jobs_routes.export_runner, "enqueue", fake_enqueue)

    app = FastAPI()
    app.include_router(jobs_routes.router)

    with TestClient(app) as client:
        response = client.post(
            "/export-jobs",
            json={
                "name": "api export",
                "source_annotations_dir": str(ann_dir),
                "train_pct": 80,
                "val_pct": 20,
                "filter_small_boxes": True,
                "small_box_area_factor": 8,
            },
        )
        listed = client.get("/export-jobs")

    assert response.status_code == 200
    payload = response.json()
    assert enqueued == [payload["export_job_id"]]
    rec = asyncio.run(db.get_export_job(payload["export_job_id"]))
    assert rec is not None
    assert rec["status"] == "queued"
    assert rec["source_annotations_dir"] == str(ann_dir)
    assert rec["train_pct"] == 80
    assert rec["val_pct"] == 20
    assert rec["filter_small_boxes"] == 1
    assert rec["small_box_area_factor"] == 8
    assert Path(rec["output_path"]).is_relative_to(export_root)
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == payload["export_job_id"]
