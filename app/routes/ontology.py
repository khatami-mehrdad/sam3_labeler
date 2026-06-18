"""API routes for ontology generation."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

import yaml
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from app.core import db
from app.core.config import ROOT
from app.core.prompt_optimizer import ClassResult, optimize_all_classes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ontology", tags=["ontology"])

# In-memory cache for live progress of currently-running jobs only.
# Completed/failed jobs are read from the DB.
_running: dict[str, dict] = {}


@router.post("/generate")
async def generate_ontology(payload: dict) -> dict:
    """Start the automated VLM->SAM3 prompt optimization loop."""
    job_id = str(uuid.uuid4())[:8]
    classes = payload.get("classes", [])
    now = db.now_utc_iso()

    await db.insert_ontology_job({
        "id": job_id,
        "status": "running",
        "image_dir": payload.get("image_dir", ""),
        "classes_json": json.dumps(classes),
        "provider": payload.get("provider", ""),
        "model": payload.get("model", ""),
        "created_at": now,
    })
    await db.update_ontology_job(job_id, started_at=now)

    _running[job_id] = {"progress": []}

    image_dir = Path(payload["image_dir"])
    if not image_dir.is_dir():
        await db.update_ontology_job(
            job_id, status="failed",
            error=f"Directory not found: {image_dir}",
            finished_at=db.now_utc_iso(),
        )
        _running.pop(job_id, None)
        return {"job_id": job_id}

    if not classes:
        await db.update_ontology_job(
            job_id, status="failed",
            error="No object classes specified",
            finished_at=db.now_utc_iso(),
        )
        _running.pop(job_id, None)
        return {"job_id": job_id}

    async def _run() -> None:
        try:
            async def on_progress(msg: str) -> None:
                if job_id in _running:
                    _running[job_id]["progress"].append(msg)

            results = await optimize_all_classes(
                image_dir=image_dir,
                classes=classes,
                vlm_provider=payload["provider"],
                vlm_model=payload["model"],
                api_key=payload["api_key"],
                score_threshold=payload.get("score_threshold", 0.5),
                max_iterations=payload.get("max_iterations", 3),
                max_images=payload.get("max_images", 10),
                on_progress=on_progress,
            )
            serialized = [_serialize_class_result(r) for r in results]
            progress = _running.get(job_id, {}).get("progress", [])
            await db.update_ontology_job(
                job_id, status="done",
                results_json=json.dumps(serialized),
                progress_json=json.dumps(progress),
                finished_at=db.now_utc_iso(),
            )
        except Exception as exc:
            logger.exception("Ontology generation failed for job %s", job_id)
            progress = _running.get(job_id, {}).get("progress", [])
            await db.update_ontology_job(
                job_id, status="failed",
                error=str(exc),
                progress_json=json.dumps(progress),
                finished_at=db.now_utc_iso(),
            )
        _running.pop(job_id, None)

    asyncio.create_task(_run())
    return {"job_id": job_id}


@router.get("/jobs")
async def list_jobs() -> list[dict]:
    """List all ontology jobs (most recent first)."""
    rows = await db.list_ontology_jobs(limit=50)
    return [
        {
            "id": j["id"],
            "status": j["status"],
            "image_dir": j.get("image_dir", ""),
            "classes": json.loads(j["classes_json"]) if j.get("classes_json") else [],
            "has_results": bool(j.get("results_json")),
        }
        for j in rows
    ]


@router.get("/status/{job_id}")
async def get_status(job_id: str) -> dict:
    """Poll optimization progress. Live progress for running jobs, DB for completed."""
    if job_id in _running:
        row = await db.get_ontology_job(job_id)
        return {
            "id": job_id,
            "status": row["status"] if row else "running",
            "progress": _running[job_id]["progress"],
            "results": None,
            "error": None,
        }

    row = await db.get_ontology_job(job_id)
    if not row:
        return {"error": "job not found"}
    return {
        "id": row["id"],
        "status": row["status"],
        "progress": json.loads(row["progress_json"]) if row.get("progress_json") else [],
        "results": json.loads(row["results_json"]) if row.get("results_json") else None,
        "error": row.get("error"),
    }


@router.post("/save")
async def save_ontology(payload: dict) -> dict:
    """Save generated ontology to YAML."""
    entries = payload.get("entries", [])
    save_path = Path(payload["path"])
    if not save_path.is_absolute():
        save_path = ROOT / save_path

    save_path.parent.mkdir(parents=True, exist_ok=True)

    ontology = {}
    for entry in entries:
        ontology[entry["prompt"]] = entry["class_name"]

    save_path.write_text(
        yaml.dump(ontology, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {"saved": True, "path": str(save_path), "n_entries": len(ontology)}


@router.get("/overlay")
async def get_overlay(path: str) -> FileResponse:
    """Serve a mask overlay image for the UI."""
    p = Path(path)
    if not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="image/jpeg")


def _serialize_prompt_score(s) -> dict:
    return {
        "prompt": s.prompt,
        "avg_score": s.avg_score,
        "hit_rate": s.hit_rate,
        "avg_detections": s.avg_detections,
        "iteration": s.iteration,
    }


def _serialize_class_result(r: ClassResult) -> dict:
    return {
        "class_name": r.class_name,
        "keywords": r.keywords,
        "best": _serialize_prompt_score(r.best),
        "kept_prompts": [_serialize_prompt_score(s) for s in r.kept_prompts],
        "all_scores": [_serialize_prompt_score(s) for s in r.all_scores],
        "iterations_used": r.iterations_used,
        "converged": r.converged,
    }
