"""HTML page routes."""
import json

import yaml
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core import db
from app.core.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, ROOT
from app.core.estimator import estimate_pipeline_seconds, humanize

router = APIRouter()
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))


def _load_recipes() -> dict[str, dict]:
    out = {}
    for p in (ROOT / "app" / "recipes").glob("*.yaml"):
        out[p.stem] = yaml.safe_load(p.read_text())
    return out


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    labeling = [{"type": "labeling", **j} for j in await db.list_jobs(limit=100)]
    exports = [{"type": "export", **j} for j in await db.list_export_jobs(limit=100)]
    onto_raw = await db.list_ontology_jobs(limit=100)
    ontology = []
    for j in onto_raw:
        classes = json.loads(j["classes_json"]) if j.get("classes_json") else []
        ontology.append({
            "type": "ontology",
            "id": j["id"],
            "name": ", ".join(classes) or j["id"],
            "status": j["status"],
            "progress_pct": None,
            "current_stage": None,
            "output_path": j.get("image_dir", ""),
            "created_at": j["created_at"],
        })
    all_jobs = sorted(
        labeling + exports + ontology,
        key=lambda j: j.get("created_at", ""),
        reverse=True,
    )[:100]
    return templates.TemplateResponse(request, "jobs_list.html", {
        "jobs": all_jobs,
    })


@router.get("/new", response_class=HTMLResponse)
async def new_job_page(request: Request):
    recipes = _load_recipes()
    return templates.TemplateResponse(request, "new_job.html", {
        "recipes": recipes,
        "recipes_json": yaml.safe_dump(recipes),
    })


@router.get("/jobs/{job_id}/view", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str):
    rec = await db.get_job(job_id)
    if not rec:
        return HTMLResponse(f"job {job_id} not found", status_code=404)
    return templates.TemplateResponse(request, "job_detail.html", {
        "job": rec,
    })


@router.get("/visualizer", response_class=HTMLResponse)
async def visualizer_page(request: Request):
    return templates.TemplateResponse(request, "visualizer.html", {})


@router.get("/exports", response_class=HTMLResponse)
async def exports_page(request: Request):
    jobs = await db.list_jobs(limit=200)
    export_jobs = await db.list_export_jobs(limit=200)
    return templates.TemplateResponse(request, "exports.html", {
        "jobs": jobs,
        "export_jobs": export_jobs,
    })


@router.get("/ontology", response_class=HTMLResponse)
async def ontology_page(request: Request):
    rows = await db.list_ontology_jobs(limit=20)
    jobs_list = [
        {"id": j["id"], "status": j["status"],
         "image_dir": j.get("image_dir", ""),
         "classes": json.loads(j["classes_json"]) if j.get("classes_json") else []}
        for j in rows
    ]
    return templates.TemplateResponse(request, "ontology.html", {
        "openai_key": OPENAI_API_KEY,
        "anthropic_key": ANTHROPIC_API_KEY,
        "recent_jobs_json": json.dumps(jobs_list),
    })


@router.post("/estimate")
async def estimate(payload: dict):
    """Quick runtime estimator — driven by the form."""
    n_images = int(payload.get("n_images") or 0)
    n_gpus   = int(payload.get("n_gpus") or 5)
    stages = [
        {"name": "prepare_frames", "model": "ingest-images", "uses_gpu": False},
        {"name": "sam3_labeling", "model": "sam3-image"},
    ]

    res = estimate_pipeline_seconds(stages, n_images, available_gpus=n_gpus)
    return {
        "total_sec": res["total_sec"],
        "total_h":   humanize(res["total_sec"]),
        "per_stage": {k: humanize(v) for k, v in res["per_stage_sec"].items()},
    }
