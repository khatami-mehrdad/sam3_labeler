"""sam3_labeler FastAPI app entrypoint.

Run:
    bash run.sh
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core import db
from app.core.config import ROOT
from app.core.export_runner import enqueue as enqueue_export
from app.core.export_runner import start_workers as start_export_workers
from app.core.job_runner import enqueue, start_workers
from app.routes import jobs as jobs_routes
from app.routes import ontology as ontology_routes
from app.routes import ui as ui_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    start_workers()
    start_export_workers()
    # Re-enqueue any jobs that were queued/running when the server died
    for j in await db.list_active_jobs():
        await enqueue(j["id"])
    for j in await db.list_active_export_jobs():
        await enqueue_export(j["id"])
    # Mark orphaned ontology jobs (were running when server died) as failed
    for j in await db.list_active_ontology_jobs():
        await db.update_ontology_job(
            j["id"], status="failed",
            error="Server restarted while job was running",
            finished_at=db.now_utc_iso(),
        )
    yield


app = FastAPI(title="sam3_labeler", lifespan=lifespan)

# Static files (CSS / future assets)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

# Routes
app.include_router(ui_routes.router)
app.include_router(jobs_routes.router)
app.include_router(ontology_routes.router)
