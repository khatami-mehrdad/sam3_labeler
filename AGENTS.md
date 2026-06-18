# sam3_labeler Agent Context

## Purpose

`sam3_labeler` is a labeling tool that turns raw image directories or local video files into raw SAM3.1 annotation runs. It exposes a FastAPI/Jinja web UI around frame preparation, pHash dedup, and external SAM3 worker processes. Curation and dataset export are separate tasks, not part of the labeling critical path.

Keep cleanup changes conservative. This repo runs heavy GPU models and writes large per-job artifacts, so prefer small, verifiable edits over broad refactors.

## Commands

- Install locally: `python3.11 -m pip install -e .`
- Run through the wrapper: `bash run.sh`
- Run directly: `python3.11 -m uvicorn app.main:app --host 0.0.0.0 --port 8090`

`run.sh` defaults to `HOST=0.0.0.0`, `PORT=8090`, and `WORKERS=1`. Keep Uvicorn workers at `1` unless the model registry is redesigned; loaded model state is in-process.

There is currently no formal test suite (`tests/` is absent and `pyproject.toml` has no pytest dependency). For behavior changes, add focused tests or document the manual verification used.

## Architecture Map

- `app/main.py`: FastAPI entrypoint, static mount, router registration, lifespan startup, worker startup, queued/running job re-enqueue.
- `app/core/config.py`: root paths, artifact paths, server defaults, GPU policy, concurrency, optional Ultralytics fork path.
- `app/core/db.py`: SQLite schema and async persistence helpers for labeling jobs.
- `app/core/job_runner.py`: SAM3 labeling orchestration and stage progress updates.
- `app/core/frame_queue.py`: job-local SQLite frame queue used by SAM3 worker processes.
- `app/core/sam3_workers.py`: launches one SAM3 worker process per selected GPU.
- `app/routes/`: UI and job APIs for labeling runs.
- `app/stages/`: frame preparation stages for ingest, scene detection, and pHash dedup.
- `app/recipes/`: YAML presets loaded by the UI.
- `app/templates/` and `static/`: server-rendered UI and local CSS.
- `bench_gemma4.py`: standalone benchmark script, not part of the web app path.

## Pipeline Flow

The main labeling flow is:

1. `prepare_frames`
2. `phash_dedup`
3. `sam3_labeling`

Labeling outputs raw SAM3 artifacts under `sam3/`. YOLO/COCO/FiftyOne/Label Studio exports should be implemented as separate tasks that consume those raw artifacts.

## Runtime Artifacts And Context Exclusions

Avoid reading, indexing, or committing generated artifacts unless the user explicitly asks:

- `jobs/`
- `data/`
- `*.db`, `*.sqlite3`, `*.log`
- generated YOLO datasets
- downloaded videos and extracted frames
- `.venv/`, `venv/`, `__pycache__/`, `.pytest_cache/`, `*.egg-info/`
- `.claude/`

The `.gitignore` already excludes the main runtime and local-only paths.

## Important Caveats

- `app/core/config.py` derives `ROOT` from the checkout path by default. Use `SAM3_LABELER_ROOT` and `SAM3_LABELER_ALLOWED_GPUS` for host-specific overrides.
- SAM3 runs in worker subprocesses launched by `SAM3_LABELER_MODEL_PYTHON`; use `SAM3_LABELER_SAM3_REPO` and `SAM3_LABELER_CHECKPOINT` for the external model runtime.
- `bench_gemma4.py` is standalone; use `SAM3_LABELER_BENCH_IMAGE_DIR` to point it at benchmark images.
- Keep docs and UI placeholders generic. Search for local absolute paths before committing portability cleanup.
- FiftyOne and Label Studio are optional integrations but are part of `app/routes/jobs.py`; keep their dependency and runtime footprint in mind when simplifying.
- `pyproject.toml` declares the core app dependencies, but optional integration dependencies such as FiftyOne and Label Studio SDK/CLI are not currently declared.
- Per-job outputs can be large. Do not traverse `jobs/` casually.
- Model wrapper imports may look unused but are required for registry side effects.

## Cleanup Priorities

Work in small passes with clear verification:

1. Keep labeling focused on raw SAM3 annotations and masks; do not reintroduce YOLO export, semantic dedup, FiftyOne, or Label Studio into the labeling path.
2. Keep SAM3 imports/model loading inside worker processes, not the FastAPI process.
3. Add focused tests around artifact paths, frame queue claiming, worker command construction, and resume behavior.
4. Move curation/export functionality into separate routes and task runners when those workflows are built.
5. Remove old detector/SigLIP/SAM2 modules once the SAM3 path is stable and no compatibility surface needs them.

Do not combine portability, behavior changes, and large refactors in one pass.
