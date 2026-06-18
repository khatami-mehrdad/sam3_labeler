# dg-labeller

Internal DeGirum tool for turning raw image folders / local video files into
raw SAM3.1 annotation runs with boxes, scores, and optional mask sidecars.

Labeling stops at durable raw SAM3 artifacts. Curation and dataset export
(YOLO, COCO, Label Studio, FiftyOne, etc.) are separate web-app tasks.

## Quick start

### One-time install
```bash
cd /path/to/dg-labeller
python3.11 -m pip install -e .
```

### Run the server
```bash
cd /path/to/dg-labeller
bash run.sh
```

Then open `http://mercury:8090` from anywhere on the LAN, or use
`PORT=9090 bash run.sh` to choose a different port.

### Runtime configuration

The app derives its repo root from the checkout path by default. Override
host-specific values with environment variables when needed:

- `DGL_ROOT=/path/to/dg-labeller`
- `DGL_ALLOWED_GPUS=6,7`
- `DGL_MODEL_PYTHON=/path/to/sam3_env/bin/python`
- `DGL_SAM3_REPO=/path/to/external/sam3`
- `DGL_SAM3_CHECKPOINT=/path/to/sam3.1_multiplex.pt`

### Using it (UI walkthrough)

1. Click **+ New job** on the home page.
2. Give the run a name and optional description.
3. Paste source paths or use **Browse server folders**. Supports:
   - Local image directories (recursively scanned)
   - Local video files (`.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.webm`)
4. Enter SAM3 ontology rows as `prompt | class_name`.
5. Adjust SAM3 threshold, GPU IDs or GPU count, and frame-prep options.
6. **Output** — leave blank to default to `jobs/<job_id>/`, or paste any
   path you want.
7. Click **Submit labeling run** to queue the job.

You'll be redirected to the job detail page which polls every 5 seconds.

## How it works

### Labeling stages

The labeling job is intentionally narrow:

| Stage | Model | What it does |
|---|---|---|
| `prepare_frames` | — | recursively walks image folders / extracts video frames |
| `scene_detect` (optional) | PySceneDetect ContentDetector | one keyframe per scene |
| `phash_dedup` (optional) | imagehash pHash+dHash | drops near-identical frames before GPU labeling |
| `sam3_labeling` | SAM3.1 worker processes | writes raw per-image annotation JSON and optional masks |

### SAM3 workers

FastAPI does not import SAM3 or hold model state. It launches one worker
process per selected GPU using `DGL_MODEL_PYTHON`. Workers claim ready
frames from `frames.db`, label one image at a time, write raw SAM3 output,
then claim the next frame.

### Concurrency

Up to `MAX_CONCURRENT_JOBS = 3` jobs run in parallel (configurable in
`app/core/config.py`). They share GPU resources via the model registry.

### Job persistence

App job state lives in SQLite at `data/dgl.db`. Each labeling run also has
a job-local `frames.db` tracking frame states: pending/ready/labeling/labeled/failed.

## Folder layout

```
dg-labeller/
├── app/
│   ├── main.py                 entrypoint
│   ├── routes/                 jobs / ui
│   ├── core/                   config, db, frame_queue, sam3_workers, job_runner
│   ├── stages/                 ingest, scene_detect, frame_dedup
│   ├── tools/                  sam3_frame_worker.py
│   ├── recipes/                default.yaml, weapons.yaml
│   └── templates/              labeling, curation, export
├── static/
│   ├── tokens.css              -> ../.degirum-design/tokens.css
│   └── dgl.css                 local styles
├── jobs/                       per-job artifact dirs (created at runtime)
├── data/
│   ├── dgl.db                  SQLite job state
└── .degirum-design/            DeGirum design tokens (installed via skill)
```

## Adding a new recipe

Drop a YAML in `app/recipes/`. Anything in `default.yaml` is overridable.
Key fields:

```yaml
ontology:
  - class_id: 0
    class_name: pedestrian
    prompt: pedestrian
score_threshold: 0.35
save_masks: true
```

The new-run UI currently takes ontology values directly from the form; recipes
remain useful as config examples.

## Known limitations / v2 ideas

- Export tasks are placeholders; YOLO/COCO conversion should consume raw SAM3
  annotations later.
- Curation tasks are placeholders and should consume raw SAM3 annotations.
- No cancellation of running jobs (only kills on server shutdown).
