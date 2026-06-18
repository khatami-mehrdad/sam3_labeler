# sam3_labeler

Tool for turning raw image folders / local video files into
raw SAM3.1 annotation runs with boxes, scores, and optional mask sidecars.

Labeling stops at durable raw SAM3 artifacts. Curation and dataset export
(YOLO, COCO, Label Studio, FiftyOne, etc.) are separate web-app tasks.

## Prerequisites

- Linux host with NVIDIA GPU(s) and a working CUDA driver (`nvidia-smi`)
- Python 3.11+ for the sam3_labeler web app
- A separate Python 3.12+ environment for SAM3 workers (see below)
- ~15 GB free GPU memory per SAM3 worker (or half the GPU's total memory,
  whichever is larger)

## Installation

sam3_labeler uses **two Python environments**:

| Environment | Purpose | Python |
|---|---|---|
| App env | FastAPI server, frame prep, job orchestration | 3.11+ |
| Model env | SAM3 worker subprocesses only | 3.12+ (per upstream SAM3) |

The FastAPI process never imports SAM3. Labeling workers are launched with
`SAM3_LABELER_MODEL_PYTHON` and load a SAM3 checkout (`SAM3_LABELER_SAM3_REPO`) plus a local
SAM3.1 checkpoint (`SAM3_LABELER_CHECKPOINT`).

### 1. Install sam3_labeler (app env)

```bash
cd /path/to/sam3_labeler
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### 2. Install upstream SAM3 (model env)

Follow the [official SAM3 install guide](https://github.com/facebookresearch/sam3#installation)
for PyTorch/CUDA versions. A typical setup:

```bash
# Create the model env (conda or venv — upstream recommends Python 3.12)
conda create -n sam3 python=3.12 -y
conda activate sam3

# Install PyTorch with CUDA (match your driver/CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Clone and install SAM3
git clone https://github.com/facebookresearch/sam3.git /path/to/sam3
cd /path/to/sam3
pip install -e .
pip install huggingface_hub
```

Keep the clone path — sam3_labeler passes it to workers as `SAM3_LABELER_SAM3_REPO`.

Any checkout that provides `sam3.model_builder.build_sam3_image_model` works
(for example a local fork based on `facebookresearch/sam3`).

### 3. Download the SAM3.1 checkpoint from Hugging Face

sam3_labeler targets the **SAM3.1 multiplex** weights (`sam3.1_multiplex.pt`).
This is **not** the default SAM3.0 `sam3.pt` that `build_sam3_image_model()`
auto-downloads when no checkpoint is provided.

In the SAM3 repo, `download_ckpt_from_hf(version="sam3.1")` pulls
`sam3.1_multiplex.pt` from the gated [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
repo. sam3_labeler workers always pass an explicit local checkpoint and set
`load_from_HF=False`, so the file must exist on disk before labeling.

1. Request access to the gated model: [facebook/sam3.1](https://huggingface.co/facebook/sam3.1)
2. Authenticate locally:

```bash
huggingface-cli login
```

3. Download the checkpoint (pick one):

```bash
# Option A — into the sam3_labeler repo
mkdir -p /path/to/sam3_labeler/checkpoints/sam3.1
huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt \
  --local-dir /path/to/sam3_labeler/checkpoints/sam3.1

# Option B — use the Hugging Face cache path directly in .env
# After any successful download, the file is typically at:
# ~/.cache/huggingface/hub/models--facebook--sam3.1/snapshots/<hash>/sam3.1_multiplex.pt
```

The checkpoint is ~3.3 GB.

### 4. Configure environment

Create a `.env` file in the sam3_labeler repo root (loaded automatically by
`app/core/config.py`) or export these variables in your shell:

```bash
# App root (optional — defaults to the checkout path)
SAM3_LABELER_ROOT=/path/to/sam3_labeler

# SAM3 worker runtime (required for labeling)
SAM3_LABELER_MODEL_PYTHON=/path/to/sam3_env/bin/python   # or: conda run -n sam3 which python
SAM3_LABELER_SAM3_REPO=/path/to/sam3                       # SAM3 checkout (see step 2)
SAM3_LABELER_CHECKPOINT=/path/to/sam3_labeler/checkpoints/sam3.1/sam3.1_multiplex.pt

# GPU policy (optional)
SAM3_LABELER_ALLOWED_GPUS=0,1
```

Example host layout:

```bash
SAM3_LABELER_SAM3_REPO=/path/to/sam3
SAM3_LABELER_CHECKPOINT=~/.cache/huggingface/hub/models--facebook--sam3.1/snapshots/<hash>/sam3.1_multiplex.pt
```

Replace `<hash>` with the snapshot directory under the Hugging Face cache
(find it with `find ~/.cache/huggingface -name sam3.1_multiplex.pt`).

### 5. Verify the SAM3 runtime

Quick import check using the model env:

```bash
SAM3_REPO=/path/to/sam3
SAM3_PYTHON=/path/to/sam3_env/bin/python

"$SAM3_PYTHON" -c "
import sys
sys.path.insert(0, '${SAM3_REPO}')
from sam3.model_builder import build_sam3_image_model
print('SAM3 import OK')
"
```

If this fails with `SAM3 is not importable`, check `SAM3_LABELER_MODEL_PYTHON` and
`SAM3_LABELER_SAM3_REPO`. If checkpoint loading fails, confirm Hugging Face access and
that `sam3.1_multiplex.pt` exists on disk.

## Quick start

### Run the server

```bash
cd /path/to/sam3_labeler
source .venv/bin/activate   # app env
bash run.sh
```

Then open `http://$(hostname):8090` from anywhere on the LAN, or use
`PORT=9090 bash run.sh` to choose a different port.

### Runtime configuration reference

| Variable | Required | Description |
|---|---|---|
| `SAM3_LABELER_MODEL_PYTHON` | Yes (for labeling) | Python interpreter for SAM3 worker subprocesses |
| `SAM3_LABELER_SAM3_REPO` | Yes (for labeling) | Path to SAM3 checkout |
| `SAM3_LABELER_CHECKPOINT` | Yes (for SAM3.1) | Local path to `sam3.1_multiplex.pt` |
| `SAM3_LABELER_ROOT` | No | Repo root; defaults to checkout path |
| `SAM3_LABELER_ALLOWED_GPUS` | No | Comma-separated GPU IDs, e.g. `6,7` |
| `SAM3_LABELER_BROWSE_ROOT` | No | Server-side folder browser root (default `/data1/ml_data`) |

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
process per selected GPU using `SAM3_LABELER_MODEL_PYTHON`. Workers claim ready
frames from `frames.db`, label one image at a time, write raw SAM3 output,
then claim the next frame.

### Concurrency

Up to `MAX_CONCURRENT_JOBS = 3` jobs run in parallel (configurable in
`app/core/config.py`). They share GPU resources via the model registry.

### Job persistence

App job state lives in SQLite at `data/app.db`. Each labeling run also has
a job-local `frames.db` tracking frame states: pending/ready/labeling/labeled/failed.

## Folder layout

```
sam3_labeler/
├── app/
│   ├── main.py                 entrypoint
│   ├── routes/                 jobs / ui
│   ├── core/                   config, db, frame_queue, sam3_workers, job_runner
│   ├── stages/                 ingest, scene_detect, frame_dedup
│   ├── tools/                  sam3_frame_worker.py
│   ├── recipes/                default.yaml, weapons.yaml
│   └── templates/              labeling, curation, export
├── static/
│   ├── tokens.css              design token variables
│   └── app.css                 local styles
├── jobs/                       per-job artifact dirs (created at runtime)
├── data/
│   ├── app.db                  SQLite job state
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
