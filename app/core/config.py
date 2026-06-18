"""Global configuration for sam3_labeler."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def _parse_allowed_gpus(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "" or value.strip().lower() == "all":
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


ROOT = Path(os.environ.get("SAM3_LABELER_ROOT", Path(__file__).resolve().parents[2]))
JOBS_DIR = ROOT / "jobs"
EXPORTS_DIR = ROOT / "exports"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "app.db"
OPENIMAGES_CATALOG = DATA_DIR / "openimages_v7_classes.json"
BROWSE_ROOT = Path(os.environ.get("SAM3_LABELER_BROWSE_ROOT", "/data1/ml_data")).expanduser().resolve()

# Server defaults
HOST = "0.0.0.0"
PORT = 8090

# Model registry policy
GPU_HEADROOM_GB = 4.0          # always keep this much free per GPU
MODEL_IDLE_UNLOAD_SEC = 600    # auto-unload models idle longer than this
MAX_CONCURRENT_JOBS = 3

# GPU IDs available to the tool. Set SAM3_LABELER_ALLOWED_GPUS="6,7" to constrain.
# The pool auto-skips GPUs with insufficient free memory.
ALLOWED_GPUS = _parse_allowed_gpus(os.environ.get("SAM3_LABELER_ALLOWED_GPUS"))

# External model-worker runtime. FastAPI launches this Python but does not
# import SAM3 or hold model state itself.
MODEL_PYTHON = os.environ.get("SAM3_LABELER_MODEL_PYTHON")
SAM3_REPO = os.environ.get("SAM3_LABELER_SAM3_REPO")
SAM3_CHECKPOINT = os.environ.get("SAM3_LABELER_CHECKPOINT")

# VLM API keys for ontology generation (optional — can also be set in the UI)
OPENAI_API_KEY = os.environ.get("SAM3_LABELER_OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("SAM3_LABELER_ANTHROPIC_API_KEY", "")
