#!/usr/bin/env bash
# Launch dg-labeller server. Defaults to port 8090 on all interfaces so the
# whole team can reach it at http://$(hostname):8090
#
# Override: PORT=9090 bash run.sh
# SAM3 runtime: set DGL_MODEL_PYTHON, DGL_SAM3_REPO, and DGL_SAM3_CHECKPOINT here.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8090}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"   # MUST be 1 — model registry is in-process
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" && -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
fi
PYTHON="${PYTHON:-python3.11}"

echo "== dg-labeller =="
echo "  http://$(hostname):${PORT}"
echo "  python: ${PYTHON}"
echo

exec "$PYTHON" -m uvicorn app.main:app \
    --host "$HOST" --port "$PORT" --workers "$WORKERS"
