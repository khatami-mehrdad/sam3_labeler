"""Manage a SAM3 prompt-tester subprocess for ontology generation."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from app.core.config import MODEL_PYTHON, ROOT, SAM3_CHECKPOINT, SAM3_REPO
from app.core.sam3_workers import selected_gpus

logger = logging.getLogger(__name__)


class PromptTesterProcess:
    """Async wrapper around the sam3_prompt_tester subprocess."""

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def _ensure_running(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process

        gpus = selected_gpus(count=1)
        gpu = gpus[0] if gpus else 0
        python = MODEL_PYTHON or sys.executable
        script = ROOT / "app" / "tools" / "sam3_prompt_tester.py"

        cmd = [python, str(script), "--device", "cuda", "--dtype", "bf16"]
        if SAM3_REPO:
            cmd.extend(["--sam3-repo", str(SAM3_REPO)])
        if SAM3_CHECKPOINT:
            cmd.extend(["--checkpoint", str(SAM3_CHECKPOINT)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

        logger.info("Starting SAM3 prompt tester on GPU %d", gpu)
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Wait for "ready" signal
        ready_line = await asyncio.wait_for(self._process.stdout.readline(), timeout=120)
        ready = json.loads(ready_line.decode())
        if ready.get("status") != "ready":
            raise RuntimeError(f"Prompt tester failed to start: {ready}")
        logger.info("SAM3 prompt tester ready")
        return self._process

    async def test_prompts(
        self,
        prompts: list[str],
        images: list[str | Path],
        score_threshold: float = 0.3,
        overlay_dir: str | Path | None = None,
    ) -> list[dict]:
        """Send prompts + images to tester, return results."""
        async with self._lock:
            proc = await self._ensure_running()

            if overlay_dir is None:
                overlay_dir = tempfile.mkdtemp(prefix="dgl_overlays_")

            request = {
                "prompts": prompts,
                "images": [str(p) for p in images],
                "score_threshold": score_threshold,
                "overlay_dir": str(overlay_dir),
            }

            proc.stdin.write((json.dumps(request) + "\n").encode())
            await proc.stdin.drain()

            response_line = await asyncio.wait_for(proc.stdout.readline(), timeout=300)
            response = json.loads(response_line.decode())

            if "error" in response:
                raise RuntimeError(f"Prompt tester error: {response['error']}")

            return response["results"]

    async def shutdown(self) -> None:
        if self._process and self._process.returncode is None:
            try:
                self._process.stdin.write(b'{"command": "quit"}\n')
                await self._process.stdin.drain()
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except Exception:
                self._process.kill()
            self._process = None


# Singleton
_instance: PromptTesterProcess | None = None


def get_prompt_tester() -> PromptTesterProcess:
    global _instance
    if _instance is None:
        _instance = PromptTesterProcess()
    return _instance
