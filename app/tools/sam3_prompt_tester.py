#!/usr/bin/env python3
"""Test SAM3 text prompts on images and generate mask overlays.

Runs as a subprocess — FastAPI never imports SAM3 directly.
Communicates via JSON over stdin/stdout: reads one JSON object per line,
writes one JSON response per line.

Protocol:
  Request:  {"prompts": ["brown box", "cardboard box"],
             "images": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
             "score_threshold": 0.3,
             "overlay_dir": "/tmp/overlays"}
  Response: {"results": [
               {"prompt": "brown box", "avg_score": 0.55, "hit_rate": 0.8,
                "avg_detections": 1.6, "per_image": [
                  {"image": "...", "n_detections": 2, "best_score": 0.62,
                   "overlay": "/tmp/overlays/brown_box_img1.jpg"},
                  ...
                ]},
               ...
             ]}

Send {"command": "quit"} to shut down.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", file=sys.stderr, flush=True)


@dataclass
class Detection:
    score: float
    box_xyxy: list[float]
    mask: np.ndarray | None = None


class Sam3Tester:
    """Lightweight SAM3 wrapper for prompt testing."""

    def __init__(self, sam3_repo: str | None, checkpoint: str | None, device: str, dtype: str):
        if sam3_repo:
            sys.path.insert(0, str(Path(sam3_repo).resolve()))

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        import torch

        model_kwargs: dict[str, Any] = {}
        if checkpoint:
            model_kwargs["checkpoint_path"] = str(checkpoint)
            model_kwargs["load_from_HF"] = False
        model_kwargs["device"] = device

        self.device = device
        self.autocast_dtype = None
        dtype_map = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype)
        if torch_dtype and torch_dtype in {torch.bfloat16, torch.float16}:
            self.autocast_dtype = torch_dtype

        model = build_sam3_image_model(**model_kwargs)
        self.processor = Sam3Processor(model, device=self.device)
        self._text_cache: dict[str, dict[str, Any]] = {}

    def _ctx(self) -> Any:
        if self.autocast_dtype is None or self.device == "cpu":
            return nullcontext()
        import torch
        return torch.autocast(self.device.split(":")[0], dtype=self.autocast_dtype)

    def test_prompt(self, prompt: str, image_path: Path, score_threshold: float) -> list[Detection]:
        image = Image.open(image_path).convert("RGB")
        with self._ctx():
            state = self.processor.set_image(image)
            text_outputs = self._text_for(prompt)
            state["backbone_out"].update(text_outputs)
            if "geometric_prompt" not in state:
                state["geometric_prompt"] = self.processor.model._get_dummy_prompt()
            output = self.processor._forward_grounding(state)

        boxes = _to_numpy(output.get("boxes"))
        scores = _to_numpy(output.get("scores"))
        masks = _to_numpy(output.get("masks"))
        if boxes is None or scores is None:
            return []

        boxes = np.atleast_2d(boxes)
        scores = np.atleast_1d(scores).astype(float)
        detections = []
        for i, score in enumerate(scores):
            if score < score_threshold or i >= len(boxes):
                continue
            mask = None
            if masks is not None and i < len(masks):
                mask = np.asarray(masks[i]).squeeze() > 0
            detections.append(Detection(
                score=float(score),
                box_xyxy=[float(v) for v in boxes[i].tolist()],
                mask=mask,
            ))
        return detections

    def _text_for(self, prompt: str) -> dict[str, Any]:
        if prompt not in self._text_cache:
            with self._ctx():
                self._text_cache[prompt] = self.processor.model.backbone.forward_text(
                    [prompt], device=self.device,
                )
        return self._text_cache[prompt]


def render_overlay(image_path: Path, detections: list[Detection], prompt: str) -> Image.Image:
    """Draw SAM3 detections as semi-transparent masks + boxes on the original image."""
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    for det in detections:
        if det.mask is not None:
            mask_rgba = np.zeros((*det.mask.shape, 4), dtype=np.uint8)
            mask_rgba[det.mask] = [0, 200, 0, 90]
            mask_img = Image.fromarray(mask_rgba, "RGBA")
            overlay = Image.alpha_composite(overlay, mask_img)

    draw = ImageDraw.Draw(overlay)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box_xyxy]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 200), width=2)
        label = f"{det.score:.2f}"
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill=(255, 255, 255, 220))

    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result


def _safe_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)[:60]


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def handle_request(tester: Sam3Tester, req: dict) -> dict:
    prompts = req["prompts"]
    images = [Path(p) for p in req["images"]]
    threshold = req.get("score_threshold", 0.3)
    overlay_dir = Path(req["overlay_dir"]) if req.get("overlay_dir") else None

    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for prompt in prompts:
        per_image = []
        total_score = 0.0
        total_dets = 0
        hits = 0

        for img_path in images:
            dets = tester.test_prompt(prompt, img_path, threshold)
            best_score = max((d.score for d in dets), default=0.0)
            overlay_path = None
            if overlay_dir:
                overlay_img = render_overlay(img_path, dets, prompt)
                fname = f"{_safe_filename(prompt)}_{img_path.stem}.jpg"
                overlay_path = str(overlay_dir / fname)
                overlay_img.save(overlay_path, quality=85)

            per_image.append({
                "image": str(img_path),
                "n_detections": len(dets),
                "best_score": best_score,
                "overlay": overlay_path,
            })
            if dets:
                hits += 1
                total_dets += len(dets)
                total_score += sum(d.score for d in dets)

        n = len(images)
        results.append({
            "prompt": prompt,
            "avg_score": total_score / max(total_dets, 1),
            "hit_rate": hits / n if n else 0,
            "avg_detections": total_dets / n if n else 0,
            "per_image": per_image,
        })

    return {"results": results}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3-repo")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    log("loading SAM3 model...")
    tester = Sam3Tester(
        sam3_repo=args.sam3_repo,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    log("ready — waiting for requests on stdin")

    # Signal readiness
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({"error": "invalid JSON"}) + "\n")
            sys.stdout.flush()
            continue

        if req.get("command") == "quit":
            log("quit received")
            break

        try:
            resp = handle_request(tester, req)
        except Exception as exc:
            log(f"error: {exc}")
            resp = {"error": str(exc)}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

    log("shutdown")


if __name__ == "__main__":
    main()
