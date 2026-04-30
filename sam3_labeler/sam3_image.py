from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sam3_labeler.ontology import OntologyItem


@dataclass
class Detection:
    class_id: int
    class_name: str
    prompt: str
    score: float
    box_xyxy: list[float]
    mask: np.ndarray | None = None


class Sam3ImageLabeler:
    """Thin wrapper around SAM 3 image text prompting."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        try:
            from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[reportMissingImports]
            from sam3.model_builder import build_sam3_image_model  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "SAM 3 is not installed. Install facebookresearch/sam3 in this "
                "environment before running labeling."
            ) from exc

        import torch  # type: ignore[reportMissingImports]

        model_kwargs = {}
        if checkpoint_path is not None:
            model_kwargs["checkpoint_path"] = str(checkpoint_path)
            model_kwargs["load_from_HF"] = False
        if device is not None:
            model_kwargs["device"] = device

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = None
        if dtype is not None:
            dtype_map = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
            }
            if dtype not in dtype_map:
                raise ValueError(f"Unsupported dtype: {dtype}")
            if dtype_map[dtype] in {torch.bfloat16, torch.float16}:
                self.autocast_dtype = dtype_map[dtype]

        model = build_sam3_image_model(**model_kwargs)

        self.processor = Sam3Processor(model, device=self.device)
        self._text_cache: dict[str, dict[str, Any]] = {}

    def _inference_context(self) -> Any:
        if self.autocast_dtype is None or self.device == "cpu":
            return nullcontext()

        import torch  # type: ignore[reportMissingImports]

        return torch.autocast(self.device.split(":", maxsplit=1)[0], dtype=self.autocast_dtype)

    def cache_text_prompts(self, ontology: list[OntologyItem]) -> None:
        for item in ontology:
            self._text_outputs_for(item.prompt)

    def label_image(
        self,
        image_path: str | Path,
        ontology: list[OntologyItem],
        score_threshold: float,
        nms_iou_threshold: float | None = None,
    ) -> tuple[tuple[int, int], list[Detection]]:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        with self._inference_context():
            state = self.processor.set_image(image)

        detections: list[Detection] = []
        for item in ontology:
            output = self._set_cached_text_prompt(state=state, prompt=item.prompt)
            detections.extend(
                _parse_output(
                    output=output,
                    item=item,
                    score_threshold=score_threshold,
                )
            )

        if nms_iou_threshold is not None:
            detections = _nms_by_class(detections, nms_iou_threshold)

        return (width, height), detections

    def _set_cached_text_prompt(self, state: dict[str, Any], prompt: str) -> dict[str, Any]:
        text_outputs = self._text_outputs_for(prompt)
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.processor.model._get_dummy_prompt()
        with self._inference_context():
            return self.processor._forward_grounding(state)

    def _text_outputs_for(self, prompt: str) -> dict[str, Any]:
        if prompt not in self._text_cache:
            with self._inference_context():
                self._text_cache[prompt] = self.processor.model.backbone.forward_text(
                    [prompt],
                    device=self.device,
                )
        return self._text_cache[prompt]


def _parse_output(
    output: dict[str, Any],
    item: OntologyItem,
    score_threshold: float,
) -> list[Detection]:
    boxes = _to_numpy(output.get("boxes"))
    scores = _to_numpy(output.get("scores"))
    masks = _to_numpy(output.get("masks"))

    if boxes is None or scores is None:
        return []

    boxes = np.atleast_2d(boxes)
    scores = np.atleast_1d(scores).astype(float)

    parsed: list[Detection] = []
    for index, score in enumerate(scores):
        if score < score_threshold or index >= len(boxes):
            continue

        mask = None
        if masks is not None and index < len(masks):
            mask = np.asarray(masks[index]).squeeze() > 0

        parsed.append(
            Detection(
                class_id=item.class_id,
                class_name=item.class_name,
                prompt=item.prompt,
                score=float(score),
                box_xyxy=[float(value) for value in boxes[index].tolist()],
                mask=mask,
            )
        )

    return parsed


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def _nms_by_class(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    if iou_threshold <= 0:
        return detections

    kept: list[Detection] = []
    for class_id in sorted({det.class_id for det in detections}):
        class_detections = [det for det in detections if det.class_id == class_id]
        class_detections.sort(key=lambda det: det.score, reverse=True)
        while class_detections:
            best = class_detections.pop(0)
            kept.append(best)
            class_detections = [
                det
                for det in class_detections
                if _box_iou(best.box_xyxy, det.box_xyxy) < iou_threshold
            ]

    kept.sort(key=lambda det: (det.class_id, -det.score))
    return kept


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union
