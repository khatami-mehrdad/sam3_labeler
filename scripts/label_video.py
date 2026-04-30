from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sam3_labeler.ontology import load_ontology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label and track video concepts with SAM 3 text prompts."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="MP4/video file or directory of consecutive JPEG frames.",
    )
    parser.add_argument("--output", required=True, help="Directory for SAM 3 video outputs.")
    parser.add_argument("--ontology", required=True, help="YAML mapping of prompt: class_name.")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam3.1/sam3.1_multiplex.pt",
        help="Path to local SAM 3.1 multiplex checkpoint.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=None,
        help="Optional GPU IDs to pass to build_sam3_video_predictor.",
    )
    parser.add_argument(
        "--no-propagate",
        action="store_true",
        help="Only label the prompt frame; do not propagate through the video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sam3.model_builder import build_sam3_predictor  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "SAM 3 is not installed. Install facebookresearch/sam3 in this "
            "environment before running video labeling."
        ) from exc

    ontology = load_ontology(args.ontology)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"SAM 3.1 checkpoint not found: {checkpoint_path}")

    if args.gpus is None:
        video_predictor = build_sam3_predictor(
            version="sam3.1",
            checkpoint_path=str(checkpoint_path),
            use_fa3=False,
            use_rope_real=False,
        )
    else:
        video_predictor = build_sam3_predictor(
            version="sam3.1",
            checkpoint_path=str(checkpoint_path),
            gpus_to_use=args.gpus,
            use_fa3=False,
            use_rope_real=False,
        )

    prompt_summaries = []
    for item in ontology:
        prompt_dir = output_dir / f"{item.class_id:03d}_{item.class_name}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        start_response = video_predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": args.video,
            }
        )
        session_id = start_response["session_id"]
        _write_json(prompt_dir / "start_session.json", start_response)

        add_response = video_predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": args.frame_index,
                "text": item.prompt,
            }
        )
        _save_torch(prompt_dir / "prompt_frame_outputs.pt", add_response)
        _write_json(prompt_dir / "prompt_frame_summary.json", _summarize_response(add_response))

        if not args.no_propagate:
            outputs_per_frame = {}
            for response in video_predictor.handle_stream_request(
                request={
                    "type": "propagate_in_video",
                    "session_id": session_id,
                }
            ):
                outputs_per_frame[response["frame_index"]] = response["outputs"]
            _save_torch(prompt_dir / "tracked_outputs_per_frame.pt", outputs_per_frame)
            tracked_frame_count = len(outputs_per_frame)
        else:
            tracked_frame_count = 0

        prompt_summaries.append(
            {
                "prompt": item.prompt,
                "class_name": item.class_name,
                "class_id": item.class_id,
                "session_id": session_id,
                "prompt_frame": args.frame_index,
                "tracked_frame_count": tracked_frame_count,
                "output_dir": str(prompt_dir),
            }
        )

    _write_json(output_dir / "summary.json", prompt_summaries)
    print(f"Wrote video labeling outputs to {output_dir}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _save_torch(path: Path, payload: Any) -> None:
    import torch  # type: ignore[reportMissingImports]

    torch.save(payload, path)


def _summarize_response(response: dict[str, Any]) -> dict[str, Any]:
    summary = {key: value for key, value in response.items() if key != "outputs"}
    outputs = response.get("outputs")
    if isinstance(outputs, dict):
        summary["output_keys"] = sorted(str(key) for key in outputs.keys())
    else:
        summary["output_type"] = type(outputs).__name__
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return repr(value)


if __name__ == "__main__":
    main()
