"""Automated VLM→SAM3 prompt refinement loop for ontology generation."""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from app.core import vlm_client
from app.core.prompt_tester import get_prompt_tester

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class PromptScore:
    prompt: str
    avg_score: float
    hit_rate: float
    avg_detections: float
    iteration: int
    overlay_paths: list[str] = field(default_factory=list)


@dataclass
class ClassResult:
    class_name: str
    keywords: str
    best: PromptScore
    kept_prompts: list[PromptScore]
    all_scores: list[PromptScore]
    iterations_used: int
    converged: bool


def _describe_failures(results: list[dict]) -> str:
    no_dets = sum(1 for r in results if r["avg_detections"] == 0)
    low_score = [r for r in results if 0 < r["avg_score"] < 0.4]
    total = len(results)
    parts = []
    if no_dets:
        parts.append(f"{no_dets}/{total} prompts produced zero detections")
    if low_score:
        avg = sum(r["avg_score"] for r in low_score) / len(low_score)
        parts.append(f"{len(low_score)} prompts had low scores (avg {avg:.2f})")
    return "; ".join(parts) or "All prompts produced some detections"


def _collect_overlay_paths(results: list[dict], prompt: str) -> list[str]:
    for r in results:
        if r["prompt"] == prompt:
            return [
                pi["overlay"] for pi in r.get("per_image", [])
                if pi.get("overlay")
            ]
    return []


def _list_images(folder: Path, recursive: bool = True, max_images: int = 10) -> list[Path]:
    if recursive:
        images = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in DEFAULT_EXTENSIONS
        )
    else:
        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in DEFAULT_EXTENSIONS
        )
    return images[:max_images]


async def optimize_class(
    class_name: str,
    keywords: str,
    exemplar_images: list[Path],
    vlm_provider: str,
    vlm_model: str,
    api_key: str,
    score_threshold: float = 0.5,
    max_iterations: int = 3,
    sam3_score_threshold: float = 0.3,
    on_progress: callable | None = None,
) -> ClassResult:
    """Run the automated refinement loop for one object class."""
    tester = get_prompt_tester()
    all_scores: list[PromptScore] = []
    keep_multiple_prompts: set[str] = set()
    prev_results: list[dict] | None = None
    best_overlay_paths: list[str] = []

    for iteration in range(1, max_iterations + 1):
        step = f"[{class_name}] iter {iteration}/{max_iterations}"

        # --- Step 1: VLM generates candidates ---
        if on_progress:
            await on_progress(f"{step}: generating prompts via VLM...")

        if iteration == 1:
            candidates = await vlm_client.generate_initial(
                images=exemplar_images,
                keywords=keywords,
                provider=vlm_provider,
                model=vlm_model,
                api_key=api_key,
            )
        else:
            overlay_paths_for_vlm = [Path(p) for p in best_overlay_paths if Path(p).exists()]
            candidates = await vlm_client.generate_refined(
                exemplar_images=exemplar_images,
                overlay_images=overlay_paths_for_vlm,
                keywords=keywords,
                previous_results=prev_results,
                score_threshold=score_threshold,
                failure_summary=_describe_failures(prev_results),
                provider=vlm_provider,
                model=vlm_model,
                api_key=api_key,
            )

        prompts = [c["prompt"] for c in candidates if c.get("prompt")]
        for c in candidates:
            if c.get("keep_multiple") and c.get("prompt"):
                keep_multiple_prompts.add(c["prompt"])
        if not prompts:
            logger.warning("%s: VLM returned no prompts in iteration %d", class_name, iteration)
            if on_progress:
                await on_progress(f"{step}: VLM returned no prompts, stopping")
            break

        # Log VLM candidates
        if on_progress:
            await on_progress(f"{step}: VLM proposed {len(prompts)} prompts:")
            for i, c in enumerate(candidates):
                if c.get("prompt"):
                    multi = " [keep_multiple]" if c.get("keep_multiple") else ""
                    await on_progress(f"  {i+1}. \"{c['prompt']}\"{multi}")

        # --- Step 2: Test prompts with SAM3 ---
        if on_progress:
            await on_progress(f"{step}: testing on SAM3...")

        overlay_dir = tempfile.mkdtemp(prefix=f"sam3_labeler_onto_{class_name}_iter{iteration}_")
        sam3_results = await tester.test_prompts(
            prompts=prompts,
            images=[str(p) for p in exemplar_images],
            score_threshold=sam3_score_threshold,
            overlay_dir=overlay_dir,
        )

        # Log SAM3 scores
        if on_progress:
            await on_progress(f"{step}: SAM3 scores:")
            sorted_results = sorted(sam3_results, key=lambda r: r["avg_score"], reverse=True)
            for r in sorted_results:
                bar = "█" * int(r["avg_score"] * 20) + "░" * (20 - int(r["avg_score"] * 20))
                await on_progress(
                    f"  {bar} {r['avg_score']:.2f}  "
                    f"hit={r['hit_rate']:.0%}  "
                    f"dets={r['avg_detections']:.1f}  "
                    f"\"{r['prompt']}\""
                )

        for r in sam3_results:
            all_scores.append(PromptScore(
                prompt=r["prompt"],
                avg_score=r["avg_score"],
                hit_rate=r["hit_rate"],
                avg_detections=r["avg_detections"],
                iteration=iteration,
                overlay_paths=_collect_overlay_paths(sam3_results, r["prompt"]),
            ))

        # --- Step 3: Check convergence ---
        round_best = max(sam3_results, key=lambda r: r["avg_score"])
        best_overlay_paths = _collect_overlay_paths(sam3_results, round_best["prompt"])

        if round_best["avg_score"] >= score_threshold:
            if on_progress:
                await on_progress(
                    f'{step}: ✓ converged — "{round_best["prompt"]}" '
                    f'scored {round_best["avg_score"]:.2f} (≥ {score_threshold})'
                )
            break
        else:
            if on_progress:
                await on_progress(
                    f'{step}: best {round_best["avg_score"]:.2f} < {score_threshold} threshold, '
                    f'{"refining..." if iteration < max_iterations else "max iterations reached"}'
                )

        # --- Step 4: Prepare for next iteration ---
        prev_results = [
            {"prompt": r["prompt"], "avg_score": r["avg_score"], "hit_rate": r["hit_rate"]}
            for r in sam3_results
        ]

    # Select prompts to keep: best + any VLM-flagged keep_multiple that scored well
    overall_best = max(all_scores, key=lambda s: s.avg_score) if all_scores else PromptScore(
        prompt=keywords, avg_score=0, hit_rate=0, avg_detections=0, iteration=0,
    )
    kept = [overall_best]
    min_keep_score = sam3_score_threshold
    for s in all_scores:
        if s.prompt == overall_best.prompt:
            continue
        if s.prompt in keep_multiple_prompts and s.avg_score >= min_keep_score:
            kept.append(s)

    if on_progress and len(kept) > 1:
        await on_progress(
            f"[{class_name}] keeping {len(kept)} prompts for better coverage"
        )

    converged = overall_best.avg_score >= score_threshold
    return ClassResult(
        class_name=class_name,
        keywords=keywords,
        best=overall_best,
        kept_prompts=kept,
        all_scores=all_scores,
        iterations_used=min(iteration, max_iterations) if all_scores else 0,
        converged=converged,
    )


async def optimize_all_classes(
    image_dir: Path,
    classes: list[str],
    vlm_provider: str,
    vlm_model: str,
    api_key: str,
    score_threshold: float = 0.5,
    max_iterations: int = 3,
    max_images: int = 10,
    on_progress: callable | None = None,
) -> list[ClassResult]:
    """
    Run optimization for each requested class using images from image_dir.

    Args:
        image_dir: Folder containing reference images (searched recursively).
        classes: List of object names to detect, e.g. ["shipping label", "barcode"].
        max_images: Cap the number of images used for testing (default 10).
    """
    images = _list_images(image_dir, recursive=True, max_images=max_images)
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    if on_progress:
        await on_progress(f"Found {len(images)} images in {image_dir}")

    results: list[ClassResult] = []
    for class_name in classes:
        if on_progress:
            await on_progress(f"Starting class: {class_name} ({len(images)} images)")

        result = await optimize_class(
            class_name=class_name,
            keywords=class_name,
            exemplar_images=images,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            api_key=api_key,
            score_threshold=score_threshold,
            max_iterations=max_iterations,
            on_progress=on_progress,
        )
        results.append(result)

    return results
