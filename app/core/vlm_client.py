"""VLM client for ontology prompt generation (OpenAI + Anthropic)."""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

INITIAL_SYSTEM_PROMPT = """\
You are an expert at generating text prompts for SAM3 (Segment Anything Model 3), \
an open-vocabulary object segmentation model.

SAM3 detects ALL instances of an object in an image based on a short noun phrase \
(1-4 words). Prompt quality dramatically affects detection accuracy.

Good prompts are:
- Short noun phrases (1-4 words): "brown cardboard box", not "a box sitting on the floor"
- Specific over generic: "stainless steel mixing bowl" >> "bowl"
- Visual-feature focused: mention color, material, shape, texture when distinctive
- Contrastive: different enough from other objects in the scene to avoid false matches

Bad prompts:
- Full sentences or descriptions
- Abstract concepts ("mess", "clutter")
- Overly generic ("object", "thing", "item")
"""

_TIMEOUT = 90


def _encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file."""
    data = base64.b64encode(path.read_bytes()).decode()
    ext = path.suffix.lower().lstrip(".")
    media_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}
    media_type = f"image/{media_map.get(ext, 'jpeg')}"
    return data, media_type


def _build_initial_user_content_text(keywords: str, n_images: int) -> str:
    return (
        f"I want to detect these objects: {keywords}\n\n"
        f"I've attached {n_images} reference images showing the target objects.\n\n"
        "Generate exactly 10 candidate SAM3 text prompts for this object class, "
        "ranked from most likely to work to least likely.\n\n"
        "IMPORTANT: The target object may appear with visual variations across images "
        "(different sizes, angles, colors, materials). If you see variations that a "
        "single prompt can't cover, suggest multiple complementary prompts and mark "
        'them with "keep_multiple": true. These will ALL be kept in the final ontology '
        "mapped to the same class, giving better coverage.\n\n"
        'Return ONLY a JSON array:\n'
        '[{"prompt": "...", "confidence": 5, "keep_multiple": false, "reasoning": "..."}, ...]'
    )


def _build_refine_user_content_text(
    keywords: str,
    previous_results: list[dict],
    score_threshold: float,
    failure_summary: str,
) -> str:
    results_text = "\n".join(
        f'  - "{r["prompt"]}": avg_score={r["avg_score"]:.2f}, hit_rate={r["hit_rate"]:.0%}'
        for r in previous_results
    )
    best = max(previous_results, key=lambda r: r["avg_score"])
    return (
        f"I want to detect these objects: {keywords}\n\n"
        "The reference images are attached (originals first, then SAM3 mask overlays).\n\n"
        f"In the previous round, I tested these SAM3 prompts:\n{results_text}\n\n"
        f'Best prompt: "{best["prompt"]}" scored {best["avg_score"]:.2f} '
        f"(target: >= {score_threshold})\n"
        f"Failure analysis: {failure_summary}\n\n"
        "Based on what SAM3 actually detected (shown in the overlay images), "
        "generate 10 NEW candidate prompts. Focus on fixing what went wrong:\n"
        "- If SAM3 missed the target entirely: try more generic or alternative phrasings\n"
        "- If SAM3 found the wrong object: try more specific or distinctive phrasings\n"
        "- If scores are low but masks look partially right: try slight variations\n\n"
        "Do NOT repeat prompts that already scored below 0.3.\n\n"
        "If different prompts each catch a DIFFERENT subset of the target objects, "
        'mark them with "keep_multiple": true — they will all be kept in the final '
        "ontology mapped to the same class for better coverage.\n\n"
        'Return ONLY a JSON array:\n'
        '[{"prompt": "...", "confidence": 5, "keep_multiple": false, "reasoning": "..."}, ...]'
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_initial(
    images: list[Path],
    keywords: str,
    provider: str,
    model: str,
    api_key: str,
) -> list[dict]:
    """First-round prompt generation from exemplar images + keywords."""
    user_text = _build_initial_user_content_text(keywords, len(images))
    return await _call_vlm(
        images=images,
        user_text=user_text,
        provider=provider,
        model=model,
        api_key=api_key,
    )


async def generate_refined(
    exemplar_images: list[Path],
    overlay_images: list[Path],
    keywords: str,
    previous_results: list[dict],
    score_threshold: float,
    failure_summary: str,
    provider: str,
    model: str,
    api_key: str,
) -> list[dict]:
    """Refinement round: VLM sees originals + SAM3 mask overlays + scores."""
    user_text = _build_refine_user_content_text(
        keywords, previous_results, score_threshold, failure_summary,
    )
    all_images = list(exemplar_images) + list(overlay_images)
    return await _call_vlm(
        images=all_images,
        user_text=user_text,
        provider=provider,
        model=model,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

async def _call_vlm(
    images: list[Path],
    user_text: str,
    provider: str,
    model: str,
    api_key: str,
) -> list[dict]:
    provider = provider.lower()
    if provider in ("openai", "gpt"):
        raw = await _call_openai(images, user_text, model, api_key)
    elif provider in ("anthropic", "claude"):
        raw = await _call_anthropic(images, user_text, model, api_key)
    else:
        raise ValueError(f"Unknown VLM provider: {provider}")
    return _parse_json_response(raw)


_OPENAI_REASONING_PREFIXES = ("gpt-5.5", "gpt-5.2", "o1", "o3", "o4")


def _openai_body(model: str, content: list[dict]) -> dict:
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": INITIAL_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    }
    is_reasoning = any(model.startswith(p) for p in _OPENAI_REASONING_PREFIXES)
    if is_reasoning:
        body["max_completion_tokens"] = 2048
        body["reasoning_effort"] = "low"
    else:
        body["max_completion_tokens"] = 2048
        body["temperature"] = 0.3
    return body


async def _call_openai(
    images: list[Path], user_text: str, model: str, api_key: str,
) -> str:
    content: list[dict] = []
    for img_path in images:
        b64, media = _encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media};base64,{b64}"},
        })
    content.append({"type": "text", "text": user_text})

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=_openai_body(model, content),
        )
        if resp.status_code != 200:
            body = resp.text
            logger.error("OpenAI API error %d: %s", resp.status_code, body)
            resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _call_anthropic(
    images: list[Path], user_text: str, model: str, api_key: str,
) -> str:
    content: list[dict] = []
    for img_path in images:
        b64, media = _encode_image(img_path)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media, "data": b64},
        })
    content.append({"type": "text", "text": user_text})

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "system": INITIAL_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 2048,
                "temperature": 0.3,
            },
        )
        if resp.status_code != 200:
            body = resp.text
            logger.error("Anthropic API error %d: %s", resp.status_code, body)
            resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _parse_json_response(raw: str) -> list[dict]:
    """Extract JSON array from VLM response (may contain markdown fences)."""
    text = raw.strip()
    if "```" in text:
        # Strip markdown code fences
        lines = text.split("\n")
        inside = False
        cleaned: list[str] = []
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside:
                cleaned.append(line)
        text = "\n".join(cleaned).strip()

    # Find JSON array in the text
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        logger.warning("Could not find JSON array in VLM response: %s", text[:200])
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from VLM response: %s", text[:200])
        return []
