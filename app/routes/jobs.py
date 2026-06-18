"""HTTP routes for job CRUD and progress polling."""
import base64
import io
import json
import mimetypes
import uuid
from pathlib import Path

import numpy as np
import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from PIL import Image

from app.core import db, export_runner, job_runner
from app.core.config import BROWSE_ROOT, EXPORTS_DIR, JOBS_DIR

router = APIRouter()
VIDEO_EXTS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
ELFINDER_VOLUME_ID = "l1_"
HIDDEN_NAMES = {".cursor", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
ANNOTATION_EXT = ".json"


def _new_job_id(name: str) -> str:
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    safe = "".join(c if c.isalnum() else "_" for c in (name or "job"))[:30]
    return f"{ts}_{safe}_{short}"


def _browse_path(raw_path: str | None) -> Path:
    root = BROWSE_ROOT.resolve()
    path = root if not raw_path else Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path != root and root not in path.parents:
        raise HTTPException(400, "path is outside the browse root")
    if not path.exists() or not path.is_dir():
        raise HTTPException(404, "browse path not found")
    return path


def _resolve_path_within_roots(raw_path: str, roots: list[Path], *, label: str) -> Path:
    if not raw_path:
        raise HTTPException(400, f"{label} is required")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise HTTPException(400, f"{label} must be an absolute path")
    path = path.resolve()
    resolved_roots = [root.resolve() for root in roots]
    if not any(path == root or root in path.parents for root in resolved_roots):
        raise HTTPException(400, f"{label} is outside allowed roots")
    return path


def _validate_existing_file(raw_path: str, roots: list[Path], *, label: str) -> Path:
    path = _resolve_path_within_roots(raw_path, roots, label=label)
    if not path.exists() or not path.is_file():
        raise HTTPException(404, f"{label} not found")
    return path


def _validate_existing_dir(raw_path: str, roots: list[Path], *, label: str) -> Path:
    path = _resolve_path_within_roots(raw_path, roots, label=label)
    if not path.exists() or not path.is_dir():
        raise HTTPException(404, f"{label} not found")
    return path


def _jobs_path(raw_path: str | None) -> Path:
    root = JOBS_DIR.resolve()
    path = root if not raw_path else Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path != root and root not in path.parents:
        raise HTTPException(400, "path is outside the jobs root")
    if not path.exists() or not path.is_dir():
        raise HTTPException(404, "jobs path not found")
    return path


def _is_hidden_path(path: Path) -> bool:
    return path.name.startswith(".") or path.name in HIDDEN_NAMES


def _is_visible_source(path: Path) -> bool:
    return path.is_dir() or (path.is_file() and path.suffix.lower() in VIDEO_EXTS)


def _is_visible_annotation(path: Path) -> bool:
    return path.is_dir() or (path.is_file() and path.suffix.lower() == ANNOTATION_EXT)


def _hash_path_for_root(root: Path, path: Path) -> str:
    root = root.resolve()
    path = path.resolve()
    rel = "/" if path == root else path.relative_to(root).as_posix()
    encoded = base64.b64encode(rel.encode()).decode()
    encoded = encoded.replace("+", "-").replace("/", "_").replace("=", ".").rstrip(".")
    return ELFINDER_VOLUME_ID + encoded


def _hash_path(path: Path) -> str:
    return _hash_path_for_root(BROWSE_ROOT, path)


def _path_from_hash_for_root(root: Path, target: str | None) -> Path:
    root = root.resolve()
    if not target:
        return root
    if not target.startswith(ELFINDER_VOLUME_ID):
        raise HTTPException(400, "invalid file browser target")
    encoded = target[len(ELFINDER_VOLUME_ID):]
    encoded = encoded.replace("-", "+").replace("_", "/")
    encoded += "=" * (-len(encoded) % 4)
    try:
        rel = base64.b64decode(encoded.encode()).decode()
    except Exception as exc:
        raise HTTPException(400, "invalid file browser target") from exc
    path = root if rel == "/" else (root / rel)
    path = path.resolve()
    if path != root and root not in path.parents:
        raise HTTPException(400, "path is outside the browse root")
    if not path.exists():
        raise HTTPException(404, "file browser path not found")
    return path


def _path_from_hash(target: str | None) -> Path:
    return _path_from_hash_for_root(BROWSE_ROOT, target)


def _has_visible_child_dir(path: Path) -> bool:
    try:
        return any(
            child.is_dir() and not _is_hidden_path(child)
            for child in path.iterdir()
        )
    except OSError:
        return False


def _elfinder_info(path: Path, root: Path | None = None) -> dict:
    root = (root or BROWSE_ROOT).resolve()
    stat = path.stat()
    is_dir = path.is_dir()
    mime = "directory" if is_dir else (mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    info = {
        "name": path.name or str(path),
        "hash": _hash_path_for_root(root, path),
        "mime": mime,
        "ts": int(stat.st_mtime),
        "size": 0 if is_dir else stat.st_size,
        "read": 1,
        "write": 1,
        "locked": 0,
        "path": str(path),
    }
    if path == root:
        info["name"] = root.name or str(root)
        info["volumeid"] = ELFINDER_VOLUME_ID
        info["phash"] = ""
    else:
        info["phash"] = _hash_path_for_root(root, path.parent)
    if is_dir:
        info["dirs"] = 1 if _has_visible_child_dir(path) else 0
    return info


def _elfinder_children(
    path: Path,
    *,
    dirs_only: bool = False,
    root: Path | None = None,
    visible_predicate=_is_visible_source,
) -> list[dict]:
    root = (root or BROWSE_ROOT).resolve()
    children = []
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            if _is_hidden_path(child) or not visible_predicate(child):
                continue
            if dirs_only and not child.is_dir():
                continue
            children.append(_elfinder_info(child, root=root))
        except OSError:
            continue
    return children


def _validate_sources(sources: list[str]) -> None:
    for source in sources:
        source = str(source).strip()
        if not source:
            continue
        if source.lower().startswith(("http://", "https://")):
            raise HTTPException(400, "sources must be image folders or video files, not URLs")
        suffix = Path(source).suffix.lower()
        if suffix and suffix not in VIDEO_EXTS:
            raise HTTPException(400, f"unsupported source file type: {source}")


def _validate_sam3_annotations_dir(path: Path) -> None:
    json_files = sorted(path.glob("*.json"))
    if not json_files:
        raise HTTPException(400, "annotations directory contains no .json files")
    try:
        sample = json.loads(json_files[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "first annotation file is not valid JSON") from exc
    if not isinstance(sample, dict):
        raise HTTPException(400, "annotation JSON has invalid shape")
    missing = [key for key in ("image", "width", "height", "annotations") if key not in sample]
    if missing:
        raise HTTPException(400, f"annotation JSON is missing required keys: {', '.join(missing)}")


@router.get("/browse", response_class=JSONResponse)
async def browse(path: str | None = Query(default=None)):
    current = _browse_path(path)
    root = BROWSE_ROOT.resolve()
    entries = []
    for child in sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            if _is_hidden_path(child) or not _is_visible_source(child):
                continue
        except OSError:
            continue
        entries.append({
            "name": child.name,
            "path": str(child),
            "type": "dir" if child.is_dir() else "video",
            "selectable": True,
        })
    return {
        "root": str(root),
        "path": str(current),
        "parent": str(current.parent) if current != root else None,
        "entries": entries,
    }


@router.get("/browse/annotations", response_class=JSONResponse)
async def browse_annotations(path: str | None = Query(default=None)):
    current = _jobs_path(path)
    root = JOBS_DIR.resolve()
    entries = []
    for child in sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        try:
            if _is_hidden_path(child) or not _is_visible_annotation(child):
                continue
        except OSError:
            continue
        entries.append({
            "name": child.name,
            "path": str(child),
            "type": "dir" if child.is_dir() else "json",
            "selectable": True,
        })
    return {
        "root": str(root),
        "path": str(current),
        "parent": str(current.parent) if current != root else None,
        "entries": entries,
    }


@router.get("/elfinder", response_class=JSONResponse)
async def elfinder_connector(
    cmd: str = Query(default="open"),
    target: str | None = Query(default=None),
    init: str | None = Query(default=None),
    tree: str | None = Query(default=None),
):
    root = BROWSE_ROOT.resolve()
    try:
        if cmd == "open":
            current = _path_from_hash(target)
            if not current.is_dir():
                current = current.parent
            files = [_elfinder_info(current), *_elfinder_children(current)]
            if current != root:
                files.insert(0, _elfinder_info(root))
            payload = {
                "cwd": _elfinder_info(current),
                "files": files,
                "netDrivers": [],
                "options": {
                    "path": str(current),
                    "separator": "/",
                    "disabled": [
                        "archive", "chmod", "copy", "cut", "duplicate", "edit",
                        "extract", "mkdir", "mkfile", "paste", "put", "rename",
                        "resize", "rm", "upload",
                    ],
                },
            }
            if init:
                payload["api"] = "2.1"
            return payload
        if cmd == "tree":
            current = _path_from_hash(target)
            return {"tree": [_elfinder_info(root), *_elfinder_children(current, dirs_only=True)]}
        if cmd == "parents":
            current = _path_from_hash(target)
            tree_entries = [_elfinder_info(root)]
            ancestors = [
                parent for parent in current.parents
                if parent != root and root in parent.parents
            ]
            for parent in reversed(ancestors):
                tree_entries.append(_elfinder_info(parent))
                tree_entries.extend(_elfinder_children(parent, dirs_only=True))
            tree_entries.append(_elfinder_info(current))
            tree_entries.extend(_elfinder_children(current, dirs_only=True))
            return {"tree": tree_entries}
        if cmd == "ls":
            current = _path_from_hash(target)
            return {"list": [child["name"] for child in _elfinder_children(current)]}
        if cmd == "info":
            current = _path_from_hash(target)
            return {"files": [_elfinder_info(current)]}
    except HTTPException as exc:
        return {"error": exc.detail}
    return {"error": "errUnknownCmd"}


@router.get("/elfinder/annotations", response_class=JSONResponse)
async def elfinder_annotations_connector(
    cmd: str = Query(default="open"),
    target: str | None = Query(default=None),
    init: str | None = Query(default=None),
):
    root = BROWSE_ROOT.resolve()
    try:
        if cmd == "open":
            current = _path_from_hash_for_root(root, target)
            if not current.is_dir():
                current = current.parent
            files = [
                _elfinder_info(current, root=root),
                *_elfinder_children(
                    current,
                    root=root,
                    visible_predicate=_is_visible_annotation,
                ),
            ]
            if current != root:
                files.insert(0, _elfinder_info(root, root=root))
            payload = {
                "cwd": _elfinder_info(current, root=root),
                "files": files,
                "netDrivers": [],
                "options": {
                    "path": str(current),
                    "separator": "/",
                    "disabled": [
                        "archive", "chmod", "copy", "cut", "duplicate", "edit",
                        "extract", "mkdir", "mkfile", "paste", "put", "rename",
                        "resize", "rm", "upload",
                    ],
                },
            }
            if init:
                payload["api"] = "2.1"
            return payload
        if cmd == "tree":
            current = _path_from_hash_for_root(root, target)
            return {
                "tree": [
                    _elfinder_info(root, root=root),
                    *_elfinder_children(
                        current,
                        dirs_only=True,
                        root=root,
                        visible_predicate=_is_visible_annotation,
                    ),
                ]
            }
        if cmd == "parents":
            current = _path_from_hash_for_root(root, target)
            tree_entries = [_elfinder_info(root, root=root)]
            ancestors = [
                parent for parent in current.parents
                if parent != root and root in parent.parents
            ]
            for parent in reversed(ancestors):
                tree_entries.append(_elfinder_info(parent, root=root))
                tree_entries.extend(
                    _elfinder_children(
                        parent,
                        dirs_only=True,
                        root=root,
                        visible_predicate=_is_visible_annotation,
                    )
                )
            tree_entries.append(_elfinder_info(current, root=root))
            tree_entries.extend(
                _elfinder_children(
                    current,
                    dirs_only=True,
                    root=root,
                    visible_predicate=_is_visible_annotation,
                )
            )
            return {"tree": tree_entries}
        if cmd == "ls":
            current = _path_from_hash_for_root(root, target)
            return {
                "list": [
                    child["name"] for child in _elfinder_children(
                        current,
                        root=root,
                        visible_predicate=_is_visible_annotation,
                    )
                ]
            }
        if cmd == "info":
            current = _path_from_hash_for_root(root, target)
            return {"files": [_elfinder_info(current, root=root)]}
    except HTTPException as exc:
        return {"error": exc.detail}
    return {"error": "errUnknownCmd"}


@router.get("/sam3/annotation-preview", response_class=JSONResponse)
async def sam3_annotation_preview(path: str = Query(...)):
    annotation_path = _validate_existing_file(path, [BROWSE_ROOT], label="annotation path")
    if annotation_path.suffix.lower() != ANNOTATION_EXT:
        raise HTTPException(400, "annotation path must end with .json")
    try:
        record = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "annotation file is not valid JSON") from exc
    if not isinstance(record, dict):
        raise HTTPException(400, "annotation file has invalid shape")
    return record


@router.get("/sam3/image")
async def sam3_image(path: str = Query(...)):
    image_path = _validate_existing_file(path, [BROWSE_ROOT], label="image path")
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    return Response(content=image_path.read_bytes(), media_type=mime)


@router.get("/sam3/mask-overlay")
async def sam3_mask_overlay(
    annotation_path: str = Query(...),
    annotation_index: int = Query(..., ge=0),
):
    ann_path = _validate_existing_file(annotation_path, [BROWSE_ROOT], label="annotation path")
    try:
        record = json.loads(ann_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "annotation file is not valid JSON") from exc

    annotations = record.get("annotations") or []
    if not isinstance(annotations, list) or annotation_index >= len(annotations):
        raise HTTPException(400, "annotation index is out of range")
    if not annotations[annotation_index].get("has_mask"):
        raise HTTPException(404, "annotation does not have a mask")

    mask_slot = sum(
        1 for ann in annotations[:annotation_index + 1]
        if ann.get("has_mask")
    ) - 1

    raw_mask_file = record.get("mask_file")
    if not raw_mask_file:
        raise HTTPException(404, "mask file not available")
    mask_file = _validate_existing_file(str(raw_mask_file), [BROWSE_ROOT], label="mask file")

    try:
        with np.load(mask_file) as loaded:
            masks = np.asarray(loaded["masks"])
    except Exception as exc:
        raise HTTPException(400, "failed to read mask file") from exc
    if masks.ndim < 3 or mask_slot >= masks.shape[0]:
        raise HTTPException(404, "mask index not found")

    mask = np.asarray(masks[mask_slot]).squeeze()
    if mask.ndim != 2:
        raise HTTPException(400, "mask has invalid shape")
    active = mask > 0
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[..., 1] = 220
    overlay[..., 2] = 255
    overlay[..., 3] = active.astype(np.uint8) * 110

    image = Image.fromarray(overlay, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@router.post("/jobs", response_class=JSONResponse)
async def create_job(payload: dict):
    """Body: {name, description, recipe_overrides, output_dir?}.

    `recipe_overrides` is a dict that overlays the chosen recipe yaml.
    """
    name = (payload.get("name") or "").strip() or "untitled"
    description = (payload.get("description") or "").strip()

    cfg = dict(payload.get("recipe_overrides") or {})
    if "sources" not in cfg or not cfg["sources"]:
        raise HTTPException(400, "sources is required (at least one image folder or video file)")
    _validate_sources(cfg["sources"])
    if not (cfg.get("ontology") or cfg.get("prompts") or cfg.get("detection_prompts")):
        raise HTTPException(400, "at least one SAM3 prompt/ontology item is required")

    job_id = _new_job_id(name)
    out_dir = payload.get("output_dir") or str(JOBS_DIR / job_id)

    await db.insert_job({
        "id":          job_id,
        "name":        name,
        "description": description,
        "config_yaml": yaml.safe_dump(cfg),
        "status":      "queued",
        "created_at":  db.now_utc_iso(),
        "output_path": out_dir,
    })
    await job_runner.enqueue(job_id)
    return {"job_id": job_id, "output_path": out_dir}


@router.post("/export-jobs", response_class=JSONResponse)
async def create_export_job(payload: dict):
    name = (payload.get("name") or "").strip() or "yolo_export"
    annotations_dir = _validate_existing_dir(
        str(payload.get("source_annotations_dir") or ""),
        [JOBS_DIR],
        label="source annotations directory",
    )
    _validate_sam3_annotations_dir(annotations_dir)

    train_pct = int(payload.get("train_pct", 90))
    val_pct = int(payload.get("val_pct", 10))
    filter_small_boxes = bool(payload.get("filter_small_boxes", False))
    small_box_area_factor = float(payload.get("small_box_area_factor", 10))
    if train_pct < 0 or val_pct < 0 or train_pct + val_pct != 100:
        raise HTTPException(400, "train_pct and val_pct must be non-negative and sum to 100")
    if small_box_area_factor <= 0:
        raise HTTPException(400, "small_box_area_factor must be greater than 0")

    job_id = _new_job_id(name)
    out_dir = Path(payload.get("output_dir") or (EXPORTS_DIR / job_id)).expanduser()
    if not out_dir.is_absolute():
        out_dir = EXPORTS_DIR / out_dir
    out_dir = out_dir.resolve()
    export_root = EXPORTS_DIR.resolve()
    if out_dir != export_root and export_root not in out_dir.parents:
        raise HTTPException(400, "output_dir is outside export root")

    await db.insert_export_job({
        "id": job_id,
        "name": name,
        "status": "queued",
        "source_annotations_dir": str(annotations_dir),
        "output_path": str(out_dir),
        "train_pct": train_pct,
        "val_pct": val_pct,
        "filter_small_boxes": int(filter_small_boxes),
        "small_box_area_factor": small_box_area_factor,
        "created_at": db.now_utc_iso(),
    })
    await export_runner.enqueue(job_id)
    return {"export_job_id": job_id, "output_path": str(out_dir)}


@router.get("/export-jobs", response_class=JSONResponse)
async def list_export_jobs():
    return await db.list_export_jobs(limit=200)


@router.get("/export-jobs/{job_id}", response_class=JSONResponse)
async def get_export_job(job_id: str):
    rec = await db.get_export_job(job_id)
    if not rec:
        raise HTTPException(404, "export job not found")
    return rec


@router.get("/jobs", response_class=JSONResponse)
async def list_jobs():
    rows = await db.list_jobs(limit=200)
    return rows


@router.get("/jobs/{job_id}", response_class=JSONResponse)
async def get_job(job_id: str):
    rec = await db.get_job(job_id)
    if not rec:
        raise HTTPException(404, "job not found")
    return rec


@router.get("/jobs/{job_id}/logs/{stage}", response_class=JSONResponse)
async def get_log(job_id: str, stage: str):
    rec = await db.get_job(job_id)
    if not rec:
        raise HTTPException(404, "job not found")
    log = Path(rec["output_path"]) / "logs" / f"{stage}.log"
    if not log.exists():
        return {"text": ""}
    text = log.read_text(errors="replace")[-5000:]
    return {"text": text}
