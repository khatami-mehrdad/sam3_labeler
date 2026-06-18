"""SQLite job persistence. Async via aiosqlite."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from app.core.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    config_yaml  TEXT NOT NULL,
    status       TEXT NOT NULL,            -- queued|running|done|failed|cancelled
    current_stage TEXT,
    progress_pct REAL DEFAULT 0,
    eta_seconds  INTEGER,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT,
    output_path  TEXT NOT NULL,
    error        TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status_created
    ON jobs(status, created_at DESC);

CREATE TABLE IF NOT EXISTS export_jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    source_annotations_dir TEXT NOT NULL,
    output_path TEXT NOT NULL,
    train_pct INTEGER NOT NULL,
    val_pct INTEGER NOT NULL,
    filter_small_boxes INTEGER NOT NULL DEFAULT 0,
    small_box_area_factor REAL NOT NULL DEFAULT 10,
    current_stage TEXT,
    progress_pct REAL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_export_jobs_status_created
    ON export_jobs(status, created_at DESC);

CREATE TABLE IF NOT EXISTS ontology_jobs (
    id            TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    image_dir     TEXT NOT NULL,
    classes_json  TEXT NOT NULL,
    provider      TEXT,
    model         TEXT,
    progress_json TEXT,
    results_json  TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    error         TEXT
);

CREATE INDEX IF NOT EXISTS idx_ontology_jobs_status_created
    ON ontology_jobs(status, created_at DESC);
"""


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(SCHEMA)
        await _ensure_column(db, "export_jobs", "filter_small_boxes", "INTEGER NOT NULL DEFAULT 0")
        await _ensure_column(db, "export_jobs", "small_box_area_factor", "REAL NOT NULL DEFAULT 10")
        await db.commit()


async def _ensure_column(db: aiosqlite.Connection, table: str, column: str, definition: str) -> None:
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        rows = await cur.fetchall()
    if column not in {row[1] for row in rows}:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


@asynccontextmanager
async def get_db():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def insert_job(job: dict) -> None:
    fields = ["id", "name", "description", "config_yaml", "status",
              "created_at", "output_path"]
    placeholders = ",".join("?" * len(fields))
    async with get_db() as db:
        await db.execute(
            f"INSERT INTO jobs ({','.join(fields)}) VALUES ({placeholders})",
            tuple(job[f] for f in fields),
        )
        await db.commit()


async def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    set_clause = ",".join(f"{k}=?" for k in fields)
    async with get_db() as db:
        await db.execute(
            f"UPDATE jobs SET {set_clause} WHERE id=?",
            (*fields.values(), job_id),
        )
        await db.commit()


async def get_job(job_id: str) -> dict | None:
    async with get_db() as db:
        async with db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def list_jobs(limit: int = 100) -> list[dict]:
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def list_active_jobs() -> list[dict]:
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM jobs WHERE status IN ('queued','running') "
            "ORDER BY created_at ASC"
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def insert_export_job(job: dict) -> None:
    fields = [
        "id", "name", "status", "source_annotations_dir", "output_path",
        "train_pct", "val_pct", "filter_small_boxes", "small_box_area_factor", "created_at",
    ]
    placeholders = ",".join("?" * len(fields))
    async with get_db() as db:
        await db.execute(
            f"INSERT INTO export_jobs ({','.join(fields)}) VALUES ({placeholders})",
            tuple(job[f] for f in fields),
        )
        await db.commit()


async def update_export_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    set_clause = ",".join(f"{k}=?" for k in fields)
    async with get_db() as db:
        await db.execute(
            f"UPDATE export_jobs SET {set_clause} WHERE id=?",
            (*fields.values(), job_id),
        )
        await db.commit()


async def get_export_job(job_id: str) -> dict | None:
    async with get_db() as db:
        async with db.execute("SELECT * FROM export_jobs WHERE id=?", (job_id,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def list_export_jobs(limit: int = 100) -> list[dict]:
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM export_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def list_active_export_jobs() -> list[dict]:
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM export_jobs WHERE status IN ('queued','running') "
            "ORDER BY created_at ASC"
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


# --- ontology jobs ---

async def insert_ontology_job(job: dict) -> None:
    fields = ["id", "status", "image_dir", "classes_json", "provider",
              "model", "created_at"]
    placeholders = ",".join("?" * len(fields))
    async with get_db() as db:
        await db.execute(
            f"INSERT INTO ontology_jobs ({','.join(fields)}) VALUES ({placeholders})",
            tuple(job[f] for f in fields),
        )
        await db.commit()


async def update_ontology_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    set_clause = ",".join(f"{k}=?" for k in fields)
    async with get_db() as db:
        await db.execute(
            f"UPDATE ontology_jobs SET {set_clause} WHERE id=?",
            (*fields.values(), job_id),
        )
        await db.commit()


async def get_ontology_job(job_id: str) -> dict | None:
    async with get_db() as db:
        async with db.execute("SELECT * FROM ontology_jobs WHERE id=?", (job_id,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def list_ontology_jobs(limit: int = 100) -> list[dict]:
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM ontology_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def list_active_ontology_jobs() -> list[dict]:
    async with get_db() as db:
        async with db.execute(
            "SELECT * FROM ontology_jobs WHERE status IN ('queued','running') "
            "ORDER BY created_at ASC"
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]
