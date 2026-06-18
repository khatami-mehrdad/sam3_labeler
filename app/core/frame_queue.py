"""Job-local SQLite queue for prepared frames."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    frame_id        TEXT PRIMARY KEY,
    path            TEXT NOT NULL,
    origin          TEXT,
    source_kind     TEXT,
    width           INTEGER,
    height          INTEGER,
    status          TEXT NOT NULL, -- pending|ready|labeling|labeled|failed
    annotation_path TEXT,
    mask_file       TEXT,
    error           TEXT,
    worker_id       TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_frames_status_created
    ON frames(status, created_at);
"""


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def upsert_frame(db_path: Path, row: dict[str, Any], status: str = "ready") -> None:
    ts = now_utc_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO frames
                (frame_id, path, origin, source_kind, width, height, status,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(frame_id) DO UPDATE SET
                path=excluded.path,
                origin=excluded.origin,
                source_kind=excluded.source_kind,
                width=excluded.width,
                height=excluded.height,
                status=CASE
                    WHEN frames.status IN ('labeled', 'labeling') THEN frames.status
                    ELSE excluded.status
                END,
                updated_at=excluded.updated_at
            """,
            (
                str(row["id"]),
                str(row["path"]),
                row.get("origin"),
                row.get("source_kind"),
                int(row.get("width") or 0),
                int(row.get("height") or 0),
                status,
                ts,
                ts,
            ),
        )
        conn.commit()


def upsert_frames(db_path: Path, rows: list[dict[str, Any]], status: str = "ready") -> None:
    for row in rows:
        upsert_frame(db_path, row, status=status)


def claim_ready_frame(db_path: Path, worker_id: str) -> dict[str, Any] | None:
    ts = now_utc_iso()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM frames
            WHERE status = 'ready'
            ORDER BY created_at, frame_id
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            """
            UPDATE frames
            SET status='labeling', worker_id=?, updated_at=?, error=NULL
            WHERE frame_id=?
            """,
            (worker_id, ts, row["frame_id"]),
        )
        conn.commit()
        return dict(row)


def mark_labeled(db_path: Path, frame_id: str, annotation_path: str, mask_file: str | None) -> None:
    ts = now_utc_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE frames
            SET status='labeled', annotation_path=?, mask_file=?, error=NULL, updated_at=?
            WHERE frame_id=?
            """,
            (annotation_path, mask_file, ts, frame_id),
        )
        conn.commit()


def mark_failed(db_path: Path, frame_id: str, error: str) -> None:
    ts = now_utc_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE frames
            SET status='failed', error=?, updated_at=?
            WHERE frame_id=?
            """,
            (error[:4000], ts, frame_id),
        )
        conn.commit()


def counts(db_path: Path) -> dict[str, int]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM frames GROUP BY status"
        ).fetchall()
    return {row["status"]: int(row["n"]) for row in rows}


def has_open_work(db_path: Path) -> bool:
    current = counts(db_path)
    return bool(current.get("pending", 0) or current.get("ready", 0) or current.get("labeling", 0))
