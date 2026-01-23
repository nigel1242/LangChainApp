# modules/progress_store.py

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path("quiz_stats.db")


def _get_conn() -> sqlite3.Connection:
    """
    Open a SQLite connection to the stats DB.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_stats_db() -> None:
    """
    Ensure the quiz_stats.db file and attempts table exist.
    """
    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                ts_ms INTEGER NOT NULL,
                question TEXT,
                chosen TEXT,
                correct TEXT,
                is_correct INTEGER NOT NULL,
                topic TEXT,
                source_hint TEXT,
                difficulty TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def record_attempt(
    *,
    subject: str,
    question: str,
    chosen: str,
    correct: str,
    is_correct: int,
    topic: str = "",
    source_hint: str = "",
    difficulty: str = "",
    ts_ms: int | None = None,
) -> None:
    """
    Insert one quiz attempt row into the DB.

    is_correct: 1 for correct, 0 for wrong.
    ts_ms: Unix timestamp in milliseconds. If None, we fill it with current time.
    """
    import time as _time

    if ts_ms is None:
        ts_ms = int(_time.time() * 1000)

    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO attempts (
                subject, ts_ms, question, chosen, correct,
                is_correct, topic, source_hint, difficulty
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject,
                int(ts_ms),
                question,
                chosen,
                correct,
                int(is_correct),
                topic or "",
                source_hint or "",
                difficulty or "",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_attempts_for_subject(subject: str) -> List[Dict[str, Any]]:
    """
    Return all attempts for a given subject as a list of dicts.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(
            """
            SELECT
                subject,
                ts_ms,
                question,
                chosen,
                correct,
                is_correct,
                topic,
                source_hint,
                difficulty
            FROM attempts
            WHERE subject = ?
            ORDER BY ts_ms ASC, id ASC
            """,
            (subject,),
        ).fetchall()

        return [dict(r) for r in rows]
    finally:
        conn.close()


def clear_subject_stats(subject: str) -> None:
    """
    Delete ALL attempts for a particular subject.
    """
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM attempts WHERE subject = ?", (subject,))
        conn.commit()
    finally:
        conn.close()
