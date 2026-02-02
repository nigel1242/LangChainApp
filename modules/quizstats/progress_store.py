# modules/progress_store.py

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path("quiz_stats.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_stats_db() -> None:
    """
    Ensure the quiz_stats.db file and attempts table exist.
    """
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                subject TEXT,
                topic TEXT,
                question TEXT,
                chosen TEXT,
                correct TEXT,
                is_correct INTEGER,
                source_hint TEXT,
                difficulty TEXT,
                ts_ms INTEGER
            )
    """)
        conn.commit()
    finally:
        conn.close()


def record_attempt(
    *,
    user_id: int,
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
    import time as _time

    if ts_ms is None:
        ts_ms = int(_time.time() * 1000)

    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO attempts (
                user_id, subject, ts_ms, question, chosen, correct,
                is_correct, topic, source_hint, difficulty
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,    # ADDED: mapping to the user_id column
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


def get_attempts_for_subject(subject: str, user_id: int) -> List[Dict[str, Any]]:
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
            WHERE subject = ? AND user_id = ?  -- ADDED user_id filter
            ORDER BY ts_ms ASC, id ASC
            """,
            (subject, user_id), # ADDED user_id to parameters
        ).fetchall()

        return [dict(r) for r in rows]
    finally:
        conn.close()


def clear_subject_stats(subject: str, user_id: int) -> None:
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM attempts WHERE subject = ? AND user_id = ?", (subject, user_id))
        conn.commit()
    finally:
        conn.close()