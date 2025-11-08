# modules/quiz_store.py
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Any, List, Union
from tinydb import TinyDB
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

DEFAULT_DB_FILE = "quiz_results.json"

def backend_name() -> str:
    return "tinydb"

def _normalize_path(path: Union[str, None]) -> str:
    return str(Path(path or DEFAULT_DB_FILE).resolve())

def init_db(path: str | None = None) -> str:
    db_path = _normalize_path(path)
    TinyDB(db_path, storage=CachingMiddleware(JSONStorage)).close()
    return db_path

def record_result(db_path: str, topic: str, difficulty: str, correct: bool, time_taken_ms: int, source_ids: List[str]) -> None:
    db_path = _normalize_path(db_path)
    with TinyDB(db_path, storage=CachingMiddleware(JSONStorage)) as db:
        table = db.table("results")
        table.insert({
            "ts": int(time.time() * 1000),
            "topic": topic,
            "difficulty": difficulty,
            "correct": bool(correct),
            "time_taken_ms": int(time_taken_ms),
            "source_ids": list(source_ids or []),
        })
        db.storage.flush()

def fetch_stats(db_path: str) -> Dict[str, Any]:
    db_path = _normalize_path(db_path)
    with TinyDB(db_path, storage=CachingMiddleware(JSONStorage)) as db:
        rows = db.table("results").all()

    if not rows:
        return {"by_topic": [], "by_diff": [], "overall": (0, 0.0)}

    from collections import defaultdict
    by_topic_map, by_diff_map = defaultdict(list), defaultdict(list)
    total, total_ms = 0, 0
    for r in rows:
        total += 1
        total_ms += int(r.get("time_taken_ms", 0))
        by_topic_map[r.get("topic", "General")].append(1 if r.get("correct") else 0)
        by_diff_map[r.get("difficulty", "easy")].append(1 if r.get("correct") else 0)

    def agg(d):
        out = []
        for k, vals in d.items():
            n = len(vals)
            acc = (sum(vals) / n) * 100.0 if n else 0.0
            out.append((k, acc, n))
        out.sort(key=lambda x: (-x[2], x[0]))
        return out

    return {
        "by_topic": agg(by_topic_map),
        "by_diff": agg(by_diff_map),
        "overall": (total, (total_ms / total) if total else 0.0),
    }
