from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Folder where all subjects live, e.g. subjects/DVISANLY, subjects/MATH, etc.
SUBJECTS_DIR = os.path.join(BASE_DIR, "subjects")
os.makedirs(SUBJECTS_DIR, exist_ok=True)

# TinyDB file to store subject metadata
SUBJECT_DB_PATH = os.path.join(BASE_DIR, "subjects_db.json")


# -----------------------------
# Safe JSON storage (same idea as quiz_store)
# -----------------------------
class SafeJSONStorage(JSONStorage):
    """
    JSONStorage that doesn't crash if file is empty/corrupted.
    """

    def read(self) -> Dict[str, Any] | None:
        try:
            return super().read()
        except Exception:
            return None


_db: TinyDB | None = None


def _get_db() -> TinyDB:
    global _db
    if _db is None:
        _db = TinyDB(SUBJECT_DB_PATH, storage=SafeJSONStorage)
    return _db


# -----------------------------
# Public API
# -----------------------------
def init_subject_db() -> TinyDB:
    """
    Ensure DB exists and return it. Call this once at startup.
    """
    return _get_db()


def _normalize_name(name: str) -> str:
    return (name or "").strip()


def list_subjects() -> List[Dict[str, Any]]:
    """
    Return a list of all subjects, each as {"name": ..., "meta": {...}}.
    """
    db = _get_db()
    table = db.table("subjects")
    rows = table.all()

    # Guarantee "name" key exists
    subjects: List[Dict[str, Any]] = []
    for r in rows:
        nm = r.get("name")
        if not nm:
            continue
        subjects.append({"name": nm, "meta": r.get("meta", {})})

    # Sort alphabetically by name
    subjects.sort(key=lambda x: x["name"].lower())
    return subjects


def ensure_subject_folders(subject_name: str) -> str:
    """
    Ensure folders:
        subjects/<subject>/,
        subjects/<subject>/raw,
        subjects/<subject>/pdf,
        subjects/<subject>/images

    Returns the root path for the subject.
    """
    subject_name = _normalize_name(subject_name)
    if not subject_name:
        raise ValueError("Subject name cannot be empty in ensure_subject_folders().")

    root = os.path.join(SUBJECTS_DIR, subject_name)
    raw = os.path.join(root, "raw")
    pdf = os.path.join(root, "pdf")
    images = os.path.join(root, "images")

    os.makedirs(raw, exist_ok=True)
    os.makedirs(pdf, exist_ok=True)
    os.makedirs(images, exist_ok=True)

    return root


def create_subject_if_missing(name: str) -> Tuple[bool, str]:
    """
    Create a subject record + folders if it doesn't already exist.

    Returns (ok: bool, message: str)
    """
    name = _normalize_name(name)
    if not name:
        return False, "Subject name cannot be empty."

    db = _get_db()
    table = db.table("subjects")
    q = Query()

    existing = table.get(q.name == name)
    if existing:
        # Still ensure folders exist
        ensure_subject_folders(name)
        return False, f"Subject '{name}' already exists."

    # Insert minimal record
    table.insert({"name": name, "meta": {}})
    ensure_subject_folders(name)
    return True, f"Subject '{name}' created."


def get_subject_meta(name: str) -> Dict[str, Any] | None:
    """
    Return metadata dict for a subject, or None if not found.
    """
    name = _normalize_name(name)
    if not name:
        return None

    db = _get_db()
    table = db.table("subjects")
    q = Query()
    row = table.get(q.name == name)
    if not row:
        return None
    return row.get("meta") or {}


def save_subject_meta(name: str, meta: Dict[str, Any]) -> None:
    """
    Upsert metadata for a subject.
    """
    name = _normalize_name(name)
    if not name:
        raise ValueError("Subject name cannot be empty in save_subject_meta().")

    db = _get_db()
    table = db.table("subjects")
    q = Query()
    row = table.get(q.name == name)

    if row is None:
        # Create new subject entry if missing
        table.insert({"name": name, "meta": dict(meta or {})})
    else:
        table.update({"meta": dict(meta or {})}, q.name == name)
