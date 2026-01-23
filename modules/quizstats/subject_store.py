#subject_store.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple
import shutil
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

def _normalize_name(name: str) -> str:
    return (name or "").strip()


def ensure_subject_folders(subject_name: str) -> str:
    """
    Ensure folders:
        subjects/<subject>/,
        subjects/<subject>/raw,
        subjects/<subject>/images

    Returns the root path for the subject.
    """
    subject_name = _normalize_name(subject_name)
    if not subject_name:
        raise ValueError("Subject name cannot be empty in ensure_subject_folders().")

    root = os.path.join(SUBJECTS_DIR, subject_name)
    raw = os.path.join(root, "raw")
    images = os.path.join(root, "images")

    os.makedirs(raw, exist_ok=True)
    os.makedirs(images, exist_ok=True)

    return root