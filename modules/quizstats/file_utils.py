import os
import json
import streamlit as st
import fitz  # PyMuPDF
import platform
from typing import List, Dict, Any

# ------------------ Platform check ------------------

IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    import win32com.client
    import pythoncom

# ------------------ Paths ------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SUBJECTS_BASE = os.path.join(BASE_DIR, "modules", "subjects")

# ------------ subject folder helpers ------------

def get_user_root(user_id: int) -> str:
    """Helper to get the root directory for a specific user."""
    path = os.path.join(SUBJECTS_BASE, f"user_{user_id}")
    os.makedirs(path, exist_ok=True)
    return path

def ensure_subject_folders(subject_name: str, user_id: int) -> str:
    """
    Ensure modules/subjects/user_{id}/<subject_name>/raw exists.
    """
    user_root = get_user_root(user_id)
    subject_root = os.path.join(user_root, subject_name)
    raw_dir = os.path.join(subject_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    return subject_root

def save_uploaded_files(subject_name: str, files, user_id: int) -> List[str]:
    """
    Saves files into the user-isolated subject folder.
    """
    subject_root = ensure_subject_folders(subject_name, user_id)
    raw_dir = os.path.join(subject_root, "raw")

    saved_paths: List[str] = []

    for f in files:
        dest = os.path.join(raw_dir, f.name)
        try:
            f.seek(0)
        except Exception:
            pass

        with open(dest, "wb") as out:
            out.write(f.read())
        saved_paths.append(dest)

    return saved_paths

# ------------ extraction helpers ------------

def convert_to_pdf(input_path: str) -> str | None:
    """
    Converts DOCX / PPTX to PDF on Windows.
    On non-Windows systems, returns None with a user-friendly message.
    """
    abs_input = os.path.abspath(input_path)
    pdf_output = os.path.splitext(abs_input)[0] + ".pdf"
    ext = os.path.splitext(abs_input)[1].lower()

    if os.path.exists(pdf_output):
        return pdf_output

    if not IS_WINDOWS:
        st.warning(
            "DOCX/PPTX conversion is not supported on cloud deployments. "
            "Please upload PDFs instead."
        )
        return None

    pythoncom.CoInitialize()
    try:
        if ext == ".pptx":
            app = win32com.client.DispatchEx("PowerPoint.Application")
            obj = app.Presentations.Open(abs_input, True, False, False)
            obj.SaveAs(pdf_output, 32)

        elif ext == ".docx":
            app = win32com.client.DispatchEx("Word.Application")
            obj = app.Documents.Open(abs_input, ReadOnly=True, Visible=False)
            obj.SaveAs(pdf_output, 17)

        else:
            return None

        obj.Close()
        app.Quit()
        return pdf_output

    except Exception as e:
        st.error(f"Conversion failed: {e}")
        return None

    finally:
        pythoncom.CoUninitialize()

def extract_text_from_path(file_path: str) -> str:
    """Extracts text from a specific PDF path."""
    text = ""

    if not os.path.exists(file_path):
        return ""

    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc, start=1):
            txt = page.get_text("text")
            if txt:
                text += f"--- Page {i} ---\n{txt}\n"
        doc.close()
    except Exception as e:
        print(f"Error extracting text: {e}")

    return text

# ------------ context search helpers ------------

def get_subject_index_entries(subject_name: str, user_id: int) -> List[Dict[str, Any]]:
    """Return index entries for a specific user's subject."""
    subject_root = ensure_subject_folders(subject_name, user_id)
    idx_path = os.path.join(subject_root, "page_index.json")

    if not os.path.isfile(idx_path):
        return []

    try:
        with open(idx_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data.get("entries", [])
    except Exception:
        return []

# ------------ rendering helpers ------------

def render_pdf_page_image(pdf_path: str, page_num: int):
    """Renders a PDF page to image bytes (path provided must be user-isolated)."""
    if not os.path.exists(pdf_path):
        return None

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        doc.close()
        return img_data
    except Exception as e:
        print(f"Rendering error: {e}")
        return None
