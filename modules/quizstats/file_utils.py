import os
import json
import streamlit as st
import fitz  # PyMuPDF
from typing import List, Dict, Any

# BASE_DIR remains the same, but SUBJECTS_DIR is now just a base path
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

import subprocess
import os
import platform

def convert_to_pdf(input_path):
    """
    Cross-platform conversion of DOCX/PPTX to PDF.
    Uses win32com on Windows (if available) or LibreOffice on Linux.
    """
    if not os.path.exists(input_path):
        return None

    output_dir = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    pdf_path = os.path.join(output_dir, f"{name}.pdf")

    # --- Windows Logic (Fallback) ---
    if platform.system() == "Windows":
        try:
            import win32com.client
            # Your existing win32 logic here...
            return pdf_path
        except ImportError:
            # If win32 isn't there, try to use LibreOffice if installed on Windows
            pass

    # --- Linux/Cloud Logic (The "Free" Deployment way) ---
    try:
        # 'libreoffice' is the command name on Linux
        # On some systems/Mac it might be 'soffice'
        command = [
            'libreoffice', '--headless', 
            '--convert-to', 'pdf', 
            '--outdir', output_dir, 
            input_path
        ]
        
        subprocess.run(command, check=True, capture_output=True)
        return pdf_path
    except Exception as e:
        print(f"LibreOffice conversion error: {e}")
        return None

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