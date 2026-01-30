# modules/quizstats/file_utils.py

import os
import json
import streamlit as st
import fitz  # PyMuPDF
import win32com.client
import pythoncom

from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
SUBJECTS_DIR = os.path.join(BASE_DIR, "subjects")
os.makedirs(SUBJECTS_DIR, exist_ok=True)

# ------------ extension sets ------------

TEXT_FILE_EXTS = {".txt", ".md"}
DOCX_EXTS = {".docx"}
PPTX_EXTS = {".pptx"}
CSV_EXTS = {".csv"}
PDF_EXTS = {".pdf", ".PDF"}


# ------------ subject folder helpers ------------

def ensure_subject_folders(subject_name: str, user_id: int = None):
    """
    Ensure subjects/user_X/<subject_name>/raw exists.
    All uploaded files (PDF, PPTX, DOCX, etc.) will be stored in this single folder.
    Returns the root path for that subject.
    """
    if user_id is None:
        # Try to get from session state
        user_id = st.session_state.get("user_id")
    
    if user_id is not None:
        # User-specific path
        root = os.path.join(SUBJECTS_DIR, f"user_{user_id}", subject_name)
    else:
        # Fallback to old path (for backward compatibility)
        root = os.path.join(SUBJECTS_DIR, subject_name)
    
    raw = os.path.join(root, "raw")
    os.makedirs(raw, exist_ok=True)
    return root


def save_uploaded_files(subject_name: str, files, user_id: int = None) -> List[str]:
    """
    Saves uploaded files into subjects/user_X/<name>/raw (user-specific folder).
    Returns list of saved paths.
    """
    if user_id is None:
        # Try to get from session state
        user_id = st.session_state.get("user_id")
    
    root = ensure_subject_folders(subject_name, user_id)
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    saved: List[str] = []

    for f in files:
        name = f.name
        dest = os.path.join(raw_dir, name)

        # Reset file pointer (Streamlit sometimes reuses objects)
        try:
            f.seek(0)
        except Exception:
            pass

        with open(dest, "wb") as out:
            out.write(f.read())
        saved.append(dest)

    return saved

def convert_to_pdf(input_path: str) -> str:
    abs_input = os.path.abspath(input_path)
    pdf_output = os.path.splitext(abs_input)[0] + ".pdf"
    ext = os.path.splitext(abs_input)[1].lower()
    
    if os.path.exists(pdf_output): return pdf_output

    pythoncom.CoInitialize() 
    try:
        if ext == ".pptx":
            app = win32com.client.DispatchEx("PowerPoint.Application")
            obj = app.Presentations.Open(abs_input, True, False, False)
            obj.SaveAs(pdf_output, 32) # 32 = ppFixedFormatTypePDF
        elif ext == ".docx":
            app = win32com.client.DispatchEx("Word.Application")
            obj = app.Documents.Open(abs_input, ReadOnly=True, Visible=False)
            obj.SaveAs(pdf_output, 17) # 17 = wdExportFormatPDF
        
        obj.Close()
        app.Quit()
        return pdf_output
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        return None
    finally:
        pythoncom.CoUninitialize()

def extract_text_from_path(file_path: str) -> str:
    """
    Reads a PDF from a physical disk path and extracts text with page markers.
    Used for standardized PDF-only processing.
    """
    text = ""
    if not os.path.exists(file_path):
        return ""
        
    try:
        # Open the PDF file from the disk path
        doc = fitz.open(file_path)
        for i, page in enumerate(doc, start=1):
            txt = page.get_text("text")
            if txt:
                # We add these markers so the RAG knows exactly which page it's on
                text += f"--- Page {i} ---\n{txt}\n"
        doc.close()
    except Exception as e:
        print(f"Error extracting text from path {file_path}: {e}")
        
    return text

# ------------ context search helpers ------------

def _load_subject_index(subject_name: str, user_id: int = None) -> Dict[str, Any]:
    if user_id is None:
        user_id = st.session_state.get("user_id")
    
    root = ensure_subject_folders(subject_name, user_id)
    idx_path = os.path.join(root, "page_index.json")
    if not os.path.isfile(idx_path):
        return {"entries": []}
    try:
        with open(idx_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"entries": []}

def get_subject_index_entries(subject_name: str, user_id: int = None) -> List[Dict[str, Any]]:
    """
    Public helper for quiz engine: return the raw index entries
    (pdf_page, pptx_slide, docx_chunk, etc.) for a subject.
    """
    if user_id is None:
        user_id = st.session_state.get("user_id")
    
    idx = _load_subject_index(subject_name, user_id)
    return idx.get("entries", [])


# ------------ rendering helpers ------------

def render_pdf_page_image(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    # page_num is 1-indexed from your RAG, Fitz is 0-indexed
    page = doc.load_page(page_num - 1) 
    
    # Increase zoom for better quality (2.0 = 2x resolution)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    
    # Convert to bytes for st.image
    img_data = pix.tobytes("png")
    doc.close()
    return img_data