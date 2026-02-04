import os
import json
import streamlit as st
import fitz  # PyMuPDF
import platform
import subprocess
from typing import List, Dict, Any

# BASE_DIR remains the same
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) 
SUBJECTS_BASE = os.path.join(BASE_DIR, "modules", "subjects")

# ------------ subject folder helpers ------------

def get_user_root(user_id: int) -> str:
    path = os.path.join(SUBJECTS_BASE, f"user_{user_id}")
    os.makedirs(path, exist_ok=True)
    return path

def ensure_subject_folders(subject_name: str, user_id: int) -> str:
    user_root = get_user_root(user_id)
    subject_root = os.path.join(user_root, subject_name)
    raw_dir = os.path.join(subject_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    return subject_root

def save_uploaded_files(subject_name: str, files, user_id: int) -> List[str]:
    subject_root = ensure_subject_folders(subject_name, user_id)
    raw_dir = os.path.join(subject_root, "raw")
    saved_paths = []
    for f in files:
        dest = os.path.join(raw_dir, f.name)
        f.seek(0)
        with open(dest, "wb") as out:
            out.write(f.read())
        saved_paths.append(dest)
    return saved_paths

# ------------ extraction helpers ------------

def convert_to_pdf(input_path: str) -> str:
    """
    Attempts to convert DOCX/PPTX to PDF based on the Operating System.
    """
    if not os.path.exists(input_path): return None
    
    abs_input = os.path.abspath(input_path)
    pdf_output = os.path.splitext(abs_input)[0] + ".pdf"
    ext = os.path.splitext(abs_input)[1].lower()
    
    if ext == ".pdf": return abs_input
    if os.path.exists(pdf_output): return pdf_output

    # --- Case 1: Windows (Use Microsoft Office via Win32) ---
    if platform.system() == "Windows":
        try:
            import win32com.client
            import pythoncom
            pythoncom.CoInitialize()
            
            if ext == ".docx":
                word = win32com.client.DispatchEx("Word.Application")
                doc = word.Documents.Open(abs_input, ReadOnly=True, Visible=False)
                doc.SaveAs(pdf_output, 17) # 17 = wdExportFormatPDF
                doc.Close()
                word.Quit()
                return pdf_output
            elif ext == ".pptx":
                ppt = win32com.client.DispatchEx("PowerPoint.Application")
                pres = ppt.Presentations.Open(abs_input, True, False, False)
                pres.SaveAs(pdf_output, 32) # 32 = ppSaveAsPDF
                pres.Close()
                ppt.Quit()
                return pdf_output
        except Exception as e:
            print(f"Windows Office conversion failed: {e}")
            # If Win32 fails, it will fall through to attempt LibreOffice

    # --- Case 2: Linux/Cloud or Fallback (Use LibreOffice) ---
    try:
        # On some Windows installs it might be 'soffice', on Linux usually 'libreoffice'
        cmd = 'libreoffice' if platform.system() != "Windows" else 'soffice'
        command = [
            cmd, '--headless', 
            '--convert-to', 'pdf', 
            '--outdir', os.path.dirname(abs_input), 
            abs_input
        ]
        subprocess.run(command, check=True, capture_output=True)
        if os.path.exists(pdf_output):
            return pdf_output
    except Exception as e:
        print(f"PDF Conversion totally failed: {e}")
    
    return None

def extract_text_from_path(file_path: str) -> str:
    """
    Tries to extract text directly from the file. 
    If direct extraction fails, falls back to PDF conversion.
    """
    if not os.path.exists(file_path): return ""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    # 1. Direct Extraction (Avoids needing Word/Office installed for RAG)
    try:
        if ext == ".docx":
            from docx import Document as DocxReader
            doc = DocxReader(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".pptx":
            from pptx import Presentation
            prs = Presentation(file_path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
    except Exception as e:
        print(f"Direct extraction failed for {ext}: {e}")

    # 2. PDF Extraction (Standard)
    if ext == ".pdf" or (not text.strip()):
        target_path = file_path if ext == ".pdf" else convert_to_pdf(file_path)
        if target_path and os.path.exists(target_path):
            try:
                doc = fitz.open(target_path)
                for i, page in enumerate(doc, start=1):
                    txt = page.get_text("text")
                    if txt:
                        text += f"--- Page {i} ---\n{txt}\n"
                doc.close()
            except Exception as e:
                print(f"PyMuPDF extraction failed: {e}")

    return text

# ------------ rendering & index helpers ------------

def render_pdf_page_image(pdf_path: str, page_num: int):
    if not os.path.exists(pdf_path): return None
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1) 
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        doc.close()
        return img_data
    except Exception: return None

def get_subject_index_entries(subject_name: str, user_id: int) -> List[Dict[str, Any]]:
    subject_root = ensure_subject_folders(subject_name, user_id)
    idx_path = os.path.join(subject_root, "page_index.json")
    if not os.path.isfile(idx_path): return []
    try:
        with open(idx_path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("entries", [])
    except Exception: return []