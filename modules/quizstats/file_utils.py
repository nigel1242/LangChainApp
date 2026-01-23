# modules/file_utils.py

import os
import json
from typing import List, Dict, Any

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation
import pandas as pd
import win32com.client
import pythoncom
from langchain_core.documents import Document

from modules.quizstats.subject_store import SUBJECTS_DIR

# ------------ extension sets ------------

TEXT_FILE_EXTS = {".txt", ".md"}
DOCX_EXTS = {".docx"}
PPTX_EXTS = {".pptx"}
CSV_EXTS = {".csv"}
PDF_EXTS = {".pdf", ".PDF"}


# ------------ subject folder helpers ------------

def ensure_subject_folders(subject_name: str):
    """
    Ensure subjects/<subject_name>/raw exists.
    All uploaded files (PDF, PPTX, DOCX, etc.) will be stored in this single folder.
    Returns the root path for that subject.
    """
    root = os.path.join(SUBJECTS_DIR, subject_name)
    raw = os.path.join(root, "raw")
    os.makedirs(raw, exist_ok=True)
    return root


def save_uploaded_files(subject_name: str, files) -> List[str]:
    """
    Saves uploaded files into subjects/<name>/raw (single folder for all types).
    Returns list of saved paths.
    """
    root = ensure_subject_folders(subject_name)
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

import win32com.client
import pythoncom
import streamlit as st

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



# ------------ text extraction helpers ------------

def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1252", errors="ignore") as fh:
            return fh.read()


def _extract_text_pdf(path: str) -> str:
    doc = fitz.open(path)
    text_parts: List[str] = []
    for page in doc:
        txt = page.get_text("text")
        if txt and txt.strip():
            text_parts.append(txt)
    doc.close()
    return "\n".join(text_parts)


def _extract_text_docx(path: str) -> str:
    """
    Extract text from a .docx file.

    - Skips temp files like '~$something.docx'
    - Includes:
        * Paragraph text
        * Table cell text
        * Image 'alt text' / descriptions where available
    """
    fname = os.path.basename(path)
    if fname.startswith("~$"):
        return ""

    try:
        doc = DocxDocument(path)
    except PackageNotFoundError:
        return ""
    except Exception:
        return ""

    parts: List[str] = []

    # Paragraphs
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())

    # Tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                txt = cell.text.strip()
                if txt:
                    parts.append(txt)

    # Image descriptions / alt text (best-effort)
    try:
        for shp in doc.inline_shapes:
            try:
                docPr = shp._inline.docPr
                descr = docPr.get("descr")
                title = docPr.get("title")
                for t in (title, descr):
                    if t and t.strip():
                        parts.append(t.strip())
            except Exception:
                continue
    except Exception:
        pass

    return "\n".join(parts)


def _docx_sections_for_index(path: str) -> List[str]:
    """
    Produce smaller 'sections' from a DOCX file to use for context matching.
    Group paragraphs into chunks of ~250+ chars.
    """
    try:
        doc = DocxDocument(path)
    except Exception:
        return []

    chunks: List[str] = []
    buffer: List[str] = []
    length = 0

    def flush():
        nonlocal buffer, length
        if buffer:
            text = " ".join(buffer).strip()
            if text:
                chunks.append(text)
        buffer = []
        length = 0

    # paragraphs
    for p in doc.paragraphs:
        txt = (p.text or "").strip()
        if not txt:
            continue
        buffer.append(txt)
        length += len(txt)
        if length >= 250:
            flush()

    # tables as additional chunks
    for table in doc.tables:
        tbuf: List[str] = []
        for row in table.rows:
            for cell in row.cells:
                txt = cell.text.strip()
                if txt:
                    tbuf.append(txt)
        if tbuf:
            chunks.append(" ".join(tbuf))

    flush()
    return chunks


def _extract_text_pptx(path: str) -> str:
    """
    Extract text from a .pptx presentation: slide by slide.
    Also includes shape 'alt text' / description where available.
    """
    try:
        prs = Presentation(path)
    except Exception:
        return ""

    parts: List[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        slide_text: List[str] = []

        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                slide_text.append(shape.text)

            try:
                alt = getattr(shape, "alternative_text", "") or ""
                if alt.strip():
                    slide_text.append(alt.strip())
            except Exception:
                pass

        if slide_text:
            parts.append(f"--- Slide {i} ---\n" + "\n".join(slide_text))

    return "\n\n".join(parts)


def _pptx_slide_texts_for_index(path: str) -> List[str]:
    """Return a list of per-slide texts for indexing."""
    try:
        prs = Presentation(path)
    except Exception:
        return []

    slides_text: List[str] = []
    for slide in prs.slides:
        slide_text: List[str] = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                slide_text.append(shape.text)
            try:
                alt = getattr(shape, "alternative_text", "") or ""
                if alt.strip():
                    slide_text.append(alt.strip())
            except Exception:
                pass
        if slide_text:
            slides_text.append(" ".join(slide_text))
        else:
            slides_text.append("")
    return slides_text


def _extract_text_csv(path: str) -> str:
    try:
        df = pd.read_csv(path)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error reading CSV: {e}"


def _pdf_page_texts_for_index(path: str) -> List[str]:
    """Return per-page text for a PDF."""
    try:
        doc = fitz.open(path)
    except Exception:
        return []
    pages: List[str] = []
    for page in doc:
        txt = page.get_text("text") or ""
        pages.append(txt)
    doc.close()
    return pages


# ------------ context search helpers ------------

def _load_subject_index(subject_name: str) -> Dict[str, Any]:
    root = ensure_subject_folders(subject_name)
    idx_path = os.path.join(root, "page_index.json")
    if not os.path.isfile(idx_path):
        return {"entries": []}
    try:
        with open(idx_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"entries": []}

def get_subject_index_entries(subject_name: str) -> List[Dict[str, Any]]:
    """
    Public helper for quiz engine: return the raw index entries
    (pdf_page, pptx_slide, docx_chunk, etc.) for a subject.
    """
    idx = _load_subject_index(subject_name)
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