# modules/file_utils.py

import os
import io
import re
import json
import tempfile
from typing import List, Tuple, Optional, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
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


def _ext(path: str) -> str:
    return os.path.splitext(path)[1]


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



# ------------ text extraction helpers ------------

def extract_text_from_file(path: str) -> list[Document]:
    ext = os.path.splitext(path)[1].lower()
    file_name = os.path.basename(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs: list[Document] = []

    # ---------- PDF ----------
    if ext == ".pdf":
        pages = _pdf_page_texts_for_index(path)
        for page_num, page_text in enumerate(pages, start=1):
            for chunk in splitter.split_text(page_text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "file": file_name,
                            "page": page_num,
                            "type": "pdf_page"
                        }
                    )
                )

    # ---------- PPTX ----------
    elif ext == ".pptx":
        slides = _pptx_slide_texts_for_index(path)
        for slide_num, slide_text in enumerate(slides, start=1):
            for chunk in splitter.split_text(slide_text):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "file": file_name,
                            "slide": slide_num,
                            "type": "pptx_slide"
                        }
                    )
                )

    # ---------- DOCX ----------
    elif ext == ".docx":
        sections = _docx_sections_for_index(path)
        for section in sections:
            for chunk in splitter.split_text(section):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "file": file_name,
                            "type": "docx_section"
                        }
                    )
                )

    # ---------- TXT ----------
    elif ext == ".txt":
        text = _read_text_file(path)
        for chunk in splitter.split_text(text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "file": file_name,
                        "type": "txt"
                    }
                )
            )

    # ---------- CSV ----------
    elif ext == ".csv":
        text = _extract_text_csv(path)
        for chunk in splitter.split_text(text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "file": file_name,
                        "type": "csv"
                    }
                )
            )

    return docs

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


# ------------ corpus builder + index ------------

def extract_corpus_for_subject(subject_name: str) -> str:
    """
    Concatenate all text extracted from subjects/<subject>/raw as a single corpus string
    AND build a page_index.json with fine-grained contexts from all files.

    raw/: .txt, .md, .docx, .pptx, .csv, .pdf
    """
    root = ensure_subject_folders(subject_name)
    raw_dir = os.path.join(root, "raw")
    corpus_parts: List[str] = []
    index_entries: List[Dict[str, Any]] = []

    if os.path.isdir(raw_dir):
        for fname in os.listdir(raw_dir):
            path = os.path.join(raw_dir, fname)
            ext = _ext(path)

            # Plain text files
            if ext in TEXT_FILE_EXTS:
                text = _read_text_file(path)
                corpus_parts.append(text)
                if text.strip():
                    index_entries.append({
                        "type": "text_file",
                        "file": fname,
                        "file_path": path,
                        "section": 1,
                        "text": text,
                    })

            # DOCX
            elif ext in DOCX_EXTS:
                text = _extract_text_docx(path)
                if text.strip():
                    corpus_parts.append(text)
                sections = _docx_sections_for_index(path)
                for i, sec in enumerate(sections, start=1):
                    if sec.strip():
                        index_entries.append({
                            "type": "docx_chunk",
                            "file": fname,
                            "file_path": path,
                            "section": i,
                            "text": sec,
                        })

            # PPTX
            elif ext in PPTX_EXTS:
                text = _extract_text_pptx(path)
                if text.strip():
                    corpus_parts.append(text)
                slide_texts = _pptx_slide_texts_for_index(path)
                for i, sec in enumerate(slide_texts, start=1):
                    if sec.strip():
                        index_entries.append({
                            "type": "pptx_slide",
                            "file": fname,
                            "file_path": path,
                            "slide": i,
                            "text": sec,
                        })

            # CSV
            elif ext in CSV_EXTS:
                text = _extract_text_csv(path)
                corpus_parts.append(text)
                if text.strip():
                    index_entries.append({
                        "type": "csv_file",
                        "file": fname,
                        "file_path": path,
                        "section": 1,
                        "text": text,
                    })

            # PDF – now also stored in raw/
            elif ext in PDF_EXTS:
                text = _extract_text_pdf(path)
                if text.strip():
                    corpus_parts.append(text)
                page_texts = _pdf_page_texts_for_index(path)
                for i, sec in enumerate(page_texts, start=1):
                    if sec.strip():
                        index_entries.append({
                            "type": "pdf_page",
                            "file": fname,
                            "file_path": path,
                            "page": i,
                            "text": sec,
                        })

    # Save index JSON
    index_obj = {"entries": index_entries}
    index_path = os.path.join(root, "page_index.json")
    try:
        with open(index_path, "w", encoding="utf-8") as fh:
            json.dump(index_obj, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[file_utils] Failed to save index for {subject_name}: {e}")

    # Join only non-empty chunks for corpus
    return "\n".join([c for c in corpus_parts if c and c.strip()])



# ------------ context search helpers ------------

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "over",
    "between", "when", "where", "what", "which", "does", "mean", "data",
    "visualisation", "visualization", "analytics", "chart", "graph", "table",
    "figure", "slide", "axis", "value", "values", "point", "points", "using",
    "sample", "example", "shown", "following", "below", "return", "lab", "exercise"
}


def _tokenise_hint(hint: str) -> List[str]:
    """
    Turn a question / hint into lowercased tokens, removing short junk
    and generic stopwords.
    """
    if not hint:
        return []
    tokens = re.findall(r"[A-Za-z0-9]+", hint)
    cleaned = [
        t.lower()
        for t in tokens
        if len(t) > 2 and t.lower() not in _STOPWORDS
    ]
    return cleaned


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



def _context_entry_key(entry: Dict[str, Any]) -> str:
    """
    Build a simple unique key for an index entry so we can avoid reusing
    the same slide/page/section across multiple questions.
    """
    etype = entry.get("type", "unknown")
    fname = entry.get("file", "unknown")
    if etype == "pdf_page":
        idx = entry.get("page", 0)
    elif etype == "pptx_slide":
        idx = entry.get("slide", 0)
    else:
        idx = entry.get("section", 0)
    return f"{etype}::{fname}::{idx}"


def find_relevant_context_for_text(
    subject_name: str,
    hint: str,
    used_keys: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find the single most relevant context entry (PDF page, PPT slide, DOCX chunk, etc.)
    based on token overlap with the hint.

    - Scores all entries by token overlap
    - Prefers entries NOT in used_keys (so you don't keep getting the same slide)
    - Enforces a MINIMUM score so we don't display random slides like
      'Return to Lab Ex' when the match is weak.
    """
    idx = _load_subject_index(subject_name)
    entries: List[Dict[str, Any]] = idx.get("entries", [])
    if not entries:
        return None

    tokens = _tokenise_hint(hint)
    if not tokens:
        return None

    used_keys = used_keys or set()

    scored: List[Tuple[int, Dict[str, Any], str]] = []
    for entry in entries:
        text = (entry.get("text") or "").lower()
        if len(text.strip()) < 40:
            continue  # skip ultra-short bits

        score = sum(1 for t in tokens if t in text)
        if score <= 0:
            continue

        key = _context_entry_key(entry)
        scored.append((score, entry, key))

    if not scored:
        return None

    scored.sort(key=lambda tup: tup[0], reverse=True)
    max_score = scored[0][0]

    # filter to "good enough" candidates
    candidates: List[Tuple[int, Dict[str, Any], str]] = []
    for score, entry, key in scored:
        if score < max(2, len(tokens) // 3):
            continue
        if score < max_score * 0.5:
            continue
        candidates.append((score, entry, key))

    if not candidates:
        return None

    # Prefer unused entries
    for score, entry, key in candidates:
        if key not in used_keys:
            return entry

    # Fallback: if all good ones are used, return the top candidate anyway
    return candidates[0][1]


def find_relevant_pdf_page_for_text(
    subject_name: str,
    hint: str,
) -> Optional[Tuple[str, int]]:
    """
    Convenience wrapper that only considers PDF pages.
    Returns (pdf_path, page_number) or None.
    """
    idx = _load_subject_index(subject_name)
    entries: List[Dict[str, Any]] = idx.get("entries", [])
    if not entries:
        return None

    tokens = _tokenise_hint(hint)
    if not tokens:
        return None

    best_score = 0
    best_pdf: Optional[str] = None
    best_page: int = 1

    for entry in entries:
        if entry.get("type") != "pdf_page":
            continue

        text = (entry.get("text") or "").lower()
        if len(text.strip()) < 40:
            continue

        score = sum(1 for t in tokens if t in text)
        if score > best_score:
            best_score = score
            best_pdf = entry.get("file_path")
            best_page = entry.get("page", 1)

    if best_score <= 0 or not best_pdf:
        return None

    return best_pdf, best_page


def find_relevant_pdf_for_text(subject_name: str, hint: str) -> Optional[str]:
    """
    Backwards-compatible wrapper: return just the PDF path.
    """
    match = find_relevant_pdf_page_for_text(subject_name, hint)
    if match is None:
        return None
    pdf_path, _ = match
    return pdf_path


# ------------ rendering helpers ------------

def _pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    """
    Convert a PyMuPDF Pixmap to a PIL Image safely.
    Works across PyMuPDF versions.
    """
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img


def render_pdf_first_page_image(pdf_path: str):
    """
    Returns a PIL.Image of the first page, or None if fail.
    """
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return None

        page = doc[0]
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = _pixmap_to_pil(pix)
        return img
    except Exception as e:
        print(f"[file_utils] render_pdf_first_page_image error for {pdf_path}: {e}")
        return None
    finally:
        if doc is not None:
            doc.close()


def render_pdf_page_image(pdf_path: str, page_number: int):
    """
    Returns a PIL.Image of a specific 1-based page, or None if fail.
    """
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return None

        page_index = max(0, min(page_number - 1, len(doc) - 1))
        page = doc[page_index]
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = _pixmap_to_pil(pix)
        return img
    except Exception as e:
        print(f"[file_utils] render_pdf_page_image error for {pdf_path}: {e}")
        return None
    finally:
        if doc is not None:
            doc.close()


def render_pptx_slide_image(pptx_path: str, slide_number: int):
    powerpoint = None
    presentation = None
    try:
        pythoncom.CoInitialize()
        
        # Use DispatchEx to ensure we get a fresh, separate process
        powerpoint = win32com.client.DispatchEx("PowerPoint.Application")
        
        # Absolute path is required for COM
        abs_path = os.path.abspath(pptx_path)
        
        # Open with specific flags: ReadOnly=True, WithWindow=False
        # This prevents PowerPoint from asking "Do you want to save changes?"
        presentation = powerpoint.Presentations.Open(abs_path, ReadOnly=True, WithWindow=False)

        if not presentation:
            return None

        # Fix Indexing
        idx = int(slide_number)
        idx = max(1, min(idx, presentation.Slides.Count))
        
        slide = presentation.Slides(idx)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "slide_export.png")
            slide.Export(out_path, "PNG")
            img = Image.open(out_path)
            img.load()
            
        # Close presentation explicitly BEFORE quitting the app
        presentation.Close()
        presentation = None # Clear reference
        
        return img

    except Exception as e:
        print(f"[file_utils] PPTX Render Error: {e}")
        return None
    finally:
        # Robust cleanup to avoid "Open.Close" errors
        try:
            if presentation is not None:
                presentation.Close()
        except:
            pass
        try:
            if powerpoint is not None:
                powerpoint.Quit()
        except:
            pass
        pythoncom.CoUninitialize()