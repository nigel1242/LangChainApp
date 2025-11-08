# modules/rag_index.py
from __future__ import annotations
import fitz  # PyMuPDF
from typing import List, Iterable, Callable, Optional
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

def _extract_text_pymupdf(file_bytes: bytes, page_from: int = 1, page_to: Optional[int] = None) -> List[str]:
    pages: List[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        start = max(0, page_from - 1)
        end = doc.page_count if page_to is None else min(page_to, doc.page_count)
        for pno in range(start, end):
            t = doc.load_page(pno).get_text("text") or ""
            t = t.strip()
            if t:
                pages.append(t)
    return pages

def pdf_to_docs(uploaded_files: Iterable, page_from: int = 1, page_to: Optional[int] = None,
                on_progress: Optional[Callable[[int, int], None]] = None) -> List[str]:
    all_pages: List[str] = []
    total_files = len(uploaded_files or [])
    done = 0

    for f in uploaded_files:
        file_bytes = f.read()
        pages = _extract_text_pymupdf(file_bytes, page_from, page_to)
        all_pages.extend(pages)
        done += 1
        if on_progress:
            on_progress(done, total_files)

    splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=120, length_function=len)
    chunks = []
    for p in all_pages:
        chunks.extend(splitter.split_text(p))
    return chunks

def build_faiss(text_chunks: List[str], embeddings):
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
