# pages/01_📚_Subject_Tutor.py

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from modules.quizstats.subject_store import list_subjects, ensure_subject_folders

# ---------------- CONFIG ----------------
load_dotenv()
st.set_page_config(page_title="Subject Tutor", page_icon="📚", layout="wide")
st.title("📚 Subject Tutor – chat with your subject notes")

# ---------------- SESSION STATE ----------------
for key, default in [
    ("openai_api_key", ""),              # optional if you load from .env
    ("subject_index", None),
    ("subject_index_subject", None),
    ("subject_model", "gpt-4o-mini"),
    ("subject_question_input", ""),
    ("tts_voice", "alloy"),
    ("last_audio_bytes", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def get_openai_client() -> OpenAI:
    api_key = st.session_state.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI API key missing. Set it in Settings or .env")
    return OpenAI(api_key=api_key)


# ---------------- EMBEDDINGS WRAPPER ----------------
class OpenAIEmbeddingsLC(Embeddings):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        res = self.client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in res.data]

    def embed_query(self, query: str) -> List[float]:
        res = self.client.embeddings.create(model=self.model_name, input=query)
        return res.data[0].embedding


# ---------------- SIMPLE CHUNKER ----------------
def simple_chunks(text: str, chunk_size: int = 900, overlap: int = 200) -> List[str]:
    """
    Simple character chunking so we don't pull heavy dependencies.
    """
    text = (text or "").replace("\r\n", "\n")
    n = len(text)
    chunks: List[str] = []
    start = 0
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


# ---------------- LOAD PDFs ----------------
def load_subject_documents(subject_name: str) -> List[Document]:
    """
    Reads PDFs from: subjects/<subject>/raw/
    Extracts text per page and chunks it.
    """
    root = ensure_subject_folders(subject_name)
    raw_dir = Path(root) / "raw"
    docs: List[Document] = []

    if not raw_dir.exists():
        return docs

    # Support both .pdf and .PDF
    pdf_paths = sorted(list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*.PDF")))
    if not pdf_paths:
        return docs

    for pdf_path in pdf_paths:
        try:
            pdf_doc = fitz.open(pdf_path.as_posix())
        except Exception:
            continue

        try:
            for page_idx in range(pdf_doc.page_count):
                try:
                    page = pdf_doc[page_idx]
                    page_text = page.get_text("text") or ""
                except Exception:
                    continue

                if not page_text.strip():
                    continue

                for chunk in simple_chunks(page_text):
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_idx + 1,  # 1-based page number
                            },
                        )
                    )
        finally:
            pdf_doc.close()

    return docs


# ---------------- BUILD INDEX ----------------
def build_index_for_subject(subject_name: str) -> FAISS | None:
    docs = load_subject_documents(subject_name)
    if not docs:
        root = ensure_subject_folders(subject_name)
        st.warning(
            f"No readable text found for **{subject_name}**.\n\n"
            f"Put PDFs here: `{Path(root) / 'raw'}`\n\n"
            "If your PDFs are scanned images, PyMuPDF won't extract text (you'd need OCR)."
        )
        return None

    api_key = st.session_state.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OpenAI API key missing. Set it in Settings or .env.")
        return None

    emb = OpenAIEmbeddingsLC("text-embedding-3-small", api_key)
    index = FAISS.from_documents(docs, emb)

    st.session_state.subject_index = index
    st.session_state.subject_index_subject = subject_name
    return index


# ---------------- SPEECH TO TEXT ----------------
def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Uses OpenAI Whisper to transcribe recorded WAV bytes.
    """
    client = get_openai_client()
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                 language="en",  
            )
        return (resp.text or "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ---------------- SIDEBAR ----------------
subjects = list_subjects()
if not subjects:
    st.info("No subjects yet. Create one in Quiz Builder first.")
    st.stop()

subject_names = [s["name"] for s in subjects]

st.sidebar.header("Subject Tutor")
subject_name = st.sidebar.selectbox("Choose subject", subject_names)

openai_models = ("gpt-4o-mini", "gpt-4o")
model_name = st.sidebar.selectbox("Model for answering", openai_models)
st.session_state.subject_model = model_name

voice_options = ["alloy", "verse", "echo", "sage"]
st.session_state.tts_voice = st.sidebar.selectbox("Voice for answer audio", voice_options)

if st.sidebar.button("🔁 Build / Rebuild index"):
    with st.spinner(f"Building index for **{subject_name}**..."):
        idx = build_index_for_subject(subject_name)
    if idx is not None:
        st.success("Index built. Ask your question below 👇")


# ---------------- MAIN HEADER ----------------
st.markdown(f"### Current subject: **{subject_name}**")
st.caption("This page searches your PDFs in `subjects/<subject>/raw/` and uses them as context.")


# ---------------- INPUT: TEXT + MIC ----------------
st.markdown("### Ask with text or voice")
col_text, col_mic = st.columns([7, 1])

with col_mic:
    st.caption("🎙 Tap to record")
    audio_bytes = audio_recorder(
        text="",
        pause_threshold=1.0,
        sample_rate=41_000,
        icon_size="2x",
    )

if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
    st.session_state.last_audio_bytes = audio_bytes
    try:
        with st.spinner("Transcribing…"):
            text_from_voice = transcribe_audio_bytes(audio_bytes)
        if text_from_voice:
            st.session_state.subject_question_input = text_from_voice
    except Exception as e:
        st.error(f"Transcription failed: {e}")

with col_text:
    question = st.text_input("Ask your question:", key="subject_question_input")


# ---------------- ANSWER + CONTEXT ----------------
if question.strip():
    index = st.session_state.subject_index
    if index is None or st.session_state.subject_index_subject != subject_name:
        st.info("Please build the index for this subject first (sidebar button).")
        st.stop()

    # Retrieve top-k chunks
    try:
        docs = index.similarity_search(question, k=4)
    except Exception as e:
        st.error(f"Search failed: {e}")
        docs = []

    context = "\n\n".join(
        f"Source: {d.metadata.get('source','unknown')} (page {d.metadata.get('page','?')})\n{d.page_content}"
        for d in docs
    )

    # ✅ LaTeX-aware prompt (forces correct math formatting)
    prompt = (
        "You are a helpful tutor for a polytechnic student.\n"
        "Use the subject notes below to answer the question as accurately and clearly as possible.\n"
        "If the notes do not contain the answer, say you are not sure instead of guessing.\n\n"
        "MATH FORMATTING RULES:\n"
        "- Use LaTeX for all mathematical formulas.\n"
        "- Inline formulas must be wrapped in single dollar signs, e.g. $x^2$.\n"
        "- Display equations must be wrapped in double dollar signs, e.g. $$x^2+y^2=z^2$$.\n"
        "- Always explain formulas in words after showing them.\n"
        "- Use standard mathematical notation only.\n\n"
    )

    if context.strip():
        prompt += f"Subject notes:\n{context}\n\n"

    prompt += (
        f"Question: {question}\n\n"
        "Now give a clear answer using proper explanations and LaTeX formatting where needed."
    )

    # Call OpenAI
    try:
        client = get_openai_client()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content or ""
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        answer = ""

    st.markdown("#### 🧠 Tutor answer")
    st.markdown(answer if answer.strip() else "_(No answer returned.)_")

    # ---- Text-to-speech (optional) ----
    if answer.strip():
        try:
            audio_resp = client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=st.session_state.tts_voice,
                input=answer,
            )
            audio_out = audio_resp.read()
            st.audio(audio_out, format="audio/mp3")
        except Exception as e:
            st.info(f"(Could not generate audio answer: {e})")

    # ---- Show matching notes + PDF page previews ----
    if docs:
        st.markdown("#### 📄 Top matching notes")

        root = ensure_subject_folders(subject_name)
        raw_dir = Path(root) / "raw"

        for d in docs[:3]:
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            label = f"{src} (page {page})" if page else src

            st.markdown(f"**Source:** {label}")

            # text snippet
            raw_txt = (d.page_content or "").strip().replace("\n", " ")
            raw_txt = " ".join(raw_txt.split())
            if raw_txt:
                st.caption(raw_txt[:400] + ("..." if len(raw_txt) > 400 else ""))

            # page preview
            try:
                if src and page:
                    pdf_path = raw_dir / src
                    if not pdf_path.is_file():
                        st.warning(f"PDF not found: `{pdf_path}`")
                        continue

                    pdf_doc = fitz.open(pdf_path.as_posix())
                    try:
                        pnum = int(page)
                        if 1 <= pnum <= pdf_doc.page_count:
                            mat = fitz.Matrix(1.2, 1.2)
                            pix = pdf_doc.load_page(pnum - 1).get_pixmap(matrix=mat)
                            st.image(
                                pix.tobytes("png"),
                                caption=f"Preview: {src} – page {page}",
                                use_container_width=True,
                            )
                    finally:
                        pdf_doc.close()

            except Exception as e:
                st.info(f"(Could not render slide preview for {label}: {e})")