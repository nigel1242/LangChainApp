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
    ("openai_api_key", ""),
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
        raise RuntimeError(
            "OpenAI API key missing. Set it in the Settings page or .env file."
        )
    return OpenAI(api_key=api_key)


# ---------------- EMBEDDINGS WRAPPER ----------------
class OpenAIEmbeddingsLC(Embeddings):
    """
    Simple OpenAI embeddings wrapper so FAISS can use it.
    Uses the same API key as your main app (st.session_state.openai_api_key).
    """

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
    Very simple character-based chunking to avoid pulling in heavy libs.
    """
    text = text.replace("\r\n", "\n")
    n = len(text)
    chunks: List[str] = []
    start = 0
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


# ---------------- LOAD PDFs ----------------
def load_subject_documents(subject_name: str) -> List[Document]:
    """
    Read all PDFs in subjects/<subject>/pdf and turn them into small chunks.
    Each chunk gets metadata: source filename + 1-based page number.
    """
    root = ensure_subject_folders(subject_name)
    pdf_dir = Path(root) / "pdf"
    docs: List[Document] = []

    if not pdf_dir.exists():
        return docs

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            pdf_doc = fitz.open(pdf_path.as_posix())
        except Exception as e:
            st.warning(f"Failed to open {pdf_path.name}: {e}")
            continue

        for page_idx in range(pdf_doc.page_count):
            page = pdf_doc[page_idx]
            page_text = page.get_text("text") or ""
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

        pdf_doc.close()

    return docs


# ---------------- BUILD INDEX ----------------
def build_index_for_subject(subject_name: str) -> FAISS | None:
    docs = load_subject_documents(subject_name)
    if not docs:
        st.warning(
            f"No text found for subject **{subject_name}**. "
            f"Make sure you have PDFs in `subjects/{subject_name}/pdf`."
        )
        return None

    api_key = st.session_state.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OpenAI API key missing. Set it in the Settings page or .env file.")
        return None

    emb = OpenAIEmbeddingsLC("text-embedding-3-small", api_key=api_key)
    index = FAISS.from_documents(docs, emb)

    st.session_state.subject_index = index
    st.session_state.subject_index_subject = subject_name
    return index


# ---------------- SPEECH TO TEXT ----------------
def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Use OpenAI STT to convert recorded audio to text.

    audio_recorder_streamlit returns WAV bytes, so we save as .wav.
    """
    client = get_openai_client()

    import tempfile

    # Save as a WAV file (matches what audio_recorder_streamlit produces)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            # whisper-1 is very robust for transcription
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                # you can force English if you want:
                # language="en",
            )
        return (resp.text or "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass



# ---------------- SIDEBAR: SUBJECT + MODEL + VOICE ----------------
subjects = list_subjects()
if not subjects:
    st.info("No subjects yet. Use your Quiz/Subjects page to create one first.")
    st.stop()

subject_names = [s["name"] for s in subjects]

st.sidebar.header("Subject Tutor")
subject_name = st.sidebar.selectbox(
    "Choose subject",
    subject_names,
    index=subject_names.index(
        st.session_state.get("subject_index_subject", subject_names[0])
    )
    if st.session_state.get("subject_index_subject") in subject_names
    else 0,
)

openai_models = ("gpt-4o-mini", "gpt-4o")
model_name = st.sidebar.selectbox(
    "Model for answering",
    openai_models,
    index=openai_models.index(
        st.session_state.get("subject_model", openai_models[0])
    )
    if st.session_state.get("subject_model") in openai_models
    else 0,
)
st.session_state.subject_model = model_name

voice_options = ["alloy", "verse", "echo", "sage"]
st.session_state.tts_voice = st.sidebar.selectbox(
    "Voice for answer audio",
    voice_options,
    index=voice_options.index(st.session_state.tts_voice)
    if st.session_state.tts_voice in voice_options
    else 0,
)

if st.sidebar.button("🔁 Build / Rebuild index"):
    with st.spinner(f"Building index for **{subject_name}**..."):
        idx = build_index_for_subject(subject_name)
    if idx is not None:
        st.success("Index built. Ask your question below 👇")

st.markdown(f"### Current subject: **{subject_name}**")
st.caption("This page searches your subject PDFs and uses them as context for the model.")

# ---------------- INPUT ROW: TEXT + MIC ----------------
st.markdown("### Ask with text or voice")

col_text, col_mic = st.columns([7, 1])

# 1) MIC on the RIGHT – handle recording and update session_state BEFORE text input
with col_mic:
    st.caption("🎙 Tap to record")
    audio_bytes = audio_recorder(
        text="",
        pause_threshold=1.0,
        sample_rate=41_000,
        icon_size="2x",
    )

# If new audio is recorded, transcribe it and update the text input value
if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
    st.session_state.last_audio_bytes = audio_bytes
    try:
        with st.spinner("Transcribing…"):
            text_from_voice = transcribe_audio_bytes(audio_bytes)
        if text_from_voice:
            # this will be used as default value when text_input is created
            st.session_state.subject_question_input = text_from_voice
    except Exception as e:
        st.error(f"Transcription failed: {e}")

# 2) TEXT INPUT on the LEFT – always visible
with col_text:
    question = st.text_input(
        f"Ask something about **{subject_name}**:",
        key="subject_question_input",
    )

# ---------------- ANSWER + CONTEXT ----------------
if question.strip():
    index = st.session_state.subject_index
    if index is None or st.session_state.subject_index_subject != subject_name:
        st.info("Please build the index for this subject first (sidebar button).")
    else:
        # Retrieve top-k chunks
        try:
            docs = index.similarity_search(question, k=4)
        except Exception as e:
            st.error(f"Search failed: {e}")
            docs = []

        context = "\n\n".join(
            f"Source: {d.metadata.get('source','unknown')} "
            f"(page {d.metadata.get('page','?')})\n{d.page_content}"
            for d in docs
        )

        # Prompt: allow LaTeX + Unicode maths
        prompt = (
            "You are a helpful tutor for a polytechnic student.\n"
            "Use the subject notes below to answer the question as accurately "
            "and clearly as possible.\n"
            "If the notes do not contain the answer, say you are not sure instead "
            "of guessing.\n\n"
            "MATH FORMATTING RULES:\n"
            "- You MAY use LaTeX for formulas (e.g. \\int, \\sum, fractions, etc.).\n"
            "- Put display equations in $$...$$ so they render nicely in Markdown.\n"
            "- You can also use Unicode symbols like ∫, Σ, √, π when convenient.\n"
            "- Always give a short explanation in words next to any formula.\n"
            "- Prefer step-by-step working over long paragraphs.\n\n"
        )
        if context.strip():
            prompt += f"Subject notes:\n{context}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Now give a clear answer with any maths formatted using LaTeX/Unicode as described."

        try:
            client = get_openai_client()
        except RuntimeError as e:
            st.error(str(e))
        else:
            # --- Generate answer with OpenAI ---
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
            st.markdown(answer)

            # --- Text-to-speech for the answer ---
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

        # ----- Show matching notes + slide images -----
        if docs:
            st.markdown("#### 📄 Top matching notes")

            root = ensure_subject_folders(subject_name)
            pdf_dir = Path(root) / "pdf"

            for d in docs[:3]:
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page")
                label = f"{src} (page {page})" if page else src

                st.markdown(f"**Source:** {label}")

                raw = (d.page_content or "").strip().replace("\n", " ")
                raw = " ".join(raw.split())

                allowed = set(
                    "0123456789abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    " .,;:!?()-+/*=<>[]{}^_=%"
                )
                clean = "".join(ch for ch in raw if ch in allowed or ch.isspace())
                ratio = len(clean) / max(1, len(raw))

                if len(clean) < 40 or ratio < 0.6:
                    st.caption(
                        "This page is mostly formulas/symbols – refer to the slide preview below."
                    )
                else:
                    st.caption(clean[:400] + ("..." if len(clean) > 400 else ""))

                # Render PDF page as image
                try:
                    if src and page:
                        pdf_path = pdf_dir / src
                        if pdf_path.is_file():
                            pdf_doc = fitz.open(pdf_path.as_posix())
                            pnum = int(page)
                            if 1 <= pnum <= pdf_doc.page_count:
                                mat = fitz.Matrix(1.2, 1.2)
                                pix = pdf_doc.load_page(pnum - 1).get_pixmap(matrix=mat)
                                img_bytes = pix.tobytes("png")
                                st.image(
                                    img_bytes,
                                    caption=f"Slide preview: {src} – page {page}",
                                    width="stretch",
                                )
                            pdf_doc.close()
                except Exception as e:
                    st.info(f"(Could not render slide preview for {label}: {e})")
