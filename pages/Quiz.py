import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import ollama
from openai import OpenAI

# --- Local Module Imports ---
from modules.db_manager import DBManager
from modules.login import check_authentication, login_page, logout
from modules.quizstats.file_utils import (
    save_uploaded_files,
    extract_corpus_for_subject,
    find_relevant_context_for_text,
    render_pdf_page_image,
    render_pptx_slide_image,
)
from modules.quizstats.quiz_engine import build_quiz_from_corpus
from modules.quizstats.progress_store import (
    init_stats_db,
    record_attempt,
    clear_subject_stats,
)
from modules.quizstats.subject_store import ensure_subject_folders # Keep for folder creation

# --- CONFIG ---
CHAT_DB_FILE = "chat_playground.db"
RAG_INDEX_DIR = "rag_indices"

st.set_page_config(page_title="Quiz Builder", page_icon="📝", layout="wide")
load_dotenv()

# ---------- AUTHENTICATION ----------
is_authenticated, username = check_authentication()
if not is_authenticated:
    login_page()
    st.stop()

# ---------- DB MANAGER & STATE ----------
db = DBManager(backend="sqlite", db_file=CHAT_DB_FILE)
init_stats_db()

# Sync with app.py defaults
if "current_subject" not in st.session_state:
    st.session_state.current_subject = "General"

if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {
        "uploads_buffer": [],
        "quiz": [],
        "current_idx": 0,
        "score": 0,
        "answered": {},
        "submitted": {},
        "corpus": "",
        "show_image": {},
    }

S = st.session_state.quiz_state

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(f"**Logged in as: {username}**")
    if st.button("Logout"):
        logout()
        st.rerun()
    st.markdown("---")
    
    # Subject Selector moved to Sidebar for consistency with app.py
    st.header("📚 Subject Library")
    subjects = db.load_all_subjects()
    if not subjects:
        db.add_subject("General")
        subjects = ["General"]
    
    # Dropdown to select the subject
    selected_sub = st.selectbox(
        "Select Subject", 
        subjects, 
        index=subjects.index(st.session_state.current_subject) if st.session_state.current_subject in subjects else 0
    )
    
    if selected_sub != st.session_state.current_subject:
        st.session_state.current_subject = selected_sub
        # Reset quiz when switching subjects
        S.update({"quiz": [], "current_idx": 0, "score": 0, "answered": {}, "submitted": {}})
        st.rerun()

# ---------- MAIN BODY ----------
st.title(f"📝 Quiz Builder: {st.session_state.current_subject}")

# Model Selector Mapping (Consolidated)
MODEL_MAP = {
    "llama3:8b": "Llama 3 (Base Model)",
    "qwen2.5vl:7b": "Qwen 2.5 VL (Vision Model)",
    "gpt-4o": "OpenAI GPT-4o",
}

# Fetch installed Ollama models
try:
    ollama_list = [m["model"] for m in ollama.list().get("models", [])]
    available_ollama = [m for m in ollama_list if not m.startswith("nomic")]
except:
    available_ollama = []

openai_models = ("gpt-4o",) if st.session_state.get("openai_api_key") else ()
available_models = tuple(available_ollama) + openai_models

if not available_models:
    st.error("⚠️ No models available. Check Ollama or API Keys.")
    st.stop()

# Determine default model
m_idx = 0
if st.session_state.get("selected_model") in available_models:
    m_idx = available_models.index(st.session_state.selected_model)

selected_friendly = st.selectbox(
    "🧠 Choose model for Quiz Generation",
    [MODEL_MAP.get(m, m) for m in available_models],
    index=m_idx
)
# Update technical ID back to session state
st.session_state.selected_model = available_technical_id = available_models[[MODEL_MAP.get(m, m) for m in available_models].index(selected_friendly)]

st.markdown("---")

# Ensure folder layout for the selected subject
subject_root = ensure_subject_folders(st.session_state.current_subject)

# ---------- FILE UPLOAD & PROCESSING ----------
with st.expander(f"📤 Knowledge Base: {st.session_state.current_subject}"):
    uploaded = st.file_uploader("Add files to this subject", type=["txt", "md", "pdf", "docx", "pptx", "csv"], accept_multiple_files=True)
    if uploaded:
        S["uploads_buffer"] = uploaded

    if st.button("📂 Process & Index Files"):
        if not S["uploads_buffer"]:
            st.warning("No files selected.")
        else:
            with st.spinner("Processing..."):
                save_uploaded_files(st.session_state.current_subject, S["uploads_buffer"])
                S["uploads_buffer"] = []
                st.success("Files saved to Subject Library!")
                st.rerun()

# ---------- GENERATE QUIZ ----------
if st.button("🎲 Generate 10 Questions from Subject"):
    corpus = extract_corpus_for_subject(st.session_state.current_subject)
    if not corpus.strip():
        st.warning("Knowledge base is empty. Upload files first.")
    else:
        with st.spinner("Generating MCQs..."):
            quiz = build_quiz_from_corpus(
                corpus_text=corpus,
                subject=st.session_state.current_subject,
                n_questions=10,
                model_name=st.session_state.selected_model,
            )
            S.update({
                "quiz": quiz, "current_idx": 0, "score": 0,
                "answered": {}, "submitted": {}, "corpus": corpus
            })
            st.success("Quiz Generated!")
# ---------- QUIZ PLAYER ----------
if S["quiz"]:
    st.divider()
    q_idx = S["current_idx"]
    q = S["quiz"][q_idx]

    # --- helper to decide if visual context is useful (for caption only) ---
    def _question_needs_visual(qdict):
        text = (
            (qdict.get("question", "") + " " + " ".join(qdict.get("choices", []))).lower()
        )
        visual_words = [
            "graph",
            "plot",
            "chart",
            "bar chart",
            "line chart",
            "histogram",
            "scatter",
            "table",
            "diagram",
            "figure",
            "workflow",
            "curve",
            "regression",
        ]
        return qdict.get("needs_image", False) or any(w in text for w in visual_words)

    needs_visual = _question_needs_visual(q)

    # --- 'Show related context' button ---
    st.markdown("#### 🔍 Need context?")
    show_now = S["show_image"].get(q_idx, False)
    if st.button("👁 Show related context", key=f"show_img_{q_idx}"):
        show_now = True
        S["show_image"][q_idx] = True

    if show_now:
        # Prefer the exact slide/page that was stored with the question
        ctx = q.get("context_meta") or {}
        entry = None

        if ctx.get("file_path") and ctx.get("type") in ("pdf_page", "pptx_slide"):
            entry = ctx
        else:
            # Fallback: semantic search if context_meta is missing
            img_query = (
                q.get("image_hint")
                or q.get("source_hint")
                or q.get("question", "")
            )
            entry = find_relevant_context_for_text(S["selected_subject"], img_query)

        if entry:
            etype = entry.get("type")
            fname = entry.get("file", "unknown file")

            if etype == "pdf_page":
                pdf_path = entry.get("file_path")
                page_num = entry.get("page", 1)
                img = render_pdf_page_image(pdf_path, page_num)
                if img is not None:
                    caption = f"Context from {fname} (PDF page {page_num})"
                    if needs_visual:
                        caption += " – this question is likely image-based."
                    st.image(img, caption=caption, width="stretch")
                else:
                    st.caption(f"Open PDF: `{pdf_path}` (page {page_num})")

            elif etype == "pptx_slide":
                pptx_path = entry.get("file_path")
                slide_num = entry.get("slide", 1)
                img = None
                try:
                    img = render_pptx_slide_image(pptx_path, slide_num)
                except Exception as e:
                    st.caption(f"PPTX render error: {e}")

                if img is not None:
                    caption = f"Context from {fname} (PowerPoint slide {slide_num})"
                    if needs_visual:
                        caption += " – this question is likely image-based."
                    st.image(img, caption=caption, width="stretch")
                else:
                    st.caption(
                        f"Context from {fname} (slide {slide_num}) – unable to render image."
                    )

            else:
                # Non-PDF / non-PPTX context: show a text snippet
                full_text = entry.get("text", "")
                snippet = full_text[:800] + ("..." if len(full_text) > 800 else "")
                st.markdown(f"**Context from {fname}**")
                st.write(snippet)
        else:
            st.caption("No matching context found for this question.")

    # --- Question text + options ---
    st.subheader(f"Question {S['current_idx']+1} of {len(S['quiz'])}")
    difficulty = q.get("difficulty")
    if difficulty:
        st.caption(f"Difficulty: {difficulty}")

    # Allow Markdown/LaTeX in question text
    st.markdown(q["question"])

    choice_labels = [f"{i+1}. {c}" for i, c in enumerate(q["choices"])]

    radio_key = f"mcq_{S['selected_subject']}_{q_idx}"
    picked = st.radio("Choose one:", choice_labels, index=None, key=radio_key)

    col1, col2, col3 = st.columns([1, 1, 2])

    # ---------- SUBMIT ----------
    with col1:
        if st.button("Submit answer", key=f"submit_{q_idx}"):
            if picked is None:
                st.warning("Pick an option first.")
            else:
                picked_idx = int(picked.split(".")[0]) - 1
                correct_idx = q["answer_idx"]
                is_correct = (picked_idx == correct_idx)

                S["answered"][q_idx] = {
                    "picked": picked_idx,
                    "correct": is_correct,
                    "already_counted": S["answered"].get(q_idx, {}).get(
                        "already_counted", False
                    ),
                }
                S["submitted"] = S.get("submitted", {})
                S["submitted"][q_idx] = True

                # ----- UPDATE SCORE + FEEDBACK -----
                if is_correct:
                    if not S["answered"][q_idx]["already_counted"]:
                        S["score"] += 1
                        S["answered"][q_idx]["already_counted"] = True
                    st.success("✅ Correct! Nice job.")
                else:
                    st.error("❌ Incorrect. Let's review this concept.")
                    st.caption(f"Answer: {q['choices'][correct_idx]}")

                explanation = q.get("explanation", "")
                if explanation:
                    # explanation might contain LaTeX, so use markdown
                    st.markdown(explanation)

                # ----- RECORD STATS -----
                try:
                    ts_ms = int(time.time() * 1000)
                    chosen_text = q["choices"][picked_idx]
                    correct_text = q["choices"][correct_idx]
                    topic = q.get("topic") or (q.get("image_hint") or "")
                    source_hint = q.get("source_hint") or ""
                    diff = q.get("difficulty") or "unknown"

                    record_attempt(
                        subject=S["selected_subject"],
                        question=q["question"],
                        chosen=chosen_text,
                        correct=correct_text,
                        is_correct=1 if is_correct else 0,
                        topic=topic,
                        source_hint=source_hint,
                        difficulty=diff,
                        ts_ms=ts_ms,
                    )
                except Exception as e:
                    st.warning(f"Could not record stats: {e}")

    # ---------- NEXT ----------
    with col2:
        if st.button("Next ➡️", key=f"next_{q_idx}"):
            if not S["submitted"].get(q_idx):
                st.warning("Submit your answer before moving to the next question.")
            else:
                if S["current_idx"] < len(S["quiz"]) - 1:
                    S["current_idx"] += 1
                else:
                    st.info("That was the last question.")

    # ---------- SCORE ----------
    with col3:
        st.metric("Score", f"{S['score']} / {len(S['quiz'])}")

else:
    st.info("Click **Generate 10 Questions** to start.")