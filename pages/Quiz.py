# pages/quiz.py (Revised: Model Selector in Main Body)

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
# >>> IMPORT OLLAMA AND OPENAI LIBRARIES <<<
import ollama
from openai import OpenAI
import openai # Needed for the AuthenticationError in catch blocks

# --- Local Module Imports ---
from modules.login import check_authentication, login_page, logout
from modules.functions import (
    rebuild_rag_index, # <-- Used for integrating with Subject Tutor RAG
)
from modules.quizstats.subject_store import (
    init_subject_db,
    list_subjects,
    create_subject_if_missing,
    get_subject_meta,
    save_subject_meta,
    ensure_subject_folders,
    delete_subject,
)
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

# --- GLOBAL CONSTANTS (Copied from app.py for RAG functions) ---
CHAT_DB_FILE = "chat_playground.db"
RAG_INDEX_DIR = "rag_indices"
os.makedirs(RAG_INDEX_DIR, exist_ok=True)


# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Quiz Builder", page_icon="📝", layout="wide")
load_dotenv()

# ---------- AUTHENTICATION GATE (Unified with App.py) ----------
is_authenticated, username = check_authentication()

if not is_authenticated:
    login_page()
    st.stop()
    
# ---------- SESSION ----------
# Ensure keys exist, using the structure from app.py
defaults = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "selected_model": None, # Top-level key for LLM
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Update the quiz_state initialization
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {
        "selected_subject": None,
        "new_subject_name": "",
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

# make sure stats DB exists
init_stats_db()


# ---------- SIDEBAR (Simplified to only contain Auth/Config) ----------
with st.sidebar:
    st.markdown(f"**Logged in as: {username}**")
    if st.button("Logout"):
        logout()
        st.rerun()
    st.markdown("---")

    # 1. --- API Key Expander (Copied from app.py) ---
    st.header("🔑 API Keys")
    with st.expander("Manage Keys"):
        # The key check uses the top-level session state key
        if st.session_state.openai_api_key:
            st.success("OpenAI API Key loaded from .env")
            override_key = st.text_input(
                "Override OpenAI API Key (optional)",
                type="password",
                placeholder="Leave empty to use .env key",
                key="quiz_openai_override"
            )
            if override_key:
                st.session_state.openai_api_key = override_key
                st.success("Using custom API key")
        else:
            api_key_input = st.text_input(
                "Enter your OpenAI API Key",
                type="password",
                key="quiz_openai_input"
            )
            if api_key_input:
                st.session_state.openai_api_key = api_key_input
                st.success("API key saved")
                st.rerun()
                
# ---------- SIDEBAR END ----------

# ---------- HEADER ----------
st.title("📝 Quiz Builder (Subjects + Files + 10 MCQs)")

# >>> START: UNIFIED MODEL SELECTOR (IN MAIN BODY, like app.py) <<<
# --- Available Models ---
try:
    ollama_models = tuple(m['model'] for m in ollama.list().get("models", []))
except Exception:
    ollama_models = ()
    
openai_models = ("gpt-3.5-turbo","gpt-4") if st.session_state["openai_api_key"] else ()
available_models = ollama_models + openai_models

if not available_models:
    st.warning("⚠️ No models available. Check Ollama server or OpenAI key")

# Set default selected_model if none is chosen
if available_models and st.session_state.selected_model is None:
    st.session_state.selected_model = available_models[0]
elif st.session_state.selected_model not in available_models:
    st.session_state.selected_model = available_models[0] if available_models else None

# --- Model Selector ---
selected_model = st.selectbox(
    "🧠 Choose model for Quiz Generation",
    available_models,
    index=available_models.index(st.session_state.selected_model) 
    if st.session_state.selected_model in available_models else 0,
    key="quiz_model_selector_main"
)
st.session_state.selected_model = selected_model # Update session state

st.markdown("---")
# >>> END: UNIFIED MODEL SELECTOR <<<


# ---------- SUBJECTS: CREATE / SELECT ----------
init_subject_db()

left, right = st.columns([1, 1])

with left:
    st.subheader("📚 Subjects")

    subjects = list_subjects()
    subject_names = [s["name"] for s in subjects]

    PLACEHOLDER = "— select —"
    existing_names = [PLACEHOLDER] + subject_names

    if S["selected_subject"] in subject_names:
        default_label = S["selected_subject"]
    else:
        default_label = PLACEHOLDER
    default_index = existing_names.index(default_label)

    selected_label = st.selectbox(
        "Choose existing subject",
        existing_names,
        index=default_index,
        key="subject_selector_main",
    )

    if selected_label == PLACEHOLDER:
        S["selected_subject"] = None
    else:
        S["selected_subject"] = selected_label

    # 🔻 DELETE SUBJECT BUTTON (only when one is selected)
    if S["selected_subject"]:
        st.markdown("### Delete Subject")
        if st.button(
            f"🗑️ Delete subject '{S['selected_subject']}'",
            type="secondary",
            key="delete_subject_btn",
        ):
            # 1) Clear stats for that subject
            try:
                clear_subject_stats(S["selected_subject"])
            except Exception as e:
                st.warning(f"Could not clear stats for subject: {e}")

            # 2) Delete subject from DB + disk
            ok, msg = delete_subject(S["selected_subject"]) 
            if ok:
                st.success(msg + " All quiz stats for this subject were cleared.")
                # 3) Reset local state
                S["selected_subject"] = None
                S["quiz"] = []
                S["current_idx"] = 0
                S["score"] = 0
                S["answered"] = {}
                S["submitted"] = {}
                S["corpus"] = ""
                S["show_image"] = {}
                st.rerun()
            else:
                st.error(msg)

with right:
    st.subheader("➕ New Subject")

    S["new_subject_name"] = st.text_input(
        "Create a new subject:",
        value=S["new_subject_name"],
        key="new_subject_name",
    )

    if st.button("Create Subject", key="create_subject_btn"):
        ok, msg = create_subject_if_missing(S["new_subject_name"])
        if ok:
            st.success(f"Created subject '{S['new_subject_name']}'")
            S["selected_subject"] = S["new_subject_name"]
            S["new_subject_name"] = ""
        else:
            st.error(msg)

if not S["selected_subject"]:
    st.info("Select a subject on the left or create a new one.")
    st.stop()

# Ensure folder layout
subject_root = ensure_subject_folders(S["selected_subject"])

# ---------- FILE UPLOAD ----------
st.markdown("### 📤 Upload Files (TXT, DOCX, PPTX, CSV, PDF)")
uploaded = st.file_uploader(
    "Drag and drop multiple files",
    type=["txt", "md", "pdf", "PDF", "docx", "pptx", "csv"],
    accept_multiple_files=True,
    key="quiz_uploader",
)
if uploaded:
    S["uploads_buffer"] = uploaded

if st.button("📦 Process Uploaded Files"):
    if not S["uploads_buffer"]:
        st.warning("No files selected.")
    elif not st.session_state.selected_model:
        st.error("Please select a model above before processing.")
    else:
        with st.spinner("Processing files and rebuilding RAG index..."):
            # Step 1: Save files and extract corpus
            saved = save_uploaded_files(S["selected_subject"], S["uploads_buffer"])
            S["uploads_buffer"] = []
            corpus = extract_corpus_for_subject(S["selected_subject"])
            
            # Step 2: Save metadata
            meta = get_subject_meta(S["selected_subject"]) or {}
            meta["last_indexed_ms"] = int(time.time() * 1000)
            meta["char_count"] = len(corpus)
            save_subject_meta(S["selected_subject"], meta)
            
            st.success(f"Processed {len(saved)} file(s). Indexed {len(corpus):,} characters.")
            
            # Step 3: Trigger Central RAG Index Rebuild (Crucial for Tutor page)
            try:
                # We use the subject name as the index identifier for FAISS
                rebuild_rag_index(
                    CHAT_DB_FILE, RAG_INDEX_DIR, 
                    chat_id_to_process=S["selected_subject"], 
                    selected_model=st.session_state.selected_model, 
                    ollama_models=ollama_models, 
                    openai_models=openai_models,
                    vector_db="faiss" # Assuming local FAISS is used for subject RAG
                )
                st.success("RAG Index updated for Subject Tutor.")
            except Exception as e:
                st.warning(f"Failed to rebuild RAG index for Subject Tutor: {e}")

        st.rerun()

# ---------- BUILD / START QUIZ ----------
st.markdown("### 🧠 Generate Quiz (10 MCQs)")
if st.button("🎲 Generate 10 Questions"):
    corpus = extract_corpus_for_subject(S["selected_subject"])
    if not corpus.strip():
        st.warning("No text in this subject yet. Please upload/process files first.")
    else:
        # Check central model state
        if not st.session_state.selected_model:
            st.error("Please select a model above first.")
            st.stop()
            
        quiz = build_quiz_from_corpus(
            corpus_text=corpus,
            subject=S["selected_subject"],
            n_questions=10,
            model_name=st.session_state.selected_model, # Use central state model
        )
        S["quiz"] = quiz
        S["current_idx"] = 0
        S["score"] = 0
        S["answered"] = {}
        S["submitted"] = {}
        S["corpus"] = corpus
        S["show_image"] = {}
        st.success("Quiz ready! Scroll down to answer.")

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