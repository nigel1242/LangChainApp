# pages/Quiz.py

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import ollama

# --- Local Module Imports ---
from modules.login import check_authentication, login_page, logout

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

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Quiz Builder", page_icon="📝", layout="wide")
load_dotenv()

# ---------- AUTHENTICATION ----------
is_authenticated, username = check_authentication()
if not is_authenticated:
    login_page()
    st.stop()

# ---------- SESSION DEFAULTS ----------
defaults = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "selected_model": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

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

# ensure stats DB exists
init_stats_db()
init_subject_db()


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(f"**Logged in as: {username}**")
    if st.button("Logout"):
        logout()
        st.rerun()
    st.markdown("---")

    st.header("🔑 API Keys")
    with st.expander("Manage Keys"):
        if st.session_state.openai_api_key:
            st.success("OpenAI API Key loaded")
            override_key = st.text_input(
                "Override OpenAI API Key (optional)",
                type="password",
                placeholder="Leave empty to use existing key",
                key="quiz_openai_override",
            )
            if override_key:
                st.session_state.openai_api_key = override_key.strip()
                os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
                st.success("Using custom API key")
        else:
            api_key_input = st.text_input(
                "Enter your OpenAI API Key",
                type="password",
                key="quiz_openai_input",
            )
            if api_key_input:
                st.session_state.openai_api_key = api_key_input.strip()
                os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
                st.success("API key saved")
                st.rerun()

# ---------- HEADER ----------
st.title("📝 Quiz Builder (Subjects + Files + 10 MCQs)")
st.caption("Upload files → Process → Generate 10 Questions → Answer → Stats records per question.")


# >>> START: MODEL SELECTOR (MAIN BODY) <<<
def get_model_display_name(model_id: str) -> str:
    if model_id == "gpt-3.5-turbo":
        return "OpenAI GPT-3.5 Turbo"
    if model_id == "gpt-4":
        return "OpenAI GPT-4"
    if model_id == "gpt-4o-mini":
        return "OpenAI GPT-4o Mini"
    if model_id == "gpt-4o":
        return "OpenAI GPT-4o"
    if model_id == "llama3:8b":
        return "Ollama Llama 3 (8B)"
    if model_id == "qwen2.5vl:7b":
        return "Ollama Qwen 2.5 VL (7B)"
    return model_id.replace("-", " ").title()


try:
    raw_ollama_models = (m["model"] for m in ollama.list().get("models", []))
    ollama_models = tuple(m for m in raw_ollama_models if not m.startswith("nomic-embed-text"))
except Exception:
    ollama_models = ()

openai_models = ()
if st.session_state.get("openai_api_key"):
    # add whatever you want available
    openai_models = ("gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo")

available_models = ollama_models + openai_models
display_models = [get_model_display_name(m) for m in available_models]

if not available_models:
    st.warning("⚠️ No models available. Check Ollama server or OpenAI key.")
    st.stop()

if st.session_state.selected_model in available_models:
    default_index = available_models.index(st.session_state.selected_model)
else:
    st.session_state.selected_model = available_models[0]
    default_index = 0

selected_display_name = st.selectbox(
    "🧠 Choose model for Quiz Generation",
    display_models,
    index=default_index,
    key="quiz_model_selector_main",
)

selected_index = display_models.index(selected_display_name)
selected_model_id = available_models[selected_index]
st.session_state.selected_model = selected_model_id

st.markdown("---")
# >>> END: MODEL SELECTOR <<<


# ---------- SUBJECTS: CREATE / SELECT ----------
left, right = st.columns([1, 1])

with left:
    st.subheader("📚 Subjects")

    subjects = list_subjects()
    subject_names = [s["name"] for s in subjects]

    PLACEHOLDER = "— select —"
    existing_names = [PLACEHOLDER] + subject_names

    default_label = S["selected_subject"] if S["selected_subject"] in subject_names else PLACEHOLDER
    default_index = existing_names.index(default_label)

    selected_label = st.selectbox(
        "Choose existing subject",
        existing_names,
        index=default_index,
        key="subject_selector_main",
    )

    S["selected_subject"] = None if selected_label == PLACEHOLDER else selected_label

    if S["selected_subject"]:
        st.markdown("### Delete Subject")
        if st.button(
            f"🗑️ Delete subject '{S['selected_subject']}'",
            type="secondary",
            key="delete_subject_btn",
        ):
            try:
                clear_subject_stats(S["selected_subject"])
            except Exception as e:
                st.warning(f"Could not clear stats for subject: {e}")

            ok, msg = delete_subject(S["selected_subject"])
            if ok:
                st.success(msg + " All quiz stats for this subject were cleared.")
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
            st.rerun()
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
        with st.spinner("Processing files and rebuilding index (page_index.json)..."):
            saved = save_uploaded_files(S["selected_subject"], S["uploads_buffer"])
            S["uploads_buffer"] = []

            corpus = extract_corpus_for_subject(S["selected_subject"])

            meta = get_subject_meta(S["selected_subject"]) or {}
            meta["last_indexed_ms"] = int(time.time() * 1000)
            meta["char_count"] = len(corpus)
            save_subject_meta(S["selected_subject"], meta)

            st.success(f"Processed {len(saved)} file(s). Indexed {len(corpus):,} characters.")

        st.rerun()


# ---------- BUILD / START QUIZ ----------
st.markdown("### 🧠 Generate Quiz (10 MCQs)")

if st.button("🎲 Generate 10 Questions"):
    corpus = extract_corpus_for_subject(S["selected_subject"])
    if not corpus.strip():
        st.warning("No text in this subject yet. Please upload/process files first.")
    else:
        if not st.session_state.selected_model:
            st.error("Please select a model above first.")
            st.stop()

        quiz = build_quiz_from_corpus(
            corpus_text=corpus,
            subject=S["selected_subject"],
            n_questions=10,  # ✅ ALWAYS 10
            model_name=st.session_state.selected_model,
        )

        if not quiz:
            st.error("Could not generate quiz questions. Try another model or re-process files.")
            st.stop()

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

    # helper: detect likely visual question
    def _question_needs_visual(qdict):
        text = ((qdict.get("question", "") + " " + " ".join(qdict.get("choices", []))).lower())
        visual_words = [
            "graph", "plot", "chart", "bar chart", "line chart",
            "histogram", "scatter", "table", "diagram", "figure",
            "workflow", "curve", "regression",
        ]
        return qdict.get("needs_image", False) or any(w in text for w in visual_words)

    needs_visual = _question_needs_visual(q)

    st.markdown("#### 🔍 Need context?")
    show_now = S["show_image"].get(q_idx, False)
    if st.button("👁 Show related context", key=f"show_img_{q_idx}"):
        show_now = True
        S["show_image"][q_idx] = True

    if show_now:
        ctx = q.get("context_meta") or {}
        entry = None

        if ctx.get("file_path") and ctx.get("type") in ("pdf_page", "pptx_slide", "pptx_slide", "docx_chunk", "text_file", "csv_file"):
            entry = ctx
        else:
            img_query = q.get("image_hint") or q.get("topic") or q.get("question", "")
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
                        caption += " – likely image-based."
                    st.image(img, caption=caption, use_container_width=True)
                else:
                    st.caption(f"Open PDF: `{pdf_path}` (page {page_num})")

            elif etype == "pptx_slide":
                pptx_path = entry.get("file_path")
                slide_num = entry.get("slide", 1)
                img = render_pptx_slide_image(pptx_path, slide_num)
                if img is not None:
                    caption = f"Context from {fname} (PPTX slide {slide_num})"
                    if needs_visual:
                        caption += " – likely image-based."
                    st.image(img, caption=caption, use_container_width=True)
                else:
                    st.caption(f"Context from {fname} (slide {slide_num}) – unable to render image.")

            else:
                full_text = entry.get("text", "")
                snippet = full_text[:800] + ("..." if len(full_text) > 800 else "")
                st.markdown(f"**Context from {fname}**")
                st.write(snippet)
        else:
            st.caption("No matching context found for this question.")

    # Question header
    st.subheader(f"Question {S['current_idx'] + 1} of {len(S['quiz'])}")
    difficulty = q.get("difficulty")
    if difficulty:
        st.caption(f"Difficulty: {difficulty}")

    # ✅ Allow LaTeX
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
                    "already_counted": S["answered"].get(q_idx, {}).get("already_counted", False),
                }
                S["submitted"] = S.get("submitted", {})
                S["submitted"][q_idx] = True

                if is_correct:
                    if not S["answered"][q_idx]["already_counted"]:
                        S["score"] += 1
                        S["answered"][q_idx]["already_counted"] = True
                    st.success("✅ Correct! Nice job.")
                else:
                    st.error("❌ Incorrect.")
                    st.caption(f"Answer: {q['choices'][correct_idx]}")

                explanation = q.get("explanation", "")
                if explanation:
                    st.markdown(explanation)  # ✅ LaTeX ok

                # ----- RECORD STATS -----
                try:
                    ts_ms = int(time.time() * 1000)
                    chosen_text = q["choices"][picked_idx]
                    correct_text = q["choices"][correct_idx]

                    # ✅ Topic: guaranteed non-empty
                    topic = (q.get("topic") or "").strip()
                    if not topic:
                        topic = (q.get("image_hint") or "").strip()
                    if not topic:
                        ctx = q.get("context_meta") or {}
                        fname = (ctx.get("file") or "").strip()
                        if fname:
                            topic = fname
                    if not topic:
                        topic = "General"

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
                    st.rerun()
                else:
                    st.info("That was the last question.")

    # ---------- SCORE ----------
    with col3:
        st.metric("Score", f"{S['score']} / {len(S['quiz'])}")

else:
    st.info("Click **Generate 10 Questions** to start.")
