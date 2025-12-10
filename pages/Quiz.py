# pages/quiz.py

import os
import time

import streamlit as st
from dotenv import load_dotenv

from modules.quizstats.subject_store import (
    init_subject_db,
    list_subjects,
    create_subject_if_missing,
    get_subject_meta,
    save_subject_meta,
    ensure_subject_folders,
    delete_subject,          # 🔹 NEW
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
    clear_subject_stats,     # 🔹 NEW
)

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Quiz Builder", page_icon="📝", layout="wide")
load_dotenv()

# ---------- SESSION ----------
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {
        "provider": "none",
        "model_name": "",
        "use_env_key": True,
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "selected_subject": None,
        "new_subject_name": "",
        "uploads_buffer": [],
        "quiz": [],
        "current_idx": 0,
        "score": 0,
        "answered": {},   # q_idx -> {"picked": int, "correct": bool, "already_counted": bool}
        "submitted": {},  # q_idx -> bool
        "corpus": "",
        "show_image": {},  # q_idx -> bool (user clicked 'show context')
    }

S = st.session_state.quiz_state

# make sure stats DB exists
init_stats_db()

# ---------- HEADER ----------
st.title("📝 Quiz Builder (Subjects + Files + 10 MCQs)")

# ---------- MODEL / PROVIDER BOX ----------
with st.expander("⚙️ Model / Provider (UI only for now)"):

    S["provider"] = st.radio(
        "Choose provider",
        ["none", "ollama (local)", "openai (API)"],
        index=["none", "ollama (local)", "openai (API)"].index(S["provider"]),
    )

    cols = st.columns(2)
    with cols[0]:
        S["model_name"] = st.text_input(
            "Model name",
            value=S["model_name"],
            placeholder="e.g., llama3:latest or gpt-4o-mini",
        )
    with cols[1]:
        S["use_env_key"] = st.checkbox("Use OPENAI_API_KEY from .env", value=True)
        if S["provider"] == "openai (API)" and not S["use_env_key"]:
            S["openai_api_key"] = st.text_input(
                "OpenAI API Key", value=S["openai_api_key"], type="password"
            )

    st.caption(
        "This page uses a built-in quiz generator that calls GPT-4o-mini when an "
        "OPENAI_API_KEY is set. The provider settings are kept for future integration."
    )

st.markdown("---")

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
    else:
        saved = save_uploaded_files(S["selected_subject"], S["uploads_buffer"])
        S["uploads_buffer"] = []
        corpus = extract_corpus_for_subject(S["selected_subject"])
        meta = get_subject_meta(S["selected_subject"]) or {}
        meta["last_indexed_ms"] = int(time.time() * 1000)
        meta["char_count"] = len(corpus)
        save_subject_meta(S["selected_subject"], meta)
        st.success(
            f"Processed {len(saved)} file(s). Indexed {len(corpus):,} characters."
        )

# ---------- BUILD / START QUIZ ----------
st.markdown("### 🧠 Generate Quiz (10 MCQs)")
if st.button("🎲 Generate 10 Questions"):
    corpus = extract_corpus_for_subject(S["selected_subject"])
    if not corpus.strip():
        st.warning("No text in this subject yet. Please upload/process files first.")
    else:
        quiz = build_quiz_from_corpus(
            corpus_text=corpus,
            subject=S["selected_subject"],
            n_questions=10,
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
