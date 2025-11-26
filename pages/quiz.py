import os
import time
import streamlit as st

from modules.quiz_db import (
    init_subject_db,
    list_subjects,
    create_subject_if_missing,
    get_subject_meta,
    save_subject_meta
)
from modules.file_utils import (
    ensure_subject_folders,
    save_uploaded_files,
    extract_corpus_for_subject,
    rebuild_rag_index,
    get_relevant_rag
)
from modules.quiz_engine_simple import build_quiz_from_corpus

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Quiz Builder with RAG", page_icon="📝", layout="wide")

# ---------- SESSION STATE ----------
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {
        "selected_subject": None,
        "new_subject_name": "",
        "uploads_buffer": [],
        "quiz": [],
        "current_idx": 0,
        "score": 0,
        "answered": {},  # q_idx -> {"picked": int, "correct": bool}
        "corpus": "",
        "provider": "ollama",
        "model_name": "",
        "openai_api_key": "",
        "rag_enabled": True
    }

S = st.session_state.quiz_state

# ---------- HEADER ----------
st.title("📝 Quiz Builder with RAG (SQLite + 10 MCQs)")

# ---------- MODEL / PROVIDER SELECTION ----------
with st.expander("⚙️ Model / Provider"):
    S["provider"] = st.radio(
        "Provider", ["ollama", "openai"], index=["ollama","openai"].index(S["provider"])
    )

    if S["provider"] == "ollama":
        S["model_name"] = st.selectbox("Ollama Model", ["nomic-embed-text"], index=0)
    elif S["provider"] == "openai":
        S["model_name"] = st.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
        S["openai_api_key"] = st.text_input(
            "OpenAI API Key",
            value=S.get("openai_api_key",""),
            type="password"
        )

    S["rag_enabled"] = st.checkbox("💡 Enable RAG", value=True)
    
st.markdown("---")

# ---------- SUBJECTS ----------
init_subject_db()
left, right = st.columns([1, 1])

with left:
    st.subheader("📚 Subjects")
    subjects = list_subjects()
    existing_names = ["— select —"] + [s["name"] for s in subjects]
    sel = st.selectbox("Choose existing subject", existing_names)
    S["selected_subject"] = None if sel == "— select —" else sel

with right:
    st.subheader("➕ New Subject")
    S["new_subject_name"] = st.text_input("Create a new subject", value=S["new_subject_name"])
    if st.button("Create Subject"):
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

# ---------- FILE UPLOAD ----------
subject_root = ensure_subject_folders(S["selected_subject"])

st.markdown("### 📤 Upload Files (TXT, PDF)")
uploaded = st.file_uploader(
    "Drag and drop multiple files",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    key="quiz_uploader"
)
if uploaded:
    S["uploads_buffer"] = uploaded

if st.button("📦 Process Uploaded Files"):
    if not S["uploads_buffer"]:
        st.warning("No files selected.")
    else:
        save_uploaded_files(S["selected_subject"], S["uploads_buffer"])
        S["uploads_buffer"] = []
        corpus = extract_corpus_for_subject(S["selected_subject"])
        S["corpus"] = corpus

        # Update metadata
        meta = get_subject_meta(S["selected_subject"]) or {}
        meta["last_indexed_ms"] = int(time.time() * 1000)
        meta["char_count"] = len(corpus)
        save_subject_meta(S["selected_subject"], meta)

        # Rebuild RAG index
        if S["rag_enabled"]:
            rebuild_rag_index(
                subject_name=S["selected_subject"],
                model_name=S["model_name"],
                ollama_models=("nomic-embed-text",),
                openai_models=("gpt-3.5-turbo","gpt-4")
            )

        st.success("Files processed and corpus updated!")

# ---------- BUILD QUIZ ----------
if st.button("🧠 Build 10-Question Quiz"):
    corpus = extract_corpus_for_subject(S["selected_subject"])
    S["corpus"] = corpus
    S["quiz"] = build_quiz_from_corpus(corpus, S["selected_subject"], n_questions=10)
    S["current_idx"] = 0
    S["answered"] = {}
    S["score"] = 0
    st.success("Quiz generated!")

# ---------- SHOW QUIZ ----------
if S["quiz"]:
    q = S["quiz"][S["current_idx"]]
    st.subheader(f"Question {S['current_idx'] + 1} / {len(S['quiz'])}")
    st.write(q["question"])

    # --- Optional RAG hint ---
    if S["rag_enabled"]:
        rag_context, has_docs = get_relevant_rag(
            subject_name=S["selected_subject"],
            query=q["question"],
            model_name=S["model_name"],
            ollama_models=("nomic-embed-text",),
            openai_models=("gpt-3.5-turbo","gpt-4"),
            top_k=3
        )
        if has_docs:
            st.markdown(f"**💡 RAG Hint:** {rag_context[:500]}...")  # limit chars

    picked = st.radio(
        "Choose an answer:",
        list(range(len(q["choices"]))),
        format_func=lambda i: q["choices"][i],
        key=f"qchoice_{S['current_idx']}"
    )

    if st.button("Submit Answer"):
        correct = (picked == q["answer_idx"])
        S["answered"][S["current_idx"]] = {"picked": picked, "correct": correct}
        if correct:
            S["score"] += 1
        st.success("Correct!") if correct else st.error("Incorrect!")

    st.markdown("---")
    cols = st.columns([1, 1])

    if S["current_idx"] > 0:
        if cols[0].button("⬅ Previous"):
            S["current_idx"] -= 1

    if S["current_idx"] < len(S["quiz"]) - 1:
        if cols[1].button("Next ➡"):
            S["current_idx"] += 1
    else:
        st.info(f"Final Score: **{S['score']} / {len(S['quiz'])}**")
