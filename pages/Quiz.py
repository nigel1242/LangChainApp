from __future__ import annotations
import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import ollama

# --- Local Module Imports ---
from modules.db_manager import DBManager 
from modules.functions import (
    OllamaEmbeddings, 
    OpenAIEmbeddings, 
    extract_text_from_file,
    get_relevant_rag, 
    rebuild_rag_index
)
from modules.login import check_authentication, login_page, logout
from modules.quizstats.subject_store import ensure_subject_folders
from modules.quizstats.file_utils import save_uploaded_files, render_pdf_page_image
from modules.quizstats.quiz_engine import build_quiz_from_rag
from modules.quizstats.progress_store import init_stats_db, record_attempt

# ---------- CONFIG & PATHS ----------
st.set_page_config(page_title="Quiz Builder", page_icon="📝", layout="wide")
load_dotenv()

BASE_DIR = Path(__file__).parents[1]
CHAT_DB_FILE = str(BASE_DIR / "chat_playground.db")
SUBJECTS_DIR = os.path.join("modules", "subjects") 

os.makedirs(SUBJECTS_DIR, exist_ok=True)

# Initialize DBManager
db = DBManager(
    backend="sqlite",
    db_file=CHAT_DB_FILE,
)

# ---------- AUTHENTICATION ----------
is_authenticated, username = check_authentication()
if not is_authenticated:
    login_page()
    st.stop()

# ---------- SESSION DEFAULTS ----------
if "quiz_state" not in st.session_state:
    # Get all subjects and filter out 'General'
    all_subs = db.load_all_subjects()
    available_subs = [s for s in all_subs if s != "General"]
    
    st.session_state.quiz_state = {
        # Default to the first non-General subject, or None if empty
        "selected_subject": available_subs[0] if available_subs else None,
        "quiz": [],
        "current_idx": 0,
        "score": 0,
        "submitted": {}, 
    }

S = st.session_state.quiz_state
init_stats_db()
st.title("📖 Quiz Builder")

# ---------- MODELS ----------
MODEL_MAP = {
    "llama3:8b": "Llama 3 (Base Model)",
    "qwen2.5vl:7b": "Qwen 2.5 VL (Vision Model)",
    "gpt-4o": "OpenAI GPT-4o",
}
ALLOWED_MODEL_IDS = list(MODEL_MAP.keys())

try:
    raw_ollama = [m["model"] for m in ollama.list().get("models", [])]
    ollama_models = tuple(m for m in raw_ollama if m in ALLOWED_MODEL_IDS)
except:
    ollama_models = ()

openai_models = ("gpt-4o",) if st.session_state.get("openai_api_key") else ()
available_models = ollama_models + openai_models
display_models = [MODEL_MAP.get(m, m) for m in available_models]

if not available_models:
    st.warning("⚠️ No supported models found. Please visit settings page.")
    st.stop()

# Model Selector
m_idx = 0
if st.session_state.get("selected_model") in available_models:
    m_idx = available_models.index(st.session_state.selected_model)

selected_friendly = st.selectbox("Model Selection", display_models, index=m_idx)
st.session_state.selected_model = available_models[display_models.index(selected_friendly)]

is_openai = "gpt" in st.session_state.selected_model.lower()
suffix = "openai" if is_openai else "ollama"

# ---------- SUBJECT SELECTION ----------
subject_names = [s for s in db.load_all_subjects() if s != "General"]

if not subject_names:
    st.warning("⚠️ No subjects found. Please create a subject in the main Chat App first.")
    st.stop()

# Ensure the session state index is valid for the filtered list
try:
    default_idx = subject_names.index(S["selected_subject"])
except (ValueError, KeyError):
    default_idx = 0

selected_subject = st.selectbox(
    "📚 Select Subject for Quiz",
    subject_names,
    index=default_idx
)

if selected_subject != S["selected_subject"]:
    S["selected_subject"] = selected_subject
    st.session_state.current_subject = selected_subject
    S["quiz"] = []
    st.rerun()

# Index Path Check
subject_folder = os.path.join(SUBJECTS_DIR, S["selected_subject"])
index_path = os.path.join(subject_folder, f"index_{suffix}.faiss")

if os.path.exists(index_path):
    st.success(f"✅ {suffix.upper()} Index found. Ready to generate.")
else:
    st.error(f"⚠️ Index missing. Please upload files and click 'Process RAG' below.")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(f"**User: {username}**")
    if st.button("Logout"): logout(); st.rerun()
    st.markdown("---")

# ---------- FILE PROCESSING (PDF-FIRST) ----------
with st.expander(f"📤 Knowledge Base: {S['selected_subject']}"):
    uploaded = st.file_uploader(
        "Add files", 
        type=["txt", "pdf", "docx", "pptx", "csv"], 
        accept_multiple_files=True
    )
    
    if st.button("📂 Process RAG"):
        target_sub = S["selected_subject"]
        subject_path = os.path.join(SUBJECTS_DIR, target_sub)
        
        with st.spinner(f"Processing and converting for {suffix.upper()}..."):
            ensure_subject_folders(target_sub)
            
            if uploaded:
                save_uploaded_files(target_sub, uploaded)
                
                for f in uploaded:
                    f.seek(0) 
                    orig_path = os.path.join(SUBJECTS_DIR, target_sub, "raw", f.name)
                    ext = f.name.split('.')[-1].lower()
                    final_path = orig_path

                    if ext in ["pptx", "docx"]:
                        from modules.quizstats.file_utils import convert_to_pdf
                        pdf_path = convert_to_pdf(orig_path)
                        if pdf_path and pdf_path != orig_path:
                            final_path = pdf_path
                            try:
                                os.remove(orig_path)
                            except Exception as e:
                                print(f"Cleanup error for {f.name}: {e}")
                    
                    from modules.quizstats.file_utils import extract_text_from_path
                    content = extract_text_from_path(final_path)
                    
                    final_filename = os.path.basename(final_path)
                    metadata = {
                        "file": final_filename,
                        "type": "pdf_page",
                        "subject": target_sub
                    }
                    db.add_rag_doc(target_sub, final_filename, content, metadata=metadata)
            
            rebuild_rag_index(
                CHAT_DB_FILE, 
                subject_folder, 
                target_sub, 
                st.session_state.selected_model, 
                ollama_models, 
                openai_models,
                suffix=suffix
            )
            
            st.success("✅ Synchronization Complete! Original Office files removed.")
            st.rerun()
            
# ---------- QUIZ GENERATION ----------
if st.button("🎲 Generate 10 Questions"):
    if not os.path.exists(index_path):
        st.error("Please Process & Sync files first.")
    else:
        with st.spinner("🧠 Analyzing Knowledge Base..."):
            quiz = build_quiz_from_rag(
                subject=S["selected_subject"],
                model_name=st.session_state.selected_model,
                n_questions=10,
                vector_db="faiss"
            )
            if quiz:
                S.update({"quiz": quiz, "current_idx": 0, "score": 0, "submitted": {}})
                st.rerun()
            else:
                st.error("Failed to generate questions. Try a different model.")

# ---------- QUIZ PLAYER ----------
if S["quiz"]:
    st.divider()
    
    # Check if we have finished all questions
    if S["current_idx"] >= len(S["quiz"]):
        # --- FINISHED PAGE ---
        st.success("🎉 Quiz Finished!")
        st.header(f"Final Score: {S['score']} / {len(S['quiz'])}")
    else:
        # --- ACTIVE QUIZ ---
        q_idx = S["current_idx"]
        q = S["quiz"][q_idx]
        has_submitted = S["submitted"].get(q_idx, False)

        st.subheader(f"Question {q_idx + 1}")

        with st.expander(f"🔍 Show Relevant Context", expanded=False):
            meta = q.get("context_meta", {})
            f_name = meta.get("file")
            page_num = meta.get("page", 1)
            pdf_path = os.path.join(SUBJECTS_DIR, S["selected_subject"], "raw", f_name)

            if f_name and os.path.exists(pdf_path):
                img = render_pdf_page_image(pdf_path, page_num)
                if img:
                    st.image(img, caption=f"Source: {f_name} | Page {page_num}", use_container_width=True)
                else:
                    st.warning("Rendering failed for this page.")
            else:
                st.error("Original source PDF not found.")
                st.write(q.get("explanation", "No additional context available."))

        st.markdown(f"### {q['question']}")
        
        picked = st.radio(
            "Choose one:", 
            q["choices"], 
            index=None if not has_submitted else q["choices"].index(st.session_state.get(f"q_{q_idx}")),
            key=f"q_{q_idx}",
            disabled=has_submitted
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Submit", disabled=has_submitted):
                if picked:
                    is_correct = (q["choices"].index(picked) == q["answer_idx"])
                    S["submitted"][q_idx] = True
                    if is_correct:
                        S["score"] += 1
                    
                    record_attempt(
                        subject=S["selected_subject"], 
                        question=q["question"],
                        chosen=picked, 
                        correct=q["choices"][q["answer_idx"]],
                        is_correct=1 if is_correct else 0, 
                        topic=q.get("topic", "General")
                    )
                    st.rerun()
                else:
                    st.warning("Please select an option.")

        if has_submitted:
            is_correct = (q["choices"].index(st.session_state.get(f"q_{q_idx}")) == q["answer_idx"])
            if is_correct:
                st.success("✨ Correct!")
            else:
                st.error(f"❌ Incorrect. Correct answer: {q['choices'][q['answer_idx']]}")

        with col2:
            # Change label to 'Finish' on the last question
            label = "Finish" if q_idx == len(S["quiz"]) - 1 else "Next"
            if st.button(label) and has_submitted:
                S["current_idx"] += 1
                st.rerun()
        
        with col3:
            st.metric("Progress", f"{q_idx + 1} / {len(S['quiz'])}", f"Score: {S['score']}")