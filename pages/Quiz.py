# pages/Quiz.py

from __future__ import annotations
import os
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import ollama

# --- Local Module Imports ---
from modules.functions import (
    OllamaEmbeddings, 
    OpenAIEmbeddings, 
    extract_text_from_file,
    get_relevant_rag, 
    rebuild_rag_index
)
from modules.login import check_authentication, login_page, logout
from modules.db_manager import DBManager 

from modules.quizstats.subject_store import (
    init_subject_db,
    get_subject_meta,
    save_subject_meta,
    ensure_subject_folders,
)
from modules.quizstats.file_utils import (
    save_uploaded_files,
)
from modules.quizstats.quiz_engine import build_quiz_from_rag
from modules.quizstats.progress_store import (
    init_stats_db,
    record_attempt,
)

# ---------- CONFIG & PATHS ----------
st.set_page_config(page_title="Quiz Builder", page_icon="📝", layout="wide")
load_dotenv()

BASE_DIR = Path(__file__).parents[1]
CHAT_DB_FILE = str(BASE_DIR / "chat_playground.db")
SUBJECTS_DIR = str(BASE_DIR / "modules" / "subjects")

os.makedirs(SUBJECTS_DIR, exist_ok=True)

# Initialize DBManager
db = DBManager(
    backend=st.session_state.get("chat_backend", "sqlite"),
    db_file=CHAT_DB_FILE,
)

# ---------- AUTHENTICATION ----------
is_authenticated, username = check_authentication()
if not is_authenticated:
    login_page()
    st.stop()

# ---------- SESSION DEFAULTS ----------
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = {
        "selected_subject": st.session_state.get("current_subject", "General"),
        "quiz": [],
        "current_idx": 0,
        "score": 0,
        "submitted": {},
    }

S = st.session_state.quiz_state
init_stats_db()
init_subject_db()
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
    st.warning("⚠️ No supported models found.")
    st.stop()

# Model Selector
m_idx = 0
if st.session_state.get("selected_model") in available_models:
    m_idx = available_models.index(st.session_state.selected_model)

selected_friendly = st.selectbox("Model Selection", display_models, index=m_idx)
st.session_state.selected_model = available_models[display_models.index(selected_friendly)]

# --- Determine Index Suffix based on selection ---
is_openai = "gpt" in st.session_state.selected_model.lower()
suffix = "openai" if is_openai else "ollama"

# ---------- SUBJECT SELECTION ----------
subject_names = db.load_all_subjects()
current_default_sub = st.session_state.get("current_subject", subject_names[0] if subject_names else "General")

selected_subject = st.selectbox(
    "📚 Select Subject for Quiz",
    subject_names,
    index=subject_names.index(current_default_sub) if current_default_sub in subject_names else 0
)

if selected_subject != S["selected_subject"]:
    S["selected_subject"] = selected_subject
    st.session_state.current_subject = selected_subject
    S["quiz"] = []
    st.rerun()

# --- CRITICAL PATH CHECK (SUBJECT FOLDER) ---
subject_folder = os.path.join(SUBJECTS_DIR, S["selected_subject"])
index_path = os.path.join(subject_folder, f"index_{suffix}.faiss")

if os.path.exists(index_path):
    st.success(f"✅ {suffix.upper()} Index found for '{S['selected_subject']}'. Ready to generate.")
else:
    st.error(f"⚠️ {suffix.upper()} Index missing in: modules/subjects/{S['selected_subject']}/")
    st.caption(f"Looking for: index_{suffix}.faiss")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(f"**User: {username}**")
    if st.button("Logout"): logout(); st.rerun()
    st.markdown("---")
    st.info(f"Active Subject: {S['selected_subject']}")
    st.caption(f"Mode: {suffix.upper()}")

# ---------- FILE PROCESSING ----------
with st.expander(f"📤 Knowledge Base: {S['selected_subject']}"):
    uploaded = st.file_uploader(
        "Add files", 
        type=["txt", "pdf", "docx", "pptx", "csv"], 
        accept_multiple_files=True
    )
    
    if st.button("📂 Process & Sync RAG"):
        target_sub = S["selected_subject"]
        
        with st.spinner(f"Synchronizing {target_sub} for {suffix.upper()}..."):
            # 1. Create folders if they don't exist
            ensure_subject_folders(target_sub)
            
            # 2. Save and Extract text
            if uploaded:
                save_uploaded_files(target_sub, uploaded)
                for f in uploaded:
                    # Note: You might need to import extract_text_from_file from modules.functions
                    from modules.functions import extract_text_from_file
                    content = extract_text_from_file(f)
                    db.add_rag_doc(target_sub, f.name, content)
            
            # 3. Build the index DIRECTLY in the subject folder
            # We pass the 'suffix' so rebuild_rag_index knows which file to create
            rebuild_rag_index(
                CHAT_DB_FILE, 
                subject_folder, # modules/subjects/Dava/
                target_sub, 
                st.session_state.selected_model, 
                ollama_models, 
                openai_models,
                suffix=suffix
            )
            
            st.success(f"✅ {suffix.upper()} Index Built Successfully!")
            st.rerun()
            
# ---------- QUIZ LOGIC ----------
if st.button("🎲 Generate 10 Questions"):
    if not os.path.exists(index_path):
        st.error(f"Missing {suffix.upper()} index. Please Process files first.")
    else:
        with st.spinner(f"🧠 Querying {suffix.upper()} Knowledge..."):
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
                st.error("LLM failed to generate questions.")

# ---------- QUIZ PLAYER ----------
if S["quiz"]:
    st.divider()
    q_idx = S["current_idx"]
    q = S["quiz"][q_idx]

    st.subheader(f"Question {q_idx + 1}")

    # ---------- SHOW RELEVANT CONTEXT ----------
    with st.expander(f"🔍 Show Relevant Context (Question {q_idx + 1})"):
        from modules.quizstats.file_utils import render_pdf_page_image, render_pptx_slide_image
        
        meta = q.get("context_meta", {})
        f_name = meta.get("file") or meta.get("source") or ""
        f_type = meta.get("type")
        
        # Path reconstruction: Search the 'raw' folder of the current subject
        f_path = os.path.join(SUBJECTS_DIR, S["selected_subject"], "raw", f_name)

        if f_name and os.path.exists(f_path):
            if f_type == "pptx_slide":
                slide_num = meta.get("slide", 1)
                img = render_pptx_slide_image(f_path, slide_num)
                if img:
                    # Use a container to hold the image
                    img_container = st.container()
                    
                    unique_caption = f"Source: {f_name} | Slide {slide_num} (Q:{q_idx + 1})"
                    
                    img_container.image(
                        img, 
                        caption=unique_caption, 
                        use_container_width=True
                    )
                else:
                    st.warning("Slide image could not be rendered.")
            
            elif f_type == "pdf_page":
                page_num = meta.get("page", 1)
                img = render_pdf_page_image(f_path, page_num)
                if img:
                    st.image(img, caption=f"Page {page_num} from {f_name}", use_container_width=True, key=f"img_q{q_idx}_s{page_num}")
            
            else:
                st.info(f"Source: {f_name}")
                st.write(q.get("explanation", "Refer to the source document."))
        else:
            st.error("Original source file not found.")
            st.write(q.get("explanation", "No explanation provided."))
    st.markdown(q["question"])
    
    picked = st.radio("Choose one:", q["choices"], index=None, key=f"q_{q_idx}")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Submit"):
            if picked:
                is_correct = (q["choices"].index(picked) == q["answer_idx"])
                S["submitted"][q_idx] = True
                if is_correct:
                    S["score"] += 1
                    st.success("Correct!")
                else:
                    st.error(f"Incorrect. Correct answer: {q['choices'][q['answer_idx']]}")
                
                record_attempt(
                    subject=S["selected_subject"], question=q["question"],
                    chosen=picked, correct=q["choices"][q["answer_idx"]],
                    is_correct=1 if is_correct else 0, topic=q.get("topic", "General")
                )
            else:
                st.warning("Please select an option.")

    with col2:
        if st.button("Next") and S["submitted"].get(q_idx):
            if S["current_idx"] < len(S["quiz"]) - 1:
                S["current_idx"] += 1
                st.rerun()
    
    with col3:
        st.metric("Progress", f"{S['current_idx'] + 1} / {len(S['quiz'])}", f"Score: {S['score']}")