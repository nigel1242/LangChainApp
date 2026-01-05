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
    extract_corpus_for_subject,
    find_relevant_context_for_text,
    render_pdf_page_image,
    render_pptx_slide_image,
)
from modules.quizstats.quiz_engine import build_quiz_from_rag
from modules.quizstats.progress_store import (
    init_stats_db,
    record_attempt,
)

# ---------- CONFIG & PATHS ----------
st.set_page_config(page_title="Quiz Builder", page_icon="📝", layout="wide")
load_dotenv()

# Use absolute paths to prevent "Index Not Found" errors across different pages
BASE_DIR = Path(__file__).parents[1]
CHAT_DB_FILE = str(BASE_DIR / "chat_playground.db")
RAG_INDEX_DIR = str(BASE_DIR / "rag_indices")
SUBJECTS_DIR = str(BASE_DIR / "modules" / "subjects")

os.makedirs(RAG_INDEX_DIR, exist_ok=True)
os.makedirs(SUBJECTS_DIR, exist_ok=True)

# Initialize DBManager
db = DBManager(
    backend=st.session_state.get("chat_backend", "sqlite"),
    db_file=CHAT_DB_FILE,
    qdrant_url=st.session_state.get("qdrant_url"),
    qdrant_api_key=st.session_state.get("qdrant_api_key"),
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
        "uploads_buffer": [],
        "quiz": [],
        "current_idx": 0,
        "score": 0,
        "answered": {},
        "submitted": {},
        "corpus": "",
        "show_image": {},
    }

# Sync local alias
S = st.session_state.quiz_state
init_stats_db()
init_subject_db()
st.title("📖 Quiz Builder")
# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(f"**User: {username}**")
    if st.button("Logout"): logout(); st.rerun()
    st.markdown("---")
    
    # Global sync with app.py current subject
    st.info(f"Active Subject: {st.session_state.get('current_subject', 'General')}")

# ---------- MODELS ----------
MODEL_MAP = {
    "llama3:8b": "Llama 3 (Base Model)",
    "qwen2.5vl:7b": "Qwen 2.5 VL (Vision Model)",
    "gpt-4o": "OpenAI GPT-4o",
}
ALLOWED_MODEL_IDS = list(MODEL_MAP.keys())

try:
    raw_ollama = [m["model"] for m in ollama.list().get("models", [])]
    # Filter: only show the ones in our allowed list
    ollama_models = tuple(m for m in raw_ollama if m in ALLOWED_MODEL_IDS)
except:
    ollama_models = ()

openai_models = ("gpt-4o",) if st.session_state.get("openai_api_key") else ()
# We use this variable name consistently now
available_models = ollama_models + openai_models

# Create display names based on the map
display_models = [MODEL_MAP.get(m, m) for m in available_models]

if not available_models:
    st.warning("⚠️ No supported models found. Please check your Ollama models or OpenAI API key.")
    st.stop()

# Ensure we have a valid selected_model in session state
if st.session_state.get("selected_model") not in available_models:
    st.session_state.selected_model = available_models[0]

# Determine current selection index
m_idx = available_models.index(st.session_state.selected_model)

selected_friendly = st.selectbox(
    "Model Selection", 
    display_models, 
    index=m_idx
)

# Update the technical model ID in session state
st.session_state.selected_model = available_models[display_models.index(selected_friendly)]

# ---------- SUBJECT SELECTION ----------
subject_names = db.load_all_subjects()
if not subject_names:
    st.info("No subjects found. Please create one in the main Chat page.")
    st.stop()

# Sync selection with what might have been picked in Chat
current_default_sub = st.session_state.get("current_subject", subject_names[0])

selected_subject = st.selectbox(
    "📚 Select Subject for Quiz",
    subject_names,
    index=subject_names.index(current_default_sub) if current_default_sub in subject_names else 0
)

# Update state if changed
if selected_subject != S["selected_subject"]:
    S["selected_subject"] = selected_subject
    st.session_state.current_subject = selected_subject
    S["quiz"] = [] # Clear old quiz when changing subject
    st.rerun()

# --- CRITICAL PATH CHECK ---
# This ensures Quiz.py looks at EXACTLY the same folder as app.py
folder_id = f"subject_{S['selected_subject']}"
index_path = os.path.join(RAG_INDEX_DIR, folder_id, "index.faiss")

if os.path.exists(index_path):
    st.success(f"✅ RAG Index found for '{S['selected_subject']}'. Ready to generate.")
else:
    st.error(f"⚠️ No index found at: {index_path}")
    st.caption("Go to Chat page to process files, or use the uploader below.")

# ---------- FILE PROCESSING (SYNCED) ----------
with st.expander(f"📤 Knowledge Base: {st.session_state.current_subject}"):
        uploaded = st.file_uploader("Add files", type=["txt", "pdf", "docx", "pptx", "csv"], accept_multiple_files=True)
        
        if uploaded:
            st.session_state.uploaded_files_to_process = uploaded

        if st.button("📂 Process & Sync RAG"):
            target_sub = st.session_state.current_subject
            
            with st.spinner(f"Synchronizing {target_sub}..."):
                # 1. Physical Storage (modules/subjects/<target_sub>/raw)
                ensure_subject_folders(target_sub)
                save_uploaded_files(target_sub, st.session_state.uploaded_files_to_process)
                
                # 2. Extraction & Vector Indexing
                for f in st.session_state.uploaded_files_to_process:
                    content = extract_text_from_file(f)
                    
                    if st.session_state.vector_db == "qdrant":
                        from modules.functions import RecursiveCharacterTextSplitter
                        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = splitter.split_text(content)
                        
                        # Embeddings setup based on technical model ID
                        if "gpt" in st.session_state.selected_model.lower():
                            embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_api_key)
                        else:
                            embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")

                        with st.spinner(f"Uploading {f.name} to Qdrant subject: {target_sub}..."):
                            for i, chunk in enumerate(chunks):
                                vector_data = embeddings.embed_documents([chunk])[0]
                                # ✅ target_id is now the Subject Name, ensuring shared access
                                db.add_rag_doc(
                                    target_id=target_sub, 
                                    file_name=f.name,
                                    content=chunk,
                                    vector_data=vector_data
                                )
                    else:
                        if not db.file_exists_in_rag(target_sub, f.name):
                            db.add_rag_doc(target_sub, f.name, content)

                # 3. Rebuild Local Index
                if st.session_state.vector_db == "faiss":
                    rebuild_rag_index(CHAT_DB_FILE, RAG_INDEX_DIR, target_sub, st.session_state.selected_model, ollama_models, openai_models)
                
                st.success(f"✅ Knowledge base for '{target_sub}' updated!")
                st.session_state.uploaded_files_to_process = []
                st.rerun()

# ---------- QUIZ LOGIC ----------
if st.button("🎲 Generate 10 Questions"):
    if not os.path.exists(index_path):
        st.error("Cannot generate quiz: RAG Index is missing.")
    else:
        with st.spinner("🧠 Querying Knowledge Base..."):
            quiz = build_quiz_from_rag(
                subject=S["selected_subject"],
                model_name=st.session_state.selected_model,
                n_questions=10,
                vector_db=st.session_state.get("vector_db", "faiss")
            )
            
            if quiz:
                S.update({
                    "quiz": quiz, "current_idx": 0, "score": 0, 
                    "answered": {}, "submitted": {}, "corpus": "RAG-Retrieved"
                })
                st.rerun()
            else:
                st.error("The LLM failed to generate questions from the retrieved context.")

# ---------- QUIZ PLAYER ----------
if S["quiz"]:
    st.divider()
    q_idx = S["current_idx"]
    q = S["quiz"][q_idx]

    st.subheader(f"Question {q_idx + 1}")
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