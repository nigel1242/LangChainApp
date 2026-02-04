from __future__ import annotations
import concurrent.futures
import os
import time
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import ollama
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# --- Local Module Imports ---
from modules.db_manager import DBManager 
from modules.functions import (
    OllamaEmbeddings,
    OpenAIEmbeddings,
    rebuild_rag_index
)
from modules.login import check_authentication, login_page, logout
from modules.quizstats.file_utils import convert_to_pdf, extract_text_from_path, save_uploaded_files, render_pdf_page_image
from modules.quizstats.quiz_engine import build_quiz_from_rag
from modules.quizstats.progress_store import init_stats_db, record_attempt

# ---------- CONFIG & PATHS ----------
st.set_page_config(page_title="Quiz Builder", page_icon="📝", layout="wide")

BASE_DIR = Path(__file__).parents[1]
CHAT_DB_FILE = str(BASE_DIR / "chat.db")
SUBJECTS_DIR = os.path.join("modules", "subjects") 

os.makedirs(SUBJECTS_DIR, exist_ok=True)

# ---------- AUTHENTICATION ----------
is_authenticated, username = check_authentication()
if not is_authenticated:
    login_page()
    st.stop()

user_id = st.session_state.get("user_id")

# --- API KEY RETRIEVAL (Strictly Database/Session) ---
# These were populated in login.py during the auth check
oa_key = st.session_state.get("openai_api_key", "").strip()
q_url = st.session_state.get("qdrant_url", "").strip()
q_key = st.session_state.get("qdrant_api_key", "").strip()

# Sync to environment variables for background libraries that require them
os.environ["OPENAI_API_KEY"] = oa_key
os.environ["QDRANT_URL"] = q_url
os.environ["QDRANT_API_KEY"] = q_key

openai_functional = bool(oa_key)

# ---------- POST-AUTH INITIALIZATION ----------
if "backend_verified" not in st.session_state:
    st.session_state.backend_verified = False

# Initialize DBManager
db = DBManager(
    backend="sqlite",
    user_id=user_id,
    db_file=CHAT_DB_FILE,
)
    
USER_DATA_ROOT = os.path.join("modules", "subjects", f"user_{user_id}")
os.makedirs(USER_DATA_ROOT, exist_ok=True)

# --- API KEY RETRIEVAL ---
oa_key = st.session_state.get("openai_api_key")

st.session_state.openai_api_key = oa_key
openai_functional = bool(oa_key)

# Set vector_db default if not already set (matching chat.py)
if "vector_db" not in st.session_state:
    st.session_state.vector_db = "faiss"

def thread_wrapper(fn, ctx, *args, **kwargs):
    """Injects the Streamlit context into the thread before running the function."""
    add_script_run_ctx(threading.current_thread(), ctx)
    return fn(*args, **kwargs)

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

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(f"**User: {username}**")
    if st.button("Logout"): 
        logout()
        st.rerun()
    st.markdown("---")

# Show OpenAI status warning (matching chat.py)
if not openai_functional:
    st.toast("**Warning:** OpenAI features are limited.", icon="⚠️")

# ---------- MODELS (CACHED - matching chat.py) ----------
MODEL_MAP = {
    "llama3:8b": "Llama 3 (Base Model)",
    "llava:7b": "Llava (Vision Model)",
    "gpt-4o": "OpenAI GPT-4o",
}
ALLOWED_MODEL_IDS = list(MODEL_MAP.keys())

# Cache Ollama model discovery (matching chat.py)
if "available_ollama_models" not in st.session_state:
    try:
        raw_ollama = [m["model"] for m in ollama.list().get("models", [])]
        st.session_state.available_ollama_models = tuple(m for m in raw_ollama if m in ALLOWED_MODEL_IDS)
    except:
        st.session_state.available_ollama_models = ()

ollama_models = st.session_state.available_ollama_models
openai_models = ("gpt-4o",) if openai_functional else ()
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
user_root = os.path.join("modules", "subjects", f"user_{user_id}")
subject_folder = os.path.join(user_root, S["selected_subject"])
index_path = os.path.join(subject_folder, f"index_{suffix}.faiss")

if os.path.exists(index_path):
    pass
else:
    st.error(f"⚠️ Please upload files and click 'Process RAG' below.")

# ---------- FILE PROCESSING ----------
with st.expander(f"📤 Knowledge Base: {S['selected_subject']}"):
    # 1. Subject Restriction from Chat.py
    if S['selected_subject'] == "General":
        st.info("💡 **Note:** You cannot upload documents to the 'General' subject. Create a new subject in the sidebar first.")
    else:
        uploaded_files = st.file_uploader(
            "Add files", 
            type=["pdf", "docx", "pptx"], 
            accept_multiple_files=True
        )
        
        if st.button("📂 Process RAG"):
            target_sub = S["selected_subject"]
            
            if uploaded_files:
                # 1. Path Construction
                raw_dir = os.path.join(USER_DATA_ROOT, target_sub, "raw")
                os.makedirs(raw_dir, exist_ok=True)
                existing_filenames = os.listdir(raw_dir)
                
                # 2. Availability Flags (Strictly using Session State from users.db)
                oa_key = st.session_state.get("openai_api_key", "").strip()
                use_openai = bool(oa_key)
                use_ollama = True # Assuming Ollama is always an option locally

                progress_bar = st.progress(0)
                status_text = st.empty()
                
                qdrant_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

                for i, f in enumerate(uploaded_files):
                    percent_val = (i + 1) / len(uploaded_files)
                    progress_bar.progress(percent_val)
                    
                    if f.name in existing_filenames:
                        continue

                    status_text.text(f"⏳ Processing {i+1}/{len(uploaded_files)}: {f.name}...")
                    
                    # Save file locally
                    f.seek(0)
                    save_uploaded_files(target_sub, [f], user_id=user_id)
                    original_path = os.path.join(raw_dir, f.name)
                    
                    # Convert to PDF if needed
                    ext = f.name.split('.')[-1].lower()
                    final_path = original_path
                    if ext in ["pptx", "docx"]:
                        pdf_path = convert_to_pdf(original_path)
                        if pdf_path:
                            final_path = pdf_path
                    
                    # Extract Text
                    content = extract_text_from_path(final_path)
                    
                    # --- CRITICAL SAFETY CHECK: Skip empty extractions to prevent OpenAI crash ---
                    if not content or not content.strip():
                        st.warning(f"⚠️ Could not extract text from {f.name}. Skipping...")
                        continue

                    final_filename = os.path.basename(final_path)

                    # --- PATH A: FAISS (Saves text to SQLite for later indexing) ---
                    if st.session_state.vector_db == "faiss":
                        if use_ollama:
                            db.add_rag_doc(target_id=target_sub, file_name=final_filename, 
                                           content=content, metadata={"engine": "ollama"})
                        if use_openai:
                            db.add_rag_doc(target_id=target_sub, file_name=final_filename, 
                                           content=content, metadata={"engine": "openai"})

                    # --- PATH B: QDRANT (Immediate Vector Ingestion) ---
                    elif st.session_state.vector_db == "qdrant":
                        chunks = qdrant_splitter.split_text(content)
                        for chunk_text in chunks:
                            if not chunk_text.strip(): continue
                            
                            if use_ollama:
                                ollama_embedder = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
                                v_ollama = ollama_embedder.embed_query(chunk_text)
                                db.add_rag_doc(target_id=target_sub, file_name=f.name,
                                               content=chunk_text, vector_data=v_ollama,
                                               metadata={"engine": "ollama"})
                            if use_openai:
                                openai_embedder = OpenAIEmbeddings(model_name="text-embedding-3-small", 
                                                                   api_key=oa_key)
                                v_openai = openai_embedder.embed_query(chunk_text)
                                db.add_rag_doc(target_id=target_sub, file_name=f.name,
                                               content=chunk_text, vector_data=v_openai,
                                               metadata={"engine": "openai"})

                # --- FINAL SYNC ---
                if st.session_state.vector_db == "faiss":
                    subject_path = os.path.join(USER_DATA_ROOT, target_sub)
                    if use_ollama:
                        status_text.text("🔄 Syncing Ollama FAISS index...")
                        rebuild_rag_index(CHAT_DB_FILE, subject_path, target_sub, "llama3:8b", 
                                          ollama_models, openai_models, suffix="ollama", user_id=user_id)
                    if use_openai:
                        status_text.text("🔄 Syncing OpenAI FAISS index...")
                        rebuild_rag_index(CHAT_DB_FILE, subject_path, target_sub, "gpt-4o", 
                                          ollama_models, openai_models, suffix="openai", user_id=user_id)
                
                st.success(f"✅ {target_sub} Knowledge Base Updated!")
                time.sleep(1.5)
                st.rerun()
            
# ---------- QUIZ GENERATION ----------
if st.button("🎲 Generate 10 Questions"):
    # Updated logic: Check for local file if FAISS, or check connection if Qdrant
    can_generate = True
    if st.session_state.vector_db == "faiss" and not os.path.exists(index_path):
        st.error("Please Process & Sync files first.")
        can_generate = False
    elif st.session_state.vector_db == "qdrant" and not st.session_state.get("backend_verified"):
        st.error("Qdrant is not connected. Check your settings.")
        can_generate = False

    if can_generate:
        colA, colB = st.columns([3, 1])
        with colA:
            status_ph = st.empty()
        with colB:
            timer_ph = st.empty()

        status_ph.info("🧠 Analyzing Knowledge Base...")
        start = time.perf_counter()

        quiz = None
        err = None
        
        # Capture the context of the current browser session
        ctx = get_script_run_ctx()

        # Retrieve keys from session state (populated from users.db)
        oa_key = st.session_state.get("openai_api_key", "").strip()
        q_url = st.session_state.get("qdrant_url", "").strip()
        q_key = st.session_state.get("qdrant_api_key", "").strip()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # We now pass the user-specific keys directly into the backend function
            future = executor.submit(
                thread_wrapper,
                build_quiz_from_rag,
                ctx,
                subject=S["selected_subject"],
                model_name=st.session_state.selected_model,
                n_questions=10,
                vector_db=st.session_state.vector_db, # Now dynamic (faiss or qdrant)
                user_id=user_id,
                ollama_models=ollama_models,
                openai_models=openai_models,
                # --- NEW: Pass user-specific credentials ---
                qdrant_url=q_url,
                qdrant_api_key=q_key
            )

            while not future.done():
                elapsed = time.perf_counter() - start
                timer_ph.metric("Load time", f"{elapsed:0.1f}s")
                time.sleep(0.1)

            try:
                quiz = future.result() 
            except Exception as e:
                err = e

        elapsed = time.perf_counter() - start
        timer_ph.metric("Load time", f"{elapsed:0.1f}s", "done")
        status_ph.empty()

        if err:
            st.error(f"Quiz generation crashed: {err}")
        elif quiz:
            # Clear previous answers and set up the new quiz
            S.update({"quiz": quiz, "current_idx": 0, "score": 0, "submitted": {}})
            st.success("Quiz generated successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Failed to generate questions. The AI might have returned invalid data.")

# ---------- QUIZ PLAYER ----------
@st.fragment
def play_quiz(user_id):
    S = st.session_state.quiz_state
    
    if not S["quiz"]:
        st.info("No quiz generated yet.")
        return

    st.divider()
    
    # --- FINISHED PAGE ---
    if S["current_idx"] >= len(S["quiz"]):
        st.success("🎉 Quiz Finished!")
        st.header(f"Final Score: {S['score']} / {len(S['quiz'])}")
        
        if st.button("Restart & Clear"):
            S.update({"quiz": [], "current_idx": 0, "score": 0, "submitted": {}})
            st.rerun() 
        return

    # --- ACTIVE QUIZ ---
    q_idx = S["current_idx"]
    q = S["quiz"][q_idx]
    has_submitted = S["submitted"].get(q_idx, False)

    st.subheader(f"Question {q_idx + 1}")

    # --- CONTEXT RENDERING (COMPACT) ---
    with st.expander("🔍 Show Relevant Context", expanded=False):
        meta = q.get("context_meta", {})
        f_name = meta.get("file")
        page_num = meta.get("page", 1)
        
        # Use the established USER_DATA_ROOT variable
        pdf_path = os.path.join(USER_DATA_ROOT, S["selected_subject"], "raw", f_name)

        if f_name and os.path.exists(pdf_path):
            try:
                img = render_pdf_page_image(pdf_path, page_num)
                if img:
                    # Create columns to shrink the image size
                    c1, c2 = st.columns([1, 1]) 
                    with c1:
                        st.image(img, width=400) # Fixed width for a smaller look
                        st.caption(f"📄 {f_name} | Page {page_num}")
                else:
                    st.warning("Could not render page image.")
            except Exception as e:
                st.error(f"Error loading context: {e}")
        else:
            st.info("Context metadata found, but source file is missing.")
            st.write(q.get("explanation", "No additional reasoning provided."))

    # --- THE FORM ---
    with st.form("quiz_form"):
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
            submit_btn = st.form_submit_button("Submit", disabled=has_submitted)
        with col2:
            label = "Finish" if q_idx == len(S["quiz"]) - 1 else "Next"
            next_btn = st.form_submit_button(label)
        with col3:
            st.metric("Progress", f"{q_idx + 1} / {len(S['quiz'])}", f"Score: {S['score']}")

    # --- LOGIC ---
    if submit_btn and not has_submitted:
        if picked:
            is_correct = (q["choices"].index(picked) == q["answer_idx"])
            S["submitted"][q_idx] = True
            if is_correct:
                S["score"] += 1
            
            record_attempt(
                user_id=st.session_state.user_id,
                subject=S["selected_subject"], 
                question=q["question"],
                chosen=picked, 
                correct=q["choices"][q["answer_idx"]],
                is_correct=1 if is_correct else 0, 
                topic=q.get("topic", "General")
            )
            st.rerun(scope="fragment") 
        else:
            st.warning("Please select an option.")

    if has_submitted:
        correct_text = q["choices"][q["answer_idx"]]
        user_choice = st.session_state.get(f"q_{q_idx}")
        if user_choice == correct_text:
            st.success("✨ Correct!")
        else:
            st.error(f"❌ Incorrect. Answer: {correct_text}")

        if next_btn:
            S["current_idx"] += 1
            st.rerun(scope="fragment")

# Call the fragment in your main script logic
if S["quiz"]:
    play_quiz(user_id)