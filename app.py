# app.py
from __future__ import annotations
import os
import sys
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from openai import OpenAI
import ollama
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- Local Module Imports ---
from modules.db_manager import DBManager
from modules.functions import (
    OllamaEmbeddings,
    OpenAIEmbeddings,
    extract_text_from_file,
    rebuild_rag_index,
    get_relevant_rag,
    get_b64_image,
    transcribe_audio_bytes,
    speak_text
)
from modules.login import check_authentication, login_page, logout
from modules.quizstats.subject_store import ensure_subject_folders
from modules.quizstats.file_utils import convert_to_pdf, extract_text_from_path, save_uploaded_files

try:
    from modules.qdrant_db import search_rag_docs_qdrant

    HAS_QDRANT_HELPER = True
except Exception:
    HAS_QDRANT_HELPER = False

load_dotenv()
CHAT_DB_FILE = "chat_playground.db"
SUBJECTS_DIR = os.path.join("modules", "subjects")
os.makedirs(SUBJECTS_DIR, exist_ok=True)

# ------------------ CONFIG ------------------
st.set_page_config(page_title="My Learning AI", page_icon="📚", layout="wide")

def get_subject_index_path(subject_name):
    return os.path.join(SUBJECTS_DIR, subject_name, "index.faiss")

def get_openai_client() -> OpenAI:
    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI API key missing. Set it in the sidebar.")
    return OpenAI(api_key=api_key)

def main():
    def process_assistant_completion(chat_id, full_res, used_rag):
        """Saves completion to DB and prepares TTS metadata."""
        st.session_state.last_assistant_text = full_res
        st.session_state.tts_audio_for = ""
        st.session_state.tts_audio_bytes = None
        
        db.add_message(chat_id, "assistant", full_res, used_rag=int(used_rag))
        st.session_state.messages.append({"role": "assistant", "content": full_res, "used_rag": int(used_rag)})
        st.session_state.vision_images = []
        st.rerun()
    def clear_tts():
        st.session_state.tts_audio_bytes = None
        st.session_state.tts_audio_for = ""

    # -------- AUTHENTICATION --------
    is_authenticated, username = check_authentication()
    if not is_authenticated:
        st.markdown("<style>[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
        login_page()
        st.stop()

    # -------- SESSION STATE DEFAULTS --------
    defaults = {
        "current_subject": "General", 
        "messages": [],
        "current_chat_id": None,
        "selected_model": None,
        "vision_images": [], 
        "uploaded_files_to_process": [],
        "rag_enabled": True,
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "last_assistant_text": "",
        "temp_prompt": None,
        "tts_audio_bytes": None,
        "tts_audio_for": "",
        "tts_voice": "alloy",
        "last_audio_bytes": None,
        # QDRANT SPECIFIC KEYS
        "vector_db": "faiss", 
        "chat_backend": "sqlite",
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    st.title("📖 My Learning AI")
    #db = DBManager(backend="sqlite", db_file=CHAT_DB_FILE) #initlialize sqlite backend first
    db = DBManager( # dynamic backend based on user selection
        backend=st.session_state.chat_backend,
        db_file=CHAT_DB_FILE,
        qdrant_url=st.session_state.qdrant_url,
        qdrant_api_key=st.session_state.qdrant_api_key,
    )

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown(f"**User: {username}**")
        if st.button("Logout"): logout(); st.rerun()
        st.markdown("---")

        st.header("📚 Subject Library")
        subjects = db.load_all_subjects()
        
        # Auto-create General if missing for this backend
        if "General" not in subjects:
            db.add_subject("General")
            subjects = db.load_all_subjects()
        
        selected_sub = st.selectbox(
            "Select Subject", 
            subjects, 
            index=subjects.index(st.session_state.current_subject) if st.session_state.current_subject in subjects else 0
        )
        
        if selected_sub != st.session_state.current_subject:
            # --- SUBJECT CHANGED: RESET EVERYTHING ---
            st.session_state.current_subject = selected_sub
            st.session_state.messages = []       # Clear the UI messages
            st.session_state.current_chat_id = None # Break link to old chat
            st.session_state.last_assistant_text = ""
            st.rerun() # Force a clean refresh to load the new subject's chats

        with st.expander("➕ New Subject"):
            new_sub = st.text_input("Subject Name")
            if st.button("Create Subject") and new_sub.strip():
                # 1. Update Database
                db.add_subject(new_sub.strip())
                
                # 2. Create physical folders (subjects/Name/raw and subjects/Name/images)
                # This call will now work because the import is global
                ensure_subject_folders(new_sub.strip())
                
                st.session_state.current_subject = new_sub.strip()
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.rerun()

        # --- UNIFIED BACKEND STORAGE SWITCHER ---
        st.markdown("---")
        backend_choice = st.selectbox(
            "Select Backend", 
            ["Local (FAISS)", "Remote (Qdrant)"],
            index=0 if st.session_state.vector_db == "faiss" else 1
        )
        
        # Determine target state
        target_v = "faiss" if backend_choice == "Local (FAISS)" else "qdrant"
        target_c = "sqlite" if backend_choice == "Local (FAISS)" else "qdrant"

        # Only trigger rerun if value actually changed
        if st.session_state.vector_db != target_v:
            st.session_state.vector_db = target_v
            st.session_state.chat_backend = target_c
            st.session_state.messages = [] 
            st.session_state.current_chat_id = None
            st.rerun()
        st.markdown("---")

    # ---------------- CHAT MANAGEMENT ----------------
        st.header(f"💬 {st.session_state.current_subject} Chats")
        all_chats = db.load_all_chats(st.session_state.current_subject)

        if st.button("🆕 Start New Chat"):
            st.session_state.update({"messages": [], "current_chat_id": None})
            clear_tts(); st.rerun()

        for chat_id, model_name, created_at, title in all_chats:
            col1, col2 = st.columns([4, 1])
            with col1:
                # Use the 'title' variable here for a better UI!
                if st.button(f"💬 {title}", key=f"chat_{chat_id}", use_container_width=True):
                    msgs, model = db.load_chat(chat_id)
                    st.session_state.update({"messages": msgs, "current_chat_id": chat_id, "selected_model": model})
                    clear_tts(); st.rerun()
            with col2:
                if st.button("✕", key=f"del_{chat_id}"):
                    db.delete_chat(chat_id); st.rerun()
# ---------------- MODELS ----------------
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

    openai_models = ("gpt-4o",) if st.session_state.openai_api_key else ()
    
    available_technical = ollama_models + openai_models
    display_models = [MODEL_MAP.get(m, m) for m in available_technical]

    if not available_technical:
        st.warning("⚠️ No supported models found. Download ollama models or enter your OpenAI API key in the settings page.")
        st.stop()

    # Determine selection index
    m_idx = 0
    if st.session_state.selected_model in available_technical:
        m_idx = available_technical.index(st.session_state.selected_model)

    selected_friendly = st.selectbox(
        "Model Selection", 
        display_models, 
        index=m_idx, 
        disabled=st.session_state.current_chat_id is not None
    )

    st.session_state.selected_model = available_technical[display_models.index(selected_friendly)]

    # ---------------- RAG UPLOADS ----------------
    
    with st.expander(f"📤 Knowledge Base: {st.session_state.current_subject}"):
        uploaded = st.file_uploader("Add files", type=["txt", "pdf", "docx", "pptx", "csv"], accept_multiple_files=True)
        
        if uploaded:
            st.session_state.uploaded_files_to_process = uploaded

        if st.button("📂 Process & Sync RAG"):
            target_sub = st.session_state.current_subject
            subject_path = os.path.join(SUBJECTS_DIR, target_sub)
            
            with st.spinner(f"Updating RAG database for {target_sub}..."):
                ensure_subject_folders(target_sub)
                
                for f in st.session_state.uploaded_files_to_process:
                    f.seek(0) # Reset pointer
                    
                    # 1. Save original file temporarily
                    save_uploaded_files(target_sub, [f])
                    original_path = os.path.join("modules", "subjects", target_sub, "raw", f.name)
                    
                    ext = f.name.split('.')[-1].lower()
                    final_path = original_path

                    # 2. Convert PPTX/DOCX to PDF and Cleanup
                    if ext in ["pptx", "docx"]:
                        with st.spinner(f"Converting {f.name} to visual PDF..."):
                            from modules.quizstats.file_utils import convert_to_pdf
                            pdf_path = convert_to_pdf(original_path)
                            
                            if pdf_path and pdf_path != original_path:
                                final_path = pdf_path
                                # DELETE ORIGINAL OFFICE FILE
                                try:
                                    os.remove(original_path)
                                except Exception as e:
                                    print(f"Cleanup error: {e}")
                    
                    # 3. Standardized Text Extraction from the DISK PATH
                    content = extract_text_from_path(final_path)
                    
                    # 4. Save metadata (always pointing to the PDF version)
                    final_filename = os.path.basename(final_path)
                    metadata = {
                        "file": final_filename,
                        "type": "pdf_page",
                        "subject": target_sub
                    }
                    db.add_rag_doc(target_id=target_sub, file_name=final_filename, content=content, metadata=metadata)
                
                # 5. REBUILD INDICES
                rebuild_rag_index(
                    CHAT_DB_FILE, subject_path, target_sub, 
                    "llama3:8b", ollama_models, openai_models, 
                    suffix="ollama" 
                )
                
                if st.session_state.openai_api_key:
                    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
                    rebuild_rag_index(
                        CHAT_DB_FILE, subject_path, target_sub, 
                        "gpt-4o", ollama_models, openai_models, 
                        suffix="openai"
                    )
                
                st.success(f"✅ Success! {target_sub} is now standardized to PDF.")
                st.session_state.uploaded_files_to_process = []
                st.rerun()

    # ---------------- CHAT INTERFACE ----------------
    message_container = st.container(height=500)
    for msg in st.session_state.messages:
        with message_container.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "😎"):
            st.markdown(("📄 " if msg.get("used_rag") else "") + msg["content"])

    # ---------------- ON-DEMAND TTS CONTROLS ----------------
    latest_a = st.session_state.get("last_assistant_text", "").strip()
    
    has_audio = (st.session_state.tts_audio_bytes is not None and 
                 st.session_state.tts_audio_for == latest_a)

    if latest_a:
        if not has_audio:
            if st.button("🔊 Listen"):
                with st.spinner("..."):
                    client = get_openai_client()
                    st.session_state.tts_audio_bytes = speak_text(latest_a, client, st.session_state.tts_voice)
                    st.session_state.tts_audio_for = latest_a
                    st.rerun()
        if has_audio:
            st.audio(st.session_state.tts_audio_bytes, format="audio/mp3")

    # ---------------- INPUT ROW ----------------
    st.markdown("### Ask with text or voice")
    col_text, col_mic = st.columns([7, 1])

    with col_mic:
        audio_data = audio_recorder(text="", icon_size="2x", key="recorder")
        if audio_data and audio_data != st.session_state.get("last_audio_bytes"):
            st.session_state["last_audio_bytes"] = audio_data
            try:
                client = get_openai_client()
                voice_text = transcribe_audio_bytes(audio_data, client)
                if voice_text:
                    st.session_state.temp_prompt = voice_text
                    clear_tts(); st.rerun()
            except Exception as e: st.error(f"Mic Error: {e}")

    with col_text:
        typed_prompt = st.chat_input(f"Ask about {st.session_state.current_subject}...")
        if typed_prompt:
            st.session_state.temp_prompt = typed_prompt
            clear_tts(); st.rerun()

# ---------------- PROCESSING ----------------
    if st.session_state.get("temp_prompt"):
        user_p = st.session_state.temp_prompt
        st.session_state.temp_prompt = None

        # FLAG: Check if this is a brand new chat
        is_new_chat = st.session_state.current_chat_id is None

        if is_new_chat:
            st.session_state.current_chat_id = db.create_new_chat(
                st.session_state.selected_model, 
                st.session_state.current_subject
            )
        
        chat_id = st.session_state.current_chat_id
        db.add_message(chat_id, "user", user_p)
        st.session_state.messages.append({"role": "user", "content": user_p})

        # ----- Title Generation -------
        if is_new_chat:
            from modules.functions import generate_chat_title
            client_openai = get_openai_client() if "gpt" in st.session_state.selected_model else None
            new_title = generate_chat_title(user_p, st.session_state.selected_model, client_openai)
            db.update_chat_title(chat_id, new_title)

        # --- Add user prompt first ---
        with message_container.chat_message("user", avatar="😎"):
            st.markdown(user_p)

        # --- SYNCED RAG SEARCH ---
        rag_content, has_docs = "", False
        search_id = st.session_state.current_subject
        subject_index_dir = os.path.join(SUBJECTS_DIR, search_id)

        if st.session_state.rag_enabled:
            current_model = st.session_state.selected_model.lower()
            suffix = "openai" if "gpt" in current_model else "ollama"
            # Determining which backend to search
            if st.session_state.vector_db == "qdrant" and HAS_QDRANT_HELPER:
                try:
                    rag_content, raw_docs, has_docs = get_relevant_rag(
                        CHAT_DB_FILE, 
                        subject_index_dir,
                        search_id, # Using synced ID
                        user_p, 
                        st.session_state.selected_model, 
                        ollama_models, 
                        openai_models, 
                        vector_db="qdrant",
                        qdrant_url=st.session_state.qdrant_url,
                        qdrant_api_key=st.session_state.qdrant_api_key
                    )
                except Exception as e: 
                    st.error(f"Qdrant Error: {e}")
            else:
                with st.spinner("📂 Searching Local Index..."):
                    rag_content, raw_docs, has_docs = get_relevant_rag(
                        CHAT_DB_FILE, 
                        subject_index_dir, 
                        search_id,
                        user_p, 
                        st.session_state.selected_model, 
                        ollama_models, 
                        openai_models, 
                        vector_db="faiss",
                        suffix=suffix
                    )

        # 2. PREDEFINED SYSTEM PROMPT
        if has_docs and rag_content.strip():
            # Indicator that RAG was successful
            indicator = "📄 "
            enhanced_p = f"""You are a helpful assistant. Use the following pieces of retrieved context to answer the user's question. 

If the provided context contains the answer, prioritize that information. 

---
CONTEXT:
{rag_content}
---

USER QUESTION: {user_p}"""
        else:
            indicator = ""
            enhanced_p = user_p

        # 3. LLM Interaction
        # A. OLLAMA MODELS
        if st.session_state.selected_model in ollama_models:
            images_to_send = [get_b64_image(f.getvalue()) for f in st.session_state.vision_images]
            with message_container.chat_message("assistant", avatar="🤖"):
                stream_placeholder = st.empty(); full_res = ""
                for chunk in ollama.chat(
                    model=st.session_state.selected_model, 
                    stream=True,
                    messages=[{"role": "user", "content": enhanced_p, "images": images_to_send}]
                ):
                    delta = chunk.get("message", {}).get("content", "")
                    full_res += delta
                    stream_placeholder.markdown(indicator + full_res)
                
                process_assistant_completion(chat_id, full_res, has_docs)

        # B. OPENAI MODELS
        elif st.session_state.selected_model in openai_models:
            client = get_openai_client()
            with message_container.chat_message("assistant", avatar="🤖"):
                stream_placeholder = st.empty(); full_res = ""
                # Construct history for OpenAI
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                history.append({"role": "user", "content": enhanced_p})
                
                stream = client.chat.completions.create(
                    model=st.session_state.selected_model,
                    messages=history,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        full_res += delta
                        stream_placeholder.markdown(indicator + full_res)
                
                process_assistant_completion(chat_id, full_res, has_docs)

if __name__ == "__main__":
    main()