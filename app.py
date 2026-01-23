# app.py
from __future__ import annotations
import os
import re
import sys
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from openai import OpenAI
import ollama
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

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
from modules.quizstats.file_utils import convert_to_pdf, extract_text_from_path, render_pdf_page_image, save_uploaded_files

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
    def process_assistant_completion(chat_id, full_res, used_rag, sources=None):
        """Saves completion to DB with source metadata for visual context."""
        st.session_state.last_assistant_text = full_res
        st.session_state.tts_audio_for = ""
        st.session_state.tts_audio_bytes = None
        
        # Save to DB (Ensure your DBManager.add_message can handle a sources/metadata column if possible)
        db.add_message(chat_id, "assistant", full_res, used_rag=int(used_rag))
        
        # Append to session state with sources
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_res, 
            "used_rag": int(used_rag),
            "sources": sources  # List of {'file': '...', 'page': ...}
        })
        st.session_state.vision_images = []
        st.rerun()

    def clear_tts():
        st.session_state.tts_audio_bytes = None
        st.session_state.tts_audio_for = ""
    
    def format_latex(text: str) -> str:
        """Converts common LLM LaTeX delimiters to Streamlit-friendly ones."""
        # Convert \[ ... \] to $$ ... $$ for block math
        text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
        # Convert \( ... \) to $ ... $ for inline math
        text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
        # Sometimes models use [ ] or ( ) without backslashes
        # Only use these if you notice the model consistently failing backslashes
        return text

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
        "vector_db": "faiss", 
        "chat_backend": "sqlite",
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # --------- Lock ---------
    q_url = os.getenv("QDRANT_URL", "").strip()
    q_key = os.getenv("QDRANT_API_KEY", "").strip()
    oa_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "").strip()

    using_qdrant = st.session_state.get("vector_db") == "qdrant"
    lock_ui = False
    if using_qdrant:
        if not q_url or not q_key:
            lock_ui = True
        else:
            try:
                # Attempt a lightweight connection test
                # timeout=3 prevents the app from hanging forever on a bad URL
                test_client = QdrantClient(url=q_url, api_key=q_key, timeout=3)
                test_client.get_collections() 
            except Exception as e:
                # Catch Errno 11001 (Bad URL), 401 (Bad Key), etc.
                lock_ui = True
    
    openai_functional = False
    if oa_key:
        try:
            # Lightweight test: list models to check if key is valid
            test_oa = OpenAI(api_key=oa_key)
            test_oa.models.list()
            openai_functional = True
        except:
            openai_functional = False

# ---------------- MAIN APP INTERFACE ----------------
    st.title("📖 My Learning AI")

    if lock_ui:
        st.error("⚠️ **Qdrant Backend Locked:** Please provide API credentials in Settings or switch back to **Local (FAISS)**.")
        
    if not openai_functional:
        st.warning("⚠️ OpenAI Features Disabled: Mic, TTS, and GPT-4o require a valid API key.")

    db = None
    if using_qdrant and not lock_ui:
        try:
            # Create the client and try a simple lightweight operation
            test_client = QdrantClient(url=q_url, api_key=q_key, timeout=3)
            test_client.get_collections() 
            
            # If we reach here, the "rubbish" is actually valid or the server is up
            active_backend = "sqlite" if lock_ui else st.session_state.chat_backend
            db = DBManager(backend=active_backend, db_file=CHAT_DB_FILE, qdrant_url=q_url, qdrant_api_key=q_key)
        except Exception as e:
            # This triggers if the API key is wrong or URL is fake
            lock_ui = True
            st.error(f"⚠️ **Qdrant Connection Failed:** The credentials provided are incorrect or the server is down. Error: {e}")
            
    # Fallback/Default DB initialization
    if db is None:
        db = DBManager(backend="sqlite", db_file=CHAT_DB_FILE)

    subjects = db.load_all_subjects()

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown(f"**User: {username}**")
        if st.button("Logout"): logout(); st.rerun()
        st.markdown("---")

        st.header("📚 Subject Library")
        
        if "General" not in subjects:
            db.add_subject("General")
            subjects = db.load_all_subjects()
        
        selected_sub = st.selectbox(
            "Select Subject", 
            subjects, 
            index=subjects.index(st.session_state.current_subject) if st.session_state.current_subject in subjects else 0,
            disabled=lock_ui
        )
        
        if selected_sub != st.session_state.current_subject:
            st.session_state.current_subject = selected_sub
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.session_state.last_assistant_text = ""
            st.rerun()

        with st.expander("➕ New Subject"):
            new_sub = st.text_input("Subject Name", disabled=lock_ui)
            if st.button("Create Subject") and new_sub.strip():
                db.add_subject(new_sub.strip())
                ensure_subject_folders(new_sub.strip())
                st.session_state.current_subject = new_sub.strip()
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.rerun()

        if selected_sub != "General":  # Prevent deleting the default subject
                if st.button(f"Delete {selected_sub}", type="secondary"):
                    db.delete_subject(selected_sub, rag_index_dir=SUBJECTS_DIR)
                    st.session_state.current_subject = "General"
                    st.session_state.current_chat_id = None
                    st.session_state.messages = []
                    
                    st.success(f"Subject '{selected_sub}' deleted.")
                    st.rerun()

        st.markdown("---")
        backend_choice = st.selectbox(
            "Select Backend", 
            ["Local (FAISS)", "Remote (Qdrant)"],
            index=0 if st.session_state.vector_db == "faiss" else 1
        )
        
        target_v = "faiss" if backend_choice == "Local (FAISS)" else "qdrant"
        target_c = "sqlite" if backend_choice == "Local (FAISS)" else "qdrant"

        if st.session_state.vector_db != target_v:
            st.session_state.vector_db = target_v
            st.session_state.chat_backend = target_c
            st.session_state.messages = [] 
            st.session_state.current_chat_id = None
            st.rerun()
        st.markdown("---")

        all_chats = []
        if not lock_ui:
            try:
                all_chats = db.load_all_chats(st.session_state.current_subject)
            except Exception:
                st.warning("⚠️ Could not load remote chats. Check your Qdrant URL.")

        if st.button("🆕 Start New Chat", disabled=lock_ui):
            st.session_state.update({"messages": [], "current_chat_id": None})
            clear_tts(); st.rerun()

        for chat_id, model_name, created_at, title in all_chats:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"💬 {title}", key=f"chat_{chat_id}", use_container_width=True):
                    msgs, model = db.load_chat(chat_id)
                    st.session_state.update({"messages": msgs, "current_chat_id": chat_id, "selected_model": model})
                    clear_tts(); st.rerun()
            with col2:
                if st.button("✕", key=f"del_{chat_id}"):
                    db.delete_chat(chat_id)
                    if st.session_state.get("current_chat_id") == chat_id:
                        st.session_state.messages = []
                        st.session_state.current_chat_id = None
                        st.session_state.last_assistant_text = ""
                        clear_tts()
                    st.rerun()

    # ---------------- MODELS ----------------
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

    openai_models = ("gpt-4o",) if openai_functional else ()
    available_technical = ollama_models + openai_models
    display_models = [MODEL_MAP.get(m, m) for m in available_technical]

    if not available_technical:
        st.warning("⚠️ No supported models found. Please visit settings page.")
        st.stop()

    m_idx = 0
    if st.session_state.selected_model in available_technical:
        m_idx = available_technical.index(st.session_state.selected_model)

    selected_friendly = st.selectbox(
        "Model Selection", 
        display_models, 
        index=m_idx, 
        disabled=st.session_state.current_chat_id is not None or lock_ui
    )
    st.session_state.selected_model = available_technical[display_models.index(selected_friendly)]

    # ---------------- RAG UPLOADS ----------------
    with st.expander(f"📤 Knowledge Base: {st.session_state.current_subject}"):
        uploaded = st.file_uploader("Add files", type=["txt", "pdf", "docx", "pptx", "csv"], accept_multiple_files=True)
        if uploaded:
            st.session_state.uploaded_files_to_process = uploaded

        if st.button("📂 Process RAG", disabled=lock_ui):
                target_sub = st.session_state.current_subject
                subject_path = os.path.join(SUBJECTS_DIR, target_sub)
                
                # We only need the splitter for Qdrant to prevent context length errors
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                qdrant_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                
                with st.spinner(f"Processing RAG"):
                    ensure_subject_folders(target_sub)
                    for f in st.session_state.uploaded_files_to_process:
                        f.seek(0)
                        save_uploaded_files(target_sub, [f])
                        original_path = os.path.join("modules", "subjects", target_sub, "raw", f.name)
                        
                        ext = f.name.split('.')[-1].lower()
                        final_path = original_path

                        if ext in ["pptx", "docx"]:
                            pdf_path = convert_to_pdf(original_path)
                            if pdf_path and pdf_path != original_path:
                                final_path = pdf_path
                                try: os.remove(original_path)
                                except: pass
                        
                        content = extract_text_from_path(final_path)
                        final_filename = os.path.basename(final_path)
                        
                        # --- 1. FAISS PATH (LEAVE AS IS) ---
                        # We store the WHOLE content in SQLite. 
                        # rebuild_rag_index_faiss will handle its own chunking/metadata later.
                        if st.session_state.vector_db == "faiss":
                            db.add_rag_doc(
                                target_id=target_sub, 
                                file_name=final_filename, 
                                content=content, 
                                metadata={"file": final_filename, "subject": target_sub},
                                vector_data=None 
                            )

                        # --- 2. QDRANT PATH (FIXED WITH CHUNKING) ---
                        elif st.session_state.vector_db == "qdrant":
                            chunks = qdrant_splitter.split_text(content)
                            for i, chunk_text in enumerate(chunks):
                                # Push Ollama Vector for this chunk
                                ollama_embedder = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
                                v_ollama = ollama_embedder.embed_query(chunk_text)
                                db.add_rag_doc(
                                    target_id=target_sub, 
                                    file_name=final_filename, 
                                    content=chunk_text, 
                                    vector_data=v_ollama,
                                    metadata={"engine": "ollama", "part": i}
                                )

                                # Push OpenAI Vector for this chunk
                                if st.session_state.openai_api_key:
                                    openai_embedder = OpenAIEmbeddings(
                                        model_name="text-embedding-3-small",
                                        api_key=st.session_state.openai_api_key
                                    )
                                    v_openai = openai_embedder.embed_query(chunk_text)
                                    db.add_rag_doc(
                                        target_id=target_sub, 
                                        file_name=final_filename, 
                                        content=chunk_text, 
                                        vector_data=v_openai,
                                        metadata={"engine": "openai", "part": i}
                                    )
                    
                    # REBUILD FAISS (Uses the full content stored in SQLite)
                    rebuild_rag_index(CHAT_DB_FILE, subject_path, target_sub, "llama3:8b", ollama_models, openai_models, suffix="ollama")
                    if st.session_state.openai_api_key:
                        rebuild_rag_index(CHAT_DB_FILE, subject_path, target_sub, "gpt-4o", ollama_models, openai_models, suffix="openai")
                    
                    st.success(f"✅ Knowledge Base Synced!")
                    st.session_state.uploaded_files_to_process = []
                    st.rerun()

    # ---------------- CHAT INTERFACE ----------------
    message_container = st.container(height=500)
    for msg in st.session_state.messages:
        with message_container.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "😎"):
            st.markdown(("📄 " if msg.get("used_rag") else "") + msg["content"])
            
            # New: Display Source Images
            if msg.get("sources"):
                with st.expander("🔍 View Source Material"):
                    cols = st.columns(len(msg["sources"]))
                    for i, source in enumerate(msg["sources"]):
                        f_name = source.get("file")
                        page_num = source.get("page", 1)
                        pdf_path = os.path.join(SUBJECTS_DIR, st.session_state.current_subject, "raw", f_name)
                        
                        if os.path.exists(pdf_path):
                            img = render_pdf_page_image(pdf_path, page_num)
                            if img:
                                cols[i].image(img, caption=f"{f_name} (p. {page_num})")

    # ---------------- TTS CONTROLS ----------------
    latest_a = st.session_state.get("last_assistant_text", "").strip()
    has_audio = (st.session_state.tts_audio_bytes is not None and st.session_state.tts_audio_for == latest_a)

    if latest_a:
        if not has_audio:
            if st.button("🔊 Listen", disabled=not openai_functional):
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
        if audio_data and not openai_functional:
            pass  # Mic disabled due to missing OpenAI key
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
        typed_prompt = st.chat_input(f"Ask about {st.session_state.current_subject}...", disabled=lock_ui)
        if typed_prompt:
            st.session_state.temp_prompt = typed_prompt
            clear_tts(); st.rerun()

    # ---------------- PROCESSING ----------------
    if st.session_state.get("temp_prompt"):
        user_p = st.session_state.temp_prompt
        st.session_state.temp_prompt = None 
        
        # Ensure Chat Session exists
        if st.session_state.current_chat_id is None:
            st.session_state.current_chat_id = db.create_new_chat(
                st.session_state.selected_model, 
                st.session_state.current_subject
            )
        
        chat_id = st.session_state.current_chat_id
        db.add_message(chat_id, "user", user_p)
        st.session_state.messages.append({"role": "user", "content": user_p})

        # Generate Title for new chats
        if len(st.session_state.messages) <= 1:
            from modules.functions import generate_chat_title
            client_openai = get_openai_client() if "gpt" in st.session_state.selected_model else None
            new_title = generate_chat_title(user_p, st.session_state.selected_model, client_openai)
            db.update_chat_title(chat_id, new_title)

        with message_container.chat_message("user", avatar="😎"):
            st.markdown(user_p)

        # --- SYNCED RAG SEARCH ---
        rag_content, raw_docs, has_docs = "", [], False
        sources_to_display = [] 
        search_id = st.session_state.current_subject
        
        if st.session_state.rag_enabled:
            if st.session_state.vector_db == "qdrant":
                try:
                    client = QdrantClient(url=st.session_state.qdrant_url, api_key=st.session_state.qdrant_api_key)
                    with st.spinner("🔍 Searching Knowledge Base (Qdrant)..."):
                        rag_content, raw_docs, search_success = get_relevant_rag(
                            CHAT_DB_FILE, None, search_id, user_p, 
                            st.session_state.selected_model, ollama_models, openai_models, 
                            vector_db="qdrant",
                            qdrant_url=st.session_state.qdrant_url,
                            qdrant_api_key=st.session_state.qdrant_api_key
                        )
                except Exception as e: st.error(f"Qdrant RAG Error: {e}")

            elif st.session_state.vector_db == "faiss":
                subject_index_dir = os.path.join(SUBJECTS_DIR, search_id)
                with st.spinner("🔍 Searching Knowledge Base (FAISS)..."):
                    rag_content, raw_docs, has_docs = get_relevant_rag(
                        CHAT_DB_FILE, subject_index_dir, search_id, user_p, 
                        st.session_state.selected_model, ollama_models, openai_models, 
                        vector_db="faiss"
                    )

            if raw_docs:
                has_docs = True
                seen = set()
                for doc in raw_docs:
                    m = doc.metadata
                    f_name = m.get('file')
                    p_num = m.get('page', 1)
                    identifier = f"{f_name}_{p_num}"
                    if f_name and identifier not in seen:
                        sources_to_display.append({"file": f_name, "page": p_num})
                        seen.add(identifier)

        # --- CONSTRUCT PROMPT ---
        if has_docs and rag_content.strip():
            indicator = "📄 "
            enhanced_p = f"You are a helpful assistant. Context:\n{rag_content}\n\nQuestion: {user_p}"
        else:
            indicator = ""
            enhanced_p = user_p

        # --- LLM CALL ---
        if st.session_state.selected_model in ollama_models:
            images_to_send = [get_b64_image(f.getvalue()) for f in st.session_state.vision_images]
            with message_container.chat_message("assistant", avatar="🤖"):
                stream_placeholder = st.empty(); full_res = ""
                for chunk in ollama.chat(model=st.session_state.selected_model, stream=True,
                                        messages=[{"role": "user", "content": enhanced_p, "images": images_to_send}]):
                    full_res += chunk.get("message", {}).get("content", "")
                    # Format LaTeX during streaming
                    stream_placeholder.markdown(indicator + format_latex(full_res))
                process_assistant_completion(chat_id, format_latex(full_res), has_docs, sources=sources_to_display)

        elif st.session_state.selected_model in openai_models:
            client = get_openai_client()
            with message_container.chat_message("assistant", avatar="🤖"):
                stream_placeholder = st.empty(); full_res = ""
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                history.append({"role": "user", "content": enhanced_p})
                stream = client.chat.completions.create(model=st.session_state.selected_model, messages=history, stream=True)
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_res += chunk.choices[0].delta.content
                        # Format LaTeX during streaming
                        stream_placeholder.markdown(indicator + format_latex(full_res))
                process_assistant_completion(chat_id, format_latex(full_res), has_docs, sources=sources_to_display)

if __name__ == "__main__":
    main()