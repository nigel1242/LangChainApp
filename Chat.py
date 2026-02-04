# chat.py
from __future__ import annotations
import os
import re
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import time
import ollama
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Local Module Imports ---
from modules.db_manager import DBManager
from modules.functions import (
    OllamaEmbeddings, OpenAIEmbeddings,
    rebuild_rag_index, get_relevant_rag,
    get_b64_image, transcribe_audio_bytes,
    speak_text, generate_chat_title, get_openai_client
)
from modules.login import check_authentication, login_page, logout
from modules.quizstats.file_utils import convert_to_pdf, extract_text_from_path, render_pdf_page_image, save_uploaded_files, ensure_subject_folders

CHAT_DB_FILE = "chat.db"

# ------------------ CONFIG ------------------
st.set_page_config(page_title="My Learning AI", page_icon="📚", layout="wide")

def main():
    def process_assistant_completion(chat_id, full_res, used_rag, sources=None):
        """Saves completion and CLEAR STATE for next turn."""
        st.session_state.last_assistant_text = full_res
        
        # Save to database
        db.add_message(chat_id, "assistant", full_res, used_rag=int(used_rag))
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_res,
            "used_rag": int(used_rag),
            "sources": sources
        })

        # --- CLEAR VISION STATE HERE ---
        st.session_state.vision_images = []  
        st.session_state.vision_uploader_key += 1  # This forces the uploader widget to reset
        st.session_state.temp_prompt = None 
        
        # CRITICAL: First message needs full reload to update sidebar & lock model
        # Subsequent messages only need fragment reload for speed
        is_first_message = len(st.session_state.messages) <= 2  # user + assistant
        
        if is_first_message:
            st.rerun()  # Full reload to show chat in sidebar and lock model
        else:
            st.rerun(scope="fragment")  # Fast fragment reload for subsequent messages

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
    
    def ensure_general_exists(db, user_id):
        existing_subs = db.load_all_subjects()
        if "General" not in existing_subs:
            db.add_subject("General")
            ensure_subject_folders("General", user_id)
    @st.fragment
    def chat_interface():
        """Isolated chat interface that reloads independently from sidebar"""
        
        # Get current state (read-only from parent scope)
        chat_id = st.session_state.current_chat_id
        selected_model = st.session_state.selected_model
        current_subject = st.session_state.current_subject
        
        # ---------------- CHAT INTERFACE ----------------
        message_container = st.container(height=500)
        for msg in st.session_state.messages:
            with message_container.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "😎"):
                st.markdown(("📄 " if msg.get("used_rag") else "") + msg["content"])
                
                # New: Display Source Images (only for FAISS)
                if msg.get("sources") and st.session_state.vector_db == "faiss":
                    with st.expander("🔍 View Source Material"):
                        cols = st.columns(len(msg["sources"]))
                        for i, source in enumerate(msg["sources"]):
                            f_name = source.get("file")
                            page_num = source.get("page", 1)
                            pdf_path = os.path.join(USER_DATA_ROOT, st.session_state.current_subject, "raw", f_name)
                            
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
                                st.rerun(scope="fragment")
            if has_audio:
                    st.audio(st.session_state.tts_audio_bytes, format="audio/mp3")

        # ---------------- INPUT ROW ----------------
        st.markdown("### Enter your prompt")
        # Logic to detect if the current model supports vision
        is_vision_model = any(v in selected_model.lower() for v in ["llava", "qwen", "gpt-4o"])

        uploaded_img = None

        col_text, col_mic = st.columns([7, 1])

        with col_mic:
            audio_data = audio_recorder(text="", icon_size="2x", key="recorder")
            if audio_data and not openai_functional:
                pass
            if audio_data and audio_data != st.session_state.get("last_audio_bytes"):
                st.session_state["last_audio_bytes"] = audio_data
                try:
                    client = get_openai_client()
                    voice_text = transcribe_audio_bytes(audio_data, client)
                    if voice_text:
                        st.session_state.temp_prompt = voice_text
                        clear_tts(); st.rerun(scope="fragment")
                except Exception as e: st.error(f"Mic Error: {e}")

        with col_text:
            typed_prompt = st.chat_input(f"Ask about {current_subject}...", disabled=lock_ui)
            if typed_prompt:
                st.session_state.temp_prompt = typed_prompt
                clear_tts(); st.rerun(scope="fragment")

        if is_vision_model:
            uploaded_img = st.file_uploader(
                "🖼️ Upload an image",
                type=["png", "jpg", "jpeg"],
                key=f"vision_uploader_{st.session_state.vision_uploader_key}"
            )

        if uploaded_img is not None:
            # Update the session state with the current image bytes
            st.session_state.vision_images = [uploaded_img.getvalue()]
            col_img, _ = st.columns([1, 3]) 
            with col_img:
                st.image(
                    uploaded_img,
                    width='stretch'
                )
        # ---------------- PROCESSING ----------------
        if st.session_state.get("temp_prompt"):
            user_p = st.session_state.temp_prompt
            st.session_state.temp_prompt = None
            
            # Ensure Chat Session exists
            if st.session_state.current_chat_id is None:
                st.session_state.current_chat_id = db.create_new_chat(
                    st.session_state.current_subject, st.session_state.selected_model
                )
            
            chat_id = st.session_state.current_chat_id
            db.add_message(chat_id, "user", user_p)
            st.session_state.messages.append({"role": "user", "content": user_p})

            # Generate Title for new chats
            if len(st.session_state.messages) <= 1:
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
                # RAG Search Logic (Qdrant or FAISS)
                if st.session_state.vector_db == "qdrant":
                    try:
                        with st.spinner("🔍 Searching Knowledge Base (Qdrant)..."):
                            rag_content, raw_docs, has_docs = get_relevant_rag(
                                CHAT_DB_FILE, None, search_id, user_p,
                                st.session_state.selected_model, ollama_models, openai_models,
                                vector_db="qdrant",
                                qdrant_url=st.session_state.qdrant_url,
                                qdrant_api_key=st.session_state.qdrant_api_key
                            )
                            # Ensure has_docs is set based on whether we got results
                            if raw_docs:
                                has_docs = True
                    except Exception as e: 
                        st.error(f"Qdrant RAG Error: {e}")
                elif st.session_state.vector_db == "faiss":
                    subject_index_dir = os.path.join(USER_DATA_ROOT, search_id)
                    with st.spinner("🔍 Searching Knowledge Base (FAISS)..."):
                        rag_content, raw_docs, has_docs = get_relevant_rag(
                            CHAT_DB_FILE, subject_index_dir, search_id, user_p,
                            st.session_state.selected_model, ollama_models, openai_models,
                            vector_db="faiss"
                        )

                # Only process sources for FAISS (since Qdrant won't render PDFs)
                if raw_docs and st.session_state.vector_db == "faiss":
                    has_docs = True
                    seen = set()
                    for doc in raw_docs:
                        # Determine if we are dealing with a Qdrant ScoredPoint or a LangChain Document
                        if hasattr(doc, "payload"):
                            m = doc.payload # Qdrant uses payload
                        elif hasattr(doc, "metadata"):
                            m = doc.metadata # FAISS uses metadata
                        else:
                            m = {}

                        # ALL logic using 'm' MUST be indented inside this loop
                        f_name = m.get('file') or m.get('file_name') 
                        p_num = m.get('page', 1)
                        
                        identifier = f"{f_name}_{p_num}"
                        if f_name and identifier not in seen:
                            sources_to_display.append({"file": f_name, "page": p_num})
                            seen.add(identifier)

            # --- CONSTRUCT PROMPT ---
            if has_docs and rag_content.strip():
                indicator = "📄 "
                enhanced_p = (
                    f"Context:\n{rag_content}\n\n"
                    f"Question: {user_p}\n\n"
                    "Important: Always use LaTeX for mathematical formulas. "
                    "Use $$ for block equations and $ for inline equations."
                )
            else:
                indicator = ""
                enhanced_p = user_p

            # --- LLM CALL ---
            if st.session_state.selected_model in ollama_models:
                # Only send images if there is a fresh upload
                images_to_send = [get_b64_image(img_data) for img_data in st.session_state.vision_images] \
                                        if st.session_state.vision_images else None

                with message_container.chat_message("assistant", avatar="🤖"):
                    stream_placeholder = st.empty()
                    full_res = ""

                    # Build the message dict
                    user_message = {"role": "user", "content": enhanced_p}
                    if images_to_send:
                        user_message["images"] = images_to_send

                    # Pass to Ollama
                    for chunk in ollama.chat(
                        model=st.session_state.selected_model,
                        stream=True,
                        messages=[user_message]
                    ):
                        full_res += chunk.get("message", {}).get("content", "")
                        stream_placeholder.markdown(indicator + format_latex(full_res))

                    # Save completion
                    process_assistant_completion(chat_id, format_latex(full_res), has_docs, sources=sources_to_display)

            elif st.session_state.selected_model in openai_models:
                client = get_openai_client()
                with message_container.chat_message("assistant", avatar="🤖"):
                    stream_placeholder = st.empty()
                    full_res = ""

                    # Prepare history
                    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

                    # Build current turn content
                    if st.session_state.vision_images:
                        current_turn_content = [{"type": "text", "text": enhanced_p}]
                        for img_bytes in st.session_state.vision_images:
                            b64_str = get_b64_image(img_bytes)
                            current_turn_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_str}",
                                    "detail": "auto"
                                }
                            })
                        history.append({"role": "user", "content": current_turn_content})
                    else:
                        # Text-only turn
                        history.append({"role": "user", "content": enhanced_p})

                    # Stream from OpenAI
                    try:
                        stream = client.chat.completions.create(
                            model=st.session_state.selected_model,
                            messages=history,
                            stream=True
                        )
                        for chunk in stream:
                            if chunk.choices and chunk.choices[0].delta.content:
                                full_res += chunk.choices[0].delta.content
                                stream_placeholder.markdown(indicator + format_latex(full_res))
                    except Exception as e:
                        st.error(f"OpenAI Error: {e}")

                    # Save completion
                    process_assistant_completion(chat_id, format_latex(full_res), has_docs, sources=sources_to_display)

    # 1. AUTHENTICATION
    is_authenticated, username = check_authentication()
    if not is_authenticated:
        st.markdown("<style>[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
        login_page()
        st.stop()

    user_id = st.session_state.get("user_id")

    # 2. POST-AUTH STABILIZATION GUARD
    # This block only runs once per login to lock the DB and Folders
    if not st.session_state.get("app_initialized") or "db" not in st.session_state:
        # Create user root folder
        USER_DATA_ROOT = os.path.join("modules", "subjects", f"user_{user_id}")
        os.makedirs(USER_DATA_ROOT, exist_ok=True)

        # Init DB Manager
        temp_db = DBManager(backend="sqlite", user_id=user_id, db_file=CHAT_DB_FILE)
        
        # Ensure 'General' exists
        ensure_general_exists(temp_db, user_id)
        
        # Lock state and force rerun
        st.session_state.db = temp_db
        st.session_state.app_initialized = True
        st.session_state.current_subject = "General" 
        st.rerun() 

    # 3. DEFINE VARIABLES (Reached ONLY after initialization is successful)
    db = st.session_state.db
    USER_DATA_ROOT = os.path.join("modules", "subjects", f"user_{user_id}")
    
    # --- API KEY RETRIEVAL (Database First) ---
    # Prioritize the keys pulled from users.db during login
    oa_key = st.session_state.get("openai_api_key", "").strip()
    
    openai_functional = bool(oa_key)
    st.session_state.openai_api_key = oa_key

    # --- QDRANT CONFIGURATION (Database First) ---
    q_url = st.session_state.get("qdrant_url", "").strip()
    q_key = st.session_state.get("qdrant_api_key", "").strip()

    st.session_state.qdrant_url = q_url
    st.session_state.qdrant_api_key = q_key

    # Define Qdrant/Backend status with caching
    using_qdrant = st.session_state.get("vector_db") == "qdrant"
    lock_ui = False
    
    if using_qdrant:
        if not q_url or not q_key:
            lock_ui = True
            st.session_state.qdrant_error = "Missing Qdrant credentials"
        elif not st.session_state.backend_verified:
            try:
                # Use the variables q_url and q_key retrieved above
                test_client = QdrantClient(url=q_url, api_key=q_key, timeout=2)
                test_client.get_collections()
                st.session_state.backend_verified = True
                st.session_state.qdrant_error = None
            except Exception as e:
                lock_ui = True
                st.session_state.qdrant_error = f"Connection failed: {str(e)}"

    # 4. DATA RETRIEVAL & SAFETY FALLBACKS
    subjects = db.load_all_subjects()
    if not subjects:
        subjects = ["General"]
    
    if st.session_state.current_subject not in subjects:
        st.session_state.current_subject = subjects[0]

    # 5. SET SESSION STATE DEFAULTS
    defaults = {
        "messages": [],
        "current_chat_id": None,
        "selected_model": None,
        "vision_images": [],
        "rag_enabled": True,
        "last_assistant_text": "",
        "temp_prompt": None,
        "tts_audio_bytes": None,
        "tts_audio_for": "",
        "tts_voice": "alloy",
        "last_audio_bytes": None,
        "vector_db": "faiss",
        "chat_backend": "sqlite",
        "vision_uploader_key": 0,
        "qdrant_url": q_url, # Use the local variable
        "qdrant_api_key": q_key # Use the local variable
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # 6. MAIN APP INTERFACE
    st.title("📖 My Learning AI")

    if lock_ui:
        st.error(f"⚠️ Qdrant Backend Locked: {st.session_state.get('qdrant_error', 'Check credentials in Settings.')}")
        
    if not openai_functional:
        st.toast("**Warning:** OpenAI features are limited.", icon="⚠️")

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown(f"**User: {username}**")
        if st.button("Logout"): 
            logout()
            st.rerun()
        st.markdown("---")

        st.header("📚 Subject Library")
        
        # Securely determine the index
        try:
            curr_idx = subjects.index(st.session_state.current_subject)
        except (ValueError, KeyError, IndexError):
            curr_idx = 0
            st.session_state.current_subject = subjects[0]

        selected_sub = st.selectbox(
            "Select Subject",
            options=subjects,
            index=curr_idx,
            disabled=(using_qdrant and lock_ui)
        )
        
        if selected_sub != st.session_state.current_subject:
            st.session_state.current_subject = selected_sub
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.rerun()

        with st.expander("➕ New Subject"):
            new_sub = st.text_input("Subject Name", disabled=lock_ui)
            if st.button("Create Subject") and new_sub.strip():
                db.add_subject(new_sub.strip())
                # Only create local folders when using FAISS backend
                if st.session_state.vector_db == "faiss":
                    ensure_subject_folders(new_sub.strip(), user_id)
                st.session_state.current_subject = new_sub.strip()
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.rerun()

        if selected_sub != "General":
            # 1. The initial "Delete" button
            if st.button(f"🗑️ Delete {selected_sub}", type="primary", use_container_width=True):
                st.session_state.confirm_delete = True

            # 2. The Confirmation UI
            if st.session_state.get("confirm_delete"):
                st.warning(f"⚠️ Are you sure? This will permanently delete all chats and RAG data for **{selected_sub}**.")
                
                col_yes, col_no = st.columns(2)
                
                with col_yes:
                    if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
                        db.delete_subject(selected_sub, subject_dir=USER_DATA_ROOT)
                        st.session_state.current_subject = "General"
                        st.session_state.current_chat_id = None
                        st.session_state.messages = []
                        st.session_state.confirm_delete = False
                        st.success(f"Subject '{selected_sub}' deleted.")
                        st.rerun()
                
                with col_no:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state.confirm_delete = False
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
            st.session_state.backend_verified = False  # Reset verification flag
            new_db = DBManager(
                backend=target_c, 
                user_id=user_id, 
                db_file=CHAT_DB_FILE
            )
            st.session_state.db = new_db
            ensure_general_exists(new_db, user_id)
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.rerun()
        
        # Show connection status
        if using_qdrant:
            if st.session_state.backend_verified:
                st.success("✅ Qdrant Connected")
            elif lock_ui:
                st.error(f"❌ {st.session_state.get('qdrant_error', 'Connection Error')}")
        
        st.markdown("---")

        all_chats = []
        if not lock_ui:
            try:
                all_chats = db.load_all_chats(st.session_state.current_subject)
            except Exception:
                st.warning("⚠️ Could not load remote chats. Check your Qdrant URL.")

        if st.button("🆕 Start New Chat", disabled=lock_ui, use_container_width=True):
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

    # ---------------- MODELS (CACHED) ----------------
    MODEL_MAP = {
        "llama3:8b": "Llama 3 (Base Model)",
        "llava:7b": "Llava (Vision Model)",
        "gpt-4o": "OpenAI GPT-4o",
    }
    ALLOWED_MODEL_IDS = list(MODEL_MAP.keys())

    # Cache Ollama model discovery
    if "available_ollama_models" not in st.session_state:
        try:
            raw_ollama = [m["model"] for m in ollama.list().get("models", [])]
            st.session_state.available_ollama_models = tuple(m for m in raw_ollama if m in ALLOWED_MODEL_IDS)
        except:
            st.session_state.available_ollama_models = ()

    ollama_models = st.session_state.available_ollama_models
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
        if st.session_state.current_subject == "General":
            st.info("💡 **Note:** You cannot upload documents to the 'General' subject. Create a new subject in the sidebar first.")
        else:
            if st.session_state.vector_db == "qdrant":
                st.info("📡 **Qdrant Mode:** Vectors stored remotely. PDF source rendering not available.")
            
            uploaded_files = st.file_uploader("Add files", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
            
            if st.button("📂 Process RAG", disabled=lock_ui):
                target_sub = st.session_state.current_subject
                
                if uploaded_files:
                    # Availability Flags
                    use_openai = bool(st.session_state.openai_api_key)
                    use_ollama = True 

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    qdrant_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

                    for i, f in enumerate(uploaded_files):
                        # Update Progress
                        percent_val = (i + 1) / len(uploaded_files)
                        progress_bar.progress(percent_val)
                        
                        status_text.text(f"⏳ Processing {i+1}/{len(uploaded_files)}: {f.name}...")
                        
                        # --- EXTRACT TEXT DIRECTLY FROM UPLOAD ---
                        f.seek(0)
                        file_bytes = f.read()
                        
                        # For FAISS: Save files locally
                        if st.session_state.vector_db == "faiss":
                            raw_dir = os.path.join(USER_DATA_ROOT, target_sub, "raw")
                            os.makedirs(raw_dir, exist_ok=True)
                            existing_filenames = os.listdir(raw_dir)
                            
                            # Skip duplicates
                            if f.name in existing_filenames:
                                continue
                            
                            f.seek(0)
                            save_uploaded_files(target_sub, [f], user_id=user_id)
                            original_path = os.path.join(raw_dir, f.name)
                            
                            # Convert PPTX/DOCX to PDF
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
                            
                            # Add to FAISS
                            if use_ollama:
                                db.add_rag_doc(target_id=target_sub, file_name=final_filename, 
                                            content=content, metadata={"engine": "ollama"})
                            if use_openai:
                                db.add_rag_doc(target_id=target_sub, file_name=final_filename, 
                                            content=content, metadata={"engine": "openai"})
                        
                        # For Qdrant: Process in-memory only
                        elif st.session_state.vector_db == "qdrant":
                            # Extract text from bytes (without saving to disk)
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{f.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(file_bytes)
                                tmp_path = tmp_file.name
                            
                            try:
                                # Convert if needed
                                ext = f.name.split('.')[-1].lower()
                                final_tmp_path = tmp_path
                                if ext in ["pptx", "docx"]:
                                    pdf_path = convert_to_pdf(tmp_path)
                                    if pdf_path and pdf_path != tmp_path:
                                        final_tmp_path = pdf_path
                                        try: os.remove(tmp_path)
                                        except: pass
                                
                                # Extract text
                                content = extract_text_from_path(final_tmp_path)
                                
                                # Chunk and upload to Qdrant
                                chunks = qdrant_splitter.split_text(content)
                                for chunk_text in chunks:
                                    if use_ollama:
                                        ollama_embedder = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
                                        v_ollama = ollama_embedder.embed_query(chunk_text)
                                        db.add_rag_doc(target_id=target_sub, file_name=f.name,
                                                    content=chunk_text, vector_data=v_ollama,
                                                    metadata={"engine": "ollama"})
                                    if use_openai:
                                        openai_embedder = OpenAIEmbeddings(model_name="text-embedding-3-small", 
                                                                        api_key=st.session_state.openai_api_key)
                                        v_openai = openai_embedder.embed_query(chunk_text)
                                        db.add_rag_doc(target_id=target_sub, file_name=f.name,
                                                    content=chunk_text, vector_data=v_openai,
                                                    metadata={"engine": "openai"})
                            finally:
                                # Clean up temp file
                                try: os.remove(final_tmp_path)
                                except: pass

                    # --- FINAL FAISS SYNC (only for FAISS) ---
                    if st.session_state.vector_db == "faiss":
                        subject_path = os.path.join(USER_DATA_ROOT, target_sub)
                        if use_ollama:
                            status_text.text("🔄 Rebuilding Ollama FAISS index...")
                            rebuild_rag_index(CHAT_DB_FILE, subject_path, target_sub, "llama3:8b", 
                                            ollama_models, openai_models, suffix="ollama", user_id=user_id)
                        if use_openai:
                            status_text.text("🔄 Rebuilding OpenAI FAISS index...")
                            rebuild_rag_index(CHAT_DB_FILE, subject_path, target_sub, "gpt-4o", 
                                            ollama_models, openai_models, suffix="openai", user_id=user_id)
                    
                    st.success(f"✅ Knowledge Base Processed!")
                    time.sleep(2)
                    st.rerun()

    # ============ CALL THE FRAGMENT ============
    # This isolates the chat interface for fast reloads
    chat_interface()

if __name__ == "__main__":
      main()