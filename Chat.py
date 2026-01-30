# Chat.py
from __future__ import annotations
import os
import re
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
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

load_dotenv()
CHAT_DB_FILE = "chat.db"
SUBJECTS_DIR = os.path.join("modules", "subjects")
os.makedirs(SUBJECTS_DIR, exist_ok=True)

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
        
        st.rerun()

    def clear_tts():
        st.session_state.tts_audio_bytes = None
        st.session_state.tts_audio_for = ""
     
    def format_latex(text: str) -> str:
        """Converts common LLM LaTeX delimiters to Streamlit-friendly ones."""
        text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
        text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
        return text

    # 1. AUTHENTICATION (Always first)
    is_authenticated, username = check_authentication()
    if not is_authenticated:
        st.markdown("<style>[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
        login_page()
        st.stop()

    # 2. GET USER ID FROM SESSION
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.error("⚠️ User ID not found in session. Please login again.")
        logout()
        st.stop()

    # 3. DEFINE CONFIGURATION VARIABLES
    q_url = os.getenv("QDRANT_URL", "").strip()
    q_key = os.getenv("QDRANT_API_KEY", "").strip()
    oa_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY", "")).strip()

    # Determine backend status
    using_qdrant = st.session_state.get("vector_db") == "qdrant"
    lock_ui = False

    # Perform Qdrant connectivity test if applicable
    if using_qdrant:
        if not q_url or not q_key:
            lock_ui = True
        else:
            try:
                test_client = QdrantClient(url=q_url, api_key=q_key, timeout=3)
                test_client.get_collections()
            except Exception:
                lock_ui = True
    
    # Perform OpenAI key validation
    openai_functional = False
    if oa_key:
        openai_functional = True 

    # Update session state
    st.session_state.openai_api_key = oa_key

    # 4. INITIALIZE QDRANT CLIENT
    if "qdrant_client" not in st.session_state and using_qdrant and not lock_ui:
        st.session_state.qdrant_client = QdrantClient(
            url=q_url, 
            api_key=q_key,
            timeout=10 
        )

    # 5. SET SESSION STATE DEFAULTS
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
        "qdrant_url": q_url,
        "qdrant_api_key": q_key,
        "vision_uploader_key": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # 6. MAIN APP INTERFACE
    st.title("📖 My Learning AI")

    if lock_ui:
        st.error("⚠️ Qdrant Backend Locked: Please provide API credentials in Settings or switch back to Local (FAISS).")
        
    if not openai_functional:
        st.toast("**Warning:** OpenAI features are limited.", icon="⚠️")

    # 7. INITIALIZE DB MANAGER WITH USER_ID
    db = None
    if using_qdrant and not lock_ui:
        try:
            active_backend = st.session_state.chat_backend
            db = DBManager(
                backend=active_backend, 
                user_id=user_id,  # ← Added user_id
                db_file=CHAT_DB_FILE, 
                qdrant_url=q_url, 
                qdrant_api_key=q_key
            )
        except Exception as e:
            lock_ui = True
            st.error(f"⚠️ Qdrant Connection Failed: {e}")
                
    if db is None:
        db = DBManager(
            backend="sqlite", 
            user_id=user_id,  # ← Added user_id
            db_file=CHAT_DB_FILE
        )

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

        if selected_sub != "General":
            if st.button(f"🗑️ Delete {selected_sub}", type="primary", use_container_width=True):
                st.session_state.confirm_delete = True

            if st.session_state.get("confirm_delete"):
                st.warning(f"⚠️ Are you sure? This will permanently delete all chats and RAG data for **{selected_sub}**.")
                
                col_yes, col_no = st.columns(2)
                
                with col_yes:
                    if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
                        # Use user-specific path
                        user_subjects_dir = os.path.join(SUBJECTS_DIR, f"user_{user_id}")
                        db.delete_subject(selected_sub, rag_index_dir=user_subjects_dir)
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
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.rerun()
        st.markdown("---")

        all_chats = []
        if not lock_ui:
            try:
                all_chats = db.load_all_chats_by_subject(st.session_state.current_subject)
            except Exception as e:
                st.warning(f"⚠️ Could not load chats: {e}")

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
        "llava:7b": "Llava (Vision Model)",
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
        if st.session_state.current_subject == "General":
            st.info("💡 **Note:** You cannot upload documents to the 'General' subject. Create a new subject in the sidebar first.")
        else:
            uploaded_files = st.file_uploader("Add files", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
            
            if st.button("📂 Process RAG", disabled=lock_ui):
                target_sub = st.session_state.current_subject
                
                if uploaded_files:
                    # Use user-specific directory
                    raw_dir = os.path.join("modules", "subjects", f"user_{user_id}", target_sub, "raw")
                    os.makedirs(raw_dir, exist_ok=True)
                    existing_filenames = os.listdir(raw_dir)
                    
                    use_openai = bool(st.session_state.openai_api_key)
                    use_ollama = True 

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    msg_placeholder = st.empty()
                    
                    qdrant_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

                    for i, f in enumerate(uploaded_files):
                        percent_val = (i + 1) / len(uploaded_files)
                        progress_bar.progress(percent_val)
                        
                        if f.name in existing_filenames:
                            msg_placeholder.warning(f"⚠️ {f.name} already exists. Skipping...")
                            time.sleep(1.5)
                            msg_placeholder.empty()
                            continue 

                        status_text.text(f"⏳ Processing {i+1}/{len(uploaded_files)}: {f.name}...")
                        
                        f.seek(0)
                        # Save to user-specific directory
                        save_uploaded_files(target_sub, [f])
                        # Use user-specific path for processing
                        original_path = os.path.join("modules", "subjects", f"user_{user_id}", target_sub, "raw", f.name)
                        
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

                        if st.session_state.vector_db == "faiss":
                            if use_ollama:
                                db.add_rag_doc(target_id=target_sub, file_name=final_filename, 
                                            content=content, metadata={"engine": "ollama"})
                            if use_openai:
                                db.add_rag_doc(target_id=target_sub, file_name=final_filename, 
                                            content=content, metadata={"engine": "openai"})

                        elif st.session_state.vector_db == "qdrant":
                            chunks = qdrant_splitter.split_text(content)
                            for chunk_text in chunks:
                                if use_ollama:
                                    ollama_embedder = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
                                    v_ollama = ollama_embedder.embed_query(chunk_text)
                                    db.add_rag_doc(target_id=target_sub, file_name=final_filename,
                                                content=chunk_text, vector_data=v_ollama,
                                                metadata={"engine": "ollama"})
                                if use_openai:
                                    openai_embedder = OpenAIEmbeddings(model_name="text-embedding-3-small", 
                                                                    api_key=st.session_state.openai_api_key)
                                    v_openai = openai_embedder.embed_query(chunk_text)
                                    db.add_rag_doc(target_id=target_sub, file_name=final_filename,
                                                content=chunk_text, vector_data=v_openai,
                                                metadata={"engine": "openai"})

                    if st.session_state.vector_db == "faiss":
                        # Use user-specific path
                        subject_path = os.path.join(SUBJECTS_DIR, f"user_{user_id}", target_sub)
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

    # ---------------- CHAT INTERFACE ----------------
    message_container = st.container(height=500)
    for msg in st.session_state.messages:
        with message_container.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "😎"):
            # Show RAG indicator if used
            rag_indicator = "📄 " if msg.get("used_rag") else ""
            st.markdown(rag_indicator + msg["content"])
            
            if msg.get("sources"):
                with st.expander(f"🔍 View Source Material ({len(msg['sources'])} sources)"):
                    st.caption("📚 Information retrieved from these documents:")
                    cols = st.columns(min(len(msg["sources"]), 3))  # Max 3 columns
                    for i, source in enumerate(msg["sources"]):
                        f_name = source.get("file")
                        page_num = source.get("page", 1)
                        # Use user-specific path
                        pdf_path = os.path.join(SUBJECTS_DIR, f"user_{user_id}", st.session_state.current_subject, "raw", f_name)
                        
                        col_idx = i % 3  # Cycle through columns
                        with cols[col_idx]:
                            if os.path.exists(pdf_path):
                                img = render_pdf_page_image(pdf_path, page_num)
                                if img:
                                    st.image(img, caption=f"{f_name} (p. {page_num})", use_container_width=True)
                            else:
                                st.warning(f"📄 {f_name} (p. {page_num})\n*Preview unavailable*")

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
    st.markdown("### Enter your prompt")
    is_vision_model = any(v in st.session_state.selected_model.lower() for v in ["llava", "qwen", "gpt-4o"])

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
                    clear_tts(); st.rerun()
            except Exception as e: st.error(f"Mic Error: {e}")

    with col_text:
        typed_prompt = st.chat_input(f"Ask about {st.session_state.current_subject}...", disabled=lock_ui)
        if typed_prompt:
            st.session_state.temp_prompt = typed_prompt
            clear_tts(); st.rerun()

    if is_vision_model:
        uploaded_img = st.file_uploader(
            "🖼️ Upload an image",
            type=["png", "jpg", "jpeg"],
            key=f"vision_uploader_{st.session_state.vision_uploader_key}"
        )

    if uploaded_img is not None:
        st.session_state.vision_images = [uploaded_img.getvalue()]
        col_img, _ = st.columns([1, 3]) 
        with col_img:
            st.image(uploaded_img, use_container_width=True)
            
    # ---------------- PROCESSING ----------------
    if st.session_state.get("temp_prompt"):
        user_p = st.session_state.temp_prompt
        st.session_state.temp_prompt = None
        
        if st.session_state.current_chat_id is None:
            st.session_state.current_chat_id = db.create_new_chat(
                st.session_state.current_subject,
                st.session_state.selected_model
            )
        
        chat_id = st.session_state.current_chat_id
        db.add_message(chat_id, "user", user_p)
        st.session_state.messages.append({"role": "user", "content": user_p})

        if len(st.session_state.messages) <= 1:
            client_openai = get_openai_client() if "gpt" in st.session_state.selected_model else None
            new_title = generate_chat_title(user_p, st.session_state.selected_model, client_openai)
            db.update_chat_title(chat_id, new_title)

        with message_container.chat_message("user", avatar="😎"):
            st.markdown(user_p)

        # --- RAG SEARCH ---
        rag_content, raw_docs, has_docs = "", [], False
        sources_to_display = []
        search_id = st.session_state.current_subject
        
        if st.session_state.rag_enabled:
            if st.session_state.vector_db == "qdrant":
                try:
                    with st.spinner("🔍 Searching Knowledge Base (Qdrant)..."):
                        rag_content, raw_docs, search_success = get_relevant_rag(
                            CHAT_DB_FILE, None, search_id, user_p,
                            st.session_state.selected_model, ollama_models, openai_models,
                            vector_db="qdrant",
                            qdrant_url=st.session_state.qdrant_url,
                            qdrant_api_key=st.session_state.qdrant_api_key
                        )
                except Exception as e: 
                    st.error(f"Qdrant RAG Error: {e}")
            elif st.session_state.vector_db == "faiss":
                # Use user-specific path
                subject_index_dir = os.path.join(SUBJECTS_DIR, f"user_{user_id}", search_id)
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
                    if hasattr(doc, "payload"):
                        m = doc.payload
                    elif hasattr(doc, "metadata"):
                        m = doc.metadata
                    else:
                        m = {}

                    f_name = m.get('file') or m.get('file_name') 
                    p_num = m.get('page', 1)
                    
                    identifier = f"{f_name}_{p_num}"
                    if f_name and identifier not in seen:
                        sources_to_display.append({"file": f_name, "page": p_num})
                        seen.add(identifier)
                
                # Add info message if RAG found results
                if sources_to_display:
                    print(f"✅ RAG: Found {len(sources_to_display)} unique source pages")
            else:
                print("⚠️ RAG: No documents returned from search")

        # --- CONSTRUCT PROMPT ---
        if has_docs and rag_content.strip():
            indicator = "📄 "
            enhanced_p = f"""You are a helpful assistant with access to specific reference documents.

GUIDELINES:
1. Prioritize the information in the documents below.
2. If the documents are slightly related but don't give a full answer, use them as a primary source and supplement with general knowledge, but clearly state what comes from the documents.
3. If the documents are completely unrelated to the topic, let the user know by saying "I am unable to find the information based on the documents uploaded." and try to be helpful.
4. If the user uses shorthand (like "linear ode" for "linear ordinary differential equation"), interpret their intent based on the document context.

REFERENCE DOCUMENTS:
{rag_content}

REFERENCE DOCUMENTS:
{rag_content}

USER QUESTION: {user_p}

Your answer (based STRICTLY on the documents above):"""
        else:
            indicator = ""
            enhanced_p = user_p

        # --- LLM CALL ---
        if st.session_state.selected_model in ollama_models:
            images_to_send = [get_b64_image(img_data) for img_data in st.session_state.vision_images] \
                                    if st.session_state.vision_images else None

            with message_container.chat_message("assistant", avatar="🤖"):
                stream_placeholder = st.empty()
                full_res = ""

                user_message = {"role": "user", "content": enhanced_p}
                if images_to_send:
                    user_message["images"] = images_to_send

                for chunk in ollama.chat(
                    model=st.session_state.selected_model,
                    stream=True,
                    messages=[user_message]
                ):
                    full_res += chunk.get("message", {}).get("content", "")
                    stream_placeholder.markdown(indicator + format_latex(full_res))

                process_assistant_completion(chat_id, format_latex(full_res), has_docs, sources=sources_to_display)

        elif st.session_state.selected_model in openai_models:
            client = get_openai_client()
            with message_container.chat_message("assistant", avatar="🤖"):
                stream_placeholder = st.empty()
                full_res = ""

                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

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
                    history.append({"role": "user", "content": enhanced_p})

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

                process_assistant_completion(chat_id, format_latex(full_res), has_docs, sources=sources_to_display)

if __name__ == "__main__":
    main()