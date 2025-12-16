# app.py (Working version with Qdrant Chunking fix)
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import ollama
import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from openai import OpenAI

# --- Local Module Imports ---
from modules.db_manager import DBManager
from modules.functions import (
    OllamaEmbeddings,
    OpenAIEmbeddings,
    extract_text_from_file,
    rebuild_rag_index,
    get_relevant_rag,
    RecursiveCharacterTextSplitter,
)
from modules.login import check_authentication, login_page, logout

# Optional Qdrant helper (only used if vector_db == "qdrant")
try:
    from modules.qdrant_db import search_rag_docs_qdrant

    HAS_QDRANT_HELPER = True
except Exception:
    HAS_QDRANT_HELPER = False

# ------------------ ENV + PATH ------------------
load_dotenv()

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="My Learning AI",
    page_icon="💬",
    layout="wide",
)

CHAT_DB_FILE = "chat_playground.db"
RAG_INDEX_DIR = "rag_indices"
os.makedirs(RAG_INDEX_DIR, exist_ok=True)


# ------------------ OPENAI HELPERS ------------------
def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client using the API key from session_state or .env.
    """
    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OpenAI API key missing. Set it in the sidebar or .env.")
    return OpenAI(api_key=api_key)


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """
    Use OpenAI Whisper to convert recorded audio to text.
    """
    client = get_openai_client()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en", 
            )
        return (resp.text or "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def speak_text(answer: str) -> bytes | None:
    """
    Use OpenAI TTS to turn a text answer into MP3 bytes.
    """
    answer = (answer or "").strip()
    if not answer:
        return None

    try:
        client = get_openai_client()
    except RuntimeError:
        return None

    try:
        voice = st.session_state.get("tts_voice", "alloy")
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=answer,
        )
        return resp.read()
    except Exception as e:
        st.info(f"(Could not generate audio answer: {e})")
        return None


# ---------------------- MAIN APP -----------------------
def main():
    # Placeholder definitions for audio utilities
    def clear_tts():
        st.session_state.tts_audio_bytes = None
        st.session_state.tts_audio_for = ""
    # Placeholder for message container
    message_container = st.container()

    # -------- AUTHENTICATION GATE + SIDEBAR HIDING --------
    is_authenticated, username = check_authentication()
    if not is_authenticated:
        st.markdown("""
            <style>
                [data-testid="stSidebar"] {
                    display: none;
                }
            </style>
        """, unsafe_allow_html=True)
        login_page()
        st.stop()

    # -------- SESSION STATE DEFAULTS --------
    defaults = {
        "chat_backend": "sqlite",
        "vector_db": "faiss",
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
        "messages": [],
        "current_chat_id": None,
        "selected_model": None,
        "uploaded_files_to_process": [],
        "rag_enabled": True,
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "tts_voice": "alloy",
        "last_audio_bytes": None,
        "chat_question_input": "",
        "prev_storage_backend": "Local (FAISS + SQLite)",
        "tts_audio_bytes": None,
        "tts_audio_for": "",
        "last_assistant_text": "",
        "voice_prompt": "",
        "trigger_send": False,
        "temp_prompt": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    st.title("📖 My Learning AI")

    # -------- DB MANAGER --------
    db = DBManager(
        backend=st.session_state.chat_backend,
        db_file=CHAT_DB_FILE,
        qdrant_url=st.session_state.qdrant_url,
        qdrant_api_key=st.session_state.qdrant_api_key,
    )

    # -------- Restore chat from DB if needed --------
    if st.session_state.get("current_chat_id") and not st.session_state.get("messages"):
        msgs, model = db.load_chat(st.session_state.current_chat_id)
        st.session_state["messages"] = msgs
        st.session_state["selected_model"] = model

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown(f"**Logged in as: {username}**")
        if st.button("Logout"):
            logout()
            st.rerun()
        st.markdown("---")

        st.header("Chat Sessions")
        all_chats = db.load_all_chats()

        if st.button("🆕 New Chat"):
            st.session_state.update(
                {
                    "messages": [],
                    "current_chat_id": None,
                    "selected_model": None,
                    "uploaded_files_to_process": [],
                    "chat_question_input": "",
                    "voice_prompt": "",
                    "trigger_send": False,
                }
            )
            clear_tts()
            st.session_state.last_assistant_text = ""
            st.rerun()

        # List chats
        for chat_id, model_name, _ in all_chats:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"{model_name} Chat", key=f"chat_{chat_id}"):
                    msgs, model = db.load_chat(chat_id)
                    st.session_state.update(
                        {
                            "messages": msgs,
                            "current_chat_id": chat_id,
                            "selected_model": model,
                            "chat_question_input": "",
                            "voice_prompt": "",
                            "trigger_send": False,
                        }
                    )
                    clear_tts()
                    last_a = ""
                    for m in reversed(msgs):
                        if m.get("role") == "assistant":
                            last_a = m.get("content", "") or ""
                            break
                    st.session_state.last_assistant_text = last_a

            with col2:
                if st.button("✕", key=f"delete_{chat_id}"):
                    # Use DBManager's delete function, which handles both backends
                    db.delete_chat(chat_id, rag_index_dir=RAG_INDEX_DIR)
                    st.session_state.update(
                        {
                            "messages": [],
                            "current_chat_id": None,
                            "selected_model": None,
                            "chat_question_input": "",
                            "voice_prompt": "",
                            "trigger_send": False,
                        }
                    )
                    clear_tts()
                    st.session_state.last_assistant_text = ""
                    st.rerun()

        st.markdown("---")

        # Storage backend switch
        previous_storage_backend = st.session_state.get(
            "prev_storage_backend", "Local"
        )
        st.session_state.storage_backend = st.selectbox(
            "🗄️ Storage Backend",
            ["Local", "Remote (Qdrant)"],
            index=0 if st.session_state.vector_db == "faiss" else 1,
        )

        if st.session_state.storage_backend != previous_storage_backend:
            if st.session_state.storage_backend.startswith("Local"):
                st.session_state.vector_db = "faiss"
                st.session_state.chat_backend = "sqlite"
            else:
                st.session_state.vector_db = "qdrant"
                st.session_state.chat_backend = "qdrant"

            st.session_state.prev_storage_backend = st.session_state.storage_backend
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.session_state.selected_model = None
            st.session_state.uploaded_files_to_process = []
            st.session_state.chat_question_input = ""
            st.session_state.voice_prompt = ""
            st.session_state.trigger_send = False
            clear_tts()
            st.session_state.last_assistant_text = ""
            st.rerun()

        st.markdown("---")
        st.session_state.rag_enabled = st.checkbox(
            "📚 Enable (RAG)",
            value=st.session_state.rag_enabled,
            help="When enabled, the query is embedded and searched against uploaded documents.",
        )
        st.markdown("---")

        # Qdrant info
        if st.session_state.vector_db == "qdrant":
            if st.session_state.qdrant_url:
                st.info("🔗 Qdrant URL found")
            else:
                st.warning("⚠️ Qdrant URL not found in .env")
            if st.session_state.qdrant_api_key:
                st.success(
                    f"✅ Qdrant API Key loaded (ends with ...{st.session_state.qdrant_api_key[-4:]})"
                )
            else:
                st.error("❌ Qdrant API Key not found in .env")

        # OpenAI API key
        with st.expander("🔑 API Keys"):
            if st.session_state.openai_api_key:
                st.success("✅ OpenAI API Key loaded from .env")
                override_key = st.text_input(
                    "Override OpenAI API Key (optional)",
                    type="password",
                    placeholder="Leave empty to use .env key",
                )
                if override_key:
                    st.session_state.openai_api_key = override_key
                    st.success("✅ Using custom API key")
            else:
                api_key_input = st.text_input(
                    "Enter your OpenAI API Key",
                    type="password",
                )
                if api_key_input:
                    st.session_state.openai_api_key = api_key_input
                    st.success("✅ API key saved")

    # ---------------- MODELS AVAILABLE ----------------
        # Mapping function to convert model IDs to friendly names
    def get_model_display_name(model_id: str) -> str:
        """Provides a user-friendly name for a technical model ID."""
        if model_id == "gpt-3.5-turbo":
            return "OpenAI GPT-3.5 Turbo"
        if model_id == "gpt-4":
            return "OpenAI GPT-4"
        # Customize for your other Ollama models
        if model_id == "llama3:8b":
            return "Llama 3"
        if model_id == "qwen2.5vl:7b":
            return "Qwen 2.5 VL"

    # --- Model Retrieval Logic ---
    try:
        # 1. Fetch all Ollama models
        raw_ollama_models = (m["model"] for m in ollama.list().get("models", []))
        
        # 2. FILTER: Exclude models starting with 'nomic-embed-text'
        ollama_models = tuple(m for m in raw_ollama_models if not m.startswith("nomic-embed-text"))
    except Exception:
        ollama_models = ()

    openai_models = (("gpt-3.5-turbo", "gpt-4") if st.session_state.openai_api_key else ())
    if isinstance(openai_models, tuple) and openai_models and isinstance(openai_models[0], str):
        openai_models = openai_models
    else:
        openai_models = tuple(openai_models)

    available_models = ollama_models + openai_models
    # Create a list of user-friendly names for the selectbox
    display_models = [get_model_display_name(model) for model in available_models]

    if not available_models:
        st.warning("⚠️ No models available. Check Ollama server or OpenAI key.")
        st.stop()

    # --- Model Selection Logic ---
    disable_model_select = st.session_state.current_chat_id is not None

    # Determine the initial display value
    if st.session_state.selected_model in available_models:
        default_index = available_models.index(st.session_state.selected_model)
        default_display_value = display_models[default_index]
    else:
        default_index = 0
        default_display_value = display_models[0]

    selected_display_name = st.selectbox(
        "🧠 Choose model",
        display_models,  # Use friendly names for display
        index=default_index,
        disabled=disable_model_select,
    )

    # CRITICAL: Map the selected friendly name back to the technical ID
    try:
        # Get the index of the selected display name
        selected_index = display_models.index(selected_display_name)
        # Use that index to get the technical ID from the original list
        selected_model_id = available_models[selected_index]
    except ValueError:
        # Should not happen if lists are kept in sync
        selected_model_id = available_models[0] 


    if not disable_model_select:
        st.session_state.selected_model = selected_model_id

    # ---------------- FILE UPLOADS (RAG) ----------------
    if st.session_state.selected_model:
        with st.expander("📤 Upload Documents"):
            uploaded = st.file_uploader(
                "Upload files",
                type=["txt", "pdf", "pptx", "docx", "csv"],
                accept_multiple_files=True,
                key="rag_uploader_batch",
            )
            if uploaded:
                st.session_state.uploaded_files_to_process = uploaded

        if st.session_state.uploaded_files_to_process:
            if st.button("📂 Process Uploaded Files"):
                chat_id_to_process = st.session_state.current_chat_id
                if chat_id_to_process is None:
                    # Use DBManager: db.create_new_chat
                    chat_id_to_process = db.create_new_chat(st.session_state.selected_model)
                    st.session_state.current_chat_id = chat_id_to_process
                    st.session_state.messages = []

                processed_files_in_batch = set()

                for f in st.session_state.uploaded_files_to_process:
                    if f.name in processed_files_in_batch:
                        st.warning(f"⚠️ {f.name} already exists in this batch. Skipping.")
                        continue
                    processed_files_in_batch.add(f.name)

                    content = extract_text_from_file(f)

                    # --- Setup Embeddings Model (Common for Qdrant/FAISS) ---
                    if st.session_state.selected_model in ollama_models:
                        embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
                    else:
                        embeddings = OpenAIEmbeddings(
                            model_name="text-embedding-3-large",
                            api_key=st.session_state.openai_api_key,
                        )

                    if st.session_state.vector_db == "qdrant":
                        # <<< START OF CRITICAL FIX: CHUNKING LOGIC >>>
                        # 1. CHUNK THE DOCUMENT 
                        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        chunks = splitter.split_text(content)
                        
                        all_success = True
                        
                        # 2. LOOP & EMBED EACH CHUNK
                        with st.spinner(f"Embedding and uploading {len(chunks)} chunks for {f.name}..."):
                            for i, chunk in enumerate(chunks):
                                try:
                                    # Embed the small chunk (Resolves the context length error)
                                    vector_data = embeddings.embed_documents([chunk])[0]
                                    
                                    # 3. Add chunk to Qdrant - Use DBManager: db.add_rag_doc
                                    success, _ = db.add_rag_doc(
                                        chat_id_to_process, 
                                        f"{f.name} (chunk {i+1}/{len(chunks)})", 
                                        chunk, 
                                        vector_data=vector_data
                                    )
                                    if not success: all_success = False
                                except Exception as e:
                                    st.error(f"Error processing chunk {i+1} of {f.name}: {e}")
                                    all_success = False
                        
                        if all_success:
                            sys_msg = f"📄 {f.name} indexed into Qdrant successfully ({len(chunks)} chunks)! 🎉"
                        else:
                            sys_msg = f"⚠️ Finished indexing {f.name}, but some chunks failed to upload to Qdrant."
                        
                        success = all_success # Set overall success flag based on chunking process
                        # <<< END OF CRITICAL FIX: CHUNKING LOGIC >>>

                    else:
                        # SQLite / FAISS path - Use DBManager: db.add_rag_doc
                        success, sys_msg = db.add_rag_doc(chat_id_to_process, f.name, content)

                    if success:
                        db.add_message(chat_id_to_process, "assistant", sys_msg)
                        st.session_state.messages.append({"role": "assistant", "content": sys_msg})
                    else:
                        st.warning(f"⚠️ {f.name} already uploaded in this chat. Skipping.")

                    if st.session_state.vector_db == "faiss":
                        rebuild_rag_index(
                            CHAT_DB_FILE,
                            RAG_INDEX_DIR,
                            chat_id_to_process,
                            st.session_state.selected_model,
                            ollama_models,
                            openai_models,
                            vector_db="faiss",
                        )

                st.session_state.uploaded_files_to_process = []
                st.rerun()

    # ---------------- CHAT HISTORY DISPLAY ----------------
    message_container = st.container()
    tts_placeholder = st.empty()

    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "😎"
        prefix = "📄 " if msg.get("used_rag") else ""
        with message_container.chat_message(msg["role"], avatar=avatar):
            st.markdown(prefix + msg["content"])

    # ---------------- ON-DEMAND TTS CONTROLS (Latest assistant only) ----------------
    latest = (st.session_state.get("last_assistant_text") or "").strip()
    
    has_audio_bytes = st.session_state.tts_audio_bytes is not None and st.session_state.tts_audio_for == latest

    if latest:
        c1, c_spacer = st.columns([1, 7], vertical_alignment="center")
        
        with c1:
            if not has_audio_bytes:
                if st.button("🔊 Listen", use_container_width=True):
                    st.session_state.tts_audio_bytes = speak_text(latest)
                    st.session_state.tts_audio_for = latest
                    st.rerun()

        if has_audio_bytes:
            tts_placeholder.audio(st.session_state.tts_audio_bytes, format="audio/mp3")

    # ---------------- INPUT ROW: CHAT (Enter) + MIC ----------------
    st.markdown("### Ask with text or voice")

    col_text, col_mic = st.columns([7, 1])

    with col_mic:
        st.caption("🎙 Tap to record")
        audio_bytes = audio_recorder(
            text="",
            pause_threshold=1.0,
            sample_rate=41_000,
            icon_size="2x",
        )

    if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
        st.session_state["last_audio_bytes"] = audio_bytes
        try:
            with st.spinner("Transcribing…"):
                text_from_voice = transcribe_audio_bytes(audio_bytes)
            if text_from_voice:
                with message_container.chat_message("user", avatar="🎤"):
                    st.markdown(f"**(Voice to text):** {text_from_voice}")

                st.session_state.voice_prompt = text_from_voice
                st.session_state.trigger_send = True
                st.rerun()
        except Exception as e:
            st.error(f"Transcription failed: {e}")

    with col_text:
        typed_prompt = st.chat_input(
            "Type your message and press Enter…",
            key="main_chat_input"
        )

    # --- IMMEDIATE INPUT HANDLING (Captures input, clears audio, forces quick rerun) ---
    prompt = (typed_prompt or "").strip()
    if prompt or st.session_state.get("trigger_send"):
        
        if not prompt and st.session_state.get("trigger_send"):
            prompt = (st.session_state.get("voice_prompt") or "").strip()
            st.session_state.trigger_send = False
            st.session_state.voice_prompt = ""
        
        if prompt:
            st.session_state.temp_prompt = prompt
            st.session_state.last_assistant_text = ""
            clear_tts() 
            st.rerun()


    # ---------------- HANDLE RAG/LLM PROCESSING (AFTER audio clear rerun) ----------------
    if st.session_state.get("temp_prompt"):
        prompt = st.session_state.temp_prompt
        st.session_state.temp_prompt = None

        # Create new chat if needed
        if st.session_state.current_chat_id is None:
            chat_id = db.create_new_chat(st.session_state.selected_model, first_message=prompt)
            st.session_state.current_chat_id = chat_id
        else:
            chat_id = st.session_state.current_chat_id

        # Store user message
        db.add_message(chat_id, "user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "used_rag": 0})
        with message_container.chat_message("user", avatar="😎"):
            st.markdown(prompt)

        # ---- RAG SEARCH ----
        rag_content = ""
        has_docs = False
        if st.session_state.rag_enabled:
            if st.session_state.vector_db == "qdrant" and HAS_QDRANT_HELPER:
                try:
                    if st.session_state.selected_model in ollama_models:
                        embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5") 
                        query_vector = embeddings.embed_query(prompt)
                    else:
                        embeddings = OpenAIEmbeddings(
                            model_name="text-embedding-3-large",
                            api_key=st.session_state.openai_api_key,
                        )
                        query_result = embeddings.embed_documents([prompt])
                        query_vector = query_result[0] if query_result else []

                    if query_vector:
                        results = search_rag_docs_qdrant(chat_id, query_vector, limit=5)
                        if results:
                            rag_content = "\n\n".join(results)
                            has_docs = True
                except Exception as e:
                    st.error(f"RAG Search failed: {e}")
            else:
                rag_content, has_docs = get_relevant_rag(
                    CHAT_DB_FILE,
                    RAG_INDEX_DIR,
                    chat_id,
                    prompt,
                    st.session_state.selected_model,
                    ollama_models,
                    openai_models,
                    vector_db="faiss",
                )

        if has_docs and rag_content.strip():
            enhanced_prompt = f"""You have access to background documents.
Background documents:
{rag_content}

Question: {prompt}"""
            use_rag = True
        else:
            sys_msg = (
                "⚠️ RAG could not find relevant documents. "
                "Answering based on model knowledge only."
            )
            enhanced_prompt = prompt
            use_rag = False
            
        # ---- OLLAMA RESPONSE ----
        if st.session_state.selected_model in ollama_models:
            avatar = "🤖"
            streamed_text = ""
            if not use_rag:
                st.session_state.messages.append({"role": "assistant", "content": sys_msg, "used_rag": 0})
                with message_container.chat_message("assistant", avatar=avatar):
                    st.markdown(sys_msg)

            with message_container.chat_message("assistant", avatar=avatar):
                stream_placeholder = st.empty()
                try:
                    for chunk in ollama.chat(
                        model=st.session_state.selected_model,
                        messages=[{"role": "user", "content": enhanced_prompt}],
                        stream=True,
                    ):
                        delta = (
                            chunk.get("message", {}).get("content", "")
                            or chunk.get("delta", "")
                        )
                        if delta:
                            streamed_text += delta
                            display_text = ("📄 " if use_rag else "") + streamed_text
                            stream_placeholder.markdown(display_text)

                    db.add_message(chat_id, "assistant", streamed_text, used_rag=int(use_rag))
                    st.session_state.messages.append(
                        {"role": "assistant", "content": streamed_text, "used_rag": int(use_rag)}
                    )
                    st.session_state.last_assistant_text = streamed_text
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")
                    db.add_message(chat_id, "assistant", f"[ERROR] {e}", used_rag=0)

        # ---- OPENAI RESPONSE ----
        elif st.session_state.selected_model in openai_models:
            client = OpenAI(api_key=st.session_state.openai_api_key)
            try:
                messages_for_openai = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                if use_rag:
                    for i in range(len(messages_for_openai) - 1, -1, -1):
                        if messages_for_openai[i]["role"] == "user":
                            messages_for_openai[i]["content"] = enhanced_prompt
                            break
                else:
                    st.session_state.messages.append({"role": "assistant", "content": sys_msg, "used_rag": 0})
                    with message_container.chat_message("assistant", avatar="🤖"):
                        st.markdown(sys_msg)

                streamed_text = ""
                avatar = "🤖"
                with message_container.chat_message("assistant", avatar=avatar):
                    stream = client.chat.completions.create(
                        model=st.session_state.selected_model,
                        messages=messages_for_openai,
                        temperature=0.7,
                        max_tokens=1000,
                        stream=True,
                    )
                    stream_placeholder = st.empty()
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            streamed_text += delta
                            display_text = ("📄 " if use_rag else "") + streamed_text
                            stream_placeholder.markdown(display_text)

                db.add_message(chat_id, "assistant", streamed_text, used_rag=int(use_rag))
                st.session_state.messages.append(
                    {"role": "assistant", "content": streamed_text, "used_rag": int(use_rag)}
                )
                st.session_state.last_assistant_text = streamed_text
                st.rerun()

            except openai.AuthenticationError:
                st.error("❌ Invalid OpenAI API key.")
                db.add_message(chat_id, "assistant", "[ERROR] Invalid API Key.", used_rag=0)
            except Exception as e:
                st.error(f"OpenAI Error: {e}")
                db.add_message(chat_id, "assistant", f"[ERROR] {e}", used_rag=0)


if __name__ == "__main__":
    main()