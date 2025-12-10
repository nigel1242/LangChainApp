import streamlit as st
import ollama
import os
import openai
from openai import OpenAI
import sys
from pathlib import Path

from dotenv import load_dotenv
import sqlite3

# Load environment variables
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Local Module Imports ---
from modules.login import check_authentication, login_page, logout

from modules.db_manager import DBManager

from modules.functions import (
    OllamaEmbeddings,
    OpenAIEmbeddings,
    extract_text_from_file,
    rebuild_rag_index,
    get_relevant_rag
)
        
# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="My Learning AI",
    page_icon="💬",
    layout="wide",
)

CHAT_DB_FILE = "chat_playground.db"
RAG_INDEX_DIR = "rag_indices"
os.makedirs(RAG_INDEX_DIR, exist_ok=True)

# ---------------------- MAIN APP -----------------------
def main():
    # --- Authentication Check ---
    is_authenticated, username = check_authentication()
    
    if not is_authenticated:
        login_page()
        st.stop()
    
    # --- User is authenticated, show main app ---
    
    # ------------------ Initialize session state ------------------
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
        "openai_api_key": os.getenv("OPENAI_API_KEY", "")
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Initialize database manager
    db = DBManager(
        backend=st.session_state.chat_backend,
        db_file=CHAT_DB_FILE,
        qdrant_url=st.session_state.qdrant_url,
        qdrant_api_key=st.session_state.qdrant_api_key
    )

    # --- Header with logout ---
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("💬 AI Playground with RAG (Ollama + OpenAI)")
    with col2:
        st.write(f"👤 {username}")
        if st.button("🚪 Logout"):
            logout()
            st.rerun()

    # --- Sidebar ---
    with st.sidebar:
        st.header("💾 Chat Sessions")
        
        if st.button("🆕 New Chat"):
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.session_state.selected_model = None
            st.session_state.uploaded_files_to_process = []
            st.rerun()

        # Load all chats
        all_chats = db.load_all_chats()
        for chat_id, model_name, _ in all_chats:
            if st.button(f"{model_name} Chat", key=f"chat_{chat_id}"):
                msgs, model = db.load_chat(chat_id)
                st.session_state.messages = msgs
                st.session_state.current_chat_id = chat_id
                st.session_state.selected_model = model
                st.rerun()

        st.markdown("---")

        # RAG Settings
        st.subheader("⚙️ RAG Settings")
        
        # Enable/Disable RAG
        st.session_state.rag_enabled = st.checkbox("💡 Enable RAG", value=st.session_state.rag_enabled)
        
        # Vector Database Selection
        st.session_state.vector_db = st.selectbox(
            "🗄️ Vector Database",
            ["faiss", "qdrant"],
            index=0 if st.session_state.vector_db == "faiss" else 1
        )
        
        # Show current Qdrant URL (read-only info from .env)
        if st.session_state.vector_db == "qdrant":
            if st.session_state.qdrant_url:
                st.info("🔗 Qdrant URL found")
            else:
                st.warning("⚠️ Qdrant URL not found in .env")
            if st.session_state.qdrant_api_key:
                st.success(f"✅ Qdrant API Key loaded (ends with ...{st.session_state.qdrant_api_key[-4:]})")
            else:
                st.error("❌ Qdrant API Key not found in .env")

        # OpenAI API key
        with st.expander("🔑 API Keys"):
            if st.session_state.openai_api_key:
                st.success("✅ OpenAI API Key loaded from .env")
                override_key = st.text_input(
                    "Override OpenAI API Key (optional)",
                    type="password",
                    placeholder="Leave empty to use .env key"
                )
                if override_key:
                    st.session_state.openai_api_key = override_key
                    st.success("✅ Using custom API key")
            else:
                api_key_input = st.text_input(
                    "Enter your OpenAI API Key",
                    type="password"
                )
                if api_key_input:
                    st.session_state.openai_api_key = api_key_input
                    st.success("✅ API key saved")

    # --- Available Models ---
    try:
        ollama_models = tuple(m['model'] for m in ollama.list().get("models", []))
    except Exception:
        ollama_models = ()
    openai_models = ("gpt-3.5-turbo","gpt-4") if st.session_state.openai_api_key else ()
    available_models = ollama_models + openai_models

    if not available_models:
        st.warning("⚠️ No models available. Check Ollama server or OpenAI key")
        st.stop()

    disable_model_select = st.session_state.current_chat_id is not None
    selected_model = st.selectbox(
        "🧠 Choose model",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        disabled=disable_model_select
    )
    if not disable_model_select:
        st.session_state.selected_model = selected_model

    # --- File uploader ---
    if st.session_state.selected_model:
        with st.expander("📤 Upload Documents"):
            uploaded = st.file_uploader(
                "Upload files",
                type=["txt","pdf","pptx","docx","csv"],
                accept_multiple_files=True,
                key="rag_uploader_batch"
            )
            if uploaded:
                st.session_state.uploaded_files_to_process = uploaded

        if st.session_state.uploaded_files_to_process:
            if st.button("📂 Process Uploaded Files"):
                # Create chat if it doesn't exist
                if st.session_state.current_chat_id is None:
                    st.session_state.current_chat_id = db.create_new_chat(st.session_state.selected_model)
                    st.session_state.messages = []
                
                chat_id_to_process = st.session_state.current_chat_id

                existing_docs = db.get_chat_documents(chat_id_to_process)
                processed_files_in_batch = set()

                for f in st.session_state.uploaded_files_to_process:
                    # Skip duplicates in this batch
                    if f.name in processed_files_in_batch:
                        st.warning(f"⚠️ {f.name} already exists in this batch. Skipping.")
                        continue
                    processed_files_in_batch.add(f.name)

                    content = extract_text_from_file(f)
                    db.add_rag_doc(chat_id_to_process, f.name, content)
                    sys_msg = f"📄 File uploaded: {f.name}"
                    db.add_message(chat_id_to_process, "system", sys_msg)
                    st.session_state.messages.append({"role": "system", "content": sys_msg})

                # Rebuild index with selected vector database
                rebuild_rag_index(
                    CHAT_DB_FILE, RAG_INDEX_DIR, chat_id_to_process, 
                    st.session_state.selected_model, ollama_models, openai_models,
                    vector_db=st.session_state.vector_db,
                    qdrant_url=st.session_state.qdrant_url if st.session_state.vector_db == "qdrant" else None,
                    qdrant_api_key=st.session_state.qdrant_api_key if st.session_state.vector_db == "qdrant" else None
                )
                st.session_state.uploaded_files_to_process = []
                st.rerun()

    # --- Chat history ---
    message_container = st.container()
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"]=="assistant" else "😎"
        prefix = "📄 " if msg.get("used_rag") else ""
        with message_container.chat_message(msg["role"], avatar=avatar):
            st.markdown(prefix + msg["content"])

    # --- Chat input ---
    if prompt := st.chat_input("Type your message here..."):
        use_rag = False
        enhanced_prompt = prompt
        rag_content = ""
        
        # Create new chat if needed
        if st.session_state.current_chat_id is None:
            chat_id = db.create_new_chat(st.session_state.selected_model, prompt)
            st.session_state.current_chat_id = chat_id
        else:
            chat_id = st.session_state.current_chat_id
        
        db.add_message(chat_id, "user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "used_rag": 0})
        with message_container.chat_message("user", avatar="😎"):
            st.markdown(prompt)
            
        # --- RAG Enhancement ---
        if st.session_state.rag_enabled:
            rag_content, has_docs = get_relevant_rag(
                CHAT_DB_FILE, RAG_INDEX_DIR, chat_id, prompt, 
                st.session_state.selected_model, ollama_models, openai_models,
                vector_db=st.session_state.vector_db,
                qdrant_url=st.session_state.qdrant_url if st.session_state.vector_db == "qdrant" else None,
                qdrant_api_key=st.session_state.qdrant_api_key if st.session_state.vector_db == "qdrant" else None
            )
            use_rag = has_docs
        else:
            rag_content = ""
            use_rag = False
        
        if use_rag and rag_content.strip():
            enhanced_prompt = f"""You have access to background documents that may be relevant to the user's question.
Use the documents to answer if relevant. If the documents do not contain the answer, provide the answer from your own knowledge.
Answer clearly and directly. Include explanations if needed. DO NOT reveal metadata unless explicitly asked.

Background documents:
{rag_content}

Question: {prompt}"""
            use_rag = True
        else:
            sys_msg = "⚠️ RAG could not find relevant documents. Answering based on AI knowledge."
            db.add_message(chat_id, "assistant", sys_msg, used_rag=0)
            st.session_state.messages.append({"role":"assistant","content":sys_msg,"used_rag":0})
            with message_container.chat_message("assistant", avatar="🤖"):
                st.markdown(sys_msg)
            enhanced_prompt = prompt
            use_rag = False

        # --- Ollama response ---
        if st.session_state.selected_model in ollama_models:
            avatar = "🤖"
            with message_container.chat_message("assistant", avatar=avatar):
                stream_placeholder = st.empty()
                streamed_text = ""
                try:
                    for chunk in ollama.chat(
                        model=st.session_state.selected_model,
                        messages=[{"role":"user","content":enhanced_prompt}],
                        stream=True
                    ):
                        delta = chunk.get("message", {}).get("content","") or chunk.get("delta","")
                        if delta:
                            streamed_text += delta
                            display_text = ("📄 " if use_rag else "") + streamed_text
                            stream_placeholder.markdown(display_text)
                    
                    db.add_message(chat_id, "assistant", streamed_text, used_rag=int(use_rag))
                    st.session_state.messages.append({"role": "assistant", "content": streamed_text, "used_rag": int(use_rag)})
                
                except Exception as e:
                    st.error(f"Error: {e}")
                    db.add_message(chat_id, "assistant", f"[ERROR] {e}", used_rag=0)

        # --- OpenAI response ---
        elif st.session_state.selected_model in openai_models:
            client = OpenAI(api_key=st.session_state.openai_api_key)
            try:
                messages_for_openai = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                if use_rag:
                    for i in range(len(messages_for_openai)-1, -1, -1):
                        if messages_for_openai[i]["role"]=="user":
                            messages_for_openai[i]["content"] = enhanced_prompt
                            break

                streamed_text = ""
                avatar = "🤖"
                with message_container.chat_message("assistant", avatar=avatar):
                    stream = client.chat.completions.create(
                        model=st.session_state.selected_model,
                        messages=messages_for_openai,
                        temperature=0.7,
                        max_tokens=1000,
                        stream=True
                    )
                    stream_placeholder = st.empty()
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            streamed_text += delta
                            display_text = ("📄 " if use_rag else "") + streamed_text
                            stream_placeholder.markdown(display_text)

                db.add_message(chat_id, "assistant", streamed_text, used_rag=int(use_rag))
                st.session_state.messages.append({"role": "assistant", "content": streamed_text, "used_rag": int(use_rag)})

            except openai.AuthenticationError:
                st.error("❌ Invalid OpenAI API key. Please check the key in the sidebar.")
                db.add_message(chat_id, "assistant", "[ERROR] Invalid API Key.", used_rag=0)
            except Exception as e:
                st.error(f"OpenAI Error: {e}")
                db.add_message(chat_id, "assistant", f"[ERROR] {e}", used_rag=0)

if __name__ == "__main__":
    main()