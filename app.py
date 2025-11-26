# app.py
import streamlit as st
import ollama
import os
import openai
from openai import OpenAI

# --- Local Module Imports (UPDATED) ---
from modules.db import (
    init_db, create_new_chat, add_message,
    load_all_chats, load_chat, add_rag_doc, get_chat_documents
)
from modules.rag import rebuild_rag_index, get_relevant_rag
from modules.file_processor import extract_text_from_file

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
    st.title("💬 AI Playground with RAG (Ollama + OpenAI)")
    init_db(CHAT_DB_FILE)

    # --- Session state ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "uploaded_files_to_process" not in st.session_state:
        st.session_state.uploaded_files_to_process = []
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None

    # --- Sidebar ---
    with st.sidebar:
        st.header("💾 Chat Sessions")
        if st.button("🆕 New Chat"):
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.session_state.selected_model = None
            st.session_state.uploaded_files_to_process = []
            st.rerun()

        for chat_id, model_name, _ in load_all_chats(CHAT_DB_FILE):
            if st.button(f"{model_name} Chat", key=f"chat_{chat_id}"):
                msgs, model = load_chat(CHAT_DB_FILE, chat_id)
                st.session_state.messages = msgs
                st.session_state.current_chat_id = chat_id
                st.session_state.selected_model = model

        st.markdown("---")

        st.session_state.rag_mode = st.checkbox("💡 Enable RAG", value=True)

        # OpenAI API Key input
        with st.expander("🔑 OpenAI API Key"):
            api_key_input = st.text_input(
                "Enter your OpenAI API Key",
                type="password",
                value=st.session_state.get("openai_api_key", "")
            )
            if api_key_input:
                st.session_state.openai_api_key = api_key_input
                st.success("✅ API key saved")

    # --- Available Models ---
    try:
        ollama_models = tuple(m['model'] for m in ollama.list().get("models", []))
    except Exception:
        ollama_models = ()
    openai_models = ("gpt-3.5-turbo", "gpt-4") if st.session_state.openai_api_key else ()
    available_models = ollama_models + openai_models

    if not available_models:
        st.warning("⚠️ No models available. (Check Ollama server or add OpenAI key)")
        st.stop()

    # Disable the dropdown if a chat already exists
    disable_model_select = st.session_state.current_chat_id is not None

    selected_model = st.selectbox(
        "🧠 Choose model",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        disabled=disable_model_select
    )

    if not disable_model_select:
        st.session_state.selected_model = selected_model

    message_container = st.container()

    # --- RAG File Uploader ---
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
                chat_id_to_process = create_new_chat(CHAT_DB_FILE, st.session_state.selected_model)
                st.session_state.current_chat_id = chat_id_to_process
                st.session_state.messages = []

            existing_docs = get_chat_documents(CHAT_DB_FILE, chat_id_to_process)
            processed_files_in_batch = set()

            for f in st.session_state.uploaded_files_to_process:
                if f.name in existing_docs or f.name in processed_files_in_batch:
                    st.warning(f"⚠️ {f.name} already exists. Skipping.")
                    continue

                processed_files_in_batch.add(f.name)

                content = extract_text_from_file(f)
                add_rag_doc(CHAT_DB_FILE, chat_id_to_process, f.name, content)
                sys_msg = f"📄 File uploaded: {f.name}"
                add_message(CHAT_DB_FILE, chat_id_to_process, "system", sys_msg)
                st.session_state.messages.append({"role": "system", "content": sys_msg})

            rebuild_rag_index(
                CHAT_DB_FILE, RAG_INDEX_DIR, chat_id_to_process, 
                st.session_state.selected_model, ollama_models, openai_models
            )
            st.session_state.uploaded_files_to_process = []
            st.rerun()

    # --- Display chat history ---
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "😎" if msg["role"] == "user" else "⚙️"
        prefix = "📄 " if msg.get("used_rag") else ""
        with message_container.chat_message(msg["role"], avatar=avatar):
            st.markdown(prefix + msg["content"])

    # --- Chat input ---
    if prompt := st.chat_input("Type your message here..."):
        use_rag = False
        enhanced_prompt = prompt
        rag_content = ""
        
        chat_id = st.session_state.current_chat_id or create_new_chat(CHAT_DB_FILE, st.session_state.selected_model, prompt)
        st.session_state.current_chat_id = chat_id
        
        add_message(CHAT_DB_FILE, chat_id, "user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "used_rag": 0})
        with message_container.chat_message("user", avatar="😎"):
            st.markdown(prompt)
            
        # --- RAG Enhancement ---
        if st.session_state.rag_mode:
            rag_content, has_docs = get_relevant_rag(
                CHAT_DB_FILE, RAG_INDEX_DIR, chat_id, prompt, 
                st.session_state.selected_model, ollama_models, openai_models
            )
            use_rag = has_docs
        
        if use_rag and rag_content.strip():
            enhanced_prompt = f"""You have access to background documents that may be relevant to the user's question.
            Use the documents to answer if relevant. If the documents do not contain the answer, provide the answer from your own knowledge.
            Answer clearly and directly. Include explanations if needed. Do NOT reveal metadata unless explicitly asked.
            
            Background documents:
            {rag_content}

            Question: {prompt}"""
        else:
            enhanced_prompt = prompt
            use_rag = False

        # --- Ollama Model ---
        if st.session_state.selected_model in ollama_models:
            avatar = "🤖"
            with message_container.chat_message("assistant", avatar=avatar):
                stream_placeholder = st.empty()
                streamed_text = ""
                try:
                    for chunk in ollama.chat(
                        model=st.session_state.selected_model,
                        messages=[{"role": "user", "content": enhanced_prompt}],
                        stream=True
                    ):
                        delta = chunk.get("message", {}).get("content", "") or chunk.get("delta", "")
                        if delta:
                            streamed_text += delta
                            display_text = ("📄 " if use_rag else "") + streamed_text
                            stream_placeholder.markdown(display_text)
                    
                    add_message(CHAT_DB_FILE, chat_id, "assistant", streamed_text, used_rag=int(use_rag))
                    st.session_state.messages.append({"role": "assistant", "content": streamed_text, "used_rag": int(use_rag)})
                
                except Exception as e:
                    st.error(f"Error: {e}")
                    add_message(CHAT_DB_FILE, chat_id, "assistant", f"[ERROR] {e}", used_rag=0)

        # --- OpenAI Model ---
        elif st.session_state.selected_model in openai_models:
            client = OpenAI(api_key=st.session_state.openai_api_key)
            try:
                messages_for_openai = [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                ]
                
                # Replace the last user message with the RAG-enhanced prompt
                if use_rag:
                    for i in range(len(messages_for_openai) - 1, -1, -1):
                        if messages_for_openai[i]["role"] == "user":
                            messages_for_openai[i]["content"] = enhanced_prompt
                            break
                
                streamed_text = ""
                avatar = "🤖"
                with message_container.chat_message("assistant", avatar=avatar):
                    stream_placeholder = st.empty()
                    stream = client.chat.completions.create(
                        model=st.session_state.selected_model,
                        messages=messages_for_openai,
                        temperature=0.7,
                        max_tokens=1000,
                        stream=True
                    )

                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta is not None:
                            streamed_text += delta
                            display_text = ("📄 " if use_rag else "") + streamed_text
                            stream_placeholder.markdown(display_text)

                add_message(CHAT_DB_FILE, chat_id, "assistant", streamed_text, used_rag=int(use_rag))
                st.session_state.messages.append({"role": "assistant", "content": streamed_text, "used_rag": int(use_rag)})

            except openai.AuthenticationError:
                st.error("❌ Invalid OpenAI API key. Please check the key in the sidebar.")
                add_message(CHAT_DB_FILE, chat_id, "assistant", "[ERROR] Invalid API Key.", used_rag=0)
            except Exception as e:
                st.error(f"OpenAI Error: {e}")
                add_message(CHAT_DB_FILE, chat_id, "assistant", f"[ERROR] {e}", used_rag=0)

if __name__ == "__main__":
    main()