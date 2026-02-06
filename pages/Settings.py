import streamlit as st
import ollama
from time import sleep
import os
from openai import OpenAI
from qdrant_client import QdrantClient

# Import the DB update function from your login module
from modules.login import update_user_credentials

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="System Settings",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODEL DISPLAY CONFIG ---
MODEL_DISPLAY_NAMES = {
    "nomic-embed-text:v1.5": "Nomic Embed (Embedding Model)",
    "llava:7b": "Llava (Vision Model)",
    "llama3:8b": "Llama3 8B (Base Model)",
}
MODELS_TO_OFFER = list(MODEL_DISPLAY_NAMES.keys())

def download_model(model_name):
    """Downloads model from Ollama with real-time percentage progress."""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stream = ollama.pull(model_name, stream=True)
        
        for chunk in stream:
            completed = chunk.get('completed')
            total = chunk.get('total')
            status = chunk.get('status', 'Processing...')

            if completed is not None and total is not None and total > 0:
                percent = completed / total
                progress_bar.progress(min(percent, 1.0))
                status_text.markdown(f"📥 Downloading **{model_name}**: {percent*100:.1f}% complete")
            else:
                status_text.markdown(f"🔍 **Status:** {status}")

        st.success(f"Downloaded model: {model_name}", icon="🎉")
        st.balloons()
        sleep(2)
        st.rerun()
    except Exception as e:
        st.error(f"Failed to download model: {model_name}. Error: {str(e)}", icon="😳")

def main():
    st.subheader("⚙️ System Settings", divider="red", anchor=False)

    # --- 1. API CONFIGURATION SECTION ---
    st.subheader("🔑 API Configuration", anchor=False)
    
    with st.container(border=True):
        # SOURCE OF TRUTH: Pull from session state (which was loaded from users.db at login)
        current_openai = st.session_state.get("openai_api_key", "")
        current_qdrant_url = st.session_state.get("qdrant_url", "")
        current_qdrant_key = st.session_state.get("qdrant_api_key", "")

        col1, col2 = st.columns(2)
        with col1:
            new_openai = st.text_input("OpenAI API Key", value=current_openai, type="password")
            new_qdrant_url = st.text_input("Qdrant URL", value=current_qdrant_url)
        
        with col2:
            new_qdrant_key = st.text_input("Qdrant API Key", value=current_qdrant_key, type="password")

        if st.button("💾 Save API Credentials", type="primary"):
            error_found = False
            
            # Validation for Qdrant
            if new_qdrant_url and new_qdrant_key:
                try:
                    test_q = QdrantClient(url=new_qdrant_url, api_key=new_qdrant_key, timeout=3)
                    test_q.get_collections()
                except Exception as e:
                    st.error(f"Qdrant Connection Failed: {e}")
                    error_found = True

            # Validation for OpenAI
            if new_openai:
                try:
                    test_oa = OpenAI(api_key=new_openai)
                    test_oa.models.list()
                except Exception as e:
                    st.error(f"OpenAI Key Invalid: {e}")
                    error_found = True

            if not error_found:
                # 1. PERSIST: Save to users.db via the update function
                update_user_credentials(
                    user_id=st.session_state.user_id,
                    openai_key=new_openai,
                    q_url=new_qdrant_url,
                    q_key=new_qdrant_key
                )
                
                # 2. MEMORY: Update Session State so other pages see it immediately
                st.session_state.openai_api_key = new_openai
                st.session_state.qdrant_url = new_qdrant_url
                st.session_state.qdrant_api_key = new_qdrant_key
                
                # 3. ENVIRONMENT: Update os.environ so existing client calls work
                os.environ["OPENAI_API_KEY"] = new_openai
                os.environ["QDRANT_URL"] = new_qdrant_url
                os.environ["QDRANT_API_KEY"] = new_qdrant_key

                # Reset the verification flag to force Chat.py to re-test the new connection
                st.session_state.backend_verified = False
                
                st.success("Credentials saved to your secure profile!")
                sleep(1)
                st.rerun()

    st.divider()

    # --- 2. SERVICE STATUS CHECKS ---
    # Check if the keys actually exist in the session now
    openai_ready = bool(st.session_state.get("openai_api_key"))
    
    ollama_online = True
    installed_models = []
    try:
        models_info = ollama.list()
        installed_models = [m.model for m in models_info["models"]]
    except Exception:
        ollama_online = False

    # --- 3. VOICE SELECTION SETTINGS ---
    st.subheader("🔊 Voice Selection", anchor=False)
    
    voice_options = ['alloy', 'nova', 'shimmer', 'echo', 'onyx', 'fable', 'ash', 'sage', 'coral']
    current_voice = st.session_state.get("tts_voice", "alloy")
    
    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        st.session_state.tts_voice = st.selectbox(
            "🎧 Select Voice for Answer Audio",
            voice_options,
            index=voice_options.index(current_voice) if current_voice in voice_options else 0,
            disabled=not openai_ready
        )

    with col_v2:
        st.write(" ")
        st.write(" ") 
        if st.button("▶️ Preview Voice", disabled=not openai_ready, width='stretch'):
            preview_text = f"Hello, I am the {st.session_state.tts_voice} voice. How do I sound?"
            try:
                from modules.functions import speak_text 
                client = OpenAI(api_key=st.session_state.openai_api_key)
                audio_bytes = speak_text(preview_text, client, st.session_state.tts_voice)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                st.error(f"TTS Error: {e}")

    st.divider()

    # --- 4. DOWNLOAD MODELS SECTION ---
    st.subheader("📥 Download Models", anchor=False)
    if not ollama_online:
        st.error("Ollama is not running. Please start Ollama to download models.")
    
    installed_model_bases = {m.split('@')[0] for m in installed_models}
    cols = st.columns(len(MODELS_TO_OFFER))
    
    for i, model_name in enumerate(MODELS_TO_OFFER):
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        is_installed = model_name in installed_model_bases
        
        with cols[i]:
            if is_installed:
                st.button(f"✅ {display_name}", disabled=True, width='stretch')
            else:
                if st.button(f"📥 Download {display_name}", key=f"dl_{model_name}", disabled=not ollama_online, width='stretch'):
                    download_model(model_name)

    st.divider()

    # --- 5. DELETE MODELS SECTION ---
    st.subheader("🗑️ Delete Models", anchor=False)
    if installed_models and ollama_online:
        delete_map = {MODEL_DISPLAY_NAMES.get(m.split('@')[0], m): m for m in installed_models}
        selected_del = st.selectbox("Select a model to remove", options=["-- Select a Model --"] + list(delete_map.keys()))

        if selected_del != "-- Select a Model --":
            m_to_delete = delete_map[selected_del]
            if st.button(f"Confirm Deletion of {selected_del}", type="primary", width='stretch'):
                try:
                    ollama.delete(m_to_delete)
                    st.success(f"Deleted {m_to_delete}")
                    sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

if __name__ == "__main__":
    main()