import streamlit as st
import ollama
from time import sleep
import os
from dotenv import load_dotenv, set_key

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="System Settings",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- .ENV MANAGEMENT ---
ENV_FILE = ".env"

def init_env():
    """Ensures .env exists and loads the latest values into the environment."""
    if not os.path.exists(ENV_FILE):
        with open(ENV_FILE, "w") as f:
            f.write("OPENAI_API_KEY=\nQDRANT_URL=\nQDRANT_API_KEY=\n")
    load_dotenv(ENV_FILE, override=True)

def update_env_key(key, value):
    """Writes the key-value pair to .env and updates the current session."""
    set_key(ENV_FILE, key, value)
    os.environ[key] = value

init_env()

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
        
        # Pull model with streaming enabled
        stream = ollama.pull(model_name, stream=True)
        
        for chunk in stream:
            # FIX: Check if 'completed' and 'total' are both present AND not None
            # Some chunks only contain a 'status' string like "pulling manifest"
            completed = chunk.get('completed')
            total = chunk.get('total')
            status = chunk.get('status', 'Processing...')

            if completed is not None and total is not None and total > 0:
                percent = completed / total
                progress_bar.progress(min(percent, 1.0)) # Ensure it doesn't exceed 100%
                status_text.markdown(f"📥 Downloading **{model_name}**: {percent*100:.1f}% complete")
            else:
                # Display the current status if we don't have numbers yet
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
        current_openai = os.getenv("OPENAI_API_KEY", "")
        current_qdrant_url = os.getenv("QDRANT_URL", "")
        current_qdrant_key = os.getenv("QDRANT_API_KEY", "")

        col1, col2 = st.columns(2)
        with col1:
            new_openai = st.text_input("OpenAI API Key", value=current_openai, type="password")
            new_qdrant_url = st.text_input("Qdrant URL", value=current_qdrant_url)
        
        with col2:
            new_qdrant_key = st.text_input("Qdrant API Key", value=current_qdrant_key, type="password")

        if st.button("💾 Save API Credentials", type="primary"):
            error_found = False
            if new_qdrant_url and new_qdrant_key:
                try:
                    from qdrant_client import QdrantClient
                    test_q = QdrantClient(url=new_qdrant_url, api_key=new_qdrant_key, timeout=3)
                    test_q.get_collections()
                except Exception as e:
                    st.error(f"Qdrant Connection Failed: {e}")
                    error_found = True

            if new_openai:
                try:
                    from openai import OpenAI
                    test_oa = OpenAI(api_key=new_openai)
                    test_oa.models.list()
                except Exception as e:
                    st.error(f"OpenAI Key Invalid: {e}")
                    error_found = True

            if not error_found:
                # 1. Update persistent .env file
                update_env_key("OPENAI_API_KEY", new_openai)
                update_env_key("QDRANT_URL", new_qdrant_url)
                update_env_key("QDRANT_API_KEY", new_qdrant_key)
                
                # 2. Update live Session State so other pages see it immediately
                st.session_state.openai_api_key = new_openai
                st.session_state.qdrant_url = new_qdrant_url
                st.session_state.qdrant_api_key = new_qdrant_key
                
                st.success("Credentials validated and saved!")
                sleep(1)
                st.rerun()

    st.divider()

    # --- 2. SERVICE STATUS CHECKS ---
    openai_ready = bool(os.getenv("OPENAI_API_KEY"))
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
                from Chat import speak_text 
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                audio_bytes = speak_text(preview_text, client)
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