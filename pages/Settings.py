# settings.py (Updated with Voice Selection)
import streamlit as st
import ollama
from time import sleep

st.set_page_config(
    page_title="Model management",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# New dictionary to map technical name (key) to friendly display name (value)
MODEL_DISPLAY_NAMES = {
    "nomic-embed-text:v1.5": "Nomic Embed (Embedding Model)",
    "qwen2.5vl:7b": "Qwen 2.5 VL (Vision Model)",
    "llama3:8b": "Llama3 8B (Base Model)",
}
MODELS_TO_OFFER = list(MODEL_DISPLAY_NAMES.keys())

def download_model(model_name):
    """Handles the model download process and Streamlit feedback."""
    if not model_name:
        st.warning("Model name is missing.", icon="⚠️")
        return

    try:
        # Use a placeholder/spinner while pulling the model
        with st.spinner(f"Downloading model: **{model_name}**... This may take a moment."):
            ollama.pull(model_name)
        
        st.success(f"Downloaded model: {model_name}", icon="🎉")
        st.balloons()
        sleep(1)
        st.rerun()
    except Exception as e:
        st.error(
            f"""Failed to download model: {model_name}. Error: {str(e)}""",
            icon="😳",
        )

def main():
    st.subheader("Model Management", divider="red", anchor=False)

    # --- VOICE SELECTION SETTINGS ---
    st.subheader("Text-to-Speech (TTS) Voice Selection", anchor=False)
    voice_options = [ 'alloy', 'nova', 'shimmer', 'echo', 'onyx', 'fable', 'ash', 'sage', 'coral']
    current_voice = st.session_state.get("tts_voice", "alloy")
    
    if st.button("▶️ Preview Selected Voice"):
        preview_text = f"Hello, I am the {st.session_state.tts_voice} voice. How do I sound?"
        from app import speak_text 
        audio_bytes = speak_text(preview_text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")

    st.session_state.tts_voice = st.selectbox(
        "🎧 Select Voice for Answer Audio (OpenAI TTS)",
        voice_options,
        index=voice_options.index(current_voice)
        if current_voice in voice_options
        else 0,
        help="This voice will be used for generated audio responses."
    )
    st.divider()

    # --- 1. Get List of Installed Models ---
    installed_models = []
    try:
        models_info = ollama.list()
        # installed_models contains full names like 'llama3:8b@sha256:...'
        installed_models = [m.model for m in models_info["models"]]
    except Exception as e:
        st.error(f"Could not list available models. Ensure Ollama is running. Error: {str(e)}", icon="🛑")
    
    # --- 2. Download Models Section (Buttons with Disable Logic) ---
    st.subheader("Download Models", anchor=False)
    st.info("Click to download a model. Buttons are disabled if the model is already installed.", icon="👇")
    
    # Create a set of base names (without digest) for fast lookup in the download section
    installed_model_bases = {m.split('@')[0] for m in installed_models}
    
    # Use columns to neatly arrange the download buttons
    cols = st.columns(len(MODELS_TO_OFFER))
    
    for i, model_name in enumerate(MODELS_TO_OFFER):
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        # Check if the model is already installed using the base name set
        is_installed = model_name in installed_model_bases
        
        button_key = f"download_{model_name}"
        
        if is_installed:
            button_label = f"✅ :gray[**Installed**] :gray[{display_name}]"
        else:
            button_label = f"📥 :green[**Download**] :red[{display_name}]"
            
        with cols[i]:
            # Create a button for each specific model
            if st.button(button_label, key=button_key, disabled=is_installed):
                download_model(model_name)


    st.divider()

    # --- 3. Delete Models Section (Single Select Dropdown with Friendly Names) ---
    st.subheader("Delete Models", anchor=False)
    
    if installed_models:
        # Create a dictionary: {Friendly Name (or raw name): Raw Model Name for deletion}
        delete_options_map = {}
        for m_raw in installed_models:
            
            # --- Get the base name without SHA digest ---
            base_name_with_tag = m_raw.split('@')[0]
            
            # Check if we have a friendly name for this model/tag
            if base_name_with_tag in MODEL_DISPLAY_NAMES:
                # Use the friendly name directly (FIX APPLIED HERE)
                display_key = MODEL_DISPLAY_NAMES[base_name_with_tag]
            else:
                # If it's a model not on our offered list, use the raw name
                display_key = m_raw
            
            # Handle potential collision of display keys
            if display_key in delete_options_map:
                unique_suffix = base_name_with_tag.split(':')[-1]
                if unique_suffix == base_name_with_tag: # Handles untagged models
                    unique_suffix = m_raw
                display_key = f"{display_key} (ID: {unique_suffix})"

            # Map the unique display key back to the raw name needed for ollama.delete()
            delete_options_map[display_key] = m_raw
            
        delete_display_names = list(delete_options_map.keys())

        selected_delete_display_name = st.selectbox(
            "Select a model to delete", 
            options=["-- Select a Model --"] + delete_display_names,
            key="delete_selectbox"
        )

        model_to_delete = None
        if selected_delete_display_name != "-- Select a Model --":
            model_to_delete = delete_options_map[selected_delete_display_name]

        if model_to_delete:
            if st.button(f"🗑️ :red[**Delete**] :red[{selected_delete_display_name}]", key="delete_button"):
                # Call delete function on the technical model name
                try:
                    ollama.delete(model_to_delete)
                    st.success(f"Deleted model: {model_to_delete}", icon="🎉")
                    st.balloons()
                    sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(
                        f"""Failed to delete model: {model_to_delete}. Error: {str(e)}""",
                        icon="😳",
                    )
    else:
        st.info("No models available for deletion.", icon="🦗")


if __name__ == "__main__":
    main()