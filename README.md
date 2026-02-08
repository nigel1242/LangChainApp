# My Learning AI (LangChainApp)

An interactive Streamlit app for learning with chat, document uploads, and model management. The app supports local Ollama models (GPU recommended) and OpenAI models when you prefer hosted inference.

## 🎬 Demo video

Watch the walkthrough here (embedded on the Settings page too):

[![My Learning AI demo](https://img.youtube.com/vi/ssU0zvjxg_4/0.jpg)](https://www.youtube.com/watch?v=ssU0zvjxg_4)

## ✨ Features

- **Chat-first learning experience** with conversation history.
- **Document workflows** for uploading and using content in your sessions.
- **Model management** for downloading/removing Ollama models from the UI.
- **Voice preview** for text-to-speech selections.
- **Quiz tools** to reinforce learning and practice.
- **Statistics dashboard** to track progress and usage insights.
- **Settings page** for API keys, model downloads, and app configuration.

## ✅ Python Version

!If your python is older or newer, please install python 3.11 from https://www.python.org/downloads/release/python-3110/
1. Open a new terminal
2. Run "python --version"
3. If you have Python 3.11, proceed to the next section.

## ✅ Requirements

- **Python 3.11** (recommended)
  - Download: https://www.python.org/downloads/release/python-3110/
- **Ollama** (optional, for local models)
  - Download: https://ollama.com/download

## 🛠️ Setup (Ollama)
If you have a GPU to run ollama models locally, follow the whole flow of this section.  

1. Install ollama >> https://ollama.com/download/windows
2. Ensure ollama is running
3. Open a new terminal in VSCode and create virtual environment using "py -3.11 -m venv venv".
4. Run venv using "venv\scripts\activate"
4. Install requirements using "pip install -r requirements.txt"
5. Run app using "streamlit run Chat.py"
6. Open settings, download the embedding model, visual model, and base model.
7. If you have OpenAI or Qdrant API keys, upload and save them in settings page.

## 🛠️ Setup (OpenAI ONLY)

If you don't have a GPU that can run ollama 7b LLMs, you can only use openAI models.  

1. Open a new terminal in VSCode and create virtual environment using "py -3.11 -m venv venv".
2. Run venv using "venv\scripts\activate"
3. Install requirements using "pip install -r requirements.txt"
4. Run app using "streamlit run Chat.py"
5. Open settings page, upload and save your OpenAI or Qdrant API keys,

## ❓Troubleshooting

- If you don’t see models in Settings, ensure Ollama is running or that you have an OpenAI API key stored.
- If OpenAI calls fail, verify your API key in the Settings page.
- If you don’t have API keys, the app can’t access hosted services. Open **Settings** → **API Configuration**, paste your OpenAI key (and Qdrant URL/key if you use it), then click **Save API Credentials**.
- For GPU-less setups, skip Ollama and use OpenAI models only.

## ⚙️ Settings

The **Settings** page helps you configure everything without editing files manually:

- **API Configuration**: add OpenAI and Qdrant credentials.
- **Voice Selection**: pick and preview the TTS voice.
- **Download Models**: pull Ollama models directly from the UI.
- **Delete Models**: remove unused models to save disk space.
- **Settings Walkthrough**: watch the embedded demo video for setup guidance.

### Getting API keys (if you don’t have them yet)

- **OpenAI API key**: create or view keys at https://platform.openai.com/account/api-keys  
- **Qdrant**:
  - **Cloud**: create a cluster and key at https://cloud.qdrant.io  
  - **Self-hosted**: follow setup docs at https://qdrant.tech/documentation/

After you have the keys, open **Settings → API Configuration**, paste the values, then click **Save API Credentials**.

## 🧯 Troubleshooting (full list)

- **App won’t start**: ensure Python 3.11 is installed and your venv is activated.
- **Missing packages**: run `pip install -r requirements.txt`.
- **OpenAI errors**: confirm your key in **Settings → API Configuration**.
- **Qdrant connection failed**: verify `QDRANT_URL` and `QDRANT_API_KEY`.
- **Ollama not detected**: start Ollama, then refresh the page.
- **Models not listed**: check Ollama is running and retry downloads in Settings.
- **No audio**: ensure your OpenAI key is valid and select a voice in Settings.
- **Uploads not working**: confirm files are supported and under size limits.