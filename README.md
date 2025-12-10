## Python Version:
!If your python is older or newer, please install python 3.11 from https://www.python.org/downloads/release/python-3110/
1. Open a new terminal
2. Run "python --version"
3. If you have Python 3.11, proceed to the next section.


## Setup:
If you have a GPU to run ollama models locally, follow the whole flow of this section.
Else, skip to "If you don't have a GPU"

1. Install ollama >> https://ollama.com/download/windows
2. Open terminal in VSCode
3. Pull Ollama embedder using "ollama pull nomic-embed-text". This is for the ollama embedder.
4. Pull Ollama model Qwen2.5VL using "ollama pull qwen-2.5vl". This is for multimodal mode, where you can drop images in.
5. Pull Ollama model Llama3 using "ollama pull llama3" This will be the main chat model.

If you don't have a GPU, you can only use openAI models.
Just enter your API key in the sidebar and select one of the GPT models.

1. Create virtual environment using "py -3.11 -m venv venv".
2. Run venv using "venv\scripts\activate"
3. Install requirements using "pip install -r requirements.txt"
4. Create .env file in project folder and copy paste the keys
5. Run app using "streamlit run app.py"