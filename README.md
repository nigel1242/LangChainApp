current issues:
model selection isn't locked until page is refreshed or streamlit is rerun using st.rerun(), e.g. clicking to another chat then clicking back to the chat
chat names are broken, its named as the model used
documents processed won't disappear from the uploaded files section until yoou click x to manually remove it
qwen2.5vl multimodal model needs to have image uploader for uploading images

missing features:
remove single chat button
remove all chats button
possible for multiple chat removal option?
RAG only has .txt and .pdf retrieval. need to add .docx, .xlxs, pptx, etc (need ocr to text)
quiz + statistics function
openai keys are not stored in database

current features:
ollama chatbot
ollama chatbot + RAG
openai chatbot
openai chatbot + RAG
chat persistence (history)
model streaming (types as chatbot thinks)

## Setup:

If you have a GPU to run ollama models locally, follow the whole flow of this section.
Else, skip to "If you don't have a GPU"

1. Install ollama >> https://ollama.com/download/windows
2. Open terminal in VSCode
3. cd to the file directory e.g. C:\Users\(yourname)\Downloads\My Learning AI>
4. Pull Ollama embedder using "ollama pull nomic-embed-text". This is for the ollama embedder.
5. Pull Ollama model Qwen2.5VL using "ollama pull qwen-2.5vl". This is for multimodal mode, where you can drop images in.
6. Pull Ollama model Llama3 using "ollama pull llama3" This will be the main chat model.


If you don't ave a GPU, you can only use openAI models.
Just enter your API key in the sidebar and select one of the GPT models.

1. Create virtual environment using "python -m venv venv"
2. Run venv using "venv\scripts\activate"
3. Install requirements using "pip install -r requirements.txt"
4. Run app using "streamlit run app.py"