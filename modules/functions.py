# modules/functions.py
import os
import re
import sqlite3
import streamlit as st
from typing import List
import pandas as pd
from docx import Document as DocxDocument
from pptx import Presentation
import fitz  # PyMuPDF

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
import ollama
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Dictionary to store active FAISS index references to prevent file locking
_LOADED_FAISS_INDEXES = {}

################################################################################
# ----------------- Embeddings -------------------

class OllamaEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model_name = model_name

    def embed_documents(self, texts):
        return [ollama.embeddings(model=self.model_name, prompt=t)['embedding'] for t in texts]

    def embed_query(self, query):
        return ollama.embeddings(model=self.model_name, prompt=query)['embedding']
    
class OpenAIEmbeddings(Embeddings):
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def embed_documents(self, texts):
        res = self.client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in res.data]

    def embed_query(self, query):
        res = self.client.embeddings.create(model=self.model_name, input=query)
        return res.data[0].embedding

################################################################################
# ----------------- File Processing -------------------

def extract_text_from_file(file) -> str:
    name = file.name.lower()
    text = ""

    if name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            txt = page.get_text("text")
            if txt: text += txt + "\n"
        doc.close()
    elif name.endswith(".docx"):
        doc = DocxDocument(file)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif name.endswith(".pptx"):
        prs = Presentation(file)
        slides_text = []
        for i, slide in enumerate(prs.slides, start=1):
            slide_text = [shape.text.strip() for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
            if slide_text: slides_text.append(f"--- Slide {i} ---\n" + "\n".join(slide_text))
        text = "\n".join(slides_text)
    elif name.endswith((".txt", ".md")):
        text = file.read().decode("utf-8", errors="ignore")
    elif name.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            text = df.to_string(index=False)
        except Exception as e:
            text = f"Error reading CSV: {e}"
    return text.strip()

# ----------------- TTS & Audio -------------------

def get_b64_image(file_bytes: bytes) -> str:
    import base64
    return base64.b64encode(file_bytes).decode("utf-8")

def transcribe_audio_bytes(audio_bytes: bytes, openai_client: OpenAI) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f, language="en")
        return (resp.text or "").strip()
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def speak_text(answer: str, openai_client: OpenAI, voice: str = "alloy") -> bytes | None:
    answer = (answer or "").strip()
    if not answer: return None
    try:
        resp = openai_client.audio.speech.create(model="tts-1", voice=voice, input=answer)
        return resp.read()
    except Exception as e:
        print(f"Audio generation failed: {e}")
        return None

# ----------------- RAG with FAISS (Local) -------------------

def rebuild_rag_index_faiss(db_file, subject_dir, target_name, model_name, ollama_models, openai_models, suffix=""):
    """
    Builds a FAISS index inside the subject folder.
    suffix: 'ollama' or 'openai' to prevent overwriting.
    """
    # 1. Clear memory if this index was already loaded
    if target_name in _LOADED_FAISS_INDEXES:
        del _LOADED_FAISS_INDEXES[target_name]
        
    # 2. Fetch documents from SQLite
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT content, file_name FROM rag_docs WHERE subject_name = ?", (target_name,))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows: 
        print(f"No documents found in DB for {target_name}")
        return

    # 3. Process Text into Chunks
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for content, file_name in rows:
        if content:
            for chunk in splitter.split_text(content):
                all_docs.append(Document(page_content=chunk, metadata={"source": file_name}))

    if not all_docs: return

    # 4. Determine Embeddings and Filename
    # If no suffix is passed, we guess based on model_name
    if not suffix:
        suffix = "openai" if "gpt" in model_name.lower() else "ollama"
    
    if suffix == "openai":
        embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.openai_api_key)
    else:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")

    # 5. Build and Save Index to modules/subjects/{target_name}/
    index_name = f"index_{suffix}"
    faiss_index = FAISS.from_documents(all_docs, embeddings)
    
    # Save directly into the subject folder
    faiss_index.save_local(folder_path=subject_dir, index_name=index_name)
    print(f"✅ Saved {index_name} to {subject_dir}")

def get_relevant_rag_faiss(db_file, subject_dir, target_name, query, model_name, ollama_models, openai_models, top_k=3, suffix=""):
    """
    Loads the correct FAISS index (ollama vs openai) from the subject folder.
    """
    # 1. Determine which index file to look for
    suffix = "openai" if "gpt" in model_name.lower() else "ollama"
    index_name = f"index_{suffix}"
    
    # Check if the specific index exists
    if not os.path.exists(os.path.join(subject_dir, f"{index_name}.faiss")):
        # Fallback to the old 'index.faiss' if the new naming isn't used yet
        if os.path.exists(os.path.join(subject_dir, "index.faiss")):
            index_name = "index"
        else:
            return "No RAG index found. Please process files for this model.", False

    # 2. Setup Embeddings for the search query
    if suffix == "openai":
        embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.openai_api_key)
    else:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")

    # 3. Load and Search
    try:
        faiss_index = FAISS.load_local(
            folder_path=subject_dir, 
            embeddings=embeddings, 
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        docs_with_scores = faiss_index.similarity_search_with_score(query, k=top_k)
        
        context = "\n\n".join(
            f"Source: {d.metadata.get('source','unknown')}\nContent: {d.page_content}" 
            for d, _ in docs_with_scores
        )
        return context, True
    except Exception as e:
        st.error(f"Error accessing FAISS index '{index_name}': {e}")
        return "", False

# ----------------- RAG with Qdrant (Remote) -------------------

def rebuild_rag_index_qdrant(db_file, qdrant_url, target_name, model_name, ollama_models, openai_models, qdrant_api_key=None):
    # This remains as a sync helper if you need to bulk-re-upload
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT content, file_name FROM rag_docs WHERE subject_name = ?", (target_name,))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows: return

    if model_name in ollama_models:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
    else:
        embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.openai_api_key)

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
    
    for content, file_name in rows:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(content)
        collection_name = f"rag_docs_{len(embeddings.embed_query('test'))}"
        
        # Note: Actual ingestion logic is usually handled by db.add_rag_doc 
        # but this ensures the collection is structured for the subject
        pass

def get_relevant_rag_qdrant(db_file, qdrant_url, target_id, query, model_name, ollama_models, openai_models, top_k=3, qdrant_api_key=None):
    if model_name in ollama_models:
        embeddings = OllamaEmbeddings("nomic-embed-text:v1.5")
    else:
        embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.openai_api_key)
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
    query_vector = embeddings.embed_query(query)
    
    # Correct Naming Convention: rag_docs_<vector_size>
    collection_name = f"rag_docs_{len(query_vector)}" 

    try:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="subject_name", match=qmodels.MatchValue(value=target_id))]
            ),
            limit=top_k,
            with_payload=True
        )
        
        if not response.points: return "", False

        context = "\n\n".join(f"Source: {p.payload.get('file_name')}\n{p.payload.get('content')}" for p in response.points)
        return context, True
    except Exception as e:
        print(f"Qdrant RAG Error: {e}")
        return "", False

# ----------------- Chat Title Generation -------------------

def generate_chat_title(first_message: str, model_name: str, openai_client=None) -> str:
    """Uses the selected model to summarize the first message into a title."""
    sys_prompt = "Summarize this message into a short 3-5 word chat title. Return ONLY the title text. No quotes."
    
    try:
        if "gpt" in model_name.lower() and openai_client:
            resp = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": first_message}],
                max_tokens=15
            )
            return resp.choices[0].message.content.strip().replace('"', '')
        else:
            resp = ollama.chat(model=model_name, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": first_message}])
            return resp['message']['content'].strip().replace('"', '')
    except Exception as e:
        print(f"Title Generation Error: {e}")
        return "New Chat Session"

# ----------------- Unified Dispatchers -------------------

def rebuild_rag_index(db_file, rag_index_dir=None, target_id=None, model_name=None, ollama_models=None, openai_models=None, vector_db="faiss", qdrant_url=None, qdrant_api_key=None, suffix=""):
    if vector_db == "faiss":
        rebuild_rag_index_faiss(db_file, rag_index_dir, target_id, model_name, ollama_models, openai_models, suffix)
    elif vector_db == "qdrant":
        rebuild_rag_index_qdrant(db_file, qdrant_url, target_id, model_name, ollama_models, openai_models, qdrant_api_key)

def get_relevant_rag(db_file, rag_index_dir=None, target_id=None, query=None, model_name=None, ollama_models=None, openai_models=None, top_k=3, vector_db="faiss", qdrant_url=None, qdrant_api_key=None, suffix=""):     
    if vector_db == "faiss":
        return get_relevant_rag_faiss(db_file, rag_index_dir, target_id, query, model_name, ollama_models, openai_models, top_k)
    elif vector_db == "qdrant":
        return get_relevant_rag_qdrant(db_file, qdrant_url, target_id, query, model_name, ollama_models, openai_models, top_k, qdrant_api_key)