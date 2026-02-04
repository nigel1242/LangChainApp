# modules/functions.py
import os
import sqlite3
import fitz
import streamlit as st
from pptx import Presentation
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import ollama
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Dictionary to store active FAISS index references to prevent file locking
_LOADED_FAISS_INDEXES = {}

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

# ----------------- openai -------------------

def get_openai_client():
    api_key = st.session_state.get("openai_api_key")
    if not api_key:
        raise ValueError("OpenAI API Key not found in your user profile. Please update Settings.")
    return OpenAI(api_key=api_key)

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

# ----------------- extractions -------------------

def _pdf_page_texts_for_index(path: str) -> List[str]:
    """Return per-page text for a PDF."""
    try:
        doc = fitz.open(path)
    except Exception:
        return []
    pages: List[str] = []
    for page in doc:
        txt = page.get_text("text") or ""
        pages.append(txt)
    doc.close()
    return pages



# ----------------- RAG with FAISS (Local) -------------------

def rebuild_rag_index_faiss(db_file, subject_dir, target_name, model_name, ollama_models, openai_models, suffix="", user_id=None):
    if target_name in _LOADED_FAISS_INDEXES:
        del _LOADED_FAISS_INDEXES[target_name]
        
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Pull text and filename from DB
    cursor.execute("SELECT content, file_name FROM rag_docs WHERE subject_name = ? AND user_id = ?", (target_name, user_id))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows: return

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    
    for _, file_name in rows:
        raw_path = os.path.join(subject_dir, "raw", file_name)
        ext = file_name.lower()

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

        # ---------- PDF ----------
        if ext.endswith(".pdf"):
            pages = _pdf_page_texts_for_index(raw_path)
            for page_num, page_text in enumerate(pages, start=1):
                for chunk_idx, chunk in enumerate(splitter.split_text(page_text), start=1):
                    all_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "file": file_name,
                            "file_path": raw_path,
                            "page": page_num,
                            "chunk": chunk_idx,
                            "type": "pdf_page"
                        }
                    ))

    # 3. Embeddings & Saving Logic
    oa_key = st.session_state.get("openai_api_key")
    embeddings = None

    # Priority 1: Use OpenAI if key is present
    if oa_key:
        try:
            embeddings = OpenAIEmbeddings(
                model_name="text-embedding-3-small", 
                api_key=oa_key
            )
            suffix = "openai"
            print("Using OpenAI Embeddings for FAISS...")
        except Exception as e:
            print(f"OpenAI Embedding Init Failed: {e}")

    # Priority 2: Use Ollama if OpenAI failed or key is missing
    if not embeddings:
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
            suffix = "ollama"
        except Exception:
            embeddings = None

    if not embeddings:
        st.error("❌ Indexing Failed: OpenAI key not found and Ollama is not running.")
        return

    # 4. Perform Indexing and Save
    try:
        faiss_index = FAISS.from_documents(all_docs, embeddings)
        faiss_index.save_local(folder_path=subject_dir, index_name=f"index_{suffix}")
    except Exception as e:
        st.error(f"Error during FAISS save: {e}")

    faiss_index = FAISS.from_documents(all_docs, embeddings)
    faiss_index.save_local(folder_path=subject_dir, index_name=f"index_{suffix}")
    print(f"✅ Rebuilt index_{suffix} with page/slide metadata.")

def get_relevant_rag_faiss(db_file, subject_dir, target_name, query, model_name, ollama_models, openai_models, top_k=3, suffix="", user_id=None):
    # 1. Determine index naming with Priority Logic
    oa_key = st.session_state.get("openai_api_key")
    
    if not suffix:
        # If we have an OpenAI key, check if that index exists first
        openai_path = os.path.join(subject_dir, "index_openai.faiss")
        if oa_key and os.path.exists(openai_path):
            suffix = "openai"
        else:
            # Fallback to model-based naming or Ollama
            suffix = "openai" if "gpt" in model_name.lower() else "ollama"
    
    index_name = f"index_{suffix}"
    index_path = os.path.join(subject_dir, f"{index_name}.faiss")
    
    # 2. Safety Check: Verify the file exists
    if not os.path.exists(index_path):
        # Final attempt: If we expected 'ollama' but only 'openai' exists, swap it
        alt_path = os.path.join(subject_dir, "index_openai.faiss")
        if suffix == "ollama" and os.path.exists(alt_path):
            suffix = "openai"
            index_name = "index_openai"
            index_path = alt_path
        else:
            print(f"Index not found at: {index_path}")
            return "No RAG index found. Please process files.", [], False

    # 3. Setup Embeddings
    try:
        if suffix == "openai":
            if not oa_key:
                return "OpenAI key missing for retrieval.", [], False
            embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=oa_key)
        else:
            # This will fail if Ollama isn't running
            embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
    except Exception as e:
        print(f"Embedding Setup Error: {e}")
        return "Failed to initialize embedding service.", [], False

    # 4. Load and Search
    try:
        faiss_index = FAISS.load_local(
            folder_path=subject_dir, 
            embeddings=embeddings, 
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        
        docs_with_scores = faiss_index.similarity_search_with_score(query, k=top_k)
        docs = [d for d, score in docs_with_scores]
        context = "\n\n".join([d.page_content for d in docs])
        
        # Debug printing
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"--- Result {i} (Score: {score:.4f}) ---")
            print(f"File: {doc.metadata.get('file')} | Page: {doc.metadata.get('page')}")
        
        return context, docs, True
        
    except Exception as e:
        print(f"Error loading or searching FAISS: {e}")
        return "Search failed. The index might be corrupted or incompatible.", [], False
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
    print(f"\n>>> [TERMINAL DEBUG] Starting Qdrant Search")
    print(f">>> Subject: '{target_id}' | Model: {model_name}")

    if model_name in openai_models or "gpt" in model_name.lower():
        embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.get("openai_api_key"))
    else:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
    
    try:
        query_vector = embeddings.embed_query(query)
        collection_name = f"rag_docs_{len(query_vector)}" 
        print(f">>> Generated Vector Dim: {len(query_vector)} | Targeting: {collection_name}")

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
        
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="subject_name", match=qmodels.MatchValue(value=str(target_id).strip()))]
            ),
            limit=top_k,
            with_payload=True
        )
        
        print(f">>> Qdrant returned {len(response.points)} points")
        
        if not response.points:
            return "", [], False

        context = "\n\n".join([p.payload.get('content', '') for p in response.points])
        return context, response.points, True

    except Exception as e:
        print(f">>> [TERMINAL ERROR] {e}")
        return "", [], False
    
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

def rebuild_rag_index(db_file, subject_dir=None, target_id=None, model_name=None, ollama_models=None, openai_models=None, vector_db="faiss", qdrant_url=None, qdrant_api_key=None, suffix="", user_id=None):
    if vector_db == "faiss":
        rebuild_rag_index_faiss(db_file, subject_dir, target_id, model_name, ollama_models, openai_models, suffix, user_id)
    elif vector_db == "qdrant":
        rebuild_rag_index_qdrant(db_file, qdrant_url, target_id, model_name, ollama_models, openai_models, qdrant_api_key)

def get_relevant_rag(db_file, subject_dir=None, target_id=None, query=None, model_name=None, ollama_models=None, openai_models=None, top_k=3, vector_db="faiss", qdrant_url=None, qdrant_api_key=None, suffix="", user_id=None):
    if vector_db == "faiss":
        return get_relevant_rag_faiss(db_file, subject_dir, target_id, query, model_name, ollama_models, openai_models, top_k, suffix, user_id)
    elif vector_db == "qdrant":
        return get_relevant_rag_qdrant(db_file, qdrant_url, target_id, query, model_name, ollama_models, openai_models, top_k, qdrant_api_key)