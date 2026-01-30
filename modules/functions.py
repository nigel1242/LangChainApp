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

def get_openai_client() -> OpenAI:
    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
    api_key = api_key.strip()
    if not api_key:
        raise RuntimeError("OpenAI API key missing. Set it in settings.")
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
    """Rebuild FAISS index for a specific user's subject"""
    if target_name in _LOADED_FAISS_INDEXES:
        del _LOADED_FAISS_INDEXES[target_name]
        
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # FIXED: Now filters by user_id
    if user_id is not None:
        cursor.execute("SELECT content, file_name FROM rag_docs WHERE subject_name = ? AND user_id = ?", 
                      (target_name, user_id))
    else:
        # Fallback for backward compatibility (should not happen with user isolation)
        cursor.execute("SELECT content, file_name FROM rag_docs WHERE subject_name = ?", (target_name,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print(f"⚠️ No documents found in database for subject '{target_name}' (user_id={user_id})")
        return

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    
    for _, file_name in rows:
        # The file_name in database should be the final PDF name
        raw_path = os.path.join(subject_dir, "raw", file_name)
        
        if not os.path.exists(raw_path):
            print(f"⚠️ File not found: {raw_path}")
            continue
        
        ext = file_name.lower()

        # ---------- PDF Processing ----------
        if ext.endswith(".pdf"):
            pages = _pdf_page_texts_for_index(raw_path)
            
            if not pages:
                print(f"⚠️ No text extracted from {file_name}")
                continue
            
            for page_num, page_text in enumerate(pages, start=1):
                if not page_text.strip():
                    continue  # Skip empty pages
                    
                chunks = splitter.split_text(page_text)
                for chunk_idx, chunk in enumerate(chunks, start=1):
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
            
            print(f"✅ Processed {file_name}: {len(pages)} pages")
        else:
            print(f"⚠️ Skipping unsupported file type: {file_name}")

    if not all_docs:
        print(f"⚠️ No documents could be processed for indexing (checked {len(rows)} files)")
        return

    print(f"📊 Total chunks for indexing: {len(all_docs)}")

    # 3. Embeddings & Saving
    if not suffix:
        suffix = "openai" if "gpt" in model_name.lower() else "ollama"
    
    if suffix == "openai":
        embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.openai_api_key)
    else:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")

    faiss_index = FAISS.from_documents(all_docs, embeddings)
    
    # Ensure directory exists
    os.makedirs(subject_dir, exist_ok=True)
    faiss_index.save_local(folder_path=subject_dir, index_name=f"index_{suffix}")
    print(f"✅ Rebuilt index_{suffix} with {len(all_docs)} chunks from {len(rows)} files (user_id={user_id})")

def get_relevant_rag_faiss(db_file, subject_dir, target_name, query, model_name, ollama_models, openai_models, top_k=3, suffix=""):
    # 1. Determine index naming
    if not suffix:
        suffix = "openai" if "gpt" in model_name.lower() else "ollama"
    index_name = f"index_{suffix}"
    
    # 2. Safety Check: Verify the file exists before attempting to load
    index_path = os.path.join(subject_dir, f"{index_name}.faiss")
    if not os.path.exists(index_path):
        print(f"Index not found at: {index_path}")
        return "No RAG index found. Please process files.", [], False

    # 3. Setup Embeddings
    if suffix == "openai":
        embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.get("openai_api_key"))
    else:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")

    # 4. Load and Search with Error Handling
    try:
        # Load the index
        faiss_index = FAISS.load_local(
            folder_path=subject_dir, 
            embeddings=embeddings, 
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        
        # Perform the search with scores
        docs_with_scores = faiss_index.similarity_search_with_score(query, k=top_k)
        
        # RELEVANCE FILTERING - Lower score = more similar (L2 distance)
        # Threshold depends on your embedding model, adjust if needed
        RELEVANCE_THRESHOLD = 1.5  # Tune this value (0.5-1.5 typical range)
        
        filtered_docs = [(doc, score) for doc, score in docs_with_scores if score < RELEVANCE_THRESHOLD]
        
        if not filtered_docs:
            print(f"⚠️ No relevant docs found (all scores > {RELEVANCE_THRESHOLD})")
            print(f"Closest match had score: {docs_with_scores[0][1]:.3f}")
            return "", [], False  # No relevant context found
        
        # Extract documents and build context
        docs = [d for d, score in filtered_docs]
        context = "\n\n".join([d.page_content for d in docs])
        
        # --- DEBUG SECTION ---
        print(f"\n📚 Found {len(filtered_docs)} relevant results (threshold: {RELEVANCE_THRESHOLD})")
        for i, (doc, score) in enumerate(filtered_docs):
            print(f"--- Result {i} (Score: {score:.3f}) ---")
            print(f"File: {doc.metadata.get('file')}")
            print(f"Slide: {doc.metadata.get('slide')}")
            print(f"Page: {doc.metadata.get('page')}")
            snippet = doc.page_content[:80].replace("\n", " ")
            print(f"Text Snippet: {snippet}...")
        
        return context, docs, True
        
    except Exception as e:
        print(f"Error loading or searching FAISS: {e}")
        return "", [], False

# ----------------- RAG with Qdrant (Remote) -------------------

def rebuild_rag_index_qdrant(db_file, qdrant_url, target_name, model_name, ollama_models, openai_models, qdrant_api_key=None, user_id=None):
    """Rebuild Qdrant index for a specific user's subject"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # FIXED: Now filters by user_id
    if user_id is not None:
        cursor.execute("SELECT content, file_name FROM rag_docs WHERE subject_name = ? AND user_id = ?", 
                      (target_name, user_id))
    else:
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

def rebuild_rag_index(db_file, rag_index_dir=None, target_id=None, model_name=None, ollama_models=None, openai_models=None, vector_db="faiss", qdrant_url=None, qdrant_api_key=None, suffix="", user_id=None):
    """Unified dispatcher for rebuilding RAG index (FAISS or Qdrant)"""
    if vector_db == "faiss":
        rebuild_rag_index_faiss(db_file, rag_index_dir, target_id, model_name, ollama_models, openai_models, suffix, user_id=user_id)
    elif vector_db == "qdrant":
        rebuild_rag_index_qdrant(db_file, qdrant_url, target_id, model_name, ollama_models, openai_models, qdrant_api_key, user_id=user_id)

def get_relevant_rag(db_file, rag_index_dir=None, target_id=None, query=None, model_name=None, ollama_models=None, openai_models=None, top_k=3, vector_db="faiss", qdrant_url=None, qdrant_api_key=None, suffix=""):
    if vector_db == "faiss":
        return get_relevant_rag_faiss(db_file, rag_index_dir, target_id, query, model_name, ollama_models, openai_models, top_k, suffix)
    elif vector_db == "qdrant":
        return get_relevant_rag_qdrant(db_file, qdrant_url, target_id, query, model_name, ollama_models, openai_models, top_k, qdrant_api_key)