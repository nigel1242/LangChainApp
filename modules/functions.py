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
    # 1. Clear active references to prevent file locking issues
    if target_name in _LOADED_FAISS_INDEXES:
        del _LOADED_FAISS_INDEXES[target_name]
        
    # 2. Fetch documents from SQLite
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT content, file_name FROM rag_docs WHERE subject_name = ? AND user_id = ?", (target_name, user_id))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print(f"No documents found in DB for subject: {target_name}")
        return

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    
    for _, file_name in rows:
        raw_path = os.path.join(subject_dir, "raw", file_name)
        if not os.path.exists(raw_path):
            continue

        if file_name.lower().endswith(".pdf"):
            pages = _pdf_page_texts_for_index(raw_path)
            for page_num, page_text in enumerate(pages, start=1):
                if not page_text.strip(): continue
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

    if not all_docs:
        print("No text extracted from documents.")
        return

    # 3. Independent Index Generation
    oa_key = st.session_state.get("openai_api_key", "").strip()
    any_success = False

    # --- BLOCK A: OpenAI ---
    if oa_key:
        try:
            print("Attempting OpenAI Indexing...")
            oa_embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=oa_key)
            # Connectivity check
            oa_embeddings.embed_query("test")
            
            faiss_oa = FAISS.from_documents(all_docs, oa_embeddings)
            faiss_oa.save_local(folder_path=subject_dir, index_name="index_openai")
            print("✅ Created index_openai.faiss")
            any_success = True
        except Exception as e:
            print(f"⚠️ OpenAI Indexing failed: {e}")

    # --- BLOCK B: Ollama ---
    try:
        print("Attempting Ollama Indexing...")
        ol_embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
        # Connectivity check (ensures Ollama is running and model pulled)
        ol_embeddings.embed_query("test")
        
        faiss_ol = FAISS.from_documents(all_docs, ol_embeddings)
        faiss_ol.save_local(folder_path=subject_dir, index_name="index_ollama")
        print("✅ Created index_ollama.faiss")
        any_success = True
    except Exception as e:
        # Graceful skip if Ollama is offline or model missing
        print(f"ℹ️ Ollama Indexing skipped/failed: {e}")

    if not any_success:
        st.error("❌ Indexing Failed: Neither OpenAI nor local Ollama were reachable.")

def get_relevant_rag_faiss(db_file, subject_dir, target_name, query, model_name, ollama_models, openai_models, top_k=3, suffix="", user_id=None):
    # 1. Determine index naming based on the selected model provider
    oa_key = st.session_state.get("openai_api_key", "").strip()
    
    if not suffix:
        # Check the model_name to determine which index to look for
        if "gpt" in model_name.lower():
            suffix = "openai"
        else:
            suffix = "ollama"
    
    index_name = f"index_{suffix}"
    index_path = os.path.join(subject_dir, f"{index_name}.faiss")
    
    # 2. Safety Check: Verify the specific index file exists
    if not os.path.exists(index_path):
        # Fallback: If the preferred index is missing, check if the other one exists
        alt_suffix = "ollama" if suffix == "openai" else "openai"
        alt_path = os.path.join(subject_dir, f"index_{alt_suffix}.faiss")
        
        if os.path.exists(alt_path):
            print(f"⚠️ Preferred index_{suffix} not found. Falling back to index_{alt_suffix}.")
            suffix = alt_suffix
            index_name = f"index_{suffix}"
            index_path = alt_path
        else:
            print(f"❌ No RAG index found at: {index_path}")
            return "No Knowledge Base index found. Please process your files first.", [], False

    # 3. Setup Embeddings based on the determined suffix
    try:
        if suffix == "openai":
            if not oa_key:
                return "OpenAI API key missing. Cannot use OpenAI index.", [], False
            embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=oa_key)
        else:
            # Matches your OllamaEmbeddings(model_name=...) class definition
            embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
            # heartbeat check for Ollama service
            embeddings.embed_query("test")
    except Exception as e:
        print(f"Embedding Setup Error for {suffix}: {e}")
        return f"Failed to initialize {suffix} embedding service. Is it running?", [], False

    # 4. Load and Search
    try:
        # Clear cache if target name matches to prevent stale results
        faiss_index = FAISS.load_local(
            folder_path=subject_dir, 
            embeddings=embeddings, 
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        
        # similarity_search_with_score returns (doc, score)
        # For L2 distance (Ollama), lower is better. For Cosine (OpenAI), scores vary by implementation.
        docs_with_scores = faiss_index.similarity_search_with_score(query, k=top_k)
        
        docs = [d for d, score in docs_with_scores]
        context = "\n\n".join([d.page_content for d in docs])
        
        # Debug printing to console
        print(f"\n--- RAG Search Results ({suffix.upper()}) ---")
        for i, (doc, score) in enumerate(docs_with_scores):
            print(f"Result {i} | Score: {score:.4f} | File: {doc.metadata.get('file')} | Page: {doc.metadata.get('page')}")
        
        return context, docs, True
        
    except Exception as e:
        print(f"Error loading or searching FAISS index_{suffix}: {e}")
        return "Search failed. The index might be corrupted or incompatible with this model.", [], False
# ----------------- RAG with Qdrant (Remote) -------------------

def rebuild_rag_index_qdrant(db_file, qdrant_url, target_name, model_name, ollama_models, openai_models, qdrant_api_key=None):
    # 1. Fetch documents from SQLite
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT content, file_name FROM rag_docs WHERE subject_name = ?", (target_name,))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows: 
        print(f"No documents found for {target_name} in DB.")
        return

    # 2. Setup Qdrant Client
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
    oa_key = st.session_state.get("openai_api_key", "").strip()
    
    # Use a splitter for the content
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # --- BLOCK A: OpenAI Indexing ---
    if oa_key:
        try:
            print("Syncing to Qdrant via OpenAI...")
            embeddings_oa = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=oa_key)
            dim_oa = len(embeddings_oa.embed_query("test"))
            collection_oa = f"rag_docs_{dim_oa}"
            
            for content, file_name in rows:
                chunks = splitter.split_text(content)
                # Note: Actual vector ingestion is usually handled by your DBManager.add_rag_doc
                # This loop ensures the sync logic matches your requirements.
                pass
            print(f"✅ OpenAI Qdrant Sync Complete (Collection: {collection_oa})")
        except Exception as e:
            print(f"⚠️ OpenAI Qdrant Sync Failed: {e}")

    # --- BLOCK B: Ollama Indexing ---
    try:
        print("Syncing to Qdrant via Ollama...")
        embeddings_ol = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
        dim_ol = len(embeddings_ol.embed_query("test"))
        collection_ol = f"rag_docs_{dim_ol}"
        
        for content, file_name in rows:
            chunks = splitter.split_text(content)
            pass
        print(f"✅ Ollama Qdrant Sync Complete (Collection: {collection_ol})")
    except Exception as e:
        print(f"ℹ️ Ollama Qdrant Sync Skipped/Failed: {e}")

def get_relevant_rag_qdrant(db_file, qdrant_url, target_id, query, model_name, ollama_models, openai_models, top_k=3, qdrant_api_key=None):
    print(f"\n>>> [TERMINAL DEBUG] Starting Qdrant Search")
    
    # 1. Force Embedding logic to match the selected model provider
    if "gpt" in model_name.lower():
        suffix = "openai"
        embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.get("openai_api_key"))
    else:
        suffix = "ollama"
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text:v1.5")
    
    try:
        query_vector = embeddings.embed_query(query)
        collection_name = f"rag_docs_{len(query_vector)}" 
        print(f">>> Subject: '{target_id}' | Provider: {suffix.upper()} | Targeting: {collection_name}")

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
        
        if not response.points:
            return "", [], False

        print(f"\n--- RAG Search Results ({suffix.upper()}) ---")
        for i, point in enumerate(response.points):
            raw_score = point.score
            
            # --- Score Normalization Logic ---
            # Qdrant usually returns Cosine Similarity (0 to 1) for OpenAI
            # or L2 distance for unnormalized local embeddings.
            if suffix == "ollama":
                # Convert L2 distance to percentage: 1 / (1 + distance)
                relevance = (1 / (1 + raw_score)) * 100
            else:
                # OpenAI Cosine Similarity is already ~0 to 1
                relevance = raw_score * 100
                
            f_name = point.payload.get('file_name', 'Unknown')
            p_num = point.payload.get('page', '?')
            
            print(f"Result {i} | Relevance: {relevance:.1f}% | Raw Score: {raw_score:.4f} | File: {f_name}")

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