import ollama
from openai import OpenAI
from langchain_core.embeddings import Embeddings

import os
import re
import sqlite3
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Try to import Qdrant from the correct package
try:
    from langchain_qdrant import QdrantVectorStore
    QDRANT_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.vectorstores import Qdrant as QdrantVectorStore
        QDRANT_AVAILABLE = True
    except ImportError:
        QDRANT_AVAILABLE = False

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from typing import List
import pandas as pd
from docx import Document as DocxDocument
from pptx import Presentation
import fitz  # PyMuPDF


####################################################################################
# ----------------- Ollama Embeddings -------------------
class OllamaEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model_name = model_name

    def embed_documents(self, texts):
        return [ollama.embeddings(model=self.model_name, prompt=t)['embedding'] for t in texts]

    def embed_query(self, query):
        return ollama.embeddings(model=self.model_name, prompt=query)['embedding']
    
# ----------------- OpenAI Embeddings -------------------
class OpenAIEmbeddings(Embeddings):
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def embed_documents(self, texts):
        res = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [d.embedding for d in res.data]

    def embed_query(self, query):
        res = self.client.embeddings.create(
            model=self.model_name,
            input=query
        )
        return res.data[0].embedding


#############################################################################
# ----------------- File Processing -------------------

TEXT_FILE_EXTS = {".txt", ".md"}
DOCX_EXTS = {".docx"}
PPTX_EXTS = {".pptx"}
CSV_EXTS = {".csv"}
PDF_EXTS = {".pdf"}


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def extract_text_from_file(file) -> str:
    """Extract text from uploaded file-like object."""
    name = file.name.lower()
    text = ""

    if name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            txt = page.get_text("text")
            if txt:
                text += txt + "\n"
        doc.close()

    elif name.endswith(".docx"):
        doc = DocxDocument(file)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif name.endswith(".pptx"):
        prs = Presentation(file)
        slides_text = []
        for i, slide in enumerate(prs.slides, start=1):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())
            if slide_text:
                slides_text.append(f"--- Slide {i} ---\n" + "\n".join(slide_text))
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


##################################################################################################
# ----------------- RAG Core Functions with FAISS -------------------

def rebuild_rag_index_faiss(db_file: str, rag_index_dir: str, chat_id, model_name, ollama_models, openai_models):
    """Rebuilds the FAISS vector index for a given chat."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT content, file_name FROM rag_docs WHERE chat_id = ?", (chat_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for content, file_name in rows:
        chunks = splitter.split_text(content)
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata={"source": file_name}))

    if not all_docs:
        return

    # Choose embedding model
    if model_name in ollama_models:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text")
    elif model_name in openai_models:
        if not st.session_state.openai_api_key:
            st.error("OpenAI API key is missing. Cannot build index.")
            return
        embeddings = OpenAIEmbeddings(
            model_name="text-embedding-3-small",
            api_key=st.session_state.openai_api_key
        )
    else:
        st.error(f"Unknown model type {model_name} for embeddings.")
        return

    faiss_index = FAISS.from_documents(all_docs, embeddings)
    faiss_index.save_local(os.path.join(rag_index_dir, f"chat_{chat_id}.faiss"))


def get_relevant_rag_faiss(db_file: str, rag_index_dir: str, chat_id, query, model_name, ollama_models, openai_models, top_k=3):
    """Gets relevant RAG context from the FAISS index."""
    faiss_index_path = os.path.join(rag_index_dir, f"chat_{chat_id}.faiss")
    if not os.path.exists(faiss_index_path):
        return "", False

    # Choose embedding
    if model_name in ollama_models:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text")
    elif model_name in openai_models:
        if not st.session_state.openai_api_key:
            st.error("OpenAI API key is missing. Cannot retrieve RAG context.")
            return "", False
        embeddings = OpenAIEmbeddings(
            model_name="text-embedding-3-small",
            api_key=st.session_state.openai_api_key
        )
    else:
        st.error(f"Unknown model type {model_name} for embeddings.")
        return "", False

    # Load FAISS
    faiss_index = FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Semantic similarity search
    docs_with_scores = faiss_index.similarity_search_with_score(query, k=top_k)
    relevant_docs = [doc for doc, score in docs_with_scores]

    # Filename matching (case-insensitive)
    query_lower = query.lower()
    for doc in faiss_index.docstore._dict.values():
        source = doc.metadata.get("source", "").lower()
        if source and re.search(r"\b" + re.escape(source) + r"\b", query_lower):
            if doc not in relevant_docs:
                relevant_docs.append(doc)

    if not relevant_docs:
        return "", False

    # Combine into context
    context = "\n\n".join(
        f"Source: {d.metadata.get('source','unknown')}\nContent: {d.page_content}"
        for d in relevant_docs
    )
    return context, True


##################################################################################################
# ----------------- RAG Core Functions with Qdrant -------------------

def rebuild_rag_index_qdrant(db_file: str, qdrant_url: str, chat_id, model_name, ollama_models, openai_models, qdrant_api_key=None):
    """Rebuilds the Qdrant vector index for a given chat."""
    if not QDRANT_AVAILABLE:
        st.error("Qdrant integration not available. Install: pip install langchain-qdrant")
        return
        
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT content, file_name FROM rag_docs WHERE chat_id = ?", (chat_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return

    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for content, file_name in rows:
        chunks = splitter.split_text(content)
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata={"source": file_name}))

    if not all_docs:
        return

    # Choose embedding model
    if model_name in ollama_models:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text")
    elif model_name in openai_models:
        if not st.session_state.openai_api_key:
            st.error("OpenAI API key is missing. Cannot build index.")
            return
        embeddings = OpenAIEmbeddings(
            model_name="text-embedding-3-small",
            api_key=st.session_state.openai_api_key
        )
    else:
        st.error(f"Unknown model type {model_name} for embeddings.")
        return

    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
        collection_name = f"chat_{chat_id}"

        # Delete existing collection if it exists
        try:
            client.delete_collection(collection_name)
        except:
            pass

        # Use simpler from_documents with positional args only
        vector_store = QdrantVectorStore.from_documents(
            all_docs,
            embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            prefer_grpc=False
        )
        
    except Exception as e:
        # If from_documents fails, try manual approach
        try:
            # Get embedding dimension
            sample_embedding = embeddings.embed_query("test")
            embedding_dim = len(sample_embedding)

            # Create collection manually
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )

            # Create vector store and add documents
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings
            )
            vector_store.add_documents(all_docs)
            
        except Exception as e2:
            st.error(f"Failed to create Qdrant index: {str(e2)}")
            return


def get_relevant_rag_qdrant(db_file: str, qdrant_url: str, chat_id, query, model_name, ollama_models, openai_models, top_k=3, qdrant_api_key=None):
    """Gets relevant RAG context from the Qdrant index."""
    if not QDRANT_AVAILABLE:
        return "", False
        
    collection_name = f"chat_{chat_id}"
    
    # Check if collection exists
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
        collection_info = client.get_collection(collection_name)
        
        # Check if collection has vectors
        if collection_info.points_count == 0:
            return "", False
            
    except Exception as e:
        # Collection doesn't exist or other error
        return "", False

    # Choose embedding
    if model_name in ollama_models:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text")
    elif model_name in openai_models:
        if not st.session_state.openai_api_key:
            return "", False
        embeddings = OpenAIEmbeddings(
            model_name="text-embedding-3-small",
            api_key=st.session_state.openai_api_key
        )
    else:
        return "", False

    try:
        # Load Qdrant
        qdrant_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )

        # Semantic similarity search
        relevant_docs = qdrant_store.similarity_search(query, k=top_k)

        if not relevant_docs:
            return "", False

        # Combine into context
        context = "\n\n".join(
            f"Source: {d.metadata.get('source','unknown')}\nContent: {d.page_content}"
            for d in relevant_docs
        )
        return context, True
    
    except Exception as e:
        st.error(f"Error retrieving from Qdrant: {e}")
        return "", False


##################################################################################################
# ----------------- Unified RAG Functions -------------------

def rebuild_rag_index(db_file: str, rag_index_dir: str, chat_id, model_name, ollama_models, openai_models, vector_db="faiss", qdrant_url=None, qdrant_api_key=None):
    """Unified function to rebuild RAG index with selected vector database."""
    if vector_db == "faiss":
        rebuild_rag_index_faiss(db_file, rag_index_dir, chat_id, model_name, ollama_models, openai_models)
    elif vector_db == "qdrant":
        if not qdrant_url:
            st.error("Qdrant URL is required when using Qdrant vector database.")
            return
        rebuild_rag_index_qdrant(db_file, qdrant_url, chat_id, model_name, ollama_models, openai_models, qdrant_api_key)
    else:
        st.error(f"Unknown vector database: {vector_db}")


def get_relevant_rag(db_file: str, rag_index_dir: str, chat_id, query, model_name, ollama_models, openai_models, top_k=3, vector_db="faiss", qdrant_url=None, qdrant_api_key=None):
    """Unified function to get relevant RAG context with selected vector database."""
    if vector_db == "faiss":
        return get_relevant_rag_faiss(db_file, rag_index_dir, chat_id, query, model_name, ollama_models, openai_models, top_k)
    elif vector_db == "qdrant":
        if not qdrant_url:
            st.error("Qdrant URL is required when using Qdrant vector database.")
            return "", False
        return get_relevant_rag_qdrant(db_file, qdrant_url, chat_id, query, model_name, ollama_models, openai_models, top_k, qdrant_api_key)
    else:
        st.error(f"Unknown vector database: {vector_db}")
        return "", False


# For backward compatibility - keep the old function signature
def get_relevant_rag_legacy(db_file: str, rag_index_dir: str, chat_id, query, model_name, ollama_models, openai_models, top_k=3):
    """Legacy function for backward compatibility."""
    return get_relevant_rag_faiss(db_file, rag_index_dir, chat_id, query, model_name, ollama_models, openai_models, top_k)