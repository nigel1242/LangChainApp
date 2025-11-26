import os
import re
import sqlite3
import streamlit as st
from typing import List
import pandas as pd
from docx import Document as DocxDocument
from pptx import Presentation
import fitz  # PyMuPDF

import ollama
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Qdrant import
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

TEXT_FILE_EXTS = {".txt", ".md"}
DOCX_EXTS = {".docx"}
PPTX_EXTS = {".pptx"}
CSV_EXTS = {".csv"}
PDF_EXTS = {".pdf"}

def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def extract_text_from_file(file) -> str:
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
            slide_text = [shape.text.strip() for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
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

################################################################################
# ----------------- RAG with FAISS -------------------

def rebuild_rag_index_faiss(db_file, rag_index_dir, chat_id, model_name, ollama_models, openai_models):
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
        for chunk in splitter.split_text(content):
            all_docs.append(Document(page_content=chunk, metadata={"source": file_name}))

    if not all_docs:
        return

    if model_name in ollama_models:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text")
    elif model_name in openai_models:
        if not st.session_state.openai_api_key:
            st.error("OpenAI API key is missing.")
            return
        embeddings = OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.openai_api_key)
    else:
        st.error(f"Unknown model: {model_name}")
        return

    faiss_index = FAISS.from_documents(all_docs, embeddings)
    faiss_index.save_local(os.path.join(rag_index_dir, f"chat_{chat_id}.faiss"))

def get_relevant_rag_faiss(db_file, rag_index_dir, chat_id, query, model_name, ollama_models, openai_models, top_k=3):
    faiss_index_path = os.path.join(rag_index_dir, f"chat_{chat_id}.faiss")
    if not os.path.exists(faiss_index_path):
        return "", False

    embeddings = OllamaEmbeddings("nomic-embed-text") if model_name in ollama_models else OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.openai_api_key)

    faiss_index = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    docs_with_scores = faiss_index.similarity_search_with_score(query, k=top_k)
    relevant_docs = [doc for doc, _ in docs_with_scores]

    query_lower = query.lower()
    for doc in faiss_index.docstore._dict.values():
        source = doc.metadata.get("source", "").lower()
        if source and re.search(r"\b" + re.escape(source) + r"\b", query_lower):
            if doc not in relevant_docs:
                relevant_docs.append(doc)

    if not relevant_docs:
        return "", False

    context = "\n\n".join(f"Source: {d.metadata.get('source','unknown')}\nContent: {d.page_content}" for d in relevant_docs)
    return context, True

################################################################################
# ----------------- RAG with Qdrant -------------------

def rebuild_rag_index_qdrant(db_file, qdrant_url, chat_id, model_name, ollama_models, openai_models, qdrant_api_key=None):
    if not QDRANT_AVAILABLE:
        st.error("Qdrant integration not available.")
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
        for chunk in splitter.split_text(content):
            all_docs.append(Document(page_content=chunk, metadata={"source": file_name}))

    if not all_docs:
        return

    embeddings = OllamaEmbeddings("nomic-embed-text") if model_name in ollama_models else OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.openai_api_key)

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
    collection_name = f"chat_{chat_id}"

    # Create collection if missing
    try:
        client.get_collection(collection_name)
    except:
        embedding_dim = len(embeddings.embed_query("test"))
        client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE))

    # Add documents
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
    vector_store.add_documents(all_docs)

def get_relevant_rag_qdrant(db_file, qdrant_url, chat_id, query, model_name, ollama_models, openai_models, top_k=3, qdrant_api_key=None):
    if not QDRANT_AVAILABLE:
        return "", False

    embeddings = OllamaEmbeddings("nomic-embed-text") if model_name in ollama_models else OpenAIEmbeddings(model_name="text-embedding-3-small", api_key=st.session_state.openai_api_key)
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=False)
    collection_name = f"chat_{chat_id}"

    try:
        qdrant_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
        relevant_docs = qdrant_store.similarity_search(query, k=top_k)
    except:
        return "", False

    if not relevant_docs:
        return "", False

    context = "\n\n".join(f"Source: {d.metadata.get('source','unknown')}\nContent: {d.page_content}" for d in relevant_docs)
    return context, True

################################################################################
# ----------------- Unified RAG Functions -------------------

def rebuild_rag_index(db_file, rag_index_dir=None, chat_id=None, model_name=None, ollama_models=None, openai_models=None, vector_db="faiss", qdrant_url=None, qdrant_api_key=None):
    if vector_db == "faiss":
        rebuild_rag_index_faiss(db_file, rag_index_dir, chat_id, model_name, ollama_models, openai_models)
    elif vector_db == "qdrant":
        rebuild_rag_index_qdrant(db_file, qdrant_url, chat_id, model_name, ollama_models, openai_models, qdrant_api_key)
    else:
        st.error(f"Unknown vector DB: {vector_db}")

def get_relevant_rag(db_file, rag_index_dir=None, chat_id=None, query=None, model_name=None, ollama_models=None, openai_models=None, top_k=3, vector_db="faiss", qdrant_url=None, qdrant_api_key=None):
    if vector_db == "faiss":
        return get_relevant_rag_faiss(db_file, rag_index_dir, chat_id, query, model_name, ollama_models, openai_models, top_k)
    elif vector_db == "qdrant":
        return get_relevant_rag_qdrant(db_file, qdrant_url, chat_id, query, model_name, ollama_models, openai_models, top_k, qdrant_api_key)
    else:
        st.error(f"Unknown vector DB: {vector_db}")
        return "", False
