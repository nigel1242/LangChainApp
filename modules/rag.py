# modules/rag.py
import os
import re
import sqlite3
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Local Import ---
from .embeddings import OllamaEmbeddings, OpenAIEmbeddings

# ----------------- RAG Core Functions -------------------
def rebuild_rag_index(db_file: str, rag_index_dir: str, chat_id, model_name, ollama_models, openai_models):
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

    # Ollama embedding
    if model_name in ollama_models:
        embeddings = OllamaEmbeddings(model_name="nomic-embed-text")
    # OpenAI embedding
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

def get_relevant_rag(db_file: str, rag_index_dir: str, chat_id, query, model_name, ollama_models, openai_models, top_k=3):
    """Gets relevant RAG context from the FAISS index."""
    faiss_index_path = os.path.join(rag_index_dir, f"chat_{chat_id}.faiss")
    if not os.path.exists(faiss_index_path):
        return "", False

    # --- Choose embedding ---
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

    # --- Load FAISS ---
    faiss_index = FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # --- Semantic similarity search ---
    docs_with_scores = faiss_index.similarity_search_with_score(query, k=top_k)
    relevant_docs = [doc for doc, score in docs_with_scores]

    # --- Filename matching (case-insensitive) ---
    query_lower = query.lower()
    for doc in faiss_index.docstore._dict.values():  # all docs in the index
        source = doc.metadata.get("source", "").lower()
        if source and re.search(r"\b" + re.escape(source) + r"\b", query_lower):
            if doc not in relevant_docs:
                relevant_docs.append(doc)

    if not relevant_docs:
        return "", False

    # --- Combine into context ---
    context = "\n\n".join(
        f"Source: {d.metadata.get('source','unknown')}\nContent: {d.page_content}"
        for d in relevant_docs
    )
    return context, True