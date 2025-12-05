# modules/db_manager.py
import os
import streamlit as st

# --- Import SQLite functions ---
from modules.sqlite_db import (
    init_db_sqlite,
    create_new_chat_sqlite,
    add_message_sqlite,
    load_chat_sqlite,
    load_all_chats_sqlite,
    delete_chat_sqlite,
    add_rag_doc_sqlite,
    get_chat_documents_sqlite
)

# --- Import Qdrant functions ---
from modules.qdrant_db import (
    create_new_chat_qdrant,
    add_message_qdrant,
    load_chat_qdrant,
    load_all_chats_qdrant,
    delete_chat_qdrant,
    add_rag_doc_qdrant
)

class DBManager:
    def __init__(self, backend: str, db_file: str = None, qdrant_url: str = None, qdrant_api_key: str = None):
        """
        backend: 'sqlite' or 'qdrant'
        db_file: path to SQLite database
        qdrant_url/qdrant_api_key: for Qdrant
        """
        self.backend = backend
        self.db_file = db_file
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key

        if backend == "sqlite" and db_file:
            if not os.path.exists(db_file):
                init_db_sqlite(db_file)

    # ----------------- Chats -----------------
    def create_new_chat(self, model_name: str, first_message: str = ""):
        if self.backend == "sqlite":
            return create_new_chat_sqlite(self.db_file, model_name, first_message)
        elif self.backend == "qdrant":
            return create_new_chat_qdrant(model_name)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def load_all_chats(self):
        if self.backend == "sqlite":
            return load_all_chats_sqlite(self.db_file)
        elif self.backend == "qdrant":
            return load_all_chats_qdrant()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def load_chat(self, chat_id):
        if self.backend == "sqlite":
            return load_chat_sqlite(self.db_file, chat_id)
        elif self.backend == "qdrant":
            return load_chat_qdrant(chat_id)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def delete_chat(self, chat_id, rag_index_dir: str = None):
        if self.backend == "sqlite":
            return delete_chat_sqlite(self.db_file, chat_id, rag_index_dir)
        elif self.backend == "qdrant":
            return delete_chat_qdrant(chat_id)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ----------------- Messages -----------------
    def add_message(self, chat_id, role, content, used_rag=0):
        if self.backend == "sqlite":
            return add_message_sqlite(self.db_file, chat_id, role, content, used_rag)
        elif self.backend == "qdrant":
            return add_message_qdrant(chat_id, role, content, used_rag)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ----------------- RAG Docs -----------------
    def add_rag_doc(self, chat_id, file_name, content, vector_data=None, metadata=None):
        """
        For SQLite: content is stored.
        For Qdrant: vector_data is required.
        """
        if self.backend == "sqlite":
            return add_rag_doc_sqlite(self.db_file, chat_id, file_name, content)
        elif self.backend == "qdrant":
            if vector_data is None:
                raise ValueError("vector_data is required for Qdrant")
            if metadata is None:
                metadata = {}
            return add_rag_doc_qdrant(chat_id, file_name, vector_data, content, metadata)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def get_chat_documents(self, chat_id):
        if self.backend == "sqlite":
            return get_chat_documents_sqlite(self.db_file, chat_id)
        elif self.backend == "qdrant":
            # For Qdrant, you might need a custom function; here we return empty for now
            st.warning("get_chat_documents not implemented for Qdrant yet.")
            return []
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
