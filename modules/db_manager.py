import os
import streamlit as st
import sqlite3
from qdrant_client import QdrantClient
from qdrant_client.http import models

# --- Import Updated SQLite functions ---
from modules.sqlite_db import (
    init_db_sqlite,
    add_subject_sqlite,
    load_all_subjects_sqlite,
    create_new_chat_sqlite,
    load_all_chats_by_subject_sqlite,
    add_message_sqlite,
    load_chat_sqlite,
    file_exists_sqlite,
    delete_chat_sqlite,
    delete_subject_sqlite,
    add_rag_doc_sqlite,
    get_subject_documents_sqlite
)

# --- Import Qdrant functions ---
from modules.qdrant_db import (
    create_new_chat_qdrant,
    delete_subject_qdrant, 
    load_all_chats_qdrant, 
    load_chat_qdrant, 
    add_message_qdrant, 
    delete_chat_qdrant, 
    add_rag_doc_qdrant,
    load_all_subjects_qdrant,
    create_subject_meta_qdrant,
    file_exists_qdrant
)

class DBManager:
    def __init__(self, backend: str, db_file: str = None, qdrant_url: str = None, qdrant_api_key: str = None):
        self.backend = backend
        self.db_file = db_file
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key

        if backend == "sqlite" and db_file:
            init_db_sqlite(db_file)

    # ----------------- Subject Management -----------------  
    def add_subject(self, subject_name: str):
        if self.backend == "sqlite":
            return add_subject_sqlite(self.db_file, subject_name)
        elif self.backend == "qdrant":
            return create_subject_meta_qdrant(subject_name)

    def load_all_subjects(self):
        if self.backend == "sqlite":
            return load_all_subjects_sqlite(self.db_file)
        if self.backend == "qdrant":
            return load_all_subjects_qdrant()

    def file_exists_in_rag(self, subject, filename, vector_size=4096):
        if self.backend == "sqlite":
            return file_exists_sqlite(self.db_file, subject, filename)
        if self.backend == "qdrant":
            return file_exists_qdrant(subject, filename, vector_size)


    def delete_subject(self, subject_name: str, rag_index_dir: str = None):
        if self.backend == "sqlite":
            return delete_subject_sqlite(self.db_file, subject_name, rag_index_dir)
        if self.backend == "qdrant":
                return delete_subject_qdrant(subject_name)

    # ----------------- Chat Management -----------------

    def create_new_chat(self, model_name: str, subject_name: str):
        if self.backend == "qdrant":
            return create_new_chat_qdrant(model_name, subject_name)
        return create_new_chat_sqlite(self.db_file, subject_name, model_name)

    def load_all_chats(self, subject_name: str):
        if self.backend == "qdrant":
            return load_all_chats_qdrant(subject_name)
        return load_all_chats_by_subject_sqlite(self.db_file, subject_name)
    
    def update_chat_title(self, chat_id, new_title):
        if self.backend == "qdrant":
            from qdrant_client.http import models as qmodels
            from modules.qdrant_db import client, CHAT_COLLECTION
            
            client.set_payload(
                collection_name=CHAT_COLLECTION,
                payload={"title": new_title},
                points=[chat_id]
            )
        else:
            # For SQLite
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("UPDATE chats SET title = ? WHERE id = ?", (new_title, chat_id))
            conn.commit()
            conn.close()

    def load_chat(self, chat_id):
        if self.backend == "qdrant":
            return load_chat_qdrant(chat_id)
        return load_chat_sqlite(self.db_file, chat_id)

    def delete_chat(self, chat_id):
        if self.backend == "qdrant":
            return delete_chat_qdrant(chat_id)
        return delete_chat_sqlite(self.db_file, chat_id)

    # ----------------- Message Management -----------------
    def add_message(self, chat_id, role, content, used_rag=0):
        if self.backend == "qdrant":
            return add_message_qdrant(chat_id, role, content, used_rag)
        return add_message_sqlite(self.db_file, chat_id, role, content, used_rag)

    # ----------------- RAG Document Management -----------------

    def add_rag_doc(self, target_id, file_name, content, vector_data: list = None, metadata=None):
        if self.backend == "qdrant":
            return add_rag_doc_qdrant(target_id, file_name, vector_data, content)
        
        return add_rag_doc_sqlite(self.db_file, target_id, file_name, content, metadata)

    def get_subject_documents(self, subject_name: str):
        # Default to SQLite for library listing metadata
        return get_subject_documents_sqlite(self.db_file, subject_name)