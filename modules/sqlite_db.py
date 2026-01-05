import json
import sqlite3
import os
import shutil

def init_db_sqlite(db_file: str):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. SUBJECTS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            backend_type TEXT DEFAULT 'sqlite'
        )
    """)

    # 2. CHATS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL,
            model_name TEXT,
            title TEXT DEFAULT 'New Chat',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_name) REFERENCES subjects (name) ON DELETE CASCADE
        )
    """)

    # --- SCHEMA MIGRATION ---
    # Ensures the 'title' column exists in older database files
    cursor.execute("PRAGMA table_info(chats)")
    columns = [column[1] for column in cursor.fetchall()]
    if "title" not in columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN title TEXT DEFAULT 'New Chat'")

    # 3. MESSAGES
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            used_rag INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
        )
    """)

    # 4. RAG_DOCS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rag_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL,
            file_name TEXT,
            content TEXT,
            metadata TEXT,
            FOREIGN KEY (subject_name) REFERENCES subjects (name) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

def add_subject_sqlite(db_file, name, backend_type='sqlite'):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO subjects (name, backend_type) VALUES (?, ?)", (name, backend_type))
    conn.commit()
    conn.close()

def load_all_subjects_sqlite(db_file, backend_type='sqlite'):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM subjects WHERE backend_type = ?", (backend_type,))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

def create_new_chat_sqlite(db_file: str, subject_name: str, model_name: str):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (subject_name, model_name) VALUES (?, ?)", (subject_name, model_name))
    chat_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return chat_id

def update_chat_title_sqlite(db_file: str, chat_id: int, new_title: str):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("UPDATE chats SET title = ? WHERE id = ?", (new_title, chat_id))
    conn.commit()
    conn.close()

def load_all_chats_by_subject_sqlite(db_file: str, subject_name: str):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Now includes 'title' for the sidebar
    cursor.execute("SELECT id, model_name, created_at, title FROM chats WHERE subject_name = ? ORDER BY created_at DESC", (subject_name,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def add_message_sqlite(db_file: str, chat_id, role, content, used_rag=0):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (chat_id, role, content, used_rag) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, used_rag),
    )
    conn.commit()
    conn.close()

def load_chat_sqlite(db_file: str, chat_id):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, used_rag FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
    rows = cursor.fetchall()
    cursor.execute("SELECT model_name FROM chats WHERE id = ?", (chat_id,))
    res = cursor.fetchone()
    model_name = res[0] if res else "Unknown"
    conn.close()
    return [{"role": r, "content": c, "used_rag": ur} for r, c, ur in rows], model_name

def add_rag_doc_sqlite(db_file, subject_name, file_name, content, metadata=None):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        # 1. Check for duplicates
        cursor.execute("SELECT id FROM rag_docs WHERE subject_name = ? AND file_name = ?", (subject_name, file_name))
        if cursor.fetchone(): 
            return False, f"{file_name} exists."    
        # 2. Convert dictionary to JSON string so SQLite can save it
        meta_str = json.dumps(metadata) if metadata else "{}"
        # 3. Insert into the 4 columns
        cursor.execute(
            "INSERT INTO rag_docs (subject_name, file_name, content, metadata) VALUES (?, ?, ?, ?)", 
            (subject_name, file_name, content, meta_str)
        )
        conn.commit()
        return True, "Success"
    finally: 
        conn.close()

def get_subject_documents_sqlite(db_file: str, subject_name: str):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name FROM rag_docs WHERE subject_name = ?", (subject_name,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def file_exists_sqlite(db_file, subject, filename):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    query = "SELECT 1 FROM rag_docs WHERE subject_name = ? AND file_name = ? LIMIT 1"
    cursor.execute(query, (subject, filename))
    res = cursor.fetchone()
    conn.close()
    return res is not None

def delete_chat_sqlite(db_file: str, chat_id: int):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()

def delete_subject_sqlite(db_file: str, subject_name: str, rag_index_dir: str = None):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM subjects WHERE name = ?", (subject_name,))
    conn.commit()
    conn.close()
    if rag_index_dir:
        path = os.path.join(rag_index_dir, f"subject_{subject_name}")
        if os.path.exists(path): shutil.rmtree(path)