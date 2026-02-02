import json
import sqlite3
import os
import shutil
from modules.quizstats.progress_store import clear_subject_stats

def init_db_sqlite(db_file: str):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. SUBJECTS - Added user_id and made (name, user_id) unique
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            UNIQUE(name, user_id)
        )
    """)

    # 2. CHATS - Added user_id and linked to subject_name
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            subject_name TEXT NOT NULL,
            model_name TEXT,
            title TEXT DEFAULT 'New Chat',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (subject_name) REFERENCES subjects (name) ON DELETE CASCADE
        )
    """)

    # --- SCHEMA MIGRATION ---
    cursor.execute("PRAGMA table_info(chats)")
    columns = [column[1] for column in cursor.fetchall()]
    if "title" not in columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN title TEXT DEFAULT 'New Chat'")
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN user_id INTEGER DEFAULT 1")

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
            user_id INTEGER NOT NULL,
            subject_name TEXT NOT NULL,
            file_name TEXT,
            content TEXT,
            metadata TEXT,
            FOREIGN KEY (subject_name) REFERENCES subjects (name) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

# --- SUBJECT FUNCTIONS ---

def add_subject_sqlite(db_file, name, user_id):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Corrected columns: name, user_id
    cursor.execute("INSERT OR IGNORE INTO subjects (name, user_id) VALUES (?, ?)", (name, user_id))
    conn.commit()
    conn.close()

def load_all_subjects_sqlite(db_file, user_id):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM subjects WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

# --- CHAT FUNCTIONS ---

def create_new_chat_sqlite(db_file: str, user_id: int, subject_name: str, model_name: str):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chats (user_id, subject_name, model_name) VALUES (?, ?, ?)", 
        (user_id, subject_name, model_name)
    )
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

def load_all_chats_sqlite(db_file: str, user_id: int, subject_name: str):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Isolates chats by BOTH user and subject
    cursor.execute("""
        SELECT id, model_name, created_at, title 
        FROM chats 
        WHERE user_id = ? AND subject_name = ? 
        ORDER BY created_at DESC
    """, (user_id, subject_name))
    rows = cursor.fetchall()
    conn.close()
    return rows

def delete_chat_sqlite(db_file: str, chat_id: int):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()

# --- MESSAGE FUNCTIONS ---

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

# --- RAG / KNOWLEDGE BASE FUNCTIONS ---

def add_rag_doc_sqlite(db_file, user_id, subject_name, file_name, content, metadata=None):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM rag_docs WHERE user_id = ? AND subject_name = ? AND file_name = ?", 
                       (user_id, subject_name, file_name))
        if cursor.fetchone(): 
            return False, f"{file_name} exists."    
        
        meta_str = json.dumps(metadata) if metadata else "{}"
        
        cursor.execute(
            "INSERT INTO rag_docs (user_id, subject_name, file_name, content, metadata) VALUES (?, ?, ?, ?, ?)", 
            (user_id, subject_name, file_name, content, meta_str)
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

# --- CLEANUP FUNCTIONS ---

def delete_subject_sqlite(db_file: str, subject_name: str, subject_dir: str = None):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys = ON;") 
        cursor.execute("DELETE FROM subjects WHERE name = ?", (subject_name,))
        conn.commit()
    finally:
        conn.close()
    
    try:
        clear_subject_stats(subject_name)
    except Exception as e:
        print(f"Error clearing statistics for {subject_name}: {e}")

    if subject_dir:
        path = os.path.join(subject_dir, subject_name) 
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Error deleting folder {path}: {e}")