# modules/sqlite_db.py
import json
import sqlite3
import os
import shutil

from modules.quizstats.progress_store import clear_subject_stats

def init_db_sqlite(db_file: str):
    """
    Initializes the application database with logical user isolation.
    Note: Foreign Keys to 'users' are removed because user data lives in users.db.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Enable internal foreign keys for the messages -> chats relationship
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. SUBJECTS (user_id is a logical link to users.db)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            backend_type TEXT DEFAULT 'sqlite',
            UNIQUE(name, user_id)
        )
    """)

    # 2. CHATS
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            model_name TEXT,
            title TEXT DEFAULT 'New Chat',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 3. MESSAGES (Internal foreign key is valid as both tables are in chat.db)
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
            user_id INTEGER NOT NULL,
            file_name TEXT,
            content TEXT,
            metadata TEXT
        )
    """)

    # --- SCHEMA MIGRATIONS ---
    # These ensure existing databases are updated without losing data
    cursor.execute("PRAGMA table_info(chats)")
    columns = [column[1] for column in cursor.fetchall()]
    if "title" not in columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN title TEXT DEFAULT 'New Chat'")
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE chats ADD COLUMN user_id INTEGER")

    cursor.execute("PRAGMA table_info(subjects)")
    columns = [column[1] for column in cursor.fetchall()]
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE subjects ADD COLUMN user_id INTEGER")

    cursor.execute("PRAGMA table_info(rag_docs)")
    columns = [column[1] for column in cursor.fetchall()]
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE rag_docs ADD COLUMN user_id INTEGER")

    # --- PERFORMANCE INDEXES ---
    # Critical for fast lookups in a multi-user environment
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_subjects_user_id ON subjects(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chats_subject ON chats(subject_name, user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_docs_user_id ON rag_docs(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_docs_subject ON rag_docs(subject_name, user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")

    conn.commit()
    conn.close()

def add_subject_sqlite(db_file, user_id, name, backend_type='sqlite'):
    """Add a subject for a specific user"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO subjects (name, user_id, backend_type) VALUES (?, ?, ?)", 
                   (name.lower(), user_id, backend_type))
    conn.commit()
    conn.close()

def load_all_subjects_sqlite(db_file, user_id, backend_type='sqlite'):
    """Load all subjects for a specific user"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM subjects WHERE user_id = ? AND backend_type = ?", 
                   (user_id, backend_type))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

def create_new_chat_sqlite(db_file: str, user_id: int, subject_name: str, model_name: str):
    """Create a new chat for a specific user"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (subject_name, user_id, model_name) VALUES (?, ?, ?)", 
                   (subject_name, user_id, model_name))
    chat_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return chat_id

def update_chat_title_sqlite(db_file: str, chat_id: int, user_id: int, new_title: str):
    """Update chat title (with user verification)"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("UPDATE chats SET title = ? WHERE id = ? AND user_id = ?", 
                   (new_title, chat_id, user_id))
    conn.commit()
    conn.close()

def load_all_chats_by_subject_sqlite(db_file: str, user_id: int, subject_name: str):
    """Load all chats for a specific user and subject"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, model_name, created_at, title FROM chats WHERE subject_name = ? AND user_id = ? ORDER BY created_at DESC", 
        (subject_name, user_id)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

def add_message_sqlite(db_file: str, chat_id, role, content, used_rag=0):
    """Add a message to a chat"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (chat_id, role, content, used_rag) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, used_rag),
    )
    conn.commit()
    conn.close()

def load_chat_sqlite(db_file: str, chat_id: int, user_id: int):
    """Load a chat (with user verification)"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute("SELECT model_name FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
    res = cursor.fetchone()
    
    if not res:
        conn.close()
        return [], None 
    
    model_name = res[0]
    cursor.execute("SELECT role, content, used_rag FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
    rows = cursor.fetchall()
    conn.close()
    
    return [{"role": r, "content": c, "used_rag": ur} for r, c, ur in rows], model_name

def add_rag_doc_sqlite(db_file, user_id, subject_name, file_name, content, metadata=None):
    """Add a RAG document for a specific user's subject"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT id FROM rag_docs WHERE subject_name = ? AND user_id = ? AND file_name = ?", 
            (subject_name, user_id, file_name)
        )
        if cursor.fetchone(): 
            return False, f"{file_name} exists."    
        
        meta_str = json.dumps(metadata) if metadata else "{}"
        cursor.execute(
            "INSERT INTO rag_docs (subject_name, user_id, file_name, content, metadata) VALUES (?, ?, ?, ?, ?)", 
            (subject_name, user_id, file_name, content, meta_str)
        )
        conn.commit()
        return True, "Success"
    finally: 
        conn.close()

def get_subject_documents_sqlite(db_file: str, user_id: int, subject_name: str):
    """Get all documents for a specific user's subject"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name FROM rag_docs WHERE subject_name = ? AND user_id = ?", 
                   (subject_name, user_id))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def file_exists_sqlite(db_file, user_id, subject, filename):
    """Check if file exists for a specific user's subject"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    query = "SELECT 1 FROM rag_docs WHERE subject_name = ? AND user_id = ? AND file_name = ? LIMIT 1"
    cursor.execute(query, (subject, user_id, filename))
    res = cursor.fetchone()
    conn.close()
    return res is not None

def delete_chat_sqlite(db_file: str, chat_id: int, user_id: int):
    """Delete a chat (with user verification)"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
    conn.commit()
    conn.close()

def delete_subject_sqlite(db_file: str, user_id: int, subject_name: str, rag_index_dir: str = None):
    """Delete a subject (with user verification)"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        # Manual deletion is required since cross-file ON DELETE CASCADE is impossible
        cursor.execute("DELETE FROM subjects WHERE name = ? AND user_id = ?", (subject_name, user_id))
        cursor.execute("DELETE FROM chats WHERE subject_name = ? AND user_id = ?", (subject_name, user_id))
        cursor.execute("DELETE FROM rag_docs WHERE subject_name = ? AND user_id = ?", (subject_name, user_id))
        conn.commit()
    finally:
        conn.close()
    
    try:
        clear_subject_stats(subject_name)
    except Exception as e:
        print(f"Error clearing statistics for {subject_name}: {e}")

    if rag_index_dir:
        # Isolated user path logic
        path = os.path.join(rag_index_dir, f"user_{user_id}", subject_name) 
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Error deleting folder {path}: {e}")

def get_all_subject_docs_content_sqlite(db_file: str, user_id: int, subject_name: str):
    """Get all document content for a subject (used by RAG)"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT file_name, content, metadata FROM rag_docs WHERE subject_name = ? AND user_id = ?",
        (subject_name, user_id)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"file_name": r[0], "content": r[1], "metadata": json.loads(r[2]) if r[2] else {}} for r in rows]
