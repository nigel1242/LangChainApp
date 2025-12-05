# modules/db.py
import sqlite3
import os

def init_db_sqlite(db_file: str):
    """Initializes the main chat database."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            used_rag INTEGER DEFAULT 0,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rag_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            file_name TEXT,
            content TEXT,
            FOREIGN KEY (chat_id) REFERENCES chats (id)
        )
    """)
    conn.commit()
    conn.close()

def create_new_chat_sqlite(db_file: str, model_name: str, first_message: str = ""):
    """Creates a new chat session in the database."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (model_name) VALUES (?)", (model_name,))
    chat_id = cursor.lastrowid
    if first_message:
        cursor.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, "user", first_message),
        )
    conn.commit()
    conn.close()
    return chat_id

def add_message_sqlite(db_file: str, chat_id, role, content, used_rag=0):
    """Adds a new message to a specific chat."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (chat_id, role, content, used_rag) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, used_rag),
    )
    conn.commit()
    conn.close()

def load_all_chats_sqlite(db_file: str):
    """Loads all chat sessions from the database."""
    if not os.path.exists(db_file):
        return []
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT id, model_name, created_at FROM chats ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def load_chat_sqlite(db_file: str, chat_id):
    """Loads all messages and the model name for a specific chat."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, used_rag FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
    rows = cursor.fetchall()
    cursor.execute("SELECT model_name FROM chats WHERE id = ?", (chat_id,))
    model_name = cursor.fetchone()[0]
    conn.close()
    return [{"role": r, "content": c, "used_rag": ur} for r, c, ur in rows], model_name

def add_rag_doc_sqlite(db_file: str, chat_id, file_name, content):
    """Adds a RAG document's content to the database."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO rag_docs (chat_id, file_name, content) VALUES (?, ?, ?)",
            (chat_id, file_name, content),
        )
        conn.commit()
        return True, f"📄 {file_name} added successfully! 🎉"
    except sqlite3.IntegrityError:
        # e.g., file already exists
        return False, f"{file_name} already exists."
    finally:
        conn.close()

def get_chat_documents_sqlite(db_file: str, chat_id):
    """Retrieves all document file names associated with a chat."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name FROM rag_docs WHERE chat_id = ?", (chat_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def delete_chat_sqlite(db_file: str, chat_id: int, rag_index_dir: str = None):
    """Deletes a chat, its messages, RAG docs, and FAISS index (if exists) from local SQLite."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Delete messages
    cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    # Delete RAG docs
    cursor.execute("DELETE FROM rag_docs WHERE chat_id = ?", (chat_id,))
    # Delete chat metadata
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()

    # Delete FAISS index file
    if rag_index_dir:
        faiss_file = os.path.join(rag_index_dir, f"chat_{chat_id}.faiss")
        if os.path.exists(faiss_file):
            os.remove(faiss_file)
