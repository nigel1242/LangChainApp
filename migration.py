"""
Database Migration Script
Run this ONCE to add user_id columns to existing tables
"""
import sqlite3
import os

CHAT_DB = "chat.db"

def run_migration():
    if not os.path.exists(CHAT_DB):
        print(f"Database {CHAT_DB} not found. Skipping migration.")
        return
    
    conn = sqlite3.connect(CHAT_DB)
    cur = conn.cursor()
    
    print("Starting database migration...")
    
    # Check if columns already exist
    def column_exists(table, column):
        cur.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cur.fetchall()]
        return column in columns
    
    try:
        cur.execute("PRAGMA foreign_keys = OFF;")
        
        # 1. Add user_id to subjects table
        if not column_exists('subjects', 'user_id'):
            print("Adding user_id to subjects table...")
            cur.execute("ALTER TABLE subjects ADD COLUMN user_id INTEGER")
            print("✓ Added user_id to subjects")
        else:
            print("✓ user_id already exists in subjects")
        
        # 2. Add user_id to chats table
        if not column_exists('chats', 'user_id'):
            print("Adding user_id to chats table...")
            cur.execute("ALTER TABLE chats ADD COLUMN user_id INTEGER")
            print("✓ Added user_id to chats")
        else:
            print("✓ user_id already exists in chats")
        
        # 3. Add user_id to rag_docs table
        if not column_exists('rag_docs', 'user_id'):
            print("Adding user_id to rag_docs table...")
            cur.execute("ALTER TABLE rag_docs ADD COLUMN user_id INTEGER")
            print("✓ Added user_id to rag_docs")
        else:
            print("✓ user_id already exists in rag_docs")
        
        # 4. Add user_id to attempts table (if it exists)
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attempts'")
        if cur.fetchone():
            if not column_exists('attempts', 'user_id'):
                print("Adding user_id to attempts table...")
                cur.execute("ALTER TABLE attempts ADD COLUMN user_id INTEGER")
                print("✓ Added user_id to attempts")
            else:
                print("✓ user_id already exists in attempts")
        
        # 5. Create indexes
        print("Creating indexes...")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_subjects_user_id ON subjects(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chats_subject ON chats(subject_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rag_docs_user_id ON rag_docs(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rag_docs_subject ON rag_docs(subject_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")
        
        if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attempts'").fetchone():
            cur.execute("CREATE INDEX IF NOT EXISTS idx_attempts_user_id ON attempts(user_id)")
        
        print("✓ Indexes created")
        
        cur.execute("PRAGMA foreign_keys = ON;")
        conn.commit()
        print("\n✅ Migration completed successfully!")
        print("\nNOTE: Existing data will have NULL user_id values.")
        print("You may want to assign them to a specific user or delete them.")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration()