import sqlite3

conn = sqlite3.connect("chat.db")
cur = conn.cursor()

# Check what's in rag_docs
cur.execute("SELECT id, subject_name, user_id, file_name FROM rag_docs")
rows = cur.fetchall()

print("RAG Documents in database:")
for row in rows:
    print(f"  ID: {row[0]} | Subject: {row[1]} | User: {row[2]} | File: {row[3]}")

conn.close()