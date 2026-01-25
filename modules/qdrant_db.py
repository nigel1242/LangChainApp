# modules/qdrant_db.py
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import uuid
import os

# ------------------ Init Qdrant client ------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ✅ prefer_grpc=False is critical for stable HTTPS connections on Qdrant Cloud
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=False 
)

# ------------------ Constants ------------------
CHAT_COLLECTION = "chats_meta"
MESSAGE_COLLECTION = "messages"
DOC_COLLECTION_PREFIX = "rag_docs"

# ------------------ Helper Functions ------------------
def get_doc_collection_name(vector_size: int) -> str:
    return f"{DOC_COLLECTION_PREFIX}_{vector_size}"

def ensure_collection_exists(collection_name: str, vector_size: int = None):
    """
    Ensures the collection exists and has all necessary payload indexes 
    to support filtered deletions and searches.
    """
    existing_collections = [c.name for c in client.get_collections().collections]

    if collection_name not in existing_collections:
        # If vector_size is 1, it's a metadata collection
        v_size = vector_size if vector_size else 1
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=v_size, distance=qmodels.Distance.COSINE)
        )
        
    # --- AUTO-INDEXING FOR CLOUD COMPATIBILITY ---
    # These indexes prevent "Index required but not found" errors during filtered operations.
    
    if collection_name == CHAT_COLLECTION:
        # Index 'subject' for listing chats by subject
        client.create_payload_index(collection_name, "subject", "keyword")
        
    elif collection_name == MESSAGE_COLLECTION:
        # CRITICAL: Index 'chat_id' to allow deleting all messages for a specific chat
        client.create_payload_index(collection_name, "chat_id", "keyword")
        
    elif collection_name.startswith(DOC_COLLECTION_PREFIX):
        # Index 'subject_name' for RAG searches
        client.create_payload_index(collection_name, "subject_name", "keyword")
        # NEW: Index 'chat_id' for document collections to support chat-linked RAG cleanup
        client.create_payload_index(collection_name, "chat_id", "keyword")

# ------------------ Subjects ------------------
def load_all_subjects_qdrant():
    subjects = set()
    try:
        collections = client.get_collections().collections
        for col in collections:
            if col.name.startswith(DOC_COLLECTION_PREFIX):
                offset = None
                while True:
                    records, offset = client.scroll(
                        collection_name=col.name,
                        limit=100,
                        with_payload=True,
                        offset=offset
                    )
                    for record in records:
                        s_name = record.payload.get("subject_name")
                        if s_name: subjects.add(s_name)
                    if offset is None: break
    except Exception: pass
    if not subjects: subjects.add("General")
    return sorted(list(subjects))

def create_subject_meta_qdrant(subject_name: str):
    """Creates a placeholder point so a new subject appears in the UI."""
    meta_col = f"{DOC_COLLECTION_PREFIX}_metadata"
    ensure_collection_exists(meta_col, vector_size=1)
    try:
        client.upsert(
            collection_name=meta_col,
            points=[qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.0],
                payload={
                    "subject_name": subject_name, 
                    "is_meta": True,
                    "created_at": datetime.utcnow().isoformat()
                }
            )]
        )
        return True
    except Exception as e:
        print(f"Error creating subject meta: {e}")
        return False

# ------------------ Chats & Messages ------------------
def create_new_chat_qdrant(model_name: str, subject_name: str):
    ensure_collection_exists(CHAT_COLLECTION, vector_size=1)
    chat_id = str(uuid.uuid4())
    client.upsert(
        collection_name=CHAT_COLLECTION,
        points=[qmodels.PointStruct(
            id=chat_id,
            vector=[0.0],
            payload={
                "model_name": model_name,
                "subject": subject_name,
                "title": "New Chat",
                "created_at": datetime.utcnow().isoformat()
            }
        )]
    )
    return chat_id

def load_all_chats_qdrant(subject_name: str):
    ensure_collection_exists(CHAT_COLLECTION, vector_size=1)
    chat_filter = qmodels.Filter(must=[qmodels.FieldCondition(key="subject", match=qmodels.MatchValue(value=subject_name))])
    try:
        points, _ = client.scroll(
            collection_name=CHAT_COLLECTION,
            scroll_filter=chat_filter,
            limit=100,
            with_payload=True
        )
        # Returns: (id, model_name, created_at, title)
        return [(p.id, p.payload.get("model_name"), p.payload.get("created_at"), p.payload.get("title", "New Chat")) for p in points]
    except Exception: return []

def load_chat_qdrant(chat_id):
    chat_info = client.retrieve(collection_name=CHAT_COLLECTION, ids=[chat_id])
    model_name = chat_info[0].payload.get("model_name", "Unknown") if chat_info else "Unknown"
    
    chat_filter = qmodels.Filter(must=[qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=chat_id))])
    all_messages = []
    offset = None
    while True:
        points, offset = client.scroll(collection_name=MESSAGE_COLLECTION, scroll_filter=chat_filter, limit=100, with_payload=True, offset=offset)
        for p in points: all_messages.append(p.payload)
        if offset is None: break

    all_messages.sort(key=lambda x: x.get("created_at") or "")
    return all_messages, model_name

def add_message_qdrant(chat_id: str, role: str, content: str, used_rag: int = 0):
    ensure_collection_exists(MESSAGE_COLLECTION, vector_size=1)
    client.upsert(
        collection_name=MESSAGE_COLLECTION,
        points=[qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.0],
            payload={
                "chat_id": chat_id, "role": role, "content": content,
                "used_rag": used_rag, "created_at": datetime.utcnow().isoformat()
            }
        )]
    )

# ------------------ RAG Docs ------------------
def add_rag_doc_qdrant(subject_name: str, file_name: str, vector_data: list, content: str, metadata: dict = None):
    vector_size = len(vector_data)
    collection_name = get_doc_collection_name(vector_size)
    ensure_collection_exists(collection_name, vector_size=vector_size)
    
    try:
        client.upsert(
            collection_name=collection_name,
            points=[qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector_data,
                payload={
                    "subject_name": subject_name, "file_name": file_name,
                    "content": content, "created_at": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
            )]
        )
        return True, f"📄 {file_name} added to {subject_name}!"
    except Exception as e:
        return False, str(e)

def file_exists_qdrant(subject_name: str, file_name: str, vector_size: int) -> bool:
    """
    Checks if a file has already been indexed in a specific Qdrant collection.
    """
    collection_name = get_doc_collection_name(vector_size)
    
    # 1. Verify collection exists before querying
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        return False

    # 2. Filter search for subject + filename
    result = client.scroll(
        collection_name=collection_name,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="subject_name", match=qmodels.MatchValue(value=subject_name)),
                qmodels.FieldCondition(key="file_name", match=qmodels.MatchValue(value=file_name))
            ]
        ),
        limit=1, # We only need to find one record to confirm it exists
        with_payload=False
    )
    
    # Scroll returns a tuple of (points, next_page_offset)
    return len(result[0]) > 0

# ------------------ Cleanup ------------------
def delete_chat_qdrant(chat_id: str):
    """Deletes only the chat metadata and its messages."""
    delete_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=chat_id))]
    )
    
    # 1. Delete the chat entry from metadata collection
    client.delete(collection_name=CHAT_COLLECTION, points_selector=[chat_id])
    
    # 2. Delete all messages associated with this chat
    # This requires the "chat_id" index which is created in ensure_collection_exists
    try:
        client.delete(
            collection_name=MESSAGE_COLLECTION, 
            points_selector=qmodels.FilterSelector(filter=delete_filter)
        )
    except Exception as e:
        print(f"Qdrant Message Delete Error: {e}")

    # NOTE: Do NOT loop through DOC_COLLECTION_PREFIX here. 
    # RAG docs belong to the Subject, not the Chat.

def delete_subject_qdrant(subject_name: str):
    # 1. Find all Chat IDs for this subject (needed to delete messages)
    chat_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="subject", match=qmodels.MatchValue(value=subject_name))]
    )

    chat_ids = []
    offset = None
    while True:
        # Scroll through CHAT_COLLECTION to get all point IDs (chat_ids)
        records, offset = client.scroll(
            collection_name=CHAT_COLLECTION,
            scroll_filter=chat_filter,
            limit=100,
            with_payload=False,
            offset=offset
        )
        for r in records:
            chat_ids.append(r.id)
        if offset is None: break

    # 2. Delete Messages in MESSAGE_COLLECTION
    if chat_ids:
        # We use MatchAny to delete messages for all chats of this subject at once
        msg_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="chat_id", match=qmodels.MatchAny(any=chat_ids))]
        )
        client.delete(
            collection_name=MESSAGE_COLLECTION, 
            points_selector=qmodels.FilterSelector(filter=msg_filter)
        )

    # 3. Delete Chats in CHAT_COLLECTION
    client.delete(
        collection_name=CHAT_COLLECTION, 
        points_selector=qmodels.FilterSelector(filter=chat_filter)
    )

    # 4. Delete RAG Docs & Metadata across all "rag_docs_..." collections
    doc_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="subject_name", match=qmodels.MatchValue(value=subject_name))]
    )

    all_collections = client.get_collections().collections
    for col in all_collections:
        if col.name.startswith(DOC_COLLECTION_PREFIX):
            try:
                client.delete(
                    collection_name=col.name, 
                    points_selector=qmodels.FilterSelector(filter=doc_filter)
                )
            except Exception as e:
                print(f"[qdrant_db] Could not delete subject points from {col.name}: {e}")

    return True