# modules/qdrant_db.py
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from datetime import datetime
import uuid
import streamlit as st

# ------------------ Init Qdrant client ------------------

def get_client():
    """
    Returns a QdrantClient using credentials stored in the current 
    Streamlit session (retrieved from users.db during login).
    """
    # Keys are populated in login.py during authentication/check_authentication
    url = st.session_state.get("qdrant_url", "").strip()
    api_key = st.session_state.get("qdrant_api_key", "").strip()

    if not url:
        # If no URL is found in session, we cannot connect. 
        # In a multi-user app, we raise an error rather than falling back to a global env.
        raise ConnectionError("Qdrant URL not found in user session. Please update Settings.")

    return QdrantClient(
        url=url,
        api_key=api_key if api_key else None,
        prefer_grpc=False,
        timeout=10
    )

# ------------------ Constants ------------------
CHAT_COLLECTION = "chats_meta"
MESSAGE_COLLECTION = "messages"
DOC_COLLECTION_PREFIX = "rag_docs"

# ------------------ Helper Functions ------------------
def get_doc_collection_name(vector_size: int) -> str:
    return f"{DOC_COLLECTION_PREFIX}_{vector_size}"

def ensure_collection_exists(collection_name: str, vector_size: int = None):
    """Ensures collection exists with proper payload indexes."""
    client = get_client() 
    existing_collections = [c.name for c in client.get_collections().collections]

    if collection_name not in existing_collections:
        v_size = vector_size if vector_size else 1
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=v_size, distance=qmodels.Distance.COSINE)
        )
        
    # --- AUTO-INDEXING FOR PERFORMANCE & FILTERING ---
    if collection_name == CHAT_COLLECTION:
        client.create_payload_index(collection_name, "subject", "keyword")
    elif collection_name == MESSAGE_COLLECTION:
        client.create_payload_index(collection_name, "chat_id", "keyword")
    elif collection_name.startswith(DOC_COLLECTION_PREFIX):
        client.create_payload_index(collection_name, "subject_name", "keyword")
        client.create_payload_index(collection_name, "chat_id", "keyword")

# ------------------ Subjects ------------------
def load_all_subjects_qdrant():
    """Lists all unique subject names stored across RAG collections."""
    client = get_client()
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
    client = get_client()
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
    client = get_client()
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
    client = get_client()
    ensure_collection_exists(CHAT_COLLECTION, vector_size=1)
    chat_filter = qmodels.Filter(must=[qmodels.FieldCondition(key="subject", match=qmodels.MatchValue(value=subject_name))])
    try:
        points, _ = client.scroll(
            collection_name=CHAT_COLLECTION,
            scroll_filter=chat_filter,
            limit=100,
            with_payload=True
        )
        return [(p.id, p.payload.get("model_name"), p.payload.get("created_at"), p.payload.get("title", "New Chat")) for p in points]
    except Exception: return []

def load_chat_qdrant(chat_id):
    client = get_client()
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
    client = get_client()
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
    """Upserts a document chunk and its vector to the appropriate collection."""
    client = get_client()
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
    client = get_client()
    collection_name = get_doc_collection_name(vector_size)
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        return False

    result = client.scroll(
        collection_name=collection_name,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="subject_name", match=qmodels.MatchValue(value=subject_name)),
                qmodels.FieldCondition(key="file_name", match=qmodels.MatchValue(value=file_name))
            ]
        ),
        limit=1,
        with_payload=False
    )
    return len(result[0]) > 0

# ------------------ Cleanup ------------------
def delete_chat_qdrant(chat_id: str):
    """Deletes a specific chat and all its associated messages."""
    client = get_client()
    delete_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=chat_id))]
    )
    client.delete(collection_name=CHAT_COLLECTION, points_selector=[chat_id])
    try:
        client.delete(
            collection_name=MESSAGE_COLLECTION, 
            points_selector=qmodels.FilterSelector(filter=delete_filter)
        )
    except Exception as e:
        print(f"Qdrant Message Delete Error: {e}")

def delete_subject_qdrant(subject_name: str):
    """Removes all data linked to a subject across all collections."""
    client = get_client()
    chat_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="subject", match=qmodels.MatchValue(value=subject_name))]
    )
    chat_ids = []
    offset = None
    while True:
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

    if chat_ids:
        msg_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="chat_id", match=qmodels.MatchAny(any=chat_ids))]
        )
        client.delete(
            collection_name=MESSAGE_COLLECTION, 
            points_selector=qmodels.FilterSelector(filter=msg_filter)
        )

    client.delete(
        collection_name=CHAT_COLLECTION, 
        points_selector=qmodels.FilterSelector(filter=chat_filter)
    )

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
def update_chat_title_qdrant(chat_id: str, new_title: str):
    """Updates the title of a specific chat in Qdrant."""
    client = get_client()
    client.set_payload(
        collection_name=CHAT_COLLECTION,
        payload={"title": new_title},
        points=[chat_id]
    )