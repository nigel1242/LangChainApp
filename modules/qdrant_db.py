# modules/qdrant_db.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import PointsSelector, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import uuid
import os

# ------------------ Init Qdrant client ------------------
# ✅ Always load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL:
    raise ValueError(f"❌ QDRANT_URL not found. Looked in: {ENV_PATH}")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ------------------ Constants ------------------
DOC_COLLECTION = "rag_documents"
CHAT_COLLECTION = "chats_meta"
MESSAGE_COLLECTION = "messages"
DOC_COLLECTION_PREFIX = "rag_docs"  # We will append _<size> to this

# ------------------ Helper Functions ------------------
def get_doc_collection_name(vector_size: int) -> str:
    """Generates a collection name based on vector dimension (e.g., rag_docs_1536)."""
    return f"{DOC_COLLECTION_PREFIX}_{vector_size}"


def ensure_collection_exists(collection_name: str, vector_size: int):
    # Get existing collection names
    existing_collections = [c.name for c in client.get_collections().collections]

    # If not present, create it
    if collection_name not in existing_collections:
        print(f"Creating collection: {collection_name} with size {vector_size}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        
        if collection_name.startswith(DOC_COLLECTION_PREFIX):
            print(f"Creating payload index on 'chat_id' for filtering...")
            client.create_payload_index(
                collection_name=collection_name,
                field_name="chat_id",
                field_schema=PayloadSchemaType.KEYWORD
            )

def check_file_exists_in_chat(chat_id: str, file_name: str) -> bool:
    """
    Returns True if a file with `file_name` has already been uploaded to this chat.
    """
    try:
        collections = client.get_collections().collections
        for col in collections:
            if col.name.startswith(DOC_COLLECTION_PREFIX):
                docs_res, _ = client.scroll(collection_name=col.name, limit=1000)
                for p in docs_res:
                    if p.payload and p.payload.get("chat_id") == chat_id:
                        if p.payload.get("file_name") == file_name:
                            return True
    except Exception:
        pass
    return False

# ------------------ Chats (Metadata) ------------------
def create_new_chat_qdrant(model_name: str):
    # Chat metadata collection uses dummy 1D vector
    ensure_collection_exists(CHAT_COLLECTION, vector_size=1)
    
    chat_id = str(uuid.uuid4())
    client.upsert(
        collection_name=CHAT_COLLECTION,
        points=[{
            "id": chat_id,
            "vector": [0.0],  # 1D dummy vector
            "payload": {
                "model_name": model_name,
                "created_at": datetime.utcnow().isoformat()
            }
        }]
    )
    return chat_id

def load_all_chats_qdrant():
    ensure_collection_exists(CHAT_COLLECTION, vector_size=1)

    try:
        points, _ = client.scroll(collection_name=CHAT_COLLECTION, limit=1000)
    except Exception:
        return []

    chats = []
    for p in points:
        payload = p.payload or {}
        chats.append((
            p.id,
            payload.get("model_name", "Unknown"),
            payload.get("created_at", "")
        ))
    return chats

def load_chat_qdrant(chat_id):
    ensure_collection_exists(CHAT_COLLECTION, vector_size=1)
    chat_info = client.retrieve(collection_name=CHAT_COLLECTION, ids=[chat_id])
    model_name = chat_info[0].payload.get("model_name", "Unknown") if chat_info else "Unknown"

    ensure_collection_exists(MESSAGE_COLLECTION, vector_size=1)
    all_messages = []
    offset = 0
    batch_size = 1000

    while True:
        points, next_offset = client.scroll(
            collection_name=MESSAGE_COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        if not points: break
        
        for p in points:
            # Handle potential list wrap
            if isinstance(p, list): p = p[0]
            
            payload = getattr(p, "payload", None)
            if payload and payload.get("chat_id") == chat_id:
                all_messages.append(payload)
        
        if next_offset is None: break
        offset = next_offset

    all_messages.sort(key=lambda x: x.get("created_at") or "")
    return all_messages, model_name

# ------------------ Messages ------------------
def add_message_qdrant(chat_id: str, role: str, content: str, used_rag: int = 0):
    ensure_collection_exists(MESSAGE_COLLECTION, vector_size=1)
    message_id = str(uuid.uuid4())
    client.upsert(
        collection_name=MESSAGE_COLLECTION,
        points=[
            PointStruct(
                id=message_id,
                vector=[0.0],
                payload={
                    "chat_id": chat_id,
                    "role": role,
                    "content": content,
                    "used_rag": used_rag,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
        ]
    )

# ------------------ RAG Docs (The Logic Change) ------------------

def add_rag_doc_qdrant(chat_id: str, file_name: str, vector_data: list, content: str, metadata: dict = None):
    if metadata is None:
        metadata = {}

    # 1. Determine size from the incoming data
    vector_size = len(vector_data)
    
    # 2. Get dynamic collection name (e.g. rag_docs_1536)
    collection_name = get_doc_collection_name(vector_size)
    
    # 3. Ensure that specific collection exists
    ensure_collection_exists(collection_name, vector_size=vector_size)
    
    doc_id = str(uuid.uuid4())
    
    try:
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=doc_id,
                    vector=vector_data,
                    payload={
                        "chat_id": chat_id,
                        "file_name": file_name,
                        "content": content,
                        **metadata,
                        "created_at": datetime.utcnow().isoformat()
                    }
                )
            ]
        )
        # ✅ Return success tuple
        return True, f"📄 {file_name} added successfully! 🎉"
    except Exception as e:
        return False, f"❌ Failed to add {file_name}: {e}"

def search_rag_docs_qdrant(chat_id: str, query_vector: list, limit: int = 5):
    vector_size = len(query_vector)
    collection_name = get_doc_collection_name(vector_size)
    
    print(f"--- [DEBUG] Search Triggered ---")
    print(f"1. Query Vector Size: {vector_size}")
    print(f"2. Target Collection: {collection_name}")
    
    try:
        # Check if collection exists
        client.get_collection(collection_name)
        print("3. Collection found ✅")
    except Exception:
        print(f"3. ❌ Collection {collection_name} NOT found. Returning empty.")
        return []

    # --- GLOBAL SEARCH (Filter Removed) ---
    search_filter = Filter(
        must=[FieldCondition(key="chat_id", match=MatchValue(value=chat_id))]
    )
    
    try:
        search_result = client.query_points(
            collection_name=collection_name,
            query=query_vector,       # <--- Argument is now 'query', not 'query_vector'
            query_filter=search_filter,
            limit=limit,
            with_payload=True         # <--- Explicitly request payload
        ).points                      # <--- Access .points to get the list!
        
        print(f"4. Search performed. Found {len(search_result)} results.")
        
    except Exception as e:
        print(f"4. ❌ Error during search: {e}")
        return []

    results = []
    for point in search_result:
        if point.payload and "content" in point.payload:
            file_name = point.payload.get("file_name", "Unknown")
            print(f"   -> Found match in file: {file_name}")
            content = point.payload["content"]
            results.append(f"Source ({file_name}): {content}")
    
    return results

# ------------------ Delete Chat ------------------
def delete_chat_qdrant(chat_id: str):
    # Delete from main collections
    try:
        client.delete(collection_name=CHAT_COLLECTION, points_selector=[chat_id])
    except Exception: pass

    try:
        # For messages, we must scroll and find by payload
        messages_res, _ = client.scroll(collection_name=MESSAGE_COLLECTION, limit=1000)
        msg_ids = [p.id for p in messages_res if p.payload and p.payload.get("chat_id") == chat_id]
        if msg_ids:
            client.delete(collection_name=MESSAGE_COLLECTION, points_selector=msg_ids)
    except Exception: pass

    # For docs, we don't know the vector size of the deleted chat easily.
    # Strategy: Check ALL rag_docs_* collections that exist.
    try:
        collections = client.get_collections().collections
        for col in collections:
            if col.name.startswith(DOC_COLLECTION_PREFIX):
                # Try to delete documents belonging to this chat_id in this collection
                try:
                    # We need to scroll to find points with this chat_id
                    docs_res, _ = client.scroll(collection_name=col.name, limit=1000)
                    doc_ids = [p.id for p in docs_res if p.payload and p.payload.get("chat_id") == chat_id]
                    if doc_ids:
                        client.delete(collection_name=col.name, points_selector=doc_ids)
                except Exception:
                    continue
    except Exception:
        pass