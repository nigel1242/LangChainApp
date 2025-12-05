from qdrant_client import QdrantClient
from modules.qdrant_db import get_doc_collection_name, DOC_COLLECTION_PREFIX
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env into os.environ

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# List all collections
collections = client.get_collections().collections
print("Collections in Qdrant:", [c.name for c in collections])

# Check rag_docs_* collections
for col in collections:
    if col.name.startswith(DOC_COLLECTION_PREFIX):
        points, _ = client.scroll(collection_name=col.name, limit=5)
        print(f"\nCollection: {col.name}")
        for p in points:
            print("ID:", p.id)
            print("Payload:", p.payload)
            print("Vector length:", len(p.vector) if p.vector else 0)

#THIS CODE IS TO CHECK WHATS UPLOADED