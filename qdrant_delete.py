from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Fetch all collections
collections = client.get_collections().collections
print("Collections found:", [c.name for c in collections])

# Delete all collections
for col in collections:
    try:
        client.delete_collection(col.name)
        print(f"Deleted collection: {col.name}")
    except Exception as e:
        print(f"Failed to delete {col.name}: {e}")

print("✅ All collections deleted from Qdrant.")

#THIS CODE IS TO CLEAR ALL EXISTING FILES IN QDRANT