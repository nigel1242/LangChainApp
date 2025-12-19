import os
import time
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load .env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

def reset_qdrant():
    print(f"🔗 Connecting to Qdrant at: {QDRANT_URL}")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # 1. Fetch all collections
        collections = client.get_collections().collections
        
        if not collections:
            print("⚪ No collections found. Qdrant is already empty.")
            return

        print(f"🔍 Found {len(collections)} collections: {[c.name for c in collections]}")
        print("⚠️ Starting deletion process...")

        # 2. Delete all collections
        for col in collections:
            try:
                client.delete_collection(col.name)
                print(f"🗑️ Deleted: {col.name}")
            except Exception as e:
                print(f"❌ Failed to delete {col.name}: {e}")

        # 3. Verification
        time.sleep(1) # Small delay for Qdrant to process deletions
        remaining = client.get_collections().collections
        if not remaining:
            print("\n✅ All collections, vectors, and metadata have been wiped from Qdrant.")
        else:
            print(f"\n⚠️ Warning: Some collections remain: {[r.name for r in remaining]}")

    except Exception as e:
        print(f"🚨 Connection Error: {e}")

if __name__ == "__main__":
        reset_qdrant()