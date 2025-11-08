# modules/llm_stack.py
import os
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Try to import local embeddings (ok if not installed)
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    _HAS_LOCAL = True
except Exception:
    _HAS_LOCAL = False


def get_embeddings(use_local: bool = False):
    """
    - use_local=True -> MiniLM (fast; needs torch + sentence-transformers)
    - else -> OpenAI embeddings (needs OPENAI_API_KEY)
    If local requested but unavailable, fallback to OpenAI.
    """
    if use_local and _HAS_LOCAL:
        try:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            print(f"[embeddings] Local MiniLM failed; falling back to OpenAI: {e}")

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY missing and local embeddings unavailable.\n"
            "Install torch + sentence-transformers or set OPENAI_API_KEY."
        )
    return OpenAIEmbeddings(openai_api_key=key)


def get_llm():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OPENAI_API_KEY in .env file.")
    return ChatOpenAI(
        openai_api_key=key,
        model_name="gpt-3.5-turbo",
        temperature=0.2
    )
