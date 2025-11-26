# modules/embeddings.py
import ollama
from openai import OpenAI
from langchain_core.embeddings import Embeddings

# ----------------- Ollama Embeddings -------------------
class OllamaEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model_name = model_name

    def embed_documents(self, texts):
        return [ollama.embeddings(model=self.model_name, prompt=t)['embedding'] for t in texts]

    def embed_query(self, query):
        return ollama.embeddings(model=self.model_name, prompt=query)['embedding']
    
# ----------------- OpenAI Embeddings -------------------
class OpenAIEmbeddings(Embeddings):
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def embed_documents(self, texts):
        res = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [d.embedding for d in res.data]

    def embed_query(self, query):
        res = self.client.embeddings.create(
            model=self.model_name,
            input=query
        )
        return res.data[0].embedding