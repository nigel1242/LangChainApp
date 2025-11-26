# This code is used to read the indexed files from the rag_indices files. Fill in the chatNumber variable,
# Run the code using "python test.py" and it will generate a digested_chat_x.txt file which you can
# view the contents of processed pdfs and txt files.

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

chatNumber = "1"

# Dummy embeddings for loading only
class OllamaEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model_name = model_name

    def embed_documents(self, texts):
        return [[0.0]*768 for _ in texts]

    def embed_query(self, query):
        return [0.0]*768

# Load FAISS index
faiss_path = f"rag_indices/chat_{chatNumber}.faiss"
embeddings = OllamaEmbeddings("dummy")
index = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

output_file = f"digested_chat_{chatNumber}.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Number of documents in index: {len(index.docstore._dict)}\n")
    f.write("="*80 + "\n\n")
    
    for doc_id, doc in index.docstore._dict.items():
        f.write(f"Document ID: {doc_id}\n")
        f.write(f"Content:\n{doc.page_content}\n")
        f.write(f"Metadata: {doc.metadata}\n")
        f.write("-"*80 + "\n")

print(f"All digested text has been written to {output_file}")
