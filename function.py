import os, uuid, time
from types import SimpleNamespace
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Qdrant
from langchain.schema import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from htmlTemplates import css, bot_template, user_template

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PDF_COLLECTION = "pdf_chat_collection"
CHAT_COLLECTION = "chat_history_collection"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        for page in PdfReader(pdf).pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    return CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200).split_text(text)

def load_vectorstore():
    if PDF_COLLECTION in [c.name for c in client.get_collections().collections]:
        return Qdrant(client=client, collection_name=PDF_COLLECTION, embeddings=embeddings)
    return None

def create_or_append_vectorstore(chunks):
    if PDF_COLLECTION in [c.name for c in client.get_collections().collections]:
        vs = Qdrant(client=client, collection_name=PDF_COLLECTION, embeddings=embeddings)
        vs.add_texts(chunks)
    else:
        vs = Qdrant.from_texts(chunks, embeddings, collection_name=PDF_COLLECTION, url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return vs

# Chat Functions
def ensure_chat_collection():
    if CHAT_COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(CHAT_COLLECTION, vectors_config={"size": 1536, "distance": "Cosine"})

def upsert_message(session_id, role, content):
    ensure_chat_collection()
    client.upsert(CHAT_COLLECTION, points=[{
        "id": uuid.uuid4().hex,
        "vector": embeddings.embed_query(content),
        "payload": {"session_id": session_id, "role": role, "content": content, "timestamp": int(time.time()*1000)}
    }])

# Pull all conversations from Qdrant
def load_sessions():
    try:
        points = client.scroll(CHAT_COLLECTION, limit=10000)[0]
        sessions = {}
        for p in points:
            sid = p.payload["session_id"]
            sessions.setdefault(sid, []).append({
                "role": p.payload["role"],
                "content": p.payload["content"],
                "timestamp": p.payload.get("timestamp", 0)
            })
        for sid in sessions:
            sessions[sid].sort(key=lambda x: x["timestamp"])
        return sessions
    except:
        return {}

# Title for chat history
def title_from_messages(messages, max_len=60):
    for m in messages:
        if m["role"] == "user" and m["content"]:
            t = m["content"].replace("\n", " ").strip()
            return t[:max_len] + "..." if len(t) > max_len else t
    return "Untitled Chat"

def get_conversation_chain(vectorstore):
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(),
        retriever=vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )

# Load saved conversation
def load_session(session_id, sessions, state):
    msgs = sessions.get(session_id, [])
    state.chat_history = [SimpleNamespace(role=m["role"], content=m["content"]) for m in msgs]
    vs = load_vectorstore()
    if vs:
        chain = get_conversation_chain(vs)
        chain.memory.chat_memory.messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in msgs
        ]
        state.conversation = chain

def handle_input(user_question, state):
    if not state.current_session:
        state.current_session = uuid.uuid4().hex
    
    sid = state.current_session
    upsert_message(sid, "user", user_question)
    
    if state.conversation is None:
        vs = load_vectorstore()
        if vs:
            state.conversation = get_conversation_chain(vs)
    
    res = state.conversation({"question": user_question})
    bot_text = str(res.get("answer", ""))
    
    upsert_message(sid, "assistant", bot_text)
    state.chat_history.append(SimpleNamespace(role="user", content=user_question))
    state.chat_history.append(SimpleNamespace(role="assistant", content=bot_text))

# Delete chat history
def clear_session(session_id):
    try:
        points = client.scroll(CHAT_COLLECTION, limit=10000)[0]
        remove_ids = [p.id for p in points if p.payload.get("session_id") == session_id]
        if remove_ids:
            client.delete(CHAT_COLLECTION, points_selector=remove_ids)
            return True
    except:
        pass
    return False