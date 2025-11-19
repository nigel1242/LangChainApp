import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
import os
from htmlTemplates import css, bot_template, user_template

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "pdf_chat_collection"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def get_qdrant_client():
    """Create and return a Qdrant client instance"""
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

def load_vectorstore():
    """Load existing Qdrant collection using the client"""
    try:
        client = get_qdrant_client()
        # Check if collection exists
        collections = client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if collection_exists:
            # Use the Qdrant constructor directly with the client
            vectorstore = Qdrant(
                client=client,
                collection_name=COLLECTION_NAME,
                embeddings=embeddings
            )
            return vectorstore
        else:
            return None
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        return None

def create_or_append_vectorstore(new_chunks):
    """Create new collection or add to existing one"""
    client = get_qdrant_client()
    
    # Check if collection exists
    collections = client.get_collections()
    collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
    
    if collection_exists:
        # Load existing and add new texts
        vectorstore = Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings
        )
        vectorstore.add_texts(new_chunks)
    else:
        # Create new collection
        vectorstore = Qdrant.from_texts(
            texts=new_chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
    
    return vectorstore

def get_conversation_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your PDFs :books:")

    # Try to load existing vectorstore on startup
    if st.session_state.conversation is None:
        vectorstore = load_vectorstore()
        if vectorstore:
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("Connected to existing vector database!")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation:
            handle_userinput(user_question)
        else:
            st.warning("Please upload and process PDFs first.")

    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.warning("No text found in PDFs. Make sure they are readable PDFs.")
                    else:
                        vectorstore = create_or_append_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success(f"Processed {len(text_chunks)} text chunks and stored in Qdrant.")

if __name__ == "__main__":
    main()