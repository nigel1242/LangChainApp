# modules/chat_chain.py
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def make_chat_chain(llm, vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
