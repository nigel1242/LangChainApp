import streamlit as st
import function

st.set_page_config(page_title="My Learning AI", page_icon=":books:")
st.write(function.css, unsafe_allow_html=True)

st.session_state.setdefault("conversation", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("current_session", None)

st.header("Welcome to My Learning AI")

# Load all sessions
sessions = function.load_sessions()
titles = {sid: function.title_from_messages(msgs) for sid, msgs in sessions.items()}
titles_list = ["New Chat"] + [titles[sid] for sid in sorted(titles.keys(), reverse=True)]

with st.sidebar:
    st.subheader("Chats")
    select = st.selectbox("Select chat", titles_list, index=0)
    
    if select == "New Chat":
        if st.session_state.current_session:
            st.session_state.current_session = None
            st.session_state.chat_history = []
            st.session_state.conversation = None
    else:
        sid = {v: k for k, v in titles.items()}[select]
        if st.session_state.current_session != sid:
            st.session_state.current_session = sid
            function.load_session(sid, sessions, st.session_state)
    
    st.subheader("Upload PDFs")
    pdfs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        if pdfs:
            with st.spinner("Processing..."):
                text = function.get_pdf_text(pdfs)
                chunks = function.get_text_chunks(text)
                if chunks:
                    vs = function.create_or_append_vectorstore(chunks)
                    st.session_state.conversation = function.get_conversation_chain(vs)
                    st.success(f"Processed {len(chunks)} chunks")
                else:
                    st.warning("No text found")
        else:
            st.warning("Upload PDFs first")
    
    if st.button("Clear Current Chat"):
        if st.session_state.current_session:
            if function.clear_session(st.session_state.current_session):
                st.session_state.current_session = None
                st.session_state.chat_history = []
                st.session_state.conversation = None
                st.success("Chat cleared!")
                st.experimental_rerun()
        else:
            st.warning("No chat selected")

q = st.text_input("Ask a question:")
if q:
    if not st.session_state.conversation:
        vs = function.load_vectorstore()
        if vs:
            st.session_state.conversation = function.get_conversation_chain(vs)
        else:
            st.warning("Upload PDFs first")
            st.stop()
    
    function.handle_input(q, st.session_state)

# Display chat by newest
if st.session_state.chat_history:
    st.markdown("---")
    for m in reversed(st.session_state.chat_history):
        tpl = function.user_template if m.role == "user" else function.bot_template
        st.write(tpl.replace("{{MSG}}", m.content), unsafe_allow_html=True)