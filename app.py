import streamlit as st

def main():
    st.set_page_config(page_title="Chat with multiple PDFs")
    st.header("Chat with multiple PDFs")
    st.text_input("Ask a question!")

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your PDFs here")
        st.button("Process")



if __name__ == '__main__':
    main()

