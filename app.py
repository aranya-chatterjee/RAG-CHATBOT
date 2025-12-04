import streamlit as st
from src.data_loader import load_documents
from src.vectorStore import VectorStoreManager
from src.ChunkAndEmbed import EmbeddingPipeline
from src.search import RAGSearch

st.set_page_config(page_title="RAG-Chatbot ", layout="wide")
st.title("RAG-Chatbot ")

if "document" not in st.session_state:
    st.session_state.document = None
    st.session_state.chats = []

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a TXT file", type="txt")

if uploaded_file:
    st.session_state.document = uploaded_file.read().decode("utf-8")
    st.session_state.chats.clear()
    st.sidebar.success("Document loaded!")

if st.session_state.document:
    st.subheader("Current Document")
    st.text_area(
        "Document Preview",
        value=st.session_state.document[:2000],
        height=200,
        disabled=True,
    )

    st.subheader("Ask Masterji About the Document")
    input_msg = st.text_input("Ask a question:", key="chat_input")

    if st.button("Ask Masterji"):
        question = input_msg.strip()
        if not question:
            st.warning("Please enter a question!")
        else:
            # Use your actual RAG/LLM function from src
            answer = get_answer(st.session_state.document, question)
            st.session_state.chats.append(
                {"user": question, "assistant": answer}
            )

    if st.session_state.chats:
        st.subheader("Chat History")
        for c in st.session_state.chats:
            st.markdown(f"**You:** {c['user']}")
            st.markdown(f"**Masterji:** {c['assistant']}\n---")
else:
    st.info("Please upload a TXT document to start chatting.")
