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
uploaded_file = st.sidebar.file_uploader("Choose any file", type=None)  # ACCEPT ALL FILES NOW

if uploaded_file:
    # Try to decode as UTF-8 if possible; else store bytes
    try:
        file_content = uploaded_file.read()
        try:
            st.session_state.document = file_content.decode("utf-8")
        except Exception:
            st.session_state.document = file_content  # Keep as bytes for non-text files
        st.session_state.chats.clear()
        st.sidebar.success(f"Document '{uploaded_file.name}' loaded!")
        st.session_state.filename = uploaded_file.name
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

if st.session_state.document is not None:
    st.subheader("Current Document")
    # Preview: try showing text, fallback to showing type and filename
    preview_text = (
        st.session_state.document[:2000]
        if isinstance(st.session_state.document, str)
        else str(st.session_state.document)[:200]  # preview raw bytes
    )
    st.text_area(
        "Document Preview",
        value=preview_text,
        height=200,
        disabled=True,
    )
    st.write(f"Filename: {st.session_state.get('filename', 'Unknown')}")

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
    st.info("Please upload a document to start chatting.")
