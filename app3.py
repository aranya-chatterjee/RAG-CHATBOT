import streamlit as st
import tempfile
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import load_documents
from ChunkAndEmbed import EmbeddingPipeline
from vectorStore import VectorStoreManager
from search import RAGSearch

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "rag_search" not in st.session_state:
    st.session_state.rag_search = None
if "chats" not in st.session_state:
    st.session_state.chats = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "processing_error" not in st.session_state:
    st.session_state.processing_error = None
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = None

def clear_session():
    st.session_state.rag_search = None
    st.session_state.chats = []
    st.session_state.processed_file = None
    st.session_state.processing_error = None
    st.session_state.vector_manager = None
    st.rerun()

def process_document(uploaded_file):
    try:
        st.info(f"Processing file: {uploaded_file.name}")
        temp_dir = tempfile.mkdtemp(prefix="rag_")
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        st.success(f"File saved to: {file_path}")

        docs = load_documents(data_dir)
        if not docs or len(docs) == 0:
            st.session_state.processing_error = "No valid text extracted from the document!"
            st.error("No text found in the document. Try a different file type or content.")
            return None, None, temp_dir

        embedding_pipeline = EmbeddingPipeline()
        vector_manager = VectorStoreManager(
            persist_path=os.path.join(temp_dir, "faiss_store"),
            embed_model="all-MiniLM-L6-v2"
        )
        st.info("Building vector store (embedding & indexing)...")
        vector_manager.build_vector_store(docs)
        st.session_state.vector_manager = vector_manager

        st.success("Vector store built!")

        # Get Groq API key from secrets (recommended on Streamlit Cloud)
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            st.session_state.processing_error = "GROQ_API_KEY not found! Set it in Streamlit secrets."
            st.error("Missing API key. Set GROQ_API_KEY in secrets.")
            return None, None, temp_dir

        rag_search = RAGSearch(
            persist_dir=os.path.join(temp_dir, "faiss_store"),
            data_dir=data_dir,
            embedding_model="all-MiniLM-L6-v2",
            llm_model="llama2-70b", # Use supported Groq model!
            groq_api_key=api_key,
            vector_store=vector_manager
        )
        st.success("RAGSearch object initialized with vector store.")

        # Test the vector store by retrieving something!
        test_vec = vector_manager.query("test", top_k=1)
        st.write("DEBUG: Test vector query result:", test_vec)
        if test_vec is None:
            st.error("Vector store query returned None - check index building.")
        elif len(test_vec) == 0:
            st.warning("Vector store built, but no chunks returned for a sample query.")

        return rag_search, uploaded_file.name, temp_dir

    except Exception as e:
        st.session_state.processing_error = str(e)
        st.error(f"❌ Error in processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Chatbot")
    st.markdown("---")
    api_key = st.secrets.get("GROQ_API_KEY")
    if api_key:
        st.success("✅ API Key: Found in secrets")
    else:
        st.error("❌ API Key: Missing")
        st.info("Add GROQ_API_KEY to your Streamlit secrets!")

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'txt', 'csv', 'docx', 'xlsx', 'json']
    )
    if uploaded_file:
        st.info(f"📄 Selected: {uploaded_file.name}")
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                rag_search, filename, temp_dir = process_document(uploaded_file)
                if rag_search:
                    st.session_state.rag_search = rag_search
                    st.session_state.processed_file = filename
                    st.session_state.chats = []
                    st.session_state.processing_error = None
                    st.success(f"✅ '{filename}' ready!")
                    st.balloons()
                else:
                    st.error(f"❌ Error: {st.session_state.processing_error}")
    st.markdown("---")
    if st.session_state.processed_file:
        st.subheader("Current Document")
        st.info(f"📊 {st.session_state.processed_file}")
        if st.button("🗑️ Clear"):
            clear_session()
    st.markdown("---")
    st.caption("💡 Upload a document and ask questions!")

# Main content
st.title("💬 RAG Chatbot")

if st.session_state.rag_search and st.session_state.vector_manager:
    st.subheader(f"Chatting about: {st.session_state.processed_file}")
    st.write("DEBUG: vector_manager.object", st.session_state.vector_manager)
    st.write("DEBUG: vector_manager.vector_store", getattr(st.session_state.vector_manager, "vector_store", None))

    # Chat history
    for chat in st.session_state.chats:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["assistant"])

    user_input = st.chat_input(f"Ask about {st.session_state.processed_file}...")
    if user_input:
        st.session_state.chats.append({"user": user_input, "assistant": ""})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.rag_search.search(user_input)
                    st.write(answer)
                    st.session_state.chats[-1]["assistant"] = answer
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chats[-1]["assistant"] = error_msg
                    import traceback
                    traceback.print_exc()
else:
    # Welcome/Instructions
    st.markdown("""
    ## Welcome to RAG Chatbot! 🤖

    **Steps:**
    1. Upload a document (PDF, TXT, CSV, DOCX, XLSX, JSON)
    2. Click "Process Document"
    3. Ask questions about the document

    **Setup required:**
    1. Get your free API key from [Groq Console](https://console.groq.com)
    2. Add `GROQ_API_KEY = "your_key_here"` to Streamlit secrets!
    """)
    if st.session_state.processing_error:
        st.error(f"Last error: {st.session_state.processing_error}")

st.markdown("---")
st.caption("🤖 Powered by Groq AI | Built with Streamlit")
