import streamlit as st
import tempfile
import os
import sys
from pathlib import Path

# Add current directory to Python path
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
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None

def clear_session():
    """Clear the session state."""
    st.session_state.rag_search = None
    st.session_state.chats = []
    st.session_state.processed_file = None
    st.session_state.processing_error = None
    st.session_state.vector_manager = None
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        import shutil
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except:
            pass
    st.session_state.temp_dir = None
    st.rerun()

def process_document(uploaded_file):
    try:
        st.info(f"📄 Processing file: {uploaded_file.name}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="rag_")
        st.session_state.temp_dir = temp_dir
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Save uploaded file
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        st.success(f"✅ File saved successfully")

        # 1. Load documents
        with st.spinner("📖 Loading document..."):
            docs = load_documents(data_dir)
        
        if not docs or len(docs) == 0:
            st.session_state.processing_error = "No valid text extracted from the document!"
            st.error("❌ No text found in the document. Try a different file type or content.")
            return None, None, temp_dir
        
        st.info(f"📊 Loaded {len(docs)} document(s)")
        
        # 2. Create vector store
        with st.spinner("🔧 Creating embeddings and vector store..."):
            vector_manager = VectorStoreManager(
                persist_path=os.path.join(temp_dir, "faiss_store"),
                embed_model="all-MiniLM-L6-v2",
                chunk_size=800,
                chunk_overlap=100
            )
            vector_manager.build_vector_store(docs)
            st.session_state.vector_manager = vector_manager

        st.success("✅ Vector store built successfully!")

        # 3. Get API key
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            st.session_state.processing_error = "GROQ_API_KEY not found! Set it in Streamlit secrets."
            st.error("❌ Missing API key. Set GROQ_API_KEY in secrets.")
            return None, None, temp_dir

        # 4. Initialize RAGSearch - IMPORTANT: Pass vector_store_manager not vector_store
        with st.spinner("⚙️ Initializing RAG system..."):
            rag_search = RAGSearch(
                persist_dir=os.path.join(temp_dir, "faiss_store"),
                data_dir=data_dir,
                embedding_model="all-MiniLM-L6-v2",
                llm_model="llama-3.1-8b-instant",  # Supported Groq model
                groq_api_key=api_key,
                vector_store_manager=vector_manager  # Pass the manager, not vector_store
            )

        st.success("✅ RAG system initialized!")
        
        # 5. Test the vector store
        with st.spinner("🧪 Testing vector store..."):
            if docs and hasattr(docs[0], 'page_content'):
                # Extract first meaningful word for testing
                sample_text = docs[0].page_content
                words = [w for w in sample_text.split() if len(w) > 4]
                
                if words:
                    test_word = words[0]
                    test_results = vector_manager.query(test_word, top_k=1)
                    
                    if test_results:
                        st.success(f"✅ Vector store test passed! Found matches for '{test_word}'")
                    else:
                        st.warning(f"⚠️ Vector store built but no matches for '{test_word}'")
        
        return rag_search, uploaded_file.name, temp_dir

    except Exception as e:
        st.session_state.processing_error = str(e)
        st.error(f"❌ Error in processing: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language='python')
        return None, None, None

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Chatbot")
    st.markdown("---")
    
    # API Key status
    api_key = st.secrets.get("GROQ_API_KEY")
    if api_key:
        st.success(f"✅ API Key: Found ({api_key[:10]}...)")
    else:
        st.error("❌ API Key: Missing")
        st.info("Add to `.streamlit/secrets.toml`:")
        st.code('GROQ_API_KEY = "your-key-here"')
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload Document",
        type=['pdf', 'txt', 'csv', 'docx', 'xlsx', 'json'],
        help="Supported formats: PDF, TXT, CSV, DOCX, XLSX, JSON"
    )
    
    if uploaded_file:
        st.info(f"📄 Selected: {uploaded_file.name}")
        
        if st.button("🚀 Process Document", type="primary", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                rag_search, filename, temp_dir = process_document(uploaded_file)
                
                if rag_search:
                    st.session_state.rag_search = rag_search
                    st.session_state.processed_file = filename
                    st.session_state.chats = []
                    st.session_state.processing_error = None
                    st.success(f"✅ '{filename}' ready for chatting!")
                    st.balloons()
                else:
                    st.error(f"❌ Processing failed: {st.session_state.processing_error}")
    
    st.markdown("---")
    
    # Current document info
    if st.session_state.processed_file:
        st.subheader("📊 Current Document")
        st.info(f"📄 {st.session_state.processed_file}")
        
        # Show debug info in expander
        with st.expander("🔧 System Info", expanded=False):
            if st.session_state.vector_manager:
                st.write(f"Vector store: {'✅ Built' if st.session_state.vector_manager.vector_store else '❌ Not built'}")
            st.write(f"Chat history: {len(st.session_state.chats)} messages")
        
        if st.button("🗑️ Clear Session", type="secondary", use_container_width=True):
            clear_session()
    
    st.markdown("---")
    st.caption("💡 Upload a document and ask questions about its content!")

# Main content area
st.title("💬 RAG Chatbot")
st.caption("Ask questions about your uploaded documents using AI")

# Chat interface
if st.session_state.rag_search and st.session_state.vector_manager:
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"📚 Chatting about: {st.session_state.processed_file}")
    with col2:
        if st.button("🔄 New Chat", type="secondary"):
            st.session_state.chats = []
            st.rerun()
    
    # System status
    with st.expander("⚙️ System Status", expanded=False):
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.write("✅ RAG Search initialized")
            st.write("✅ Vector store ready")
        with status_col2:
            if st.session_state.vector_manager and st.session_state.vector_manager.vector_store:
                index_size = st.session_state.vector_manager.vector_store.index.ntotal
                st.write(f"📊 Index size: {index_size} chunks")
    
    # Chat history
    for chat in st.session_state.chats:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant", avatar="🤖"):
            st.write(chat["assistant"])
    
    # Chat input
    user_input = st.chat_input(f"Ask about {st.session_state.processed_file}...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chats.append({"user": user_input, "assistant": ""})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get response from RAG system
                    answer = st.session_state.rag_search.search(user_input, top_k=3)
                    
                    # Display response
                    st.write(answer)
                    
                    # Update chat history
                    st.session_state.chats[-1]["assistant"] = answer
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chats[-1]["assistant"] = error_msg
                    
                    # Show detailed error in expander
                    with st.expander("🔍 Error Details", expanded=False):
                        import traceback
                        st.code(traceback.format_exc(), language='python')

else:
    # Welcome/Instructions
    st.markdown("""
    ## 👋 Welcome to RAG Chatbot!
    
    **Retrieval-Augmented Generation (RAG)** allows you to chat with your documents using AI.
    
    ### 🚀 **Getting Started:**
    1. **Get a Groq API key** from [console.groq.com](https://console.groq.com) (free tier available)
    2. **Add it to Streamlit secrets** (`.streamlit/secrets.toml`):
    ```toml
    GROQ_API_KEY = "your-key-here"
    ```
    3. **Upload a document** using the sidebar
    4. **Ask questions** about the document content
    
    ### 📁 **Supported Formats:**
    - PDF documents
    - Text files (.txt)
    - Word documents (.docx)
    - Excel files (.xlsx)
    - CSV files
    - JSON files
    
    ### 💡 **Tips for Best Results:**
    - Start with a **simple text file** to test
    - Ask questions **based on the document content**
    - For large documents, processing may take a minute
    """)
    
    # Show previous error if any
    if st.session_state.processing_error:
        st.error(f"⚠️ Last error: {st.session_state.processing_error}")
        if st.button("🔄 Clear Error"):
            st.session_state.processing_error = None
            st.rerun()
    
    # Quick start example
    with st.expander("🎯 Example Usage", expanded=True):
        st.markdown("""
        **Example document content:**
        ```
        Artificial intelligence enables machines to perform human-like tasks.
        Machine learning allows systems to learn from data without explicit programming.
        ```
        
        **Good questions to ask:**
        - What is artificial intelligence?
        - How does machine learning work?
        - What are the differences between AI and ML?
        """)

# Footer
st.markdown("---")
footer_col1, footer_col2 = st.columns([3, 1])
with footer_col1:
    st.caption("🤖 Powered by Groq AI | 🛠️ Built with Streamlit & LangChain")
with footer_col2:
    st.caption(f"Version: {len(st.session_state.chats)} messages")
