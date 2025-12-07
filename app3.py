import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
import shutil

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

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

def clear_session():
    """Clear all session state"""
    st.session_state.rag_search = None
    st.session_state.chats = []
    st.session_state.processed_file = None
    st.session_state.processing_error = None
    st.rerun()

def process_document_simple(uploaded_file):
    """Simple document processing that always works"""
    try:
        print(f"Processing file: {uploaded_file.name}")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="rag_")
        
        # Create data directory
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        print(f"File saved to: {file_path}")
        
        # Import and initialize RAGSearch
        from search import RAGSearch
        
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            # Try loading from .env
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("GROQ_API_KEY")
            except:
                pass
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        print(f"API key found: {api_key[:10]}...")
        
        # Initialize RAGSearch
        rag_search = RAGSearch(
            persist_dir=os.path.join(temp_dir, "faiss_store"),
            data_dir=data_dir,
            embedding_model="all-MiniLM-L6-v2",
            llm_model="llama-3.1-8b-instant",
            groq_api_key=api_key
        )
        
        print("RAGSearch initialized successfully")
        
        # Test it works
        test_result = rag_search.search("test")
        print(f"Test search result: {test_result[:50]}...")
        
        return rag_search, uploaded_file.name, temp_dir
        
    except Exception as e:
        print(f"Error in process_document_simple: {e}")
        import traceback
        traceback.print_exc()
        raise e

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Chatbot")
    st.markdown("---")
    
    # Check API key status
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
        except:
            pass
    
    if api_key:
        st.success("✅ API Key: Found")
    else:
        st.error("❌ API Key: Missing")
        st.info("Create `.env` file with: GROQ_API_KEY=your_key")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'txt', 'csv', 'docx', 'xlsx', 'json']
    )
    
    if uploaded_file:
        st.info(f"📄 Selected: {uploaded_file.name}")
        
        if st.button("🚀 Process Document", type="primary"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    rag_search, filename, temp_dir = process_document_simple(uploaded_file)
                    
                    # Store in session state
                    st.session_state.rag_search = rag_search
                    st.session_state.processed_file = filename
                    st.session_state.chats = []
                    st.session_state.processing_error = None
                    
                    st.success(f"✅ '{filename}' ready!")
                    st.balloons()
                    
                except Exception as e:
                    st.session_state.processing_error = str(e)
                    st.error(f"❌ Error: {str(e)}")
    
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

if st.session_state.rag_search:
    # Chat interface
    st.subheader(f"Chatting about: {st.session_state.processed_file}")
    
    # Show chat history
    for chat in st.session_state.chats:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["assistant"])
    
    # Chat input
    user_input = st.chat_input(f"Ask about {st.session_state.processed_file}...")
    
    if user_input:
        # Add user message
        st.session_state.chats.append({"user": user_input, "assistant": ""})
        
        # Show user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # DEBUG: Check rag_search
                    print(f"DEBUG: rag_search = {st.session_state.rag_search}")
                    print(f"DEBUG: Has search method? {hasattr(st.session_state.rag_search, 'search')}")
                    
                    # Get answer
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
    # Welcome screen
    st.markdown("""
    ## Welcome to RAG Chatbot! 🤖
    
    **How to use:**
    1. Upload a document (PDF, TXT, CSV, DOCX, XLSX, JSON)
    2. Click "Process Document"
    3. Ask questions about the document
    
    **Setup required:**
    1. Get free API key from [Groq Console](https://console.groq.com)
    2. Create `.env` file with: `GROQ_API_KEY=your_key_here`
    """)
    
    # Show error if any
    if st.session_state.processing_error:
        st.error(f"Last error: {st.session_state.processing_error}")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Groq AI | Built with Streamlit")
