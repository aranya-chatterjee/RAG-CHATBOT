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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
        background-color: #f8f9fa;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin: 10px 0;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0D47A1;
    }
</style>
""", unsafe_allow_html=True)

# Function to check API key from multiple sources
def check_api_key():
    """Check if API key is available from any source"""
    try:
        # Priority 1: Streamlit Secrets (for Streamlit Cloud)
        if hasattr(st, 'secrets'):
            if 'GROQ_API_KEY' in st.secrets:
                api_key = st.secrets['GROQ_API_KEY']
                if api_key and len(api_key.strip()) > 10:
                    st.session_state.api_key_source = "Streamlit Secrets"
                    return True, api_key
        
        # Priority 2: Environment variable
        api_key = os.getenv("GROQ_API_KEY")
        if api_key and len(api_key.strip()) > 10:
            st.session_state.api_key_source = "Environment Variable"
            return True, api_key
        
        # Priority 3: .env file
        try:
            from dotenv import load_dotenv
            # Try multiple .env locations
            env_paths = [
                '.env',
                '../.env',
                os.path.join(os.path.dirname(__file__), '.env'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
            ]
            
            for env_path in env_paths:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    api_key = os.getenv("GROQ_API_KEY")
                    if api_key and len(api_key.strip()) > 10:
                        st.session_state.api_key_source = f".env file ({env_path})"
                        return True, api_key
        except ImportError:
            pass  # dotenv not installed
        except Exception as e:
            print(f"Error loading .env: {e}")
        
        return False, None
        
    except Exception as e:
        print(f"Error checking API key: {e}")
        return False, None

# Initialize session state
def init_session_state():
    if "rag_search" not in st.session_state:
        st.session_state.rag_search = None
    if "chats" not in st.session_state:
        st.session_state.chats = []
    if "processed_file" not in st.session_state:
        st.session_state.processed_file = None
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "processing_error" not in st.session_state:
        st.session_state.processing_error = None
    if "file_preview" not in st.session_state:
        st.session_state.file_preview = None
    if "api_key_status" not in st.session_state:
        st.session_state.api_key_status = "unknown"
    if "api_key_source" not in st.session_state:
        st.session_state.api_key_source = None

init_session_state()

# Check API key on startup
api_key_available, api_key = check_api_key()
if api_key_available:
    st.session_state.api_key_status = "available"
else:
    st.session_state.api_key_status = "missing"

def cleanup_temp_files():
    """Clean up temporary files"""
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")

def clear_session():
    """Clear all session state"""
    cleanup_temp_files()
    
    # Reset session state
    st.session_state.rag_search = None
    st.session_state.chats = []
    st.session_state.processed_file = None
    st.session_state.processing_done = False
    st.session_state.temp_dir = None
    st.session_state.processing_error = None
    st.session_state.file_preview = None
    
    st.rerun()

def get_file_preview(file_bytes, filename):
    """Get preview of uploaded file"""
    try:
        # Try to decode as text for text-based files
        if filename.endswith('.txt') or filename.endswith('.csv'):
            try:
                content = file_bytes.decode('utf-8', errors='ignore')
                return content[:500] + ("..." if len(content) > 500 else "")
            except:
                pass
        
        # For other files, just show info
        file_size = len(file_bytes)
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024*1024:
            size_str = f"{file_size/1024:.1f} KB"
        else:
            size_str = f"{file_size/(1024*1024):.1f} MB"
        
        ext = os.path.splitext(filename)[1].lower()
        file_types = {
            '.pdf': 'PDF Document',
            '.txt': 'Text File',
            '.csv': 'CSV File',
            '.docx': 'Word Document',
            '.xlsx': 'Excel File',
            '.json': 'JSON File'
        }
        
        file_type = file_types.get(ext, 'Unknown File Type')
        return f"{file_type} - {size_str}"
        
    except Exception as e:
        return f"File preview not available: {str(e)}"

def process_document(uploaded_file):
    """Process the uploaded document"""
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="rag_chatbot_")
        st.session_state.temp_dir = temp_dir  # Store in session state
        
        # Save the uploaded file
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Get file bytes
        file_bytes = uploaded_file.getvalue()
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
        
        # Create data directory
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Move file to data directory
        new_file_path = os.path.join(data_dir, uploaded_file.name)
        os.rename(file_path, new_file_path)
        
        # Get file preview
        st.session_state.file_preview = get_file_preview(file_bytes, uploaded_file.name)
        
        # Import RAGSearch here to avoid circular imports
        from search import RAGSearch
        
        # Get API key to pass to RAGSearch
        api_key_available, api_key = check_api_key()
        if not api_key_available:
            raise ValueError("API key not found. Please check your configuration.")
        
        # Initialize RAG with the temporary directory
        rag_search = RAGSearch(
            persist_dir=os.path.join(temp_dir, "faiss_store"),
            data_dir=data_dir,
            embedding_model="all-MiniLM-L6-v2",
            llm_model="gemma2-9b-it",
            groq_api_key=api_key  # Pass the API key directly
        )
        
        return rag_search, uploaded_file.name
        
    except Exception as e:
        # Cleanup on error
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Chatbot")
    st.markdown("---")
    
    # API Key Status Display
    if st.session_state.api_key_status == "available":
        st.markdown(f"""
        <div class="success-box">
        ✅ **API Key Status:** Available
        <small>Source: {st.session_state.api_key_source}</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-box">
        ⚠️ **API Key Status:** Missing
        <small>Required for document processing</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Instructions for setting up API key
    with st.expander("🔑 How to set API key", expanded=False):
        st.markdown("""
        **For Local Development:**
        1. Create `.env` file in project root
        2. Add: `GROQ_API_KEY=your_key_here`
        
        **For Streamlit Cloud:**
        1. Go to app settings → Secrets
        2. Add:
        ```toml
        GROQ_API_KEY = "your_key_here"
        ```
        
        **Get Free API Key:**
        - Visit [Groq Console](https://console.groq.com)
        - Sign up for free account
        - Create API key
        """)
    
    st.markdown("---")
    
    # File Upload Section
    st.subheader("📁 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'csv', 'docx', 'xlsx', 'json'],
        label_visibility="collapsed",
        help="Supported: PDF, TXT, CSV, DOCX, XLSX, JSON"
    )
    
    if uploaded_file:
        # Show file info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ **{uploaded_file.name}**")
        with col2:
            file_size = len(uploaded_file.getvalue())
            st.caption(f"{file_size/1024:.1f} KB" if file_size < 1024*1024 else f"{file_size/(1024*1024):.1f} MB")
        
        # Check API key before allowing processing
        if st.session_state.api_key_status != "available":
            st.warning("⚠️ Please set up API key first (see instructions above)")
        else:
            # Process button
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Process the document
                        rag_search, filename = process_document(uploaded_file)
                        
                        # Update session state
                        st.session_state.rag_search = rag_search
                        st.session_state.processed_file = filename
                        st.session_state.chats = []
                        st.session_state.processing_done = True
                        st.session_state.processing_error = None
                        
                        st.success(f"✅ Document processed successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.session_state.processing_error = str(e)
                        st.error(f"❌ Error: {str(e)}")
                        # Cleanup temp files on error
                        cleanup_temp_files()
    
    # Show current document
    if st.session_state.processed_file:
        st.markdown("---")
        st.subheader("📊 Current Document")
        st.info(f"**File:** {st.session_state.processed_file}")
        
        # Show preview if available
        if st.session_state.file_preview:
            with st.expander("📋 Preview", expanded=False):
                st.text(st.session_state.file_preview)
        
        # Clear button
        if st.button("🗑️ Clear Document", use_container_width=True):
            clear_session()
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    
    # Settings
    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of document chunks to use for answering questions"
    )
    
    st.markdown("---")
    
    # Deployment info
    st.caption("""
    **Deployment:**
    - Local: Use `.env` file
    - Streamlit Cloud: Use app secrets
    """)

# Main content
st.markdown('<h1 class="main-header">RAG Chatbot</h1>', unsafe_allow_html=True)

# Check if we have a processed document
if st.session_state.rag_search and st.session_state.processing_done:
    # Chat interface
    st.subheader(f"💬 Chat about: {st.session_state.processed_file}")
    
    # Display chat history
    for chat in st.session_state.chats:
        if chat["user"]:
            with st.chat_message("user"):
                st.write(chat["user"])
        if chat["assistant"]:
            with st.chat_message("assistant"):
                st.write(chat["assistant"])
    
    # Chat input
    user_input = st.chat_input(f"Ask a question about {st.session_state.processed_file}...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chats.append({"user": user_input, "assistant": ""})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        # Generate response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                try:
                    # Try different method names
                    if hasattr(st.session_state.rag_search, 'search'):
                        answer = st.session_state.rag_search.search(user_input, top_k=top_k)
                    elif hasattr(st.session_state.rag_search, 'search_and_summarize'):
                        answer = st.session_state.rag_search.search_and_summarize(user_input, top_k=top_k)
                    elif hasattr(st.session_state.rag_search, 'query'):
                        answer = st.session_state.rag_search.query(user_input, top_k=top_k)
                    else:
                        answer = "Error: RAG system doesn't have a search method"
                        st.write(answer)
                        st.session_state.chats[-1]["assistant"] = answer
                except Exception as e:
                    error_msg = f"⚠️ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chats[-1]["assistant"] = error_msg
    
        
        # Generate response
        # with st.chat_message("assistant"):
        #     with st.spinner("Thinking..."):
        #         try:
        #             # Get answer from RAG system
        #             answer = st.session_state.rag_search.search(user_input, top_k=top_k)
        #             st.write(answer)
        #             st.session_state.chats[-1]["assistant"] = answer
        #         except Exception as e:
        #             error_msg = f"⚠️ Error generating response: {str(e)}"
        #             st.error(error_msg)
        #             st.session_state.chats[-1]["assistant"] = error_msg
    
    # Clear chat button
    if st.session_state.chats:
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.chats = []
                st.rerun()

else:
    # Welcome/Instructions screen
    if st.session_state.api_key_status == "available":
        st.markdown("""
        <div class="info-box">
        <h3>📚 Welcome to RAG Chatbot!</h3>
        <p>✅ API key detected. Upload a document to start chatting!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-box">
        <h3>⚠️ API Key Required</h3>
        <p>Please set up your GROQ_API_KEY to use the chatbot.</p>
        <p>See sidebar for instructions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show error if any
    if st.session_state.processing_error:
        st.markdown(f'<div class="error-box">❌ Error: {st.session_state.processing_error}</div>', unsafe_allow_html=True)
    
    # Features grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📄 **Supported Files**
        - PDF documents
        - Text files (.txt)
        - CSV files
        - Word documents (.docx)
        - Excel files (.xlsx)
        - JSON files
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 **How It Works**
        1. Upload any supported document
        2. System analyzes and indexes content
        3. Ask questions in natural language
        4. Get accurate, context-aware answers
        """)
    
    with col3:
        st.markdown("""
        ### 🚀 **Quick Start**
        1. Get API key from [Groq Console](https://console.groq.com)
        2. Set API key (sidebar instructions)
        3. Upload document in sidebar 👈
        4. Start chatting!
        """)
    
    # Example questions
    st.markdown("---")
    st.subheader("💡 Example Questions You Can Ask")
    
    examples = [
        "What is the main topic of this document?",
        "Summarize the key points",
        "Find information about [specific topic]",
        "What are the conclusions?",
        "Extract important dates or figures"
    ]
    
    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            if st.button(example, key=f"example_{i}"):
                # If we have a document, add to chat
                if st.session_state.rag_search:
                    st.session_state.chats.append({"user": example, "assistant": ""})
                    st.rerun()
                else:
                    st.info("Please upload and process a document first!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🤖 Powered by Groq AI & FAISS | Built with Streamlit</p>
    <p>Upload documents and chat with AI assistant</p>
</div>
""", unsafe_allow_html=True)
