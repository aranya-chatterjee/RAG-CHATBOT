import streamlit as st
import tempfile
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import load_documents
from vectorStore import VectorStoreManager
from ChunkAndEmbed import EmbeddingPipeline
from search import RAGSearch

st.set_page_config(page_title="RAG-Chatbot", layout="wide")
st.title("📚 RAG-Chatbot")

# Initialize session state
if "rag_search" not in st.session_state:
    st.session_state.rag_search = None
if "chats" not in st.session_state:
    st.session_state.chats = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# Sidebar for file upload
st.sidebar.header("📁 Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file", 
    type=['pdf', 'txt', 'csv', 'docx', 'xlsx', 'json']
)

# Function to process uploaded file
def process_uploaded_file(uploaded_file):
    """Process the uploaded file and create RAG pipeline."""
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Write file content
            if uploaded_file.type.startswith('text/'):
                content = uploaded_file.getvalue().decode('utf-8')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                content = uploaded_file.getvalue()
                with open(file_path, 'wb') as f:
                    f.write(content)
            
            # Create a data directory for the loader
            data_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # Move file to data directory
            new_file_path = os.path.join(data_dir, uploaded_file.name)
            os.rename(file_path, new_file_path)
            
            # Process with RAG
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Initialize RAG with the temporary data directory
                rag_search = RAGSearch(
                    persist_dir=os.path.join(temp_dir, "faiss_store"),
                    data_dir=data_dir,
                    embedding_model="all-MiniLM-L6-v2",
                    llm_model="gemma2-9b-it"  # or "llama3-70b-8192", "mixtral-8x7b-32768"
                )
            
            return rag_search, uploaded_file.name
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None

# Handle file upload
if uploaded_file and uploaded_file.name != st.session_state.get('processed_file'):
    rag_search, filename = process_uploaded_file(uploaded_file)
    
    if rag_search:
        st.session_state.rag_search = rag_search
        st.session_state.processed_file = filename
        st.session_state.chats = []  # Clear previous chats
        st.session_state.processing_done = True
        st.sidebar.success(f"✅ '{filename}' loaded and processed!")
    else:
        st.sidebar.error("Failed to process the file")

# Display current document info
if st.session_state.processed_file:
    st.sidebar.subheader("Current Document")
    st.sidebar.write(f"📄 **File:** {st.session_state.processed_file}")
    
    # Clear button
    if st.sidebar.button("🔄 Clear Document"):
        st.session_state.rag_search = None
        st.session_state.processed_file = None
        st.session_state.chats = []
        st.session_state.processing_done = False
        st.rerun()

# Main chat interface
if st.session_state.rag_search:
    st.header("💬 Chat with Masterji")
    
    # Display chat history
    for chat in st.session_state.chats:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["assistant"])
        st.divider()
    
    # Chat input
    user_input = st.chat_input("Ask a question about the document...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chats.append({"user": user_input, "assistant": ""})
        
        # Generate response
        with st.spinner("Masterji is thinking..."):
            try:
                response = st.session_state.rag_search.search(user_input, top_k=3)
                st.session_state.chats[-1]["assistant"] = response
            except Exception as e:
                st.session_state.chats[-1]["assistant"] = f"Error: {str(e)}"
        
        # Rerun to display the new response
        st.rerun()
    
    # Clear chat history button
    if st.session_state.chats:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chats = []
            st.rerun()

else:
    # Welcome message
    st.markdown("""
    ## 👋 Welcome to RAG-Chatbot!
    
    **How to use:**
    1. **Upload a document** using the file uploader in the sidebar 👈
    2. Wait for the document to be processed
    3. Start asking questions about your document!
    
    **Supported file types:**
    - 📄 PDF documents
    - 📝 Text files (.txt)
    - 📊 CSV files
    - 📑 Word documents (.docx)
    - 📈 Excel files (.xlsx)
    - 📋 JSON files
    
    **Features:**
    - Document analysis and question answering
    - Chat history preservation
    - Fast response using RAG (Retrieval Augmented Generation)
    """)
    
    # Placeholder for demo
    with st.expander("💡 Quick Tips"):
        st.markdown("""
        - For best results, upload documents with clear text content
        - The system works best with documents under 50 pages
        - Try questions like:
          * "What is the main topic of this document?"
          * "Summarize the key points"
          * "Find information about [specific topic]"
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
top_k = st.sidebar.slider("Number of chunks to retrieve", 1, 10, 3)

# Update RAG search settings if needed
if st.session_state.rag_search:
    # You can add settings adjustment here if needed
    pass

# Add requirements reminder
st.sidebar.markdown("""
---
**Requirements:**
Make sure you have:
1. `.env` file with `GROQ_API_KEY`
2. Required packages installed
""")