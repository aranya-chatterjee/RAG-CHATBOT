import streamlit as st
from src.data_loader import load_documents
from src.ChunkAndEmbed import EmbeddingPipeline
from src.vectorStore import VectorStoreManager
from src.search import RAGSearch
import os

# ===== Utility for API key via Streamlit secrets ===== #
def get_groq_api_key():
    # Streamlit secrets (Streamlit Cloud: settings -> secrets -> TOML)
    # should include: GROQ_API_KEY = "your-key-here"
    return st.secrets.get("GROQ_API_KEY")

# ===== Session state initialization ===== #
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "embedding_pipeline" not in st.session_state:
    st.session_state.embedding_pipeline = None
if "rag_search" not in st.session_state:
    st.session_state.rag_search = None
if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

# ===== App Layout ===== #
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🦙 RAG Chatbot: Search Your Documents + LLM Fallback")

st.info("Upload documents, ask questions, get context-aware answers using Groq's LLM. "
        "API key is securely loaded from Streamlit secrets.")

# ===== File Upload ===== #
uploaded_files = st.file_uploader(
    "Upload .pdf, .txt, .csv, .docx, .xlsx, .json documents here", 
    type=["pdf", "txt", "csv", "docx", "xlsx", "json"],
    accept_multiple_files=True
)

persist_dir = "faiss_store"
data_dir = "uploaded_data"
os.makedirs(data_dir, exist_ok=True)

# ===== Saving Uploaded Files ===== #
if uploaded_files:
    st.session_state.doc_uploaded = True
    # Save every file locally
    saved_paths = []
    for file in uploaded_files:
        save_path = os.path.join(data_dir, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(save_path)
    st.success(f"Saved {len(saved_paths)} document(s) for processing.")

# ===== Build Vector Store ===== #
if st.session_state.doc_uploaded:
    if st.button("Process & Embed Documents"):
        st.info("Loading documents...")
        docs = load_documents(data_dir)
        if not docs:
            st.error("No valid documents found. Please upload at least one supported file.")
        else:
            st.success(f"{len(docs)} documents loaded!")
            # Create RAGSearch (LLM fallback)
            api_key = get_groq_api_key()
            if api_key:
                st.session_state.rag_search = RAGSearch(
                    persist_dir=persist_dir,
                    embedding_model="all-MiniLM-L6-v2",
                    llm_model="gemma2-9b-it",
                    data_dir=data_dir,
                    groq_api_key=api_key
                )
            else:
                st.error("GROQ API Key missing from Streamlit secrets! Please add it.")

# ===== Query Interface ===== #
if st.session_state.vector_manager and st.session_state.rag_search:
    st.subheader("Search Your Documents")

    query_text = st.text_area("Ask a question about your documents (or general questions)")
    top_k = st.slider("Number of document results (top_k)", 1, 10, 3)

    if st.button("Search"):
        # First try vector search
        results = st.session_state.vector_manager.query(query_text, top_k=top_k)
        if results:
            st.write(f"Top {len(results)} context results:")
            for r in results:
                content = r.get("page_content", str(r))
                meta = r.get("metadata", {})
                st.markdown(f"- **Content:** `{content[:250]}...`")
                if meta:
                    st.json(meta)
            # Optionally: Run LLM on the found context—advanced (can prepend to prompt)
        else:
            st.warning("No results found in vector store. Using LLM fallback.")

        # Always provide the LLM fallback (regardless of vector search success)
        llm_response = st.session_state.rag_search.search(query_text)
        st.markdown("### LLM (Groq) Answer:")
        st.write(llm_response)

st.markdown("""
---
**How secrets are used?**  
> Create a Streamlit secrets with:  
> ```
> GROQ_API_KEY = "your-groq-api-key-here"
> ```
> in `secrets.toml` via Streamlit cloud settings.
""")
