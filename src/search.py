import os 
# from dotenv import load_dotenv
from src.vectorStore import VectorStoreManager
from src.ChunkAndEmbed import EmbeddingPipeline
from src.data_loader import load_documents
from langchain_groq import ChatGroq
# from langchain_core.documents import Documents 

# load_dotenv()


class RAGSearch:
   def __init__(
    self,
    persist_dir: str = "faiss_store",
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_model: str = "gemma2-9b-it",
    data_dir: str = "data",
    groq_api_key: str = None  # ADD THIS LINE
):
    self.persist_dir = persist_dir
    self.embedding_model = embedding_model
    self.data_dir = data_dir

    # Get API key (priority: passed key > environment > .env > secrets)
    if groq_api_key and groq_api_key.strip():
        api_key = groq_api_key.strip()
        print("[INFO] Using provided API key")
    else:
        # Try to get from environment or .env
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("GROQ_API_KEY")
            except:
                pass
        
        if not api_key:
            # Try Streamlit secrets
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
                    api_key = st.secrets['GROQ_API_KEY']
            except:
                pass
    
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY. Please provide it.")

    # Initialize FAISS Vector Store
    self.vectorstore = VectorStoreManager(persist_dir, embedding_model)

    # Load or build FAISS index
    # self._initialize_vectorstore()

    # Initialize LLM
    self.llm = ChatGroq(
        groq_api_key=api_key,
        model_name=llm_model
    )
    print(f"[INFO] Groq LLM initialized: {llm_model}")
   

    

    

    # def _initialize_vectorstore(self):
    #     """Load FAISS index if present, else build from documents."""
    #     faiss_path = os.path.join(self.persist_dir, "faiss.index")
    #     metadata_path = os.path.join(self.persist_dir, "metadata.pkl")

    #     if os.path.exists(faiss_path) and os.path.exists(metadata_path):
    #         print("[INFO] Loading existing FAISS index...")
    #         self.vectorstore.load()
    #     else:
    #         print("[INFO] No FAISS index found. Building a new one...")
    #         docs = load_documents(self.data_dir)
    #         if not docs:
    #             raise ValueError("❌ No documents found to build vectorstore.")
    #         # self.vectorstore.build_from_documents(docs)
    #         print("[INFO] Vectorstore built successfully!")

   

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
       
        # Retrieve similar chunks
        results = self.vectorstore.query(query, top_k=top_k)

        if not results:
            return "No relevant documents found."

        # Extract text metadata
        context_chunks = [
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata")
        ]

        # Filter empty text
        context_chunks = [c for c in context_chunks if c]

        if not context_chunks:
            return "No relevant text found in retrieved documents."

        context = "\n\n".join(context_chunks)

        # Build LLM prompt
        prompt = f"""
You are a helpful assistant. Summarize the following information
for the user query: **{query}**

Context:
{context}

Provide a clean, concise answer:
"""

        # Call the LLM
        response = self.llm.invoke(prompt)

        return response.content.strip()




if __name__ == "__main__":
    rag = RAGSearch()

    user_query = "What is attention mechanism?"
    summary = rag.search_and_summarize(user_query, top_k=3)

    print("\n=== Summary ===")
    print(summary)







