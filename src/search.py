import os
from dotenv import load_dotenv
from vectorStore import VectorStoreManager
from ChunkAndEmbed import EmbeddingPipeline
from data_loader import load_documents
# from langchain_groq import ChatGroq

# Load environment variables once
load_dotenv()

class RAGSearch:
    """
    Retrieval-Augmented Generation (RAG) search leveraging a GROQ LLM.
    """
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
        data_dir: str = "data",
        groq_api_key: str = None
    ):
        print(f"[RAGSearch] Initializing with data_dir: {data_dir}")

        # Get API key
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")
        self.api_key = api_key

        self.data_dir = data_dir
        self.llm_model = llm_model
        print(f"[RAGSearch] Initialization complete")

    def search(self, query: str, top_k: int = 3) -> str:
        
        try:
            
        # 1. Vector retrieval (search similar chunks)
            if hasattr(self, 'vector_store') and self.vector_store is not None:
                relevant_docs = self.vector_store.query(query, top_k=top_k)
                if relevant_docs:
                    docs_context = "\n\n".join([doc["page_content"] for doc in relevant_docs if "page_content" in doc])
                    prompt = f"""You are a helpful assistant. Use the following document excerpts to answer the user's question:
    {docs_context}

    User question: {query}

    Answer as informatively as possible based only on this document context."""
                else:
                    prompt = f"User question: {query}\n(No relevant document content found.)"
            else:
                prompt = f"User question: {query}\n(No vector store available.)"

        # 2. Pass to LLM
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.llm_model,
                temperature=0.1,
                max_tokens=512
            )
            response = llm.invoke(prompt)
            return getattr(response, "content", str(response)).strip()
        except Exception as e:
            return f"[ERROR] {e}"
# Test function
if __name__ == "__main__":
    print("Testing RAGSearch...")

    # Load environment variable, if not already loaded
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        print(f"API key found: {api_key[:10]}...")

        rag = RAGSearch(
            data_dir="test_data",
            groq_api_key=api_key,
            llm_model="gemma2-9b-it"
        )

        # Test search
        result = rag.search("What is artificial intelligence?")
        print(f"Test result: {result}")
    else:
        print("ERROR: No GROQ_API_KEY found in .env file")






