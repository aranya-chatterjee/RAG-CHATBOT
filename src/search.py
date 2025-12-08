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
        groq_api_key: str = None,
        vector_store=None  
    ):
        print(f"[RAGSearch] Initializing with data_dir: {data_dir}")

        # Get API key
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")
        self.api_key = api_key

        self.data_dir = data_dir
        self.llm_model = llm_model
        self.vector_store = vector_store
        print(f"[RAGSearch] Initialization complete")

    def search(self, query: str, top_k: int = 3) -> str:
        try:
            if self.vector_store is not None:
            # DEBUG: Check vector store
                print(f"[DEBUG] Vector store type: {type(self.vector_store)}")
            
            # Try to query
                results = self.vector_store.query(query, top_k=top_k)
                print(f"[DEBUG] Query returned {len(results)} results")
            
                if results:
                # Extract content - handle different result types
                    context_parts = []
                    for r in results:
                        if hasattr(r, 'page_content'):
                            context_parts.append(r.page_content)
                        elif isinstance(r, dict):
                            context_parts.append(r.get('page_content', str(r)))
                        else:
                            context_parts.append(str(r))
                
                    docs_context = "\n---\n".join(context_parts[:3])  # Limit to 3
                
                    prompt = f"""Use this information to answer the question:

    {docs_context}

    Question: {query}

    Answer:"""
            else:
                prompt = f"""I couldn't find information about {query} in the document.

Please try asking about something in the document or rephrase your question."""
        
        

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









