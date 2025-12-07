# src/search.py
import os 
from dotenv import load_dotenv
from vectorStore import VectorStoreManager
from ChunkAndEmbed import EmbeddingPipeline
from data_loader import load_documents
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemma2-9b-it",
        data_dir: str = "data",
        groq_api_key: str = None  # Your main.py is passing this parameter
    ):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.data_dir = data_dir
        
        # Get API key
        if groq_api_key:
            api_key = groq_api_key
        elif os.getenv("GROQ_API_KEY"):
            api_key = os.getenv("GROQ_API_KEY")
        else:
            raise ValueError("GROQ_API_KEY not found")
        
        # Initialize vector store
        self.vectorstore = VectorStoreManager(persist_dir, embedding_model)
        
        # Load or build vector store
        self._initialize_vectorstore()
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=llm_model
        )
        print(f"[INFO] Groq LLM initialized: {llm_model}")
    
    def _initialize_vectorstore(self):
        """Load existing or build new vector store"""
        # Try to load existing
        if os.path.exists(os.path.join(self.persist_dir, "faiss_index.idx")):
            print("[INFO] Loading existing FAISS index...")
            self.vectorstore.load()
        else:
            print("[INFO] Building new vector store...")
            if os.path.exists(self.data_dir):
                docs = load_documents(self.data_dir)
                if docs:
                    # Get embedding pipeline
                    embedding_pipeline = EmbeddingPipeline(model_name=self.embedding_model)
                    # Build vector store
                    self.vectorstore.build_vector_store(docs, embedding_pipeline)
                else:
                    print("[WARNING] No documents found")
    
    # THIS IS THE CRITICAL METHOD - main.py calls search()
    def search(self, query: str, top_k: int = 5) -> str:
        """Search method that your main.py is calling"""
        print(f"[INFO] Searching for: {query}")
        
        try:
            # First, we need to search the vector store
            # Check what method vectorStore has
            embedding_pipeline = EmbeddingPipeline(model_name=self.embedding_model)
            
            # Try different possible method names in vectorStore
            if hasattr(self.vectorstore, 'search'):
                results = self.vectorstore.search(query, embedding_pipeline, top_k=top_k)
            elif hasattr(self.vectorstore, 'query'):
                results = self.vectorstore.query(query, embedding_pipeline, top_k=top_k)
            else:
                # Last resort: try to call whatever method exists
                for method_name in ['search', 'query', 'find']:
                    if hasattr(self.vectorstore, method_name):
                        method = getattr(self.vectorstore, method_name)
                        results = method(query, embedding_pipeline, top_k=top_k)
                        break
                else:
                    return "Error: Vector store has no search method"
            
            if not results:
                return "No relevant information found in the document."
            
            # Extract text from results
            context_parts = []
            for result in results:
                # Handle different result formats
                if hasattr(result, 'page_content'):
                    context_parts.append(result.page_content)
                elif isinstance(result, dict):
                    if 'page_content' in result:
                        context_parts.append(result['page_content'])
                    elif 'text' in result:
                        context_parts.append(result['text'])
                    elif 'content' in result:
                        context_parts.append(result['content'])
                else:
                    context_parts.append(str(result))
            
            # Combine context
            context = "\n\n".join(context_parts)
            
            # Create prompt for LLM
            prompt = f"""Based on this document content:

{context}

Question: {query}

Answer the question using only information from the document. If the document doesn't contain the answer, say so.

Answer:"""
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    # Your existing code might have this method - keep it for compatibility
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Alias for search() for compatibility"""
        return self.search(query, top_k)


# Simple test
if __name__ == "__main__":
    print("Testing RAGSearch compatibility...")
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Try to load from .env
        try:
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
        except:
            pass
    
    if api_key:
        print("✅ API key found")
        try:
            # Test initialization
            rag = RAGSearch(
                persist_dir="test_faiss",
                data_dir="test_data",
                groq_api_key=api_key
            )
            print("✅ RAGSearch initialized")
            
            # Test search method
            result = rag.search("Test query")
            print(f"✅ Search method works: {result[:50]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ No API key found. Please set GROQ_API_KEY")
