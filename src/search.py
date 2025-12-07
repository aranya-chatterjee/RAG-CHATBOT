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
        groq_api_key: str = None
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
            raise ValueError("GROQ_API_KEY not found. Please set it in .env file")
        
        # Initialize embedding pipeline (store it as instance variable)
        self.embedding_pipeline = EmbeddingPipeline(model_name=embedding_model)
        
        # Initialize vector store
        self.vectorstore = VectorStoreManager(
            persist_path=persist_dir,
            embed_model=embedding_model
        )
        
        # Load or build vector store
        self._initialize_vectorstore()
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=llm_model,
            temperature=0.1,
            max_tokens=1024
        )
        print(f"[INFO] RAGSearch initialized successfully")
    
    def _initialize_vectorstore(self):
        """Load existing or build new vector store"""
        try:
            # Try to load existing vector store
            if self.vectorstore.load():
                print("[INFO] Loaded existing vector store")
                return True
        except Exception as e:
            print(f"[INFO] Could not load existing vector store: {e}")
        
        # Build new vector store
        print("[INFO] Building new vector store...")
        
        if not os.path.exists(self.data_dir):
            print(f"[WARNING] Data directory not found: {self.data_dir}")
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"[INFO] Created empty data directory")
            return False
        
        try:
            # Load documents
            docs = load_documents(self.data_dir)
            if not docs:
                print("[WARNING] No documents found to build vector store")
                return False
            
            print(f"[INFO] Loaded {len(docs)} documents")
            
            # Build vector store - CORRECTED: Only pass documents
            # The embedding_pipeline should be accessed from self.vectorstore
            self.vectorstore.build_vector_store(docs)
            print("[INFO] Vector store built successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to build vector store: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> str:
        """Search method that your main.py calls"""
        print(f"[INFO] Searching for: {query}")
        
        try:
            # Search the vector store - check what method it has
            if hasattr(self.vectorstore, 'search'):
                # If vectorstore.search() exists
                results = self.vectorstore.search(query, top_k=top_k)
            elif hasattr(self.vectorstore, 'query'):
                # If vectorstore.query() exists
                results = self.vectorstore.query(query, top_k=top_k)
            else:
                # Try to find any search-like method
                for method_name in ['search', 'query', 'find_similar', 'retrieve']:
                    if hasattr(self.vectorstore, method_name):
                        method = getattr(self.vectorstore, method_name)
                        # Try calling with different parameter combinations
                        try:
                            results = method(query, top_k=top_k)
                            break
                        except:
                            try:
                                results = method(query, self.embedding_pipeline, top_k=top_k)
                                break
                            except:
                                continue
                else:
                    return "Error: Could not find a search method in vector store"
            
            if not results:
                return "I couldn't find any relevant information in the document to answer your question."
            
            # Extract text from results
            context_parts = []
            for result in results:
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
            if len(context) > 3000:
                context = context[:3000] + "..."
            
            # Create prompt
            prompt = f"""Based on the following document excerpts, answer the question:

{document_content}

Question: {query}

Instructions:
1. Answer using ONLY information from the document
2. If the document doesn't have the answer, say so
3. Be clear and concise

Answer:"""
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
            print(f"[INFO] Generated answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg


# Test function
if __name__ == "__main__":
    print("=" * 50)
    print("Testing RAGSearch")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
        except:
            pass
    
    if api_key:
        print("✅ API key found")
        
        # Create test directory
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create test document
        test_file = os.path.join(data_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("This is a test document about artificial intelligence.")
        
        try:
            # Test initialization
            rag = RAGSearch(
                persist_dir=os.path.join(temp_dir, "faiss_store"),
                data_dir=data_dir,
                groq_api_key=api_key
            )
            print("✅ RAGSearch initialized")
            
            # Test search method
            result = rag.search("What is this document about?")
            print(f"✅ Search test: {result}")
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    else:
        print("❌ GROQ_API_KEY not found")
        print("Please create .env file with: GROQ_API_KEY=your_key_here")
