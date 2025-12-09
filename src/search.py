import os
from dotenv import load_dotenv
from vectorStore import VectorStoreManager
from ChunkAndEmbed import EmbeddingPipeline
from data_loader import load_documents
from langchain_groq import ChatGroq

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
        vector_store_manager=None  # Changed from vector_store to vector_store_manager
    ):
        print(f"[RAGSearch] Initializing...")

        # Get API key
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment or parameters")
        self.api_key = api_key

        self.llm_model = llm_model
        self.vector_store_manager = vector_store_manager  # Store the manager
        
        print(f"[RAGSearch] Initialization complete")
        if vector_store_manager:
            print(f"[RAGSearch] Vector store manager type: {type(vector_store_manager)}")

    def search(self, query: str, top_k: int = 3) -> str:
        try:
            print(f"[RAGSearch] Searching for: '{query}'")
            
            if self.vector_store_manager is not None:
                print(f"[DEBUG] Vector store manager available")
                
                # Query using the manager's query method
                results = self.vector_store_manager.query(query, top_k=top_k)
                print(f"[DEBUG] Query returned {len(results)} results")
                
                if results and len(results) > 0:
                    # Extract content from results
                    context_parts = []
                    for r in results:
                        if hasattr(r, 'page_content'):
                            context_parts.append(r.page_content)
                        elif isinstance(r, dict) and 'page_content' in r:
                            context_parts.append(r['page_content'])
                        else:
                            context_parts.append(str(r)[:500])
                    
                    # Join contexts
                    docs_context = "\n\n---\n\n".join(context_parts[:top_k])
                    
                    # Create prompt
                    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context. 
If the answer cannot be found in the context, say "I don't have enough information to answer that question."

CONTEXT:
{docs_context}

QUESTION: {query}

ANSWER (based only on context):"""
                    
                    print(f"[DEBUG] Context length: {len(docs_context)} characters")
                    
                else:
                    # No results found
                    prompt = f"""I searched the document but couldn't find specific information about: {query}

Please try asking about something else in the document or rephrase your question."""
                    
            else:
                # No vector store available
                prompt = f"""No document has been loaded yet. Please upload a document first.

Your question was: {query}"""

            # Call LLM
            print(f"[RAGSearch] Calling LLM...")
            llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.llm_model,
                temperature=0.1,
                max_tokens=512
            )
            
            response = llm.invoke(prompt)
            
            # Extract content safely
            if hasattr(response, 'content'):
                answer = response.content
            elif hasattr(response, 'text'):
                answer = response.text
            else:
                answer = str(response)
            
            return answer.strip()
            
        except Exception as e:
            print(f"[RAGSearch ERROR] {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"


# def main():
#     """Example usage for testing."""
#     print("=== RAG Search Test ===")
    
#     # Check API key
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         print("ERROR: GROQ_API_KEY not found in environment variables")
#         print("Add it to your .env file or set it as an environment variable")
#         return
    
#     print(f"API key found: {api_key[:10]}...")
    
#     # Check if data directory exists
#     data_dir = "data"
#     if not os.path.exists(data_dir):
#         print(f"Creating sample data directory: {data_dir}")
#         os.makedirs(data_dir, exist_ok=True)
        
#         # Create a sample document
#         sample_content = """Artificial intelligence (AI) is the simulation of human intelligence in machines.
# Machine learning is a subset of AI that enables systems to learn from data.
# Deep learning uses neural networks with multiple layers for complex pattern recognition."""
        
#         with open(os.path.join(data_dir, "sample.txt"), "w") as f:
#             f.write(sample_content)
#         print("Created sample.txt in data directory")
    
#     try:
#         # Load documents
#         print(f"Loading documents from {data_dir}...")
#         documents = load_documents(data_dir)
#         print(f"Loaded {len(documents)} documents")
        
#         if not documents:
#             print("No documents loaded. Please add files to the data directory.")
#             return
        
#         # Create vector store manager and build vector store
#         print("Building vector store...")
#         vector_manager = VectorStoreManager(
#             persist_path="test_faiss_store",
#             embed_model="all-MiniLM-L6-v2"
#         )
#         vector_manager.build_vector_store(documents)
        
#         # Initialize RAGSearch - Pass the vector_store_manager
#         rag_search = RAGSearch(
#             persist_dir="test_faiss_store",
#             embedding_model="all-MiniLM-L6-v2",
#             llm_model="llama-3.1-8b-instant",
#             data_dir=data_dir,
#             groq_api_key=api_key,
#             vector_store_manager=vector_manager  # Pass the manager, not the FAISS object
#         )
        
#         # Test queries
#         test_queries = [
#             "What is artificial intelligence?",
#             "Explain machine learning",
#             "What is deep learning?",
#             "what is software engineering?"
#         ]
        
#         print("\n" + "="*50)
#         print("Testing RAG Search:")
#         print("="*50)
        
#         for query in test_queries:
#             print(f"\nQuery: {query}")
#             answer = rag_search.search(query, top_k=2)
#             print(f"Answer: {answer[:200]}...")
            
#         print("\n✅ RAG Search test complete!")
        
#     except Exception as e:
#         print(f"\n❌ Error during test: {e}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()








