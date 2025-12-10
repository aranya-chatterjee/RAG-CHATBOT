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
        vector_store_manager=None  
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
                
           
                results = self.vector_store_manager.query(query, top_k=top_k)
                print(f"[DEBUG] Query returned {len(results)} results")
                
                if results and len(results) > 0:
                   
                    context_parts = []
                    for r in results:
                        if hasattr(r, 'page_content'):
                            context_parts.append(r.page_content)
                        elif isinstance(r, dict) and 'page_content' in r:
                            context_parts.append(r['page_content'])
                        else:
                            context_parts.append(str(r)[:500])
                    
                    
                    docs_context = "\n\n---\n\n".join(context_parts[:top_k])
                    
                    
                    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context. 
If the answer cannot be found in the context, say "I don't have enough information to answer that question."

CONTEXT:
{docs_context}

QUESTION: {query}

ANSWER (based only on context):"""
                    
                    print(f"[DEBUG] Context length: {len(docs_context)} characters")
                    
                else:
                    
                    prompt = f"""I searched the document but couldn't find specific information about: {query}

Please try asking about something else in the document or rephrase your question."""
                    
            else:
                
                prompt = f"""No document has been loaded yet. Please upload a document first.

Your question was: {query}"""

            
            print(f"[RAGSearch] Calling LLM...")
            llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.llm_model,
                temperature=0.1,
                max_tokens=512
            )
            
            response = llm.invoke(prompt)
            
            
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




