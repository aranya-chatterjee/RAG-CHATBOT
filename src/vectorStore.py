import os 
from typing import List, Any
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import numpy as np 
import pickle 
from sentence_transformers import SentenceTransformer 
from ChunkAndEmbed import EmbeddingPipeline

class SentenceTransformerEmbeddings(Embeddings):
    """Proper Embeddings class for LangChain"""
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.model.encode([text])[0].tolist()

class VectorStoreManager:
    def __init__(self, persist_path: str="faiss_store", embed_model:str="all-MiniLM-L6-v2"):
        self.persist_path = persist_path
        self.embed_model = embed_model
        self.embedding_pipeline = EmbeddingPipeline(model_name=self.embed_model)
        self.vector_store = None
        print(f"[VectorStore] Initialized")
    
    def build_vector_store(self, documents: List[Any]):
        """Build vector store that actually works"""
        print(f"[1/3] Chunking documents...")
        chunks = self.embedding_pipeline.chunk_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        if not chunks:
            print("ERROR: No chunks!")
            return
        
        print(f"[2/3] Generating embeddings...")
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        print(f"Embeddings shape: {embeddings.shape}")
        
        print(f"[3/3] Creating FAISS store...")
        
        # Create proper Embeddings object
        embedding_function = SentenceTransformerEmbeddings(self.embedding_pipeline.model)
        
        # Create metadata
        metadatas = [{"chunk_id": i, "source": "doc"} for i in range(len(chunks))]
        
        # Use from_texts with proper Embeddings object
        self.vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_function,  # Use Embeddings object
            metadatas=metadatas
        )
        
        print(f"✅ Created FAISS store with {len(chunks)} documents")
        
        # TEST
        self._test_query(chunks[0])
    
    def _test_query(self, sample_chunk):
        """Test with actual content"""
        print("\n🔍 Testing vector store...")
        
        # Extract words from sample
        words = [w for w in sample_chunk.split() if len(w) > 3]
        if words:
            # Try the first word
            test_word = words[0]
            print(f"Testing query: '{test_word}'")
            
            results = self.query(test_word, top_k=1)
            print(f"Results found: {len(results)}")
            
            if results:
                print("✅ Vector store WORKS!")
                content = results[0].page_content
                print(f"Found: {content[:100]}...")
            else:
                print("❌ No results found")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Any]:
        if not self.vector_store:
            print("No vector store!")
            return []
        
        try:
            # Use similarity_search with k parameter
            results = self.vector_store.similarity_search(query_text, k=top_k)
            return results
        except Exception as e:
            print(f"Query error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def save(self):
        if self.vector_store:
            self.vector_store.save_local(self.persist_path)
            print(f"Saved to {self.persist_path}")
    
    def load(self):
        try:
            # Create Embeddings object for loading
            embedding_function = SentenceTransformerEmbeddings(self.embedding_pipeline.model)
            
            self.vector_store = FAISS.load_local(
                folder_path=self.persist_path,
                embeddings=embedding_function,  
                allow_dangerous_deserialization=True
            )
            print(f"Loaded from {self.persist_path}")
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            import traceback
            traceback.print_exc()
            return False

