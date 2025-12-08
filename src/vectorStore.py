import os 
from typing import List, Any
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import numpy as np 
import pickle 
from sentence_transformers import SentenceTransformer 
from ChunkAndEmbed import EmbeddingPipeline

class VectorStoreManager:
    def __init__(self, persist_path: str="faiss_store", embed_model:str="all-MiniLM-L6-v2"):
        self.persist_path = persist_path
        self.embed_model = embed_model
        self.embedding_pipeline = EmbeddingPipeline(model_name=self.embed_model)
        self.vector_store = None
        print(f"[VectorStore] Initialized")
    
    def build_vector_store(self, documents: List[Any]):
        """SIMPLIFIED: Build vector store that actually works"""
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
        
        # METHOD 1: Use LangChain's from_texts (SIMPLEST)
        try:
            # Create a simple embedding function
            class Embedder:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts):
                    return self.model.encode(texts).tolist()
                
                def embed_query(self, text):
                    return self.model.encode([text])[0].tolist()
            
            embedder = Embedder(self.embedding_pipeline.model)
            
            # Create metadata
            metadatas = [{"chunk_id": i, "source": "doc"} for i in range(len(chunks))]
            
            # Use from_texts - it handles everything
            self.vector_store = FAISS.from_texts(
                texts=chunks,
                embedding=embedder,
                metadatas=metadatas
            )
            
            print(f"✅ Created FAISS store with {len(chunks)} documents")
            
        except Exception as e:
            print(f"Method 1 failed: {e}")
            # METHOD 2: Manual creation as fallback
            self._create_manually(chunks, embeddings)
        
        # TEST
        self._test_query(chunks[0])
    
    def _create_manually(self, chunks, embeddings):
        """Manual FAISS creation"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        embeddings_np = np.array(embeddings).astype('float32')
        index.add(embeddings_np)
        
        # Create docstore with proper objects
        docstore_dict = {}
        for i, chunk in enumerate(chunks):
            # Create an object with page_content attribute
            doc = type('Doc', (), {})()
            doc.page_content = chunk
            doc.metadata = {"id": i}
            docstore_dict[str(i)] = doc
        
        docstore = InMemoryDocstore(docstore_dict)
        
        # Embedding function
        class Embedder:
            def __init__(self, model):
                self.model = model
            
            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()
            
            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()
        
        embedder = Embedder(self.embedding_pipeline.model)
        
        self.vector_store = FAISS(
            embedding_function=embedder,
            index=index,
            docstore=docstore,
            index_to_docstore_id={i: str(i) for i in range(len(chunks))}
        )
        
        print(f"Created manual FAISS with {len(chunks)} docs")
    
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
                content = results[0].page_content if hasattr(results[0], 'page_content') else str(results[0])
                print(f"Found: {content[:100]}...")
            else:
                print("❌ No results - trying alternative search...")
                # Try direct embedding search
                self._test_direct_search(test_word, sample_chunk)
    
    def _test_direct_search(self, query, sample_chunk):
        """Direct embedding search for debugging"""
        try:
            # Get query embedding
            query_embedding = self.embedding_pipeline.model.encode([query])
            query_vec = np.array(query_embedding[0], dtype="float32").reshape(1, -1)
            
            # Get sample embedding
            sample_embedding = self.embedding_pipeline.model.encode([sample_chunk[:100]])
            sample_vec = np.array(sample_embedding[0], dtype="float32").reshape(1, -1)
            
            # Calculate distance
            distance = np.linalg.norm(query_vec - sample_vec)
            print(f"Distance between query and sample: {distance}")
            
            if distance < 10:  # Should be close
                print("⚠️ Embeddings are close but FAISS not finding them")
                print("Check FAISS index count:", self.vector_store.index.ntotal if self.vector_store.index else "No index")
        except Exception as e:
            print(f"Direct test error: {e}")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Any]:
        if not self.vector_store:
            print("No vector store!")
            return []
        
        try:
            # IMPORTANT: Use k=top_k (not top_k=top_k)
            results = self.vector_store.similarity_search(query_text, k=top_k)
            return results
        except Exception as e:
            print(f"Query error: {e}")
            return []
    
    def save(self):
        if self.vector_store:
            self.vector_store.save_local(self.persist_path)
            print(f"Saved to {self.persist_path}")
    
    def load(self):
        try:
            class Embedder:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts):
                    return self.model.encode(texts).tolist()
                
                def embed_query(self, text):
                    return self.model.encode([text])[0].tolist()
            
            embedder = Embedder(self.embedding_pipeline.model)
            
            self.vector_store = FAISS.load_local(
                self.persist_path,
                embedder,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded from {self.persist_path}")
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            return False
