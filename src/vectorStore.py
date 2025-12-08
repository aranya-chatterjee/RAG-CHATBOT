import os 
from typing import List, Any
import faiss
import numpy as np 
import pickle 
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from ChunkAndEmbed import EmbeddingPipeline

class VectorStoreManager:
    def __init__(self, persist_path: str="faiss_store", embed_model:str="all-MiniLM-L6-v2", chunk_size:int=1000, chunk_overlap:int=200):
        self.persist_path = persist_path
        self.embed_model = embed_model
        self.embedding_pipeline = EmbeddingPipeline(
            model_name=self.embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = None
        print(f"[VectorStore] Initialized with model: {self.embed_model}")
    
    def build_vector_store(self, documents: List[Any]):
        """Build vector store from documents using LangChain FAISS."""
        print(f"[VectorStore] Building from {len(documents)} documents...")
        
        # 1. Chunk documents
        chunks = self.embedding_pipeline.chunk_documents(documents)
        print(f"[VectorStore] Created {len(chunks)} chunks")
        
        if len(chunks) == 0:
            print("[VectorStore] ERROR: No chunks created!")
            return
        
        # 2. Create LangChain Document objects with metadata
        documents_list = []
        for i, chunk in enumerate(chunks):
            # Try to get source from original document
            source = f"chunk_{i}"
            if i < len(documents) and hasattr(documents[i], 'metadata'):
                doc_meta = documents[i].metadata
                if isinstance(doc_meta, dict) and 'source' in doc_meta:
                    source = doc_meta['source']
            
            doc = Document(
                page_content=chunk,
                metadata={"source": source, "chunk_id": i}
            )
            documents_list.append(doc)
        
        # 3. Generate embeddings for all chunks
        print("[VectorStore] Generating embeddings...")
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        
        if len(embeddings) == 0:
            print("[VectorStore] ERROR: No embeddings generated!")
            return
        
        print(f"[VectorStore] Embeddings shape: {embeddings.shape}")
        
        # 4. Create custom embeddings class for LangChain
        class SentenceTransformerEmbeddings:
            def __init__(self, model):
                self.model = model
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Embed documents using SentenceTransformer."""
                return self.model.encode(texts, show_progress_bar=False).tolist()
            
            def embed_query(self, text: str) -> List[float]:
                """Embed a query using SentenceTransformer."""
                return self.model.encode([text], show_progress_bar=False)[0].tolist()
        
        # Create embedding function
        embedding_function = SentenceTransformerEmbeddings(self.embedding_pipeline.model)
        
        # 5. Create FAISS index directly with embeddings
        print("[VectorStore] Creating FAISS index...")
        
        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        index.add(embeddings_np)
        print(f"[VectorStore] FAISS index has {index.ntotal} vectors")
        
        # 6. Create FAISS vector store using from_embeddings (EASIER METHOD)
        print("[VectorStore] Creating FAISS vector store...")
        
        # Method 1: Using from_embeddings (RECOMMENDED)
        try:
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=list(zip(chunks, embeddings_np)),
                embedding=embedding_function,
                metadatas=[doc.metadata for doc in documents_list]
            )
            print("[VectorStore] Created using FAISS.from_embeddings()")
        except Exception as e:
            print(f"[VectorStore] Method 1 failed: {e}. Trying alternative...")
            
            # Method 2: Manual creation
            self.vector_store = FAISS(
                embedding_function=embedding_function,
                index=index,
                docstore=self._create_docstore(documents_list),
                index_to_docstore_id={i: str(i) for i in range(len(documents_list))}
            )
            print("[VectorStore] Created manually")
        
        # 7. Save
        self.save()
        
        # 8. Verify
        self._verify_store(chunks)
        
        print(f"[VectorStore] Build complete. Saved to {self.persist_path}")
    
    def _create_docstore(self, documents):
        """Create a simple docstore."""
        from langchain_community.docstore.in_memory import InMemoryDocstore
        
        docstore_dict = {
            str(i): documents[i] for i in range(len(documents))
        }
        return InMemoryDocstore(docstore_dict)
    
    def _verify_store(self, chunks):
        """Verify the vector store works."""
        if len(chunks) == 0:
            return
        
        # Test with first few words from first chunk
        test_text = " ".join(chunks[0].split()[:3])
        if test_text:
            print(f"[VectorStore] Testing with query: '{test_text}'")
            results = self.query(test_text, top_k=1)
            print(f"[VectorStore] Test returned {len(results)} results")
            
            if len(results) == 0:
                print("[VectorStore] WARNING: Test query returned NO results!")
                # Try a simpler test
                print("[VectorStore] Trying simpler test...")
                for word in chunks[0].split()[:5]:
                    if len(word) > 3:  # Avoid short words
                        results = self.query(word, top_k=1)
                        print(f"  Query '{word}': {len(results)} results")
                        if len(results) > 0:
                            print(f"  SUCCESS! Found match for '{word}'")
                            break
    
    def save(self):
        """Save vector store to disk."""
        if self.vector_store is None:
            print("[VectorStore] WARNING: No vector store to save")
            return
        
        os.makedirs(self.persist_path, exist_ok=True)
        
        try:
            # Use LangChain's save_local method
            self.vector_store.save_local(self.persist_path)
            print(f"[VectorStore] Saved to {self.persist_path}")
        except Exception as e:
            print(f"[VectorStore] Error saving: {e}")
            # Fallback: Manual save
            self._save_manually()
    
    def _save_manually(self):
        """Manual save as fallback."""
        try:
            faiss.write_index(self.vector_store.index, 
                            os.path.join(self.persist_path, "index.faiss"))
            
            # Save documents
            import pickle
            with open(os.path.join(self.persist_path, "documents.pkl"), "wb") as f:
                pickle.dump({
                    "documents": [
                        {"page_content": doc.page_content, "metadata": doc.metadata}
                        for doc in self.vector_store.docstore._dict.values()
                    ]
                }, f)
            print(f"[VectorStore] Manually saved to {self.persist_path}")
        except Exception as e:
            print(f"[VectorStore] Manual save failed: {e}")
    
    def load(self):
        """Load vector store from disk."""
        try:
            # Try LangChain's load_local
            from langchain_community.vectorstores.utils import DistanceStrategy
            
            # Create embedding function
            class SentenceTransformerEmbeddings:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return self.model.encode(texts, show_progress_bar=False).tolist()
                
                def embed_query(self, text: str) -> List[float]:
                    return self.model.encode([text], show_progress_bar=False)[0].tolist()
            
            embedding_function = SentenceTransformerEmbeddings(self.embedding_pipeline.model)
            
            self.vector_store = FAISS.load_local(
                self.persist_path,
                embedding_function,
                allow_dangerous_deserialization=True
            )
            print(f"[VectorStore] Loaded from {self.persist_path}")
            return True
        except Exception as e:
            print(f"[VectorStore] Load failed: {e}")
            return False
    
    def query(self, query_text: str, top_k: int = 5) -> List[Document]:
        """Query the vector store."""
        if self.vector_store is None:
            print("[VectorStore] ERROR: Vector store not initialized")
            return []
        
        try:
            print(f"[VectorStore] Querying: '{query_text}'")
            
            # IMPORTANT: Use similarity_search, NOT similarity_search_by_vector
            results = self.vector_store.similarity_search(query_text, k=top_k)
            
            print(f"[VectorStore] Found {len(results)} results")
            
            # Debug: Show what was found
            if len(results) > 0:
                for i, r in enumerate(results[:2]):  # Show first 2
                    print(f"  Result {i}: {r.page_content[:80]}...")
            else:
                print(f"  No results found for: '{query_text}'")
                # Debug: Check what's in the index
                print(f"  Index has {self.vector_store.index.ntotal if self.vector_store.index else 0} vectors")
            
            return results
            
        except Exception as e:
            print(f"[VectorStore] Query error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: Try direct FAISS search
            return self._query_fallback(query_text, top_k)
    
    def _query_fallback(self, query_text: str, top_k: int = 5) -> List[Document]:
        """Fallback query method using direct FAISS."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_pipeline.model.encode([query_text], 
                                                                  show_progress_bar=False)
            query_vector = np.array(query_embedding[0], dtype="float32").reshape(1, -1)
            
            # Search FAISS index directly
            distances, indices = self.vector_store.index.search(query_vector, top_k)
            
            # Get documents from docstore
            results = []
            for idx in indices[0]:
                if idx != -1:  # -1 means no result
                    doc_id = self.vector_store.index_to_docstore_id.get(idx)
                    if doc_id and doc_id in self.vector_store.docstore._dict:
                        results.append(self.vector_store.docstore._dict[doc_id])
            
            print(f"[VectorStore] Fallback query found {len(results)} results")
            return results
        except Exception as e:
            print(f"[VectorStore] Fallback query failed: {e}")
            return []
    
    def get_info(self):
        """Get information about the vector store."""
        if self.vector_store is None:
            return {"status": "Not initialized"}
        
        info = {
            "status": "Initialized",
            "index_vectors": 0,
            "has_docstore": False
        }
        
        if self.vector_store.index:
            info["index_vectors"] = self.vector_store.index.ntotal
        
        if self.vector_store.docstore:
            info["has_docstore"] = True
            info["docstore_size"] = len(self.vector_store.docstore._dict)
        
        return info
