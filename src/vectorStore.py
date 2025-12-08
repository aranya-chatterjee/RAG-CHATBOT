import os 
from typing import List, Any, Tuple
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import numpy as np 
import pickle 
from sentence_transformers import SentenceTransformer 
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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        print(f"[INFO] VectorStoreManager initialized with model: {self.embed_model}")
    
    def build_vector_store(self, documents: List[Any]):
        """Build and save the vector store from documents."""
        # Create directory if it doesn't exist
        os.makedirs(self.persist_path, exist_ok=True)
        
        print(f"[INFO] Starting to build vector store from {len(documents)} documents...")
        
        # 1. Chunk documents
        chunks = self.embedding_pipeline.chunk_documents(documents)
        print(f"[INFO] Created {len(chunks)} chunks")
        
        if len(chunks) == 0:
            print("[ERROR] No chunks created from documents!")
            return
        
        # 2. Generate embeddings for chunks
        print("[INFO] Generating embeddings for chunks...")
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        print(f"[INFO] Generated embeddings shape: {embeddings.shape}")
        
        # 3. Create metadata for each chunk
        metadatas = []
        for i, chunk in enumerate(chunks):
            # Try to preserve original document source if available
            source = f"chunk_{i}"
            if i < len(documents) and hasattr(documents[i], 'metadata'):
                doc_metadata = getattr(documents[i], 'metadata', {})
                if isinstance(doc_metadata, dict) and 'source' in doc_metadata:
                    source = doc_metadata['source']
            metadatas.append({"source": source, "chunk_index": i})
        
        # 4. Create FAISS index
        print("[INFO] Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        embeddings_np = np.array(embeddings).astype('float32')
        index.add(embeddings_np)
        
        print(f"[INFO] FAISS index created with {index.ntotal} vectors")
        
        # 5. Create docstore
        docstore_dict = {
            f"doc_{i}": {
                "page_content": chunks[i],
                "metadata": metadatas[i]
            }
            for i in range(len(chunks))
        }
        docstore = InMemoryDocstore(docstore_dict)
        
        # 6. Create index_to_docstore_id mapping
        index_to_docstore_id = {i: f"doc_{i}" for i in range(len(chunks))}
        
        # 7. Create a custom embedding function wrapper
        class CustomEmbeddings:
            def __init__(self, model):
                self.model = model
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Embed multiple documents."""
                embeddings = self.model.encode(texts, show_progress_bar=False)
                return embeddings.tolist()
            
            def embed_query(self, text: str) -> List[float]:
                """Embed a single query."""
                embedding = self.model.encode([text], show_progress_bar=False)
                return embedding[0].tolist()
        
        # Create embedding function
        embedding_function = CustomEmbeddings(self.embedding_pipeline.model)
        
        # 8. Initialize FAISS vector store
        self.vector_store = FAISS(
            embedding_function=embedding_function,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        print(f"[INFO] FAISS vector store created with {len(chunks)} documents")
        
        # 9. Save to disk
        self.save()
        
        # 10. Verify the index works
        print("[INFO] Verifying index with test query...")
        if len(chunks) > 0:
            # Use first few words of first chunk as test
            test_text = chunks[0].split()[:3]
            if test_text:
                test_query = " ".join(test_text)
                test_results = self.query(test_query, top_k=1)
                print(f"[INFO] Test query '{test_query}' returned {len(test_results)} results")
        
        print(f"[INFO] Vector store built and saved to {self.persist_path}")
    
    def save(self):
        """Save the vector store to disk."""
        if self.vector_store is not None:
            os.makedirs(self.persist_path, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.vector_store.index, os.path.join(self.persist_path, "faiss_index.idx"))
            
            # Save docstore
            with open(os.path.join(self.persist_path, "docstore.pkl"), "wb") as f:
                pickle.dump(self.vector_store.docstore, f)
            
            # Save index_to_docstore_id mapping
            with open(os.path.join(self.persist_path, "index_to_docstore_id.pkl"), "wb") as f:
                pickle.dump(self.vector_store.index_to_docstore_id, f)
            
            print(f"[INFO] Vector store saved to {self.persist_path}")
        else:
            print("[WARNING] No vector store to save")
    
    def load(self):
        """Load the vector store from disk."""
        index_path = os.path.join(self.persist_path, "faiss_index.idx")
        docstore_path = os.path.join(self.persist_path, "docstore.pkl")
        mapping_path = os.path.join(self.persist_path, "index_to_docstore_id.pkl")
        
        try:
            if all(os.path.exists(p) for p in [index_path, docstore_path, mapping_path]):
                # Load FAISS index
                index = faiss.read_index(index_path)
                
                # Load docstore
                with open(docstore_path, "rb") as f:
                    docstore = pickle.load(f)
                
                # Load mapping
                with open(mapping_path, "rb") as f:
                    index_to_docstore_id = pickle.load(f)
                
                # Create embedding function
                class CustomEmbeddings:
                    def __init__(self, model):
                        self.model = model
                    
                    def embed_documents(self, texts: List[str]) -> List[List[float]]:
                        embeddings = self.model.encode(texts, show_progress_bar=False)
                        return embeddings.tolist()
                    
                    def embed_query(self, text: str) -> List[float]:
                        embedding = self.model.encode([text], show_progress_bar=False)
                        return embedding[0].tolist()
                
                embedding_function = CustomEmbeddings(self.embedding_pipeline.model)
                
                # Recreate FAISS vector store
                self.vector_store = FAISS(
                    embedding_function=embedding_function,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id
                )
                
                print(f"[INFO] Vector store loaded from {self.persist_path}")
                print(f"[INFO] Index contains {index.ntotal} vectors")
                return True
            else:
                print("[ERROR] Vector store files not found")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to load vector store: {e}")
            return False
    
    def query(self, query_text: str, top_k: int = 5) -> List[Any]:
        """Query the vector store."""
        if self.vector_store is None:
            print("[ERROR] Vector store not loaded or built")
            return []
        
        try:
            print(f"[INFO] Querying: '{query_text}'")
            
            # Method 1: Use similarity_search (recommended - uses embedding function)
            results = self.vector_store.similarity_search(query_text, k=top_k)
            
            # Method 2: Alternative - similarity_search_by_vector
            # query_embedding = self.embedding_pipeline.model.encode([query_text], show_progress_bar=False)
            # results = self.vector_store.similarity_search_by_vector(query_embedding[0], k=top_k)
            
            print(f"[INFO] Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def search(self, query_text: str, top_k: int = 5) -> List[Any]:
        """Alias for query method."""
        return self.query(query_text, top_k)
    
    def get_index_info(self) -> dict:
        """Get information about the FAISS index."""
        if self.vector_store is None or self.vector_store.index is None:
            return {"error": "No index available"}
        
        index = self.vector_store.index
        return {
            "num_vectors": index.ntotal,
            "dimension": index.d if hasattr(index, 'd') else "unknown",
            "type": str(type(index))
        }
