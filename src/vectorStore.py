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
        # Create directory if it doesn't exist
        os.makedirs(self.persist_path, exist_ok=True)
        
        print(f"[1/4] Chunking {len(documents)} documents...")
        chunks = self.embedding_pipeline.chunk_documents(documents)
        print(f"    Created {len(chunks)} chunks")
        
        if len(chunks) == 0:
            print("ERROR: No chunks created!")
            return
        
        print(f"[2/4] Generating embeddings...")
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        print(f"    Embeddings shape: {embeddings.shape}")
        
        # Create metadata for each chunk
        metadatas = []
        for i in range(len(chunks)):
            source = "unknown"
            if i < len(documents) and hasattr(documents[i], 'metadata'):
                doc_meta = getattr(documents[i], 'metadata', {})
                if isinstance(doc_meta, dict) and 'source' in doc_meta:
                    source = doc_meta['source']
            metadatas.append({"source": source, "chunk_id": i, "text": chunks[i][:50] + "..."})
        
        # CRITICAL FIX: Create proper embedding function
        class SimpleEmbeddingFunction:
            def __init__(self, model):
                self.model = model
            
            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()
            
            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()
        
        embedding_function = SimpleEmbeddingFunction(self.embedding_pipeline.model)
        
        print(f"[3/4] Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        embeddings_np = np.array(embeddings).astype('float32')
        index.add(embeddings_np)
        print(f"    Index now has {index.ntotal} vectors")
        
        # Create docstore
        docstore_dict = {}
        for i in range(len(chunks)):
            from langchain.schema import Document
            docstore_dict[str(i)] = Document(
                page_content=chunks[i],
                metadata=metadatas[i]
            )
        
        docstore = InMemoryDocstore(docstore_dict)
        
        # Create index_to_docstore_id mapping
        index_to_docstore_id = {i: str(i) for i in range(len(chunks))}
        
        print(f"[4/4] Initializing FAISS vector store...")
        # Initialize FAISS with embedding function
        self.vector_store = FAISS(
            embedding_function=embedding_function,  # THIS IS CRITICAL
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        print(f"✅ Vector store built with {len(chunks)} chunks")
        
        # TEST IMMEDIATELY
        self._test_vector_store(chunks)
        
        self.save()
        print(f"[INFO] Vector store saved at {self.persist_path}")
    
    def _test_vector_store(self, chunks):
        """Test if the vector store actually works."""
        if len(chunks) == 0:
            return
        
        print("\n🧪 Testing vector store...")
        
        # Test 1: Use first few meaningful words from first chunk
        first_chunk = chunks[0]
        words = [w for w in first_chunk.split() if len(w) > 3][:3]
        
        if words:
            test_query = " ".join(words)
            print(f"   Test query: '{test_query}'")
            results = self.query(test_query, top_k=1)
            print(f"   Results: {len(results)}")
            
            if len(results) > 0:
                print("   ✅ Vector store works!")
            else:
                print("   ❌ No results found!")
                print("   Trying individual words...")
                for word in words:
                    results = self.query(word, top_k=1)
                    print(f"     '{word}': {len(results)} results")
        
        # Test 2: Check index stats
        if self.vector_store and self.vector_store.index:
            print(f"   Index has {self.vector_store.index.ntotal} vectors")
    
    def save(self):
        if self.vector_store is not None:
            os.makedirs(self.persist_path, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.vector_store.index, 
                            os.path.join(self.persist_path, "faiss_index.idx"))
            
            # Save docstore
            with open(os.path.join(self.persist_path, "docstore.pkl"), "wb") as f:
                pickle.dump(self.vector_store.docstore, f)
            
            # Save index_to_docstore_id mapping
            with open(os.path.join(self.persist_path, "index_to_docstore_id.pkl"), "wb") as f:
                pickle.dump(self.vector_store.index_to_docstore_id, f)
            
            print(f"[INFO] Vector store saved to {self.persist_path}")
        else:
            print("[WARN] No vector store to save.")
    
    def load(self):
        index_path = os.path.join(self.persist_path, "faiss_index.idx")
        docstore_path = os.path.join(self.persist_path, "docstore.pkl")
        index_to_docstore_id_path = os.path.join(self.persist_path, "index_to_docstore_id.pkl")
        
        try:
            if all(os.path.exists(path) for path in [index_path, docstore_path, index_to_docstore_id_path]):
                # Load FAISS index
                index = faiss.read_index(index_path)
                
                # Load docstore
                with open(docstore_path, "rb") as f:
                    docstore = pickle.load(f)
                
                # Load mapping
                with open(index_to_docstore_id_path, "rb") as f:
                    index_to_docstore_id = pickle.load(f)
                
                # Create embedding function
                class SimpleEmbeddingFunction:
                    def __init__(self, model):
                        self.model = model
                    
                    def embed_documents(self, texts):
                        return self.model.encode(texts).tolist()
                    
                    def embed_query(self, text):
                        return self.model.encode([text])[0].tolist()
                
                embedding_function = SimpleEmbeddingFunction(self.embedding_pipeline.model)
                
                # Recreate FAISS
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
        if self.vector_store is None:
            print("[ERROR] Vector store not loaded.")
            return []
        
        try:
            print(f"[QUERY] Searching for: '{query_text}'")
            
            # METHOD 1: Use similarity_search (RECOMMENDED)
            # This uses the embedding_function to embed the query
            results = self.vector_store.similarity_search(query_text, k=top_k)
            
            print(f"[QUERY] Found {len(results)} results")
            
            # If no results, try fallback method
            if len(results) == 0:
                print("[QUERY] No results with similarity_search, trying direct FAISS search...")
                results = self._direct_faiss_search(query_text, top_k)
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _direct_faiss_search(self, query_text: str, top_k: int = 5) -> List[Any]:
        """Direct FAISS search as fallback."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_pipeline.model.encode([query_text], show_progress_bar=False)
            query_vector = np.array(query_embedding[0], dtype="float32").reshape(1, -1)
            
            # Search FAISS index
            distances, indices = self.vector_store.index.search(query_vector, top_k)
            
            # Get documents from docstore
            results = []
            for idx in indices[0]:
                if idx != -1:  # -1 means no result
                    doc_id = self.vector_store.index_to_docstore_id.get(idx)
                    if doc_id and hasattr(self.vector_store.docstore, '_dict'):
                        doc = self.vector_store.docstore._dict.get(doc_id)
                        if doc:
                            results.append(doc)
            
            print(f"[DIRECT FAISS] Found {len(results)} results")
            return results
        except Exception as e:
            print(f"[ERROR] Direct FAISS search failed: {e}")
            return []
    
    def search(self, query_text: str, top_k: int = 5) -> List[Any]:
        return self.query(query_text, top_k)

