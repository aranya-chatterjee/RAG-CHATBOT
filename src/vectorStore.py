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
        self.vector_store = None
    
    def build_vector_store(self, documents: List[Any]):
        # 1. Chunk documents
        chunks = self.embedding_pipeline.chunk_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        if len(chunks) == 0:
            return
        
        # 2. Generate embeddings
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        
        # 3. Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        embeddings_np = np.array(embeddings).astype('float32')
        index.add(embeddings_np)
        print(f"FAISS index has {index.ntotal} vectors")
        
        # 4. Create simple docstore (without Document class)
        docstore_dict = {}
        for i in range(len(chunks)):
            # Store as dictionary instead of Document object
            docstore_dict[str(i)] = {
                "page_content": chunks[i],
                "metadata": {"chunk_id": i, "source": "document"}
            }
        
        docstore = InMemoryDocstore(docstore_dict)
        
        # 5. Create index mapping
        index_to_docstore_id = {i: str(i) for i in range(len(chunks))}
        
        # 6. Create embedding function
        class EmbeddingFunc:
            def __init__(self, model):
                self.model = model
            
            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()
            
            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()
        
        embedding_function = EmbeddingFunc(self.embedding_pipeline.model)
        
        # 7. Create FAISS vector store
        self.vector_store = FAISS(
            embedding_function=embedding_function,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        
        # 8. Test it
        self._test_with_first_chunk(chunks[0])
    
    def _test_with_first_chunk(self, first_chunk):
        """Test query with content from first chunk."""
        words = first_chunk.split()[:3]
        if words:
            test_query = " ".join(words)
            results = self.query(test_query, top_k=1)
            print(f"Test query '{test_query}': {len(results)} results")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Any]:
        if not self.vector_store:
            return []
        
        try:
            results = self.vector_store.similarity_search(query_text, k=top_k)
            return results
        except:
            return []
    
    def save(self):
        if self.vector_store:
            os.makedirs(self.persist_path, exist_ok=True)
            faiss.write_index(self.vector_store.index, 
                            os.path.join(self.persist_path, "index.faiss"))
            print(f"Saved to {self.persist_path}")
