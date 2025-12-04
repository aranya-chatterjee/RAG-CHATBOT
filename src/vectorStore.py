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
        self.embedding_pipeline = EmbeddingPipeline(model_name=self.embed_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        print(f"[INFO] VectorStoreManager initialized with model: {self.embed_model}, chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}")
    
    def build_vector_store(self, documents: List[Any]):
        # Create directory if it doesn't exist
        os.makedirs(self.persist_path, exist_ok=True)
        
        chunks = self.embedding_pipeline.chunk_documents(documents)
        embeddings = self.embedding_pipeline.embed_chunks(chunks)
        metadatas = [{"source": f"chunk_{i}"} for i in range(len(chunks))]
        
        # adding and saving to FAISS vector store
        self.add_embeddings(chunks, np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved at {self.persist_path}")
    
    def add_embeddings(self, chunks: List[str], embeddings: np.ndarray, metadatas: List[dict]):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # Create index_to_docstore_id mapping (required by newer FAISS versions)
        index_to_docstore_id = {i: f"chunk_{i}" for i in range(len(metadatas))}
        
        # Create docstore with proper content
        docstore_dict = {}
        for i in range(len(metadatas)):
            docstore_dict[f"chunk_{i}"] = {
                "page_content": chunks[i],  # Store the actual chunk content
                "metadata": metadatas[i]
            }
        
        docstore = InMemoryDocstore(docstore_dict)
        
        # Initialize FAISS with all required parameters
        self.vector_store = FAISS(
            embedding_function=None, 
            index=index, 
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
        print(f"[INFO] Added {len(metadatas)} embeddings to the vector store.")
    
    def save(self):
        if self.vector_store is not None:
            # Create directory if it doesn't exist
            os.makedirs(self.persist_path, exist_ok=True)
            
            faiss.write_index(self.vector_store.index, os.path.join(self.persist_path, "faiss_index.idx"))
            with open(os.path.join(self.persist_path, "docstore.pkl"), "wb") as f:
                pickle.dump(self.vector_store.docstore, f)
            # Also save the index_to_docstore_id mapping
            with open(os.path.join(self.persist_path, "index_to_docstore_id.pkl"), "wb") as f:
                pickle.dump(self.vector_store.index_to_docstore_id, f)
            print(f"[INFO] Vector store saved to {self.persist_path}")
        else:
            print("[WARN] No vector store to save.")
    
    def load(self):
        index_path = os.path.join(self.persist_path, "faiss_index.idx")
        docstore_path = os.path.join(self.persist_path, "docstore.pkl")
        index_to_docstore_id_path = os.path.join(self.persist_path, "index_to_docstore_id.pkl")
        
        if all(os.path.exists(path) for path in [index_path, docstore_path, index_to_docstore_id_path]):
            index = faiss.read_index(index_path)
            with open(docstore_path, "rb") as f:
                docstore = pickle.load(f)
            with open(index_to_docstore_id_path, "rb") as f:
                index_to_docstore_id = pickle.load(f)
            
            self.vector_store = FAISS(
                embedding_function=None, 
                index=index, 
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
            print(f"[INFO] Vector store loaded from {self.persist_path}")
        else:
            print("[ERROR] Vector store files not found.")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Any]:  
        if self.vector_store is None:
            print("[ERROR] Vector store not loaded.")
            return []
        query_embedding = self.embedding_pipeline.model.encode([query_text], show_progress_bar=False)
        results = self.vector_store.similarity_search_by_vector(query_embedding[0], k=top_k)
        print(f"[INFO] Retrieved {len(results)} results for the query.")
        return results
    
    def search(self, query_text: str, top_k: int = 5) -> List[Any]:
        return self.query(query_text, top_k)

# # Example usage:
# if __name__ == "__main__":
#     from data_loader import load_documents
#     docs = load_documents("data")
#     vsm = VectorStoreManager()
#     vsm.build_vector_store(docs)
    
#     # No need to call load() immediately after build_vector_store()
#     # since the vector store is already in memory
#     results = vsm.search("example query", top_k=3)
#     for res in results:
#         print(res)
    
#     # Example of loading a previously saved vector store
#     # vsm2 = VectorStoreManager()
#     # vsm2.load()
#     # results2 = vsm2.search("another query", top_k=3)
