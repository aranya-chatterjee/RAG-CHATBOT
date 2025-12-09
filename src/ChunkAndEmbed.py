from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer 
import numpy as np 
from typing import List, Any
from data_loader import load_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            raise

    
    def chunk_documents(self, documents: List[Any]) -> List[str]:
        """Chunk documents into smaller pieces."""
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = []
        for doc in documents:
            content = getattr(doc, "page_content", None)
            if not content:
                print("Warning: Document missing 'page_content'. Skipping.")
                continue
            doc_chunks = text_splitter.split_text(content)
            chunks.extend(doc_chunks)
        print(f"Chunked documents into {len(chunks)} pieces.")
        return chunks
    def embed_chunks(self, chunks: List[str], batch_size:int=32) -> np.ndarray:
        """
        Generate embeddings for each chunk.
        """
        try:
            if not chunks:
                print("[WARNING] No chunks to embed")
                return np.array([])
            
            print(f"[INFO] Embedding {len(chunks)} chunks...")
            embeddings = self.model.encode(chunks, show_progress_bar=True, batch_size=batch_size)
            print(f"[INFO] Generated embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"[ERROR] Error in embedding chunks: {e}")
            raise
   
# if __name__ == "__main__":
    
#     docs = load_documents("data")
#     emb_pipe = EmbeddingPipeline()
#     chunks = emb_pipe.chunk_documents(docs)
#     embeddings = emb_pipe.embed_chunks(chunks)
#     print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
