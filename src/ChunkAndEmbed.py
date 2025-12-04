from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer 
import numpy as np 
from typing import List, Any
from src.data_loader import load_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"Loaded SentenceTransformer model: {model_name}")
    
    def chunk_documents(self, documents: List[Any]) -> List[str]:
        """Chunk documents into smaller pieces."""
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = []
        for doc in documents:
            doc_chunks = text_splitter.split_text(doc.page_content)
            chunks.extend(doc_chunks)
        print(f"Chunked documents into {len(chunks)} pieces.")
        return chunks
    
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for each chunk."""
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        print(f"Generated embeddings for {len(chunks)} chunks.")
        return embeddings
# if __name__ == "__main__":
    
#     docs = load_documents("data")
#     emb_pipe = EmbeddingPipeline()
#     chunks = emb_pipe.chunk_documents(docs)
#     embeddings = emb_pipe.embed_chunks(chunks)

#     print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)p
