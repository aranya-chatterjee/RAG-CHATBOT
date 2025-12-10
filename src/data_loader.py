from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
import os

def load_documents(data_dir : str) -> List[Any]:
    """Load documents from a directory using appropriate loaders based on file extensions."""
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.docx': Docx2txtLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.json': JSONLoader
    }
    
    documents = []
    data_path = Path(data_dir)
    for file_path in data_path.iterdir():
        if not file_path.is_file() or file_path.name.startswith('.'):
       
            continue
        
        ext = file_path.suffix.lower()
        if ext in loaders:
            loader_class = loaders[ext]
            try:
                loader = loader_class(str(file_path))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path.name} with {loader_class.__name__}: {e}")
        else:
            print(f"Unsupported file type: {file_path.suffix}")

    return documents
    
    

# if __name__ == "__main__":
   
    
#     docs = load_documents("data")
#     print(f"Loaded {len(docs)} documents")
#     for d in docs[:3]:   # print first 3 docs
#         print("---------")
#         print(d)


