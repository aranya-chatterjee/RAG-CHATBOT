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
        if file_path.suffix in loaders:
            loader_class = loaders[file_path.suffix]
            loader = loader_class(str(file_path))
            documents.extend(loader.load())
        else:
            print(f"Unsupported file type: {file_path.suffix}")
    
    return documents
# testing the code 

# if __name__ == "__main__":
    
#     test_path = "src\transformer.pdf"  # Replace with actual path
#     documents = load_documents("src\transformer.pdf")
#     print(f"Loaded {len(documents)} documents")
#     for doc in documents:
#         print(f"Content preview: {doc.page_content[:200]}...")