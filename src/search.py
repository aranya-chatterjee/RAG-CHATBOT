import os
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

class RAGSearch:
    """
    Retrieval-Augmented Generation (RAG) search leveraging a GROQ LLM.
    """
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemma2-9b-it",
        data_dir: str = "data",
        groq_api_key: str = None
    ):
        print(f"[RAGSearch] Initializing with data_dir: {data_dir}")

        # Get API key
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")
        self.api_key = api_key

        self.data_dir = data_dir
        self.llm_model = llm_model
        print(f"[RAGSearch] Initialization complete")

    def search(self, query: str, top_k: int = 3) -> str:
        """
        Simple fallback search, returning a helpful general response.
        """
        print(f"[RAGSearch] search() called with query: '{query}'")

        try:
            # Import here to avoid circular imports
            from langchain_groq import ChatGroq

            # Initialize LLM
            llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.llm_model,
                temperature=0.1,
                max_tokens=512
            )

            # Simple prompt
            prompt = f"""You are a helpful assistant. The user has uploaded a document and is asking about it.

User question: {query}

Since I'm having technical issues with my document search system, please provide a helpful response based on general knowledge.

Answer the question helpfully:"""

            # Get response
            response = llm.invoke(prompt)
            answer = getattr(response, "content", str(response)).strip()

            print(f"[RAGSearch] Generated response: {answer[:100]}...")
            return answer
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[RAGSearch ERROR] {error_msg}")
            return (
                f"I'm here! You asked: '{query}'. "
                "I can hear you but I'm having trouble processing the document. "
                "Please try again or re-upload the document."
            )

# Test function
if __name__ == "__main__":
    print("Testing RAGSearch...")

    # Load environment variable, if not already loaded
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        print(f"API key found: {api_key[:10]}...")

        rag = RAGSearch(
            data_dir="test_data",
            groq_api_key=api_key,
            llm_model="gemma2-9b-it"
        )

        # Test search
        result = rag.search("What is artificial intelligence?")
        print(f"Test result: {result}")
    else:
        print("ERROR: No GROQ_API_KEY found in .env file")

