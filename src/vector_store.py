import os
from typing import List, Dict, Any
import faiss
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    """
    Manages the vector database using FAISS and Google Gemini embeddings.
    """
    def __init__(self, index_path: str = "./faiss_index"):
        self.index_path = index_path
        
        # This line will read the API key from your .env file
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        self.vector_store = None
        
        # Load the index if it exists
        if os.path.exists(self.index_path):
            print(f"Loading existing FAISS index from {self.index_path}")
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("No existing FAISS index found. A new one will be created when documents are added.")
            
        print(f"VectorStore initialized. Using FAISS with Google Gemini embeddings.")

    def add_documents(self, documents: List[Dict[str, Any]]):
        if not documents:
            print("No documents to add.")
            return

        print(f"Adding {len(documents)} documents to the FAISS vector store...")
        
        if self.vector_store is None:
            # Create the vector store if it doesn't exist
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Add to the existing vector store
            self.vector_store.add_documents(documents)
        
        # Save the updated index to disk
        self.vector_store.save_local(self.index_path)
        print("Documents added and index saved successfully.")

    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        if self.vector_store is None:
            return {"documents": [], "metadatas": []}
            
        print(f"Querying the FAISS vector store with: '{query_text}'")
        
        # FAISS uses a different method for similarity search
        docs = self.vector_store.similarity_search(query_text, k=n_results)
        
        # Format the results to match the structure expected by our RAG pipeline
        results = {
            "documents": [[doc.page_content for doc in docs]],
            "metadatas": [[doc.metadata for doc in docs]]
        }
        
        return results

    def is_empty(self) -> bool:
        return self.vector_store is None