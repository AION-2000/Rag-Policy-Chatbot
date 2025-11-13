from langchain_google_genai import ChatGoogleGenerativeAI
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    """
    A Retrieval-Augmented Generation (RAG) pipeline using Google Gemini.
    """
    def __init__(self, vector_store, model_name: str = "gemini-1.5-flash"):
        self.vector_store = vector_store
        self.model_name = model_name
        
        # Initialize the Gemini Chat Model
        self.client = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True # Gemini works better this way
        )
        
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            
        print(f"RAGPipeline initialized with model: {self.model_name}")

    def answer_question(
        self, 
        question: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        n_results: int = 3
    ) -> Dict[str, Any]:
        search_results = self.vector_store.query(question, n_results=n_results)
        
        if not search_results['documents'] or not search_results['documents'][0]:
            return {
                "answer": "I'm sorry, I couldn't find any relevant information in the company policies to answer your question.",
                "sources": [],
                "context": []
            }

        context_docs = search_results['documents'][0]
        source_metas = search_results['metadatas'][0]

        # Gemini works well with a combined system/user prompt
        prompt = f"""You are a helpful assistant that answers questions about company policies.
Use ONLY the provided context to answer the question. If the answer is not in the context, 
say "I don't have enough information to answer this question based on the provided documents."
Be concise and accurate. Cite your sources using the document name when possible.

Context from company policies:
{context_docs}

Question: {question}

Please provide a clear and concise answer based on the context above."""

        # LangChain handles the history and formatting for us
        messages = [("human", prompt)]

        print("Generating response from Gemini...")
        try:
            response = self.client.invoke(messages)
            answer = response.content
            
            # Extract source information
            sources = []
            for meta in source_metas:
                source = meta.get('source', 'Unknown Document')
                # Clean up the source path to just show filename
                if os.path.exists(source):
                    source = os.path.basename(source)
                sources.append(source)
            
            # Remove duplicates while preserving order
            sources = list(dict.fromkeys(sources))

            print(f"Response generated. Sources: {sources}")
            
            return {
                "answer": answer,
                "sources": sources,
                "context": context_docs
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "answer": f"I encountered an error while generating the response: {str(e)}",
                "sources": [],
                "context": context_docs
            }