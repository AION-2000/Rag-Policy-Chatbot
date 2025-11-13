import streamlit as st
import os
import shutil
from dotenv import load_dotenv

# Import our custom modules
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Company Policy Chatbot",
    page_icon="🏢",
    layout="centered"
)

# --- Initialize Session State Variables ---
# 'conversation_history' stores the chat messages
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# 'vector_store_initialized' checks if the DB is ready
if 'vector_store_initialized' not in st.session_state:
    st.session_state.vector_store_initialized = False

# --- Helper Functions ---
def initialize_vector_store():
    """Initializes the document processor, vector store, and RAG pipeline."""
    with st.spinner("Processing documents and initializing system... This may take a few minutes."):
        processor = DocumentProcessor()
        chunks = processor.load_and_split_documents("data")
        
        if not chunks:
            st.error("No documents were processed. Please check the 'data' directory for valid PDF or TXT files.")
            st.stop()
            
        vector_store = VectorStore()
        vector_store.add_documents(chunks)
        
        rag_pipeline = RAGPipeline(vector_store)
        
        st.session_state.rag_pipeline = rag_pipeline
        st.session_state.vector_store_initialized = True
        st.success("System is ready! You can now ask questions.")
        st.rerun()

def reset_system():
    """Resets the entire system by deleting the vector store and session state."""
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    if 'rag_pipeline' in st.session_state:
        del st.session_state.rag_pipeline
    st.session_state.vector_store_initialized = False
    st.session_state.conversation_history = []
    st.success("System has been reset. Please upload new documents to begin.")
    st.rerun()

# --- Sidebar for Document Management ---
with st.sidebar:
    st.header("📄 Document Management")
    
    # Check if the data directory has files
    data_dir_exists = os.path.exists("data")
    files_in_data = os.listdir("data") if data_dir_exists else []
    
    if not st.session_state.vector_store_initialized:
        if files_in_data:
            st.info(f"Found {len(files_in_data)} file(s) in the 'data' directory.")
            if st.button("🚀 Process Documents", type="primary"):
                initialize_vector_store()
        else:
            st.warning("No documents found in the 'data' directory.")
            st.info("Please add PDF or TXT files to the 'data' folder and restart the app.")
    else:
        st.success("Vector database is initialized and ready.")
        if st.button("🗑️ Reset System"):
            reset_system()

# --- Main Chat Interface ---
st.title("🏢 Company Policy Chatbot")
st.markdown("Ask questions about company policies and get accurate answers based on the provided documents.")

# Display conversation history
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📖 Sources"):
                st.write(message["sources"])

# User input field
if st.session_state.vector_store_initialized:
    user_input = st.chat_input("Ask a question about company policies...")
    
    if user_input:
        # Add user message to conversation history and display it
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get conversation history without the latest user message for context
                history_for_llm = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.conversation_history[:-1]
                ]
                
                response = st.session_state.rag_pipeline.answer_question(
                    user_input, 
                    conversation_history=history_for_llm
                )
                
                st.markdown(response["answer"])
                
                with st.expander("📖 Sources"):
                    st.write(response["sources"])
        
        # Add assistant response to conversation history
        st.session_state.conversation_history.append({
            "role": "assistant", 
            "content": response["answer"],
            "sources": response["sources"]
        })
else:
    st.info("Please add documents to the 'data' directory and click 'Process Documents' to start chatting.")
