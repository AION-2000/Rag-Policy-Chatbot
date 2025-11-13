import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """
    Handles loading and splitting of documents from a specified directory.
    Supports PDF and TXT files.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        print(f"DocumentProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def load_and_split_documents(self, directory_path: str) -> List:
        if not os.path.exists(directory_path):
            print(f"Error: Directory '{directory_path}' not found.")
            return []

        print(f"Loading documents from: {directory_path}")
        
        # Load PDFs
        pdf_loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        
        # Load TXTs
        txt_loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )

        pdf_docs = pdf_loader.load()
        txt_docs = txt_loader.load()
        
        all_docs = pdf_docs + txt_docs
        
        if not all_docs:
            print("No documents were loaded. Please check the directory and file types.")
            return []

        print(f"Loaded {len(all_docs)} documents. Splitting into chunks...")
        chunks = self.text_splitter.split_documents(all_docs)
        print(f"Successfully split documents into {len(chunks)} chunks.")
        
        return chunks