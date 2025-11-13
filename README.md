# 🏢 RAG-based Company Policy Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about company policies accurately and efficiently, based solely on provided documents.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Project Architecture](#project-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Submission Details](#submission-details)

---

## Features

- ✅ **Document Ingestion**: Seamlessly loads and processes PDF and TXT files.
- ✅ **Intelligent Retrieval**: Uses advanced embedding models to find the most relevant document chunks for any query.
- ✅ **Context-Aware Answers**: Generates responses based strictly on retrieved information, preventing hallucination.
- ✅ **Source Citations**: Provides clear sources for every answer, allowing for easy verification.
- ✅ **Conversation Memory**: Maintains conversation history for a natural, multi-turn dialogue experience.
- ✅ **Modern UI**: Clean and intuitive chat interface built with Streamlit.

---

## Demo

*(Optional: Add a link to your demo video here)*

For a quick overview, watch the demo video showing the chatbot in action.

---

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

- **Python 3.9+**: Make sure you have Python 3.9 or newer installed.
- **Google Gemini API Key**: A valid API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- **Git**: Recommended for cloning the repository.

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/rag-policy-chatbot.git
    cd rag-policy-chatbot
    ```

2.  **Create and activate a virtual environment**
    ```bash
    # Create the virtual environment
    python -m venv .venv

    # Activate it (On Windows)
    .\.venv\Scripts\Activate.ps1

    # Activate it (On macOS/Linux)
    source .venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

This application requires a Google Gemini API key to function.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your API key to the `.env` file in the following format:
    ```
    GOOGLE_API_KEY=your_actual_google_gemini_api_key_here
    ```
    Replace `your_actual_google_gemini_api_key_here` with your key. **Do not wrap the key in quotes.**

---

## Running the Application

Once the installation and configuration are complete, you can start the application with a single command:

```bash
streamlit run app.py

# RAG Policy Chatbot

## Usage Guide
1. **Add Documents:** Place your company policy documents (PDF or TXT) into the `data/` directory.
2. **Process Documents:** In the sidebar, click the "🚀 Process Documents" button. This may take a minute.
3. **Ask Questions:** Once the system is ready, type your question into the chat input and press Enter.
4. **Review Answers:** The chatbot will provide a concise answer. Click "📖 Sources" to see which documents were used.

---

## Project Architecture

This application is built on the RAG (Retrieval-Augmented Generation) architecture.

┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────▶│   RAG Pipeline   │────▶│   LLM Response  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Vector Store   │
                       └─────────────────┘
                              ▲
                              │
                       ┌─────────────────┐
                       │ Document Store  │
                       └─────────────────┘


                       
- **Document Store:** Raw PDF/TXT files are stored in the `data/` directory.  
- **Vector Store:** When documents are processed, they are chunked and converted into numerical embeddings. These are stored in a FAISS vector database (`./faiss_index`).  
- **RAG Pipeline:** User queries are converted into embeddings; similarity search is performed in the vector store, and the most relevant chunks are sent to the LLM.  
- **LLM:** Google Gemini generates answers based only on the provided context, preventing hallucination.

---

## Technology Stack
- **Backend Framework:** Streamlit - For a fast and interactive web UI.  
- **LLM:** Google Gemini (gemini-1.5-flash) - For natural language generation.  
- **Embeddings:** Google Generative AI (embedding-001) - For converting text to vectors.  
- **Vector Database:** FAISS - For efficient and local vector similarity search.  
- **Orchestration:** LangChain - For simplifying the RAG pipeline components.

---

## Project Structure

rag-policy-chatbot/
│
├── data/                           # Storage for policy documents
│   ├── employee_handbook.pdf
│   ├── it_policy.pdf
│   └── remote_work_policy.txt
│
├── src/                            # Source code modules
│   ├── document_processor.py         # Handles loading and chunking of documents
│   ├── rag_pipeline.py              # Orchestrates the RAG process
│   └── vector_store.py              # Manages the FAISS vector database
│
├── .env                            # Environment variables (API key)
├── .gitignore                      # Files to ignore in Git
├── app.py                          # Main Streamlit application
├── README.md                       # This file
└── requirements.txt                # Python dependencies
