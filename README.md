# Local Advanced PDF RAG Pipeline

This repository contains a fully local Retrieval-Augmented Generation (RAG) pipeline designed for complex PDF documents. It extracts text, tables, and images, strictly tracks page-level metadata for citations, and utilizes a two-stage retrieval system (Bi-Encoder + Cross-Encoder) to prevent context loss.

## System Architecture

This pipeline is built entirely on local models to ensure data privacy. It relies on a FastAPI backend and a Streamlit frontend.

- **Extraction:** `pdfplumber` (Text & Tables), `PyMuPDF` (Image extraction)
- **Vision Model:** `Llava` (via Ollama) for generating semantic descriptions of embedded diagrams
- **Chunking:** `LangChain` (`RecursiveCharacterTextSplitter`) for semantic paragraph handling
- **Vector Database:** `ChromaDB` using `all-mpnet-base-v2` for rapid Bi-Encoder candidate retrieval
- **Re-Ranker:** `ms-marco-MiniLM-L-6-v2` (Cross-Encoder) for high-accuracy semantic sorting
- **LLM Engine:** `Mistral` or `Llama 3` (via Ollama) with strict prompt constraints for zero-hallucination page citations

---

## Prerequisites

Before running this software, ensure the following are installed on your machine:

1. **Python 3.9+**
2. **Ollama** – Download and install from: https://ollama.com/
3. **Required Ollama Models**

```bash
ollama pull llava
ollama pull mistral
```

> **Note:** If `mistral` does not perform well with strict prompt constraints, you can replace it with `llama3` or `qwen2.5` in `src/generator.py`.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SujithMH/RAG
cd RAG
```

### 2. Install the Required Python Dependencies

```bash
pip install -r requirements.txt
```

---

# Project Structure

```
.
├── app.py                    # FastAPI backend with /upload and /chat REST APIs
├── frontend.py               # Streamlit frontend for PDF upload and chat
├── requirements.txt          # Python dependencies
├── data/                     # Temporary storage for uploaded PDF files
├── my_local_vectordb/        # Persistent ChromaDB vector database
├── src/
│   ├── document_parser.py    # PDF parsing, image extraction, and chunking
│   ├── database.py           # ChromaDB embedding and persistence logic
│   ├── search.py             # Bi-Encoder retrieval + Cross-Encoder re-ranking
│   └── generator.py          # Prompt construction and Ollama LLM inference
└── README.md
```

### File Descriptions

| File/Folder | Description |
|-------------|-------------|
| `app.py` | FastAPI backend exposing the `/upload` and `/chat` REST endpoints with CORS enabled. |
| `frontend.py` | Streamlit user interface for uploading PDFs and interacting with the chatbot. |
| `src/document_parser.py` | Extracts text, tables, and images from PDFs and performs semantic chunking using LangChain. |
| `src/database.py` | Generates embeddings and manages persistent storage using ChromaDB. |
| `src/search.py` | Retrieves relevant document chunks using a Bi-Encoder and improves ranking using a Cross-Encoder. |
| `src/generator.py` | Builds the LLM prompt with retrieved context and queries Ollama for the final response. |
| `data/` | Temporary directory where uploaded PDF files are stored before processing. |
| `my_local_vectordb/` | Persistent directory containing the ChromaDB vector embeddings. |

---

# Running the Application

This project requires both the **FastAPI backend** and the **Streamlit frontend** to be running simultaneously.

## 1. Start the Backend Server

Open a terminal in the project root directory and run:

```bash
uvicorn app:app --reload
```

The backend API will be available at:

- API Base URL: `http://127.0.0.1:8000`
- Swagger Documentation: `http://127.0.0.1:8000/docs`

---

## 2. Launch the Streamlit Frontend

Open a **new terminal window**, navigate to the same project directory, and execute:

```bash
streamlit run frontend.py
```

Streamlit will automatically launch the application in your default web browser (typically at):

```
http://localhost:8501
```

---

## Workflow

1. Start the FastAPI backend.
2. Launch the Streamlit frontend.
3. Upload one or more PDF documents.
4. The system extracts text, tables, and image descriptions.
5. Documents are chunked and stored in ChromaDB.
6. Ask questions in the chat interface.
7. The pipeline retrieves the most relevant chunks using:
   - Bi-Encoder retrieval
   - Cross-Encoder re-ranking
8. Ollama generates a grounded answer with page-level citations.

---

## Features

- Fully local RAG pipeline
- PDF text, table, and image extraction
- Automatic image captioning using Llava
- Semantic chunking with LangChain
- ChromaDB persistent vector storage
- Two-stage retrieval (Bi-Encoder + Cross-Encoder)
- FastAPI REST backend
- Interactive Streamlit frontend
- Strict page-level citation support
- Offline and privacy-focused architecture
