import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Import your existing pipeline functions
from src.document_parser import extract_advanced_content, chunk_text
from src.database import store_chunks_in_chroma, get_chroma_collection
from src.search import search_and_rerank
from src.generator import generate_answer

app = FastAPI(title="RAG PDF API")

# Ensure the upload directory exists
os.makedirs("data", exist_ok=True)

# Define the data structure for the chat endpoint
class ChatRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_and_ingest(file: UploadFile = File(...)):
    """Accepts a PDF upload, saves it, and runs the RAG ingestion pipeline."""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    file_path = f"data/{file.filename}"
    
    # 1. Save the uploaded file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    print(f"Processing uploaded file: {file.filename}...")
    
    # 2. Run the pipeline on the new file
    try:
        raw_pages = extract_advanced_content(file_path)
        chunks = chunk_text(raw_pages, chunk_size=1000, overlap=200)
        store_chunks_in_chroma(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
        
    return {
        "status": "success",
        "filename": file.filename,
        "chunks_stored": len(chunks)
    }

@app.post("/chat")
async def chat_with_document(request: ChatRequest):
    """Takes a user query, searches ChromaDB, and generates an answer via Mistral."""
    
    collection = get_chroma_collection()
    
    # Run the retrieval and re-ranking
    retrieved_chunks = search_and_rerank(request.query, collection, initial_k=30, final_k=5)    
    if not retrieved_chunks:
        return {"answer": "No relevant context found in the uploaded documents."}
        
    # Generate the answer
    final_answer = generate_answer(request.query, retrieved_chunks)
    
    return {
        "answer": final_answer,
        "sources": [chunk["metadata"] for chunk in retrieved_chunks]
    }