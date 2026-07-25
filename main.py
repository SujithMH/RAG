import os
from src.document_parser import extract_text_from_pdf, chunk_text
from src.database import store_chunks_in_chroma
from src.search import search_database

def ingest_document(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}")
        return

    print(f"Reading {pdf_path}...")
    raw_text = extract_text_from_pdf(pdf_path)
    
    print("Chunking text...")
    chunks = chunk_text(raw_text, chunk_size=1000, overlap=200)
    
    store_chunks_in_chroma(chunks)

if __name__ == "__main__":
    target_pdf = "data/BAD601-module-5-pdf.pdf" 
    
    # 1. Only run ingestion if the database folder does not exist yet
    if not os.path.exists("./my_local_vectordb"):
        print("Database not found. Starting ingestion pipeline...")
        ingest_document(target_pdf)
    else:
        print("Database found. Skipping ingestion.")
        
    # 2. Test the search functionality
    # CHANGE THIS to a question that can actually be answered by your BAD601 PDF
    user_question = "What are the main objectives discussed in this module?" 
    
    print(f"\nSearching for: '{user_question}'")
    retrieved_chunks = search_database(user_question, n_results=3)
    
    print("\n--- Top 3 Retrieved Chunks ---")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\n[Result {i+1}]:\n{chunk}")
        print("-" * 50)