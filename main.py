import os
from src.document_parser import extract_advanced_content, chunk_text
from src.database import store_chunks_in_chroma
from src.search import search_database
from src.generator import generate_answer

def ingest_document(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}")
        return

    print(f"Reading {pdf_path}...")
    raw_text = extract_advanced_content(pdf_path)
    
    print("Chunking text...")
    chunks = chunk_text(raw_text, chunk_size=1000, overlap=200)
    
    store_chunks_in_chroma(chunks)

if __name__ == "__main__":
    target_pdf = "data/BAD601-module-5-pdf.pdf" 
    
    # 1. INGESTION PHASE
    if not os.path.exists("./my_local_vectordb"):
        print("Database not found. Starting ingestion pipeline...")
        ingest_document(target_pdf)
    else:
        print("Database found. Skipping ingestion.")
        
    # 2. RETRIEVAL PHASE
    user_question = "What are the main objectives discussed in this module?" 
    print(f"\nUser: {user_question}")
    
    retrieved_chunks = search_database(user_question, n_results=3)
    
    if not retrieved_chunks:
        print("No relevant information found in the database.")
    else:
        # 3. GENERATION PHASE
        final_answer = generate_answer(user_question, retrieved_chunks)
        
        print("\n--- Final AI Answer ---")
        print(final_answer)