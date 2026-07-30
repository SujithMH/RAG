import os
from src.document_parser import extract_advanced_content, chunk_text
from src.database import store_chunks_in_chroma, get_chroma_collection
from src.search import search_and_rerank
from src.generator import generate_answer

def ingest_document(pdf_path: str):
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}")
        return None

    print(f"Reading {pdf_path}...")
    # Returns list[dict] with page-level text and metadata
    raw_pages = extract_advanced_content(pdf_path)
    
    print("Chunking text with metadata...")
    # Takes list[dict] and returns list[dict] of chunks
    chunks = chunk_text(raw_pages, chunk_size=1000, overlap=200)
    
    # Store chunks in ChromaDB and return the collection
    collection = store_chunks_in_chroma(chunks)
    return collection

if __name__ == "__main__":
    target_pdf = "data/BAD601-module-5-pdf.pdf" 
    
    # 1. INGESTION PHASE
    if not os.path.exists("./my_local_vectordb"):
        print("Database not found. Starting ingestion pipeline...")
        collection = ingest_document(target_pdf)
    else:
        print("Database found. Skipping ingestion.")
        # Load the existing collection from disk
        collection = get_chroma_collection()

    if collection is None:
        print("Error: Could not load or initialize ChromaDB collection.")
        exit(1)

    # 2. RETRIEVAL PHASE
    query = "What are the main objectives discussed in this module?"
    print(f"\nUser Query: {query}")
    
    # Search and re-rank candidates
    retrieved_chunks = search_and_rerank(query, collection, initial_k=15, final_k=3)
    
    if not retrieved_chunks:
        print("No relevant information found in the database.")
    else:
        # 3. GENERATION PHASE
        final_answer = generate_answer(query, retrieved_chunks)
        print("\nFinal Answer:\n", final_answer)