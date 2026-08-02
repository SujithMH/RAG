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
    raw_pages = extract_advanced_content(pdf_path)
    
    print("Chunking text with metadata...")
    chunks = chunk_text(raw_pages, chunk_size=1000, overlap=200)
    
    # This function must append to the existing DB, not overwrite it.
    collection = store_chunks_in_chroma(chunks)
    return collection

if __name__ == "__main__":
    target_pdf = "data/BAD601-module-5-pdf.pdf" 
    filename = os.path.basename(target_pdf)
    
    # 1. DATABASE INITIALIZATION
    # We must load the collection first to inspect its contents.
    collection = get_chroma_collection()

    if collection is None:
        print("Error: Could not load or initialize ChromaDB collection.")
        exit(1)

    # 2. INGESTION PHASE (Metadata Check)
    # Check if the database already contains chunks from this specific source file.
    existing_docs = collection.get(where={"source": filename}, limit=1)
    
    if not existing_docs or not existing_docs.get('ids'):
        print(f"'{filename}' not found in database. Starting ingestion...")
        collection = ingest_document(target_pdf)
    else:
        print(f"'{filename}' is already in the database. Skipping ingestion.")

    # 3. RETRIEVAL PHASE
    query = "What are the main objectives discussed in this module?"
    print(f"\nUser Query: {query}")
    
    # Search and re-rank candidates
    retrieved_chunks = search_and_rerank(query, collection, initial_k=15, final_k=3)
    
    if not retrieved_chunks:
        print("No relevant information found in the database.")
    else:
        # 4. GENERATION PHASE
        final_answer = generate_answer(query, retrieved_chunks)
        print("\nFinal Answer:\n", final_answer)