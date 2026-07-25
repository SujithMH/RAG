import chromadb
from sentence_transformers import SentenceTransformer

def store_chunks_in_chroma(chunks: list[str], db_path: str = "./my_local_vectordb", collection_name: str = "pdf_knowledge_base"):
    # 1. Initialize the embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # 2. Convert chunks to vectors
    print(f"Embedding {len(chunks)} chunks. This might take a moment...")
    chunk_vectors = model.encode(chunks)
    
    # 3. Initialize ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    
    # 4. Generate unique IDs and insert
    chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        embeddings=chunk_vectors.tolist(),
        ids=chunk_ids
    )
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB at {db_path}!")