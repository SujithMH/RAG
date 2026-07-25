import chromadb
from sentence_transformers import SentenceTransformer

def search_database(query: str, n_results: int = 3, db_path: str = "./my_local_vectordb", collection_name: str = "pdf_knowledge_base") -> list[str]:
    # 1. Load the exact same embedding model
    print("Loading model for query embedding...")
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # 2. Convert the user's string query into a 384-dimensional vector
    query_vector = model.encode([query])
    
    # 3. Connect to the existing local database
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    
    # 4. Execute the cosine similarity search
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=n_results
    )
    
    # 5. Extract and return just the text documents
    if not results['documents'] or not results['documents'][0]:
        return []
        
    return results['documents'][0]