from sentence_transformers import CrossEncoder

# Initialize the cross-encoder globally so it does not reload on every query.
# ms-marco-MiniLM-L-6-v2 is an industry standard for RAG re-ranking.
print("Loading Cross-Encoder model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def search_and_rerank(query: str, collection, initial_k: int = 30, final_k: int = 5) -> list[dict]:
    """
    Retrieves a broad set of candidates using a fast Bi-Encoder, 
    then scores and filters them using an accurate Cross-Encoder.
    """
    
    # 1. Ask ChromaDB for documents AND metadatas
    print(f"Fetching top {initial_k} candidates from ChromaDB...")
    results = collection.query(
        query_texts=[query],
        n_results=initial_k,
        include=["documents", "metadatas"] 
    )
    
    # Ensure results exist to prevent index out of bounds errors
    if not results or not results['documents'] or not results['documents'][0]:
        return []
        
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    # 2. Re-rank
    print(f"Re-ranking {len(documents)} candidates...")
    
    # Format the input exactly how the Cross-Encoder expects it: a list of [query, chunk] pairs
    sentence_pairs = [[query, doc] for doc in documents]
    
    # Predict returns a NumPy array of scores
    scores = cross_encoder.predict(sentence_pairs)
    
    # Zip scores, text, AND metadata together before sorting.
    # We must cast scores back to standard Python floats for easier handling, 
    # though zip will naturally handle the numpy array elements.
    scored_docs = zip(scores, documents, metadatas)
    
    # Sort by the score (the first element in the zipped tuple) in descending order
    sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    
    # Optional: Print the scores to the terminal so you can verify it is working
    print("\n--- Cross-Encoder Re-ranker Scores ---")
    for i, (score, doc, meta) in enumerate(sorted_docs[:final_k]):
        # doc[:50] gets the first 50 characters, replace cleans up newlines for the print
        snippet = doc[:50].replace('\n', ' ')
        print(f"Rank {i+1} | Score: {score:.2f} | Page: {meta.get('page')} | Snippet: {snippet}...")
    print("--------------------------------------\n")
    
    # 3. Extract and return ONLY the true top chunks as dictionaries
    top_chunks = [{"text": doc, "metadata": meta} for score, doc, meta in sorted_docs[:final_k]]
    
    return top_chunks