import ollama

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    
    context = ""
    for chunk in retrieved_chunks:
        text = chunk["text"]
        source = chunk["metadata"]["source"]
        page = chunk["metadata"]["page"]
        # Inject the metadata tags right into the LLM's context window
        context += f"[Source: {source} | Page: {page}]\n{text}\n\n"
    
    prompt = f"""You are a strict, factual expert. Answer the user's question using ONLY the context provided below. 

CRITICAL RULES:
1. You must explicitly cite the Source and Page number for every fact you state.
2. If the context only provides examples (e.g., a "Vacuum Cleaner Agent") but does not provide the actual definition or overarching types requested by the user, YOU MUST STATE: "The provided document only contains specific examples, not the formal definitions." Do not try to reverse-engineer a definition from an example.
3. If the answer cannot be found in the context at all, do not guess. State that the information is missing.

Context:
{context}

Question:
{query}
"""

    print("Generating answer via phi4-mini...")
    response = ollama.chat(
        model='phi4-mini', 
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return response['message']['content']