import ollama

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    
    context = ""
    for chunk in retrieved_chunks:
        text = chunk["text"]
        source = chunk["metadata"]["source"]
        page = chunk["metadata"]["page"]
        # Inject the metadata tags right into the LLM's context window
        context += f"[Source: {source} | Page: {page}]\n{text}\n\n"
    
    prompt = f"""You are a helpful expert. Answer the user's question using ONLY the context provided below. 
If the answer cannot be found in the context, do not guess. 

CRITICAL RULE: You must explicitly cite the Source and Page number for every fact you state in your answer.

Context:
{context}

Question:
{query}
"""

    print("Generating answer via llava...")
    response = ollama.chat(
        model='llava', 
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return response['message']['content']