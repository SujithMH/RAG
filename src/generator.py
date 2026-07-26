import ollama

def generate_answer(query: str, retrieved_chunks: list[str]) -> str:
    # 1. Combine the retrieved list of strings into one massive text block
    context = "\n\n".join(retrieved_chunks)
    
    # 2. Build the strict prompt
    prompt = f"""You are a helpful expert. Answer the user's question using ONLY the context provided below. 
If the answer cannot be found in the context, do not guess. Simply state "I cannot answer this based on the provided document."

Context:
{context}

Question:
{query}
"""

    print("Generating answer via mistral...")
    
    # 3. Send the prompt to your local Llama 3 model
    response = ollama.chat(
        model='mistral', 
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return response['message']['content']