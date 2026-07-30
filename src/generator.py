import ollama

def generate_answer(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""You are a helpful expert. Answer the user's question using ONLY the context provided below. 
If the answer cannot be found in the context, do not guess. Simply state "I cannot answer this based on the provided document."

Context:
{context}

Question:
{query}
"""

    print("Generating answer via llava...")
    
    # Change the model name here
    response = ollama.chat(
        model='llava', 
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    return response['message']['content']