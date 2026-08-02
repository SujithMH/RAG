import streamlit as st
import requests

# The URL where your FastAPI server is running
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="PDF RAG Chat", layout="centered")
st.title("📄 PDF RAG Assistant")

# --- SIDEBAR: UPLOAD PIPELINE ---
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if st.button("Upload & Ingest"):
        if uploaded_file is not None:
            with st.spinner("Parsing, extracting images, and storing vectors..."):
                # Prepare the file for a multipart HTTP POST request
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Success! Stored {data.get('chunks_stored')} chunks from {data.get('filename')}.")
                    else:
                        st.error(f"API Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is FastAPI running on port 8000?")
        else:
            st.warning("You must select a PDF file first.")

# --- MAIN CHAT INTERFACE ---
st.header("2. Chat")

# Initialize chat history in session state so it doesn't reset on every button click
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    # Immediately display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the FastAPI chat endpoint
    with st.chat_message("assistant"):
        with st.spinner("Searching database and generating response..."):
            try:
                response = requests.post(f"{API_URL}/chat", json={"query": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    sources = data.get("sources", [])
                    
                    # Format the final output to display the answer and a list of sources
                    full_response = answer
                    if sources:
                        full_response += "\n\n**Sources Used:**\n"
                        # Use a set to remove duplicate source citations
                        unique_sources = set([f"- {s.get('source')} (Page {s.get('page')})" for s in sources])
                        full_response += "\n".join(unique_sources)
                        
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Is FastAPI running on port 8000?")