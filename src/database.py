import chromadb
from chromadb.utils import embedding_functions

DB_PATH = "./my_local_vectordb"
COLLECTION_NAME = "pdf_chunks"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

def get_chroma_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME, 
        embedding_function=embedding_fn
    )

def store_chunks_in_chroma(chunks: list[dict]):
    collection = get_chroma_collection()

    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"page": chunk["page"], "source": chunk["source"]} for chunk in chunks]
    ids = [f"id_{i}" for i in range(len(chunks))]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB!")
    return collection