import chromadb


def create_vector_db(chunks):

    chroma_client = chromadb.PersistentClient(
        path=r"E:\Machine_learning\AI\RAG\PROJECT\chroma_db"
    )

    collection = chroma_client.get_or_create_collection(
        name="multi_documents",
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )

    ids = []

    for i, chunk in enumerate(chunks):

        document_id = chunk["metadata"].get("document_id")

        if document_id:
            chunk_id = f"{document_id}_{i}"
        else:
            chunk_id = f"chunk_{i}"

        ids.append(chunk_id)

    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks]
    )

    print("Documents stored in ChromaDB:", collection.count())

    return collection