from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")


def create_embeddings(chunks):

    chunk_texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(chunk_texts,normalize_embeddings=True)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    print("Number of embeddings:", len(embeddings))
    print("Embedding dimension:", len(embeddings[0]))

    return chunks