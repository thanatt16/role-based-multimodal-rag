import re
import chromadb

from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

from bm25 import create_bm25, bm25_search
from reranker import rerank


load_dotenv()

client = OpenAI()
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")


# =========================================================
# CHROMA DATABASE
# =========================================================

def get_collection():

    chroma_client = chromadb.PersistentClient(path=r"E:\Machine_learning\AI\RAG\PROJECT\chroma_db")

    collection = chroma_client.get_collection(name="multi_documents")

    return collection


# =========================================================
# LOAD CURRENT USER'S CHUNKS
# =========================================================

def load_corpus(collection, user_id):

    data = collection.get(where={"user_id": user_id},include=["documents", "metadatas"])

    chunks = []

    for chunk_id, content, metadata in zip(data["ids"],data["documents"],data["metadatas"]):
        chunks.append({"id": chunk_id,"content": content,"metadata": metadata})

    return chunks


# =========================================================
# SEMANTIC SEARCH
# =========================================================

def semantic_search(query, collection, user_id, n_results=20):

    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(query_embeddings=[query_embedding],n_results=n_results,where={"user_id": user_id})

    semantic_results = []

    for chunk_id, content, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        semantic_results.append({
            "id": chunk_id,
            "content": content,
            "metadata": metadata,
            "semantic_distance": float(distance)
        })

    return semantic_results


# =========================================================
# RECIPROCAL RANK FUSION
# =========================================================

def reciprocal_rank_fusion(semantic_results, bm25_results, k=60, top_k=10):

    rrf_scores = {}
    merged_results = {}

    # Semantic results
    for rank, result in enumerate(semantic_results, start=1):

        chunk_id = result["id"]

        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)

        if chunk_id not in merged_results:
            merged_results[chunk_id] = result.copy()
        else:
            merged_results[chunk_id].update(result)

    # BM25 results
    for rank, result in enumerate(bm25_results, start=1):

        chunk_id = result["id"]

        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)

        if chunk_id not in merged_results:
            merged_results[chunk_id] = result.copy()
        else:
            merged_results[chunk_id].update(result)

    ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    final_results = []

    for chunk_id in ranked_ids[:top_k]:

        result = merged_results[chunk_id]
        result["rrf_score"] = rrf_scores[chunk_id]
        final_results.append(result)

    return final_results


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        metadata = result["metadata"]

        source = metadata.get("source", "unknown")
        page = metadata.get("page")
        content_type = metadata.get("content_type", "unknown")

        source_info = f"[{i}] Source: {source}"

        if page:
            source_info += f", page {page}"

        source_info += f", content type: {content_type}"

        context_part = f"""
{source_info}

{result["content"]}
"""

        context_parts.append(context_part)

    return "\n\n---\n\n".join(context_parts)


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(query, context):

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=f"""
You are answering questions using the user's private knowledge base.

Rules:
- Answer only from the provided context.
- Do not use outside knowledge.
- Do not invent information.
- If the context does not contain enough information to answer the question, clearly say so.
- Cite factual claims using the citation numbers provided in the context.
- Use citations in the format [1], [2], etc.
- Place citations immediately after the specific claim they support.
- Place citations before the final punctuation, for example: Google authentication is supported [1].
- Do not group unrelated factual claims under one citation if they come from different sources.
- Never invent citation numbers.
- Use only citation numbers that exist in the provided context.
- If information comes from an image or chart and values are approximate, clearly state that they are estimates.
- Give a concise and direct answer.

QUESTION:
{query}

CONTEXT:
{context}
"""
    )

    return response.output_text


# =========================================================
# FULL RAG PIPELINE
# =========================================================

def run_rag(query, user_id):

    collection = get_collection()

    # Only chunks belonging to the current user
    chunks = load_corpus(collection, user_id)

    if not chunks:
        return {
            "answer": "You do not have any processed documents in your knowledge base yet.",
            "sources": []
        }

    # Create BM25 index only from current user's chunks
    bm25 = create_bm25(chunks)

    n_results = min(20, len(chunks))

    # Semantic retrieval
    semantic_results = semantic_search(query,collection,user_id,n_results=n_results)

    # Keyword retrieval
    bm25_results = bm25_search(query, bm25, chunks, n_results=n_results)

    # Merge both retrieval methods
    final_results = reciprocal_rank_fusion(semantic_results,bm25_results,top_k=10)

    if not final_results:
        return {
            "answer": "I could not find relevant information in your documents.",
            "sources": []
        }

    # Rerank best RRF candidates
    reranked_results = rerank(query,final_results,top_k=min(5, len(final_results)))

    # Build LLM context
    context = build_context(reranked_results)

    # Generate answer
    answer = generate_answer(query, context)

    # Find citation numbers actually used by the LLM
    citation_numbers = {
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer)
    }

    # Return only sources actually cited
    sources = []

    for i, result in enumerate(reranked_results, start=1):

        if i not in citation_numbers:
            continue

        metadata = result["metadata"]

        sources.append({
            "citation": i,
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "content_type": metadata.get("content_type"),
            "document_id": metadata.get("document_id")
        })

    return {"answer": answer,"sources": sources
}