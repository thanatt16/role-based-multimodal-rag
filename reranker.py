from sentence_transformers import CrossEncoder


reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")


def rerank(query, results, top_k=5):

    pairs = [
        (query, result["content"])
        for result in results
    ]

    scores = reranker_model.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    ranked_results = sorted(
        results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked_results[:top_k]
