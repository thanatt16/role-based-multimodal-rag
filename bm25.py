from rank_bm25 import BM25Okapi
import re



def tokenize(text):

    return re.findall(r"\b\w+\b", text.lower())


def create_bm25(chunks):

    tokenized_corpus = [
        tokenize(chunk["content"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


def bm25_search(query, bm25, chunks, n_results=20):

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = scores.argsort()[::-1][:n_results]

    results = []

    for index in ranked_indices:

        if scores[index] <= 0:
            continue

        results.append({
            "id": chunks[index]["id"],
            "content": chunks[index]["content"],
            "metadata": chunks[index]["metadata"],
            "bm25_score": float(scores[index])
        })

    return results