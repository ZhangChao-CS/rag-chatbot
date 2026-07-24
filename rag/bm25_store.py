import jieba
import numpy as np
from rank_bm25 import BM25Okapi

import config


def build_bm25_index(chunks):
    tokenized = [list(jieba.cut(chunk.page_content)) for chunk in chunks]
    return BM25Okapi(tokenized)


def bm25_retrieve(query: str, bm25_index, chunks, k=None):
    k = k or config.BM25_K
    tokenized_query = list(jieba.cut(query))
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-k:][::-1]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append(
                {"document": chunks[idx], "score": scores[idx], "index": idx}
            )
    return results
