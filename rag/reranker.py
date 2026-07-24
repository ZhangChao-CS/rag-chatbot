from functools import lru_cache

from FlagEmbedding import FlagReranker

import config
from rag.utils import cleanup_memory


@lru_cache(maxsize=1)
def get_reranker():
    return FlagReranker(config.RERANKER_MODEL, use_fp16=True)


def rerank(query: str, docs, top_k=None):
    top_k = top_k or config.FINAL_TOP_K
    if not docs:
        return docs

    docs = docs[: config.RERANK_TOP_K]
    try:
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.compute_score(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        cleanup_memory()
        return [doc for doc, _ in ranked][:top_k]
    except Exception:
        return docs[:top_k]
