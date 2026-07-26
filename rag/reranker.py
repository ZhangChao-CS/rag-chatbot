import platform

from FlagEmbedding import FlagReranker

import config
from rag.utils import cleanup_memory

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        # Mac 无 CUDA，fp16 在 CPU 上无收益且可能增加兼容问题
        use_fp16 = platform.system() != "Darwin"
        _reranker = FlagReranker(config.RERANKER_MODEL, use_fp16=use_fp16)
    return _reranker


def unload_reranker():
    global _reranker
    if _reranker is not None:
        del _reranker
        _reranker = None
        cleanup_memory()


def rerank(query: str, docs, top_k=None):
    top_k = top_k or config.FINAL_TOP_K
    if not docs:
        return docs

    docs = docs[: config.RERANK_TOP_K]
    try:
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in docs]
        batch_size = config.RERANK_BATCH_SIZE
        if len(pairs) <= batch_size:
            scores = reranker.compute_score(pairs)
        else:
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                scores.extend(reranker.compute_score(batch))
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked][:top_k]
    except Exception:
        return docs[:top_k]
    finally:
        unload_reranker()
