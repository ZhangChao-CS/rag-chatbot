import config
from rag.bm25_store import bm25_retrieve
from rag.reranker import rerank
from rag.utils import doc_id


def reciprocal_rank_fusion(vector_results, bm25_results, k=None):
    k = k or config.RRF_K
    doc_scores = {}

    for rank, (doc, _score) in enumerate(vector_results, 1):
        did = doc_id(doc)
        doc_scores[did] = doc_scores.get(did, 0) + 1 / (k + rank)

    for rank, result in enumerate(bm25_results, 1):
        did = doc_id(result["document"])
        doc_scores[did] = doc_scores.get(did, 0) + 1 / (k + rank)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    doc_dict = {}
    for doc, _ in vector_results:
        doc_dict[doc_id(doc)] = doc
    for result in bm25_results:
        doc_dict[doc_id(result["document"])] = result["document"]

    final_results = []
    for did, score in sorted_docs[: config.FUSED_TOP_K]:
        if did in doc_dict:
            final_results.append((doc_dict[did], score))
    return final_results


def retrieve_with_multi_recall(query, db, chunks, bm25_index, use_rerank=True):
    vector_docs = db.similarity_search_with_score(query, k=config.VECTOR_K)
    bm25_results = bm25_retrieve(query, bm25_index, chunks)
    fused = reciprocal_rank_fusion(vector_docs, bm25_results)
    docs = [doc for doc, _ in fused][: config.VECTOR_K]

    if use_rerank and docs:
        return rerank(query, docs)
    return docs[: config.FINAL_TOP_K]
