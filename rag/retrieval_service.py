import config
from rag.bm25_store import bm25_retrieve
from rag.reranker import rerank
from rag.retriever import reciprocal_rank_fusion

class RetrievalService:

    def __init__(self, db, chunks, bm25_index):
        self.db = db
        self.chunks = chunks
        self.bm25_index = bm25_index

    def retrieve(self, query, use_rerank=True):
        vector_docs = self.db.similarity_search_with_score(query, k=config.VECTOR_K)
        bm25_results = bm25_retrieve(query, self.bm25_index, self.chunks)
        fused = reciprocal_rank_fusion(vector_docs, bm25_results)
        docs = [doc for doc, _ in fused][: config.VECTOR_K]

        if use_rerank and docs:
            return rerank(query, docs)
        return docs[: config.FINAL_TOP_K]
