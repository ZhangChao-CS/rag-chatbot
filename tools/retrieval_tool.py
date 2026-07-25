class RetrievalTool:

    def __init__(self, retrieval_service):
        self.retrieval_service = retrieval_service

    def run(self, query, use_rerank=True):
        return self.retrieval_service.retrieve(query, use_rerank=True)
