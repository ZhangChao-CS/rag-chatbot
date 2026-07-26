from tools.base_tool import BaseTool

class RetrievalTool(BaseTool):
    def __init__(self, retrieval_service):
        self.retrieval_service = retrieval_service 

    @property
    def name(self):
        return "retrieval"

    def run(self, query, use_rerank=True):
        return self.retrieval_service.retrieve(query, use_rerank=use_rerank)
