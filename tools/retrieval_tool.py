from tools.base_tool import BaseTool
from tools.schemas import RetrievalArgs


class RetrievalTool(BaseTool):
    def __init__(self, retrieval_service, rewrite_service=None, use_rerank=True):
        self.retrieval_service = retrieval_service
        self.rewrite_service = rewrite_service
        self.use_rerank = use_rerank

    @property
    def name(self):
        return "retrieval"

    @property
    def description(self):

        return "查询本地知识库，适用于PDF、论文和文档内容检索"

    @property
    def args_schema(self):

        return RetrievalArgs

    def run(self, **kwargs):
        query = kwargs["query"]

        if self.rewrite_service:
            query = self.rewrite_service.run(query)

        docs = self.retrieval_service.retrieve(query, use_rerank=self.use_rerank)

        return self.create_result(
            observation="\n".join(doc.page_content for doc in docs), raw=docs
        )
