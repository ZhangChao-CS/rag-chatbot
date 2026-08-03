import os

from tavily import TavilyClient

from tools.base_tool import BaseTool
from tools.schemas import WebSerchArgs


class WebSearchTool(BaseTool):
    def __init__(self):

        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    @property
    def name(self):

        return "web_search"

    @property
    def description(self):

        return """
        网络搜索工具。

        适用于：
        - 最新新闻
        - 实时信息
        - 最新技术资料
        - 网络公开信息查询

        不适用于：
        - 本地PDF文档查询
        - 已知知识库查询
        """

    @property
    def args_schema(self):
        return WebSerchArgs

    def run(self, **kwargs):
        query = kwargs["query"]

        response = self.client.search(
            query=query, search_depth="advanced", max_results=5
        )

        results = response["results"]

        observation = ""

        for idx, item in enumerate(results, start=1):
            observation += f"""
            
            文档 {idx}

            标题:
            {item["title"]}

            内容:
            {item["content"]}

            来源:
            {item["url"]}

            --------------------
            """

        return {"observation": observation, "raw": results}
