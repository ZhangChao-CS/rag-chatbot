import os
import time
from typing import Optional

from tavily import TavilyClient

from agent.runtime.retry_utils import is_transient_error
from tools.base_tool import BaseTool
from tools.schemas import WebSearchArgs

MAX_ATTEMPTS = 3
RETRY_DELAY_SEC = 1.5


class WebSearchTool(BaseTool):
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("未设置 TAVILY_API_KEY 环境变量")
        self.client = TavilyClient(api_key=api_key)

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
        return WebSearchArgs

    def run(self, **kwargs):
        query = kwargs["query"]
        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5,
                )
                results = response["results"]
                observation = self._format_results(results)
                return self.create_result(observation=observation, raw=results)

            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < MAX_ATTEMPTS and is_transient_error(e):
                    delay = RETRY_DELAY_SEC * attempt
                    print(
                        f"[WebSearch] 网络错误，{delay:.1f}s 后重试 "
                        f"({attempt}/{MAX_ATTEMPTS}): {e}"
                    )
                    time.sleep(delay)
                    continue
                break

        raise RuntimeError(
            f"网络搜索失败（已重试 {MAX_ATTEMPTS} 次）: {last_error}"
        ) from last_error

    @staticmethod
    def _format_results(results: list) -> str:
        if not results:
            return "未找到相关结果"

        parts = []
        for idx, item in enumerate(results, start=1):
            parts.append(
                f"文档 {idx}\n"
                f"标题: {item['title']}\n"
                f"内容: {item['content']}\n"
                f"来源: {item['url']}\n"
                f"--------------------"
            )
        return "\n".join(parts)
