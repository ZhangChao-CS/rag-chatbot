from rag.llm import ask_llm

SUMMARIZE_PROMPT = """
将以下工具输出压缩为简短摘要（80字以内），保留关键事实（名称、数字、价格）：

{observation}

只输出摘要文本，不要 JSON，不要解释。
"""


class ObservationSummarizer:
    """Observation → Summary，供 Reflection 使用，避免 Prompt 膨胀。"""

    MAX_CHARS_BEFORE_SUMMARIZE = 500

    def summarize(self, observation: str) -> str:
        if not observation:
            return ""
        if len(observation) <= self.MAX_CHARS_BEFORE_SUMMARIZE:
            return observation.strip()

        try:
            response = ask_llm(
                SUMMARIZE_PROMPT.format(observation=observation[:3000]),
                max_tokens=150,
            )
            return response.strip()
        except Exception:  # noqa: BLE001
            return observation[:500] + "...(truncated)"
