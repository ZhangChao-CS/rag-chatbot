import json
import re

from rag.llm import ask_llm

from .prompt import REFLECTION_PROMPT
from .schema import ReflectionResult


class Reflection:
    def __init__(self):

        pass

    def run(
        self,
        question: str,
        observation: str,
    ) -> ReflectionResult:

        prompt = REFLECTION_PROMPT.format(
            question=question,
            observation=observation,
        )

        response = ask_llm(prompt)

        data = self._parse_response(response)

        return ReflectionResult(**data)

    def _parse_response(self, response: str):

        # 去Markdown

        response = response.replace("```json", "").replace("```", "").strip()

        # 提取JSON

        match = re.search(r"\{.*\}", response, re.S)

        if match is None:
            raise ValueError(f"Reflection输出不是JSON：\n{response}")

        data = json.loads(match.group())

        return data
