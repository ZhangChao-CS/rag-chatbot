from typing import Any, Dict, Optional

from agent.planning.schema import Plan, Task, TaskResult, TaskStatus, TaskType
from agent.runtime.price_utils import (
    extract_price_metadata,
    format_price_unit,
    infer_currency,
    infer_price_period,
    normalize_price_field,
)
from rag.llm import ask_llm
from rag.utils import parse_llm_json

BUILD_OUTPUT_PROMPT = """
从 Observation 摘要中提取结构化 TaskResult.output（不要复述全文）。

任务目标：{description}
使用的 Tool：{tool}
Observation 摘要：
{summary}

输出 JSON：
{{
  "summary": "一句话结论（30字内）",
  "output": {{
    "model": "",
    "price": 0.0,
    "currency": "",
    "price_period": "",
    "entities": [],
    "comparisons": [],
    "facts": {{}}
  }}
}}

规则：
- 只填摘要中明确出现的信息，禁止编造
- price 必须是纯数字
- 若涉及价格，必须填写 currency（如 美元/USD/人民币）和 price_period（如 月/年）
- 论文/RAG 对比类：entities 填 [{{name, innovation, ...}}]，comparisons 填差异点
- 只保留与任务目标相关的字段，无关字段可省略
"""


class ResultBuilder:
    """Summary → TaskResult（Reflection 阶段调用，不读 Raw Observation）。"""

    def build(
        self,
        task: Task,
        summary: str,
        tool: str,
        expression: str = "",
        plan: Optional[Plan] = None,
    ) -> TaskResult:
        if not summary or not summary.strip():
            return TaskResult(status=TaskStatus.FAILED, summary="无有效输出", output={})

        if task.task_type == TaskType.COMPUTATION:
            return self._build_computation_result(task, plan, summary, expression)

        return self._build_information_result(task, summary, tool)

    def _build_computation_result(
        self,
        task: Task,
        plan: Optional[Plan],
        summary: str,
        expression: str,
    ) -> TaskResult:
        try:
            value = float(summary.strip())
        except ValueError:
            import re
            match = re.search(r"[-+]?\d*\.?\d+", summary)
            value = float(match.group()) if match else None

        if value is None:
            return TaskResult(status=TaskStatus.FAILED, summary=summary[:100], output={})

        output: Dict[str, Any] = {"value": value}
        if expression:
            output["expression"] = expression

        if plan:
            deps = plan.get_dependency_results(task)
            if deps:
                dep = deps[-1]
                if dep.result:
                    meta = extract_price_metadata(dep.result.output, dep.result.summary)
                    output.update(meta)

        unit = format_price_unit(output.get("currency"), output.get("price_period"))
        summary_text = f"计算结果: {value} {unit}".strip()

        return TaskResult(
            status=TaskStatus.DONE,
            summary=summary_text,
            output=output,
        )

    def _build_information_result(
        self, task: Task, summary: str, tool: str
    ) -> TaskResult:
        prompt = BUILD_OUTPUT_PROMPT.format(
            description=task.description,
            tool=tool,
            summary=summary[:2000],
        )

        try:
            response = ask_llm(prompt, max_tokens=400)
            data = parse_llm_json(response)
            output = data.get("output", {})
            if not isinstance(output, dict):
                output = {}
            normalize_price_field(output)
            if not output.get("currency"):
                output["currency"] = infer_currency(output, summary)
            if not output.get("price_period"):
                output["price_period"] = infer_price_period(output, summary)
            if output.get("price") is None:
                normalize_price_field(output, output.get("model"))
            return TaskResult(
                status=TaskStatus.DONE,
                summary=data.get("summary", summary[:80]),
                output=output,
            )
        except Exception:  # noqa: BLE001
            return TaskResult(
                status=TaskStatus.DONE,
                summary=summary[:80],
                output={"text": summary[:200]},
            )
