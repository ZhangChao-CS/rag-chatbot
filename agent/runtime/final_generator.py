from typing import Optional

from agent.planning.schema import TaskType
from agent.runtime.price_utils import format_price_unit
from agent.state import AgentState
from rag.llm import ask_llm

FINAL_ANSWER_PROMPT = """
用户问题：{question}

已完成 Task 的结构化 output（唯一信息源）：
{outputs}

请基于以上 output 生成最终答案。
- 只能使用 output 中的字段，禁止编造
- [FAILED] 的任务不可用
- 若 information_collection 的 output 已含对比/结论，直接整理成答案
- 涉及数值（尤其价格）时必须带单位，从 output 的 currency、price_period 等字段读取
"""


class FinalGenerator:
    """TaskResult.output → Final Answer（不读 Raw Observation）。"""

    def generate(self, state: AgentState) -> str:
        if not state.plan:
            return "无法生成答案：无执行计划"

        outputs = state.plan.collect_outputs()
        failed = [
            t for t in state.plan.tasks
            if t.status.value == "failed"
        ]

        if not outputs and failed:
            return self._failed_message(failed)

        computation_answer = self._format_computation_answer(state)
        if computation_answer:
            return computation_answer

        outputs_text = state.plan.structured_chain()
        if failed:
            outputs_text += "\n\n失败任务:\n" + "\n".join(
                f"Task #{t.id}: {t.error or '失败'}" for t in failed
            )

        return ask_llm(
            FINAL_ANSWER_PROMPT.format(
                question=state.question,
                outputs=outputs_text,
            ),
            max_tokens=1000,
        )

    def _format_computation_answer(self, state: AgentState) -> Optional[str]:
        comp_task = None
        for task in state.plan.tasks:
            if task.task_type == TaskType.COMPUTATION and task.result:
                comp_task = task
                break
        if comp_task is None or comp_task.result.get("value") is None:
            return None

        out = comp_task.result.output
        value = out.get("value")
        model = out.get("model", "")
        original = out.get("original_price")
        unit = format_price_unit(out.get("currency"), out.get("price_period"))

        if model and original is not None and unit:
            return f"{model} 原价 {original} {unit}，调整后为 {value} {unit}"
        if model and unit:
            return f"{model} 计算结果为 {value} {unit}"
        if unit:
            return f"计算结果: {value} {unit}"
        if value is not None:
            return f"计算结果: {value}"
        return None

    @staticmethod
    def _failed_message(failed) -> str:
        lines = ["部分任务执行失败，无法完整回答："]
        for t in failed:
            lines.append(f"- Task #{t.id} {t.description}: {t.error or '失败'}")
        return "\n".join(lines)
