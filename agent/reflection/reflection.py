from typing import Optional

from agent.plan_utils import format_prior_outputs
from agent.planning.schema import Plan, PlanRepair, Task, TaskResult, TaskStatus, TaskType
from agent.reflection.result_builder import ResultBuilder
from agent.runtime.retry_utils import is_transient_error
from agent.runtime.types import ExecutionResult
from rag.llm import ask_llm
from rag.utils import parse_llm_json

from .prompt import REFLECTION_PROMPT
from .schema import ReflectionResult


class Reflection:
    """
    Raw Observation → Summary → Reflection（判 status/repair）→ ResultBuilder（结构化 output）
    """

    def __init__(self):
        self.result_builder = ResultBuilder()

    def evaluate(
        self,
        task: Task,
        execution: ExecutionResult,
        plan: Plan,
        question: str,
        summary: str,
    ) -> ReflectionResult:
        if not execution.success:
            repair = self._failure_repair(task, execution)
            return ReflectionResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                reason=execution.error or "工具执行异常",
                confidence=1.0,
                plan_repair=repair,
            )

        if task.task_type == TaskType.COMPUTATION:
            expr = execution.arguments.get("expression", "")
            result = self.result_builder.build(
                task, execution.observation, execution.tool, expr, plan=plan
            )
            if result.get("value") is not None:
                return ReflectionResult(
                    task_id=task.id,
                    status=TaskStatus.DONE,
                    reason="计算完成",
                    confidence=1.0,
                    result=result,
                )
            return ReflectionResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                reason="计算失败",
                confidence=1.0,
                result=result,
            )

        prior = format_prior_outputs(task, plan)

        prompt = REFLECTION_PROMPT.format(
            question=question,
            task=f"#{task.id} {task.description}",
            task_id=task.id,
            tool=execution.tool,
            summary=summary or "无",
            prior_outputs=prior,
        )

        response = ask_llm(prompt, max_tokens=500)
        data = parse_llm_json(response, error_prefix="Reflection")

        status = self._parse_status(data.get("status", "done"))
        repair_data = data.get("plan_repair") or {}
        plan_repair = PlanRepair(
            action=repair_data.get("action", "none"),
            task_description=repair_data.get("task_description", ""),
            task_type=self._parse_task_type(repair_data.get("task_type")),
            insert_before_task_id=repair_data.get("insert_before_task_id"),
            preferred_tool=repair_data.get("preferred_tool") or None,
        )

        result = None
        if status == TaskStatus.DONE:
            result = self.result_builder.build(task, summary, execution.tool, plan=plan)
        elif status == TaskStatus.FAILED:
            result = TaskResult(
                status=TaskStatus.FAILED,
                summary=data.get("reason", "任务未完成")[:80],
                output={},
            )

        return ReflectionResult(
            task_id=data.get("task_id", task.id),
            status=status,
            reason=data.get("reason", ""),
            confidence=float(data.get("confidence", 0.0)),
            retry=bool(data.get("retry", False)),
            result=result,
            plan_repair=plan_repair,
        )

    def _failure_repair(self, task: Task, execution: ExecutionResult) -> PlanRepair:
        error = execution.error or ""
        if task.task_type != TaskType.INFORMATION_COLLECTION:
            return PlanRepair(action="none")

        if execution.tool == "web_search" and is_transient_error(error):
            return PlanRepair(
                action="retry_task",
                preferred_tool="retrieval",
            )

        if execution.tool == "retrieval" and "未找到" in error:
            return PlanRepair(
                action="retry_task",
                preferred_tool="web_search",
            )

        return PlanRepair(action="retry_task")

    def _parse_task_type(self, raw: Optional[str]) -> Optional[TaskType]:  # noqa: UP045
        if not raw:
            return None
        mapping = {
            "information_collection": TaskType.INFORMATION_COLLECTION,
            "computation": TaskType.COMPUTATION,
        }
        return mapping.get(raw.lower().strip())

    def _parse_status(self, raw: str) -> TaskStatus:
        try:
            return TaskStatus(raw.lower().strip())
        except ValueError:
            return TaskStatus.DONE
