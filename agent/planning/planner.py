from typing import List

from agent.plan_utils import format_prior_outputs

from agent.planning.prompt import CREATE_PLAN_PROMPT
from agent.planning.schema import Plan, Task, TaskStatus, TaskType
from agent.state import AgentState
from rag.llm import ask_llm
from rag.utils import parse_llm_json


class Planner:
    """Planner 只负责任务目标与依赖，不决定 Tool。"""

    MAX_TASKS = 4

    def create_plan(self, state: AgentState) -> Plan:
        prompt = CREATE_PLAN_PROMPT.format(
            question=state.question,
            history=state.history_text or "无",
        )

        response = ask_llm(prompt, max_tokens=600)
        data = parse_llm_json(response, error_prefix="Planner")

        tasks = []
        for item in data.get("tasks", [])[: self.MAX_TASKS]:
            task_type = self._normalize_task_type(item.get("task_type", ""))
            depends_on = item.get("depends_on") or []
            tasks.append(
                Task(
                    id=item["id"],
                    description=item["description"],
                    task_type=task_type,
                    depends_on=depends_on,
                    status=TaskStatus.TODO,
                )
            )

        if not tasks:
            tasks = [
                Task(
                    id=1,
                    description=state.question,
                    task_type=TaskType.INFORMATION_COLLECTION,
                    depends_on=[],
                    status=TaskStatus.TODO,
                )
            ]
        else:
            self._auto_infer_dependencies(tasks)

        return Plan(tasks=tasks)

    def _auto_infer_dependencies(self, tasks: List[Task]) -> None:
        for i, task in enumerate(tasks):
            if task.depends_on:
                continue
            task.depends_on = [] if i == 0 else [tasks[i - 1].id]

    def _normalize_task_type(self, raw: str) -> TaskType:
        mapping = {
            "information_collection": TaskType.INFORMATION_COLLECTION,
            "information_gathering": TaskType.INFORMATION_COLLECTION,
            "knowledge_lookup": TaskType.INFORMATION_COLLECTION,
            "retrieve": TaskType.INFORMATION_COLLECTION,
            "retrieval": TaskType.INFORMATION_COLLECTION,
            "web_search": TaskType.INFORMATION_COLLECTION,
            "search": TaskType.INFORMATION_COLLECTION,
            "computation": TaskType.COMPUTATION,
            "calculate": TaskType.COMPUTATION,
            "calculator": TaskType.COMPUTATION,
        }
        return mapping.get(raw.lower().strip(), TaskType.INFORMATION_COLLECTION)

