import re
from typing import Optional, Tuple

from agent.planning.schema import Plan, Task
from agent.runtime.price_utils import extract_price_from_task, get_model_from_plan


class CalculatorRouter:
    """从依赖 Task 的 TaskResult 确定性生成 calculator expression。"""

    def build_arguments(self, task: Task, plan: Plan) -> Optional[dict]:
        price, source = self._resolve_price(task, plan)
        if price is None:
            return None

        multiplier = self._parse_multiplier(task.description)
        expression = f"{price} * {multiplier}"

        print(f"[CalculatorRouter] price={price} (from {source}), multiplier={multiplier}")
        print(f"[CalculatorRouter] expression={expression}")

        return {"expression": expression}

    def build_arguments_or_raise(self, task: Task, plan: Plan) -> dict:
        args = self.build_arguments(task, plan)
        if args is None:
            dep = self._get_primary_dep(task, plan)
            detail = "无依赖任务"
            if dep and dep.result:
                detail = f"output keys={list(dep.result.output.keys())}, summary={dep.result.summary[:80]}"
            raise ValueError(
                f"Task #{task.id} 无法从依赖 TaskResult 获取 price/value（{detail}）"
            )
        return args

    def _get_primary_dep(self, task: Task, plan: Plan) -> Optional[Task]:
        deps = plan.get_dependency_results(task)
        if not deps:
            deps = [
                t for t in plan.tasks
                if t.id < task.id and t.status.value == "done"
            ]
        return deps[-1] if deps else None

    def _resolve_price(self, task: Task, plan: Plan) -> Tuple[Optional[float], str]:
        dep_tasks = plan.get_dependency_results(task)
        if not dep_tasks:
            dep_tasks = [
                t for t in plan.tasks
                if t.id < task.id and t.status.value == "done"
            ]

        preferred_model = get_model_from_plan(plan, task.id)

        for dep in reversed(dep_tasks):
            price = extract_price_from_task(dep, preferred_model)
            if price is not None:
                return price, f"Task #{dep.id} result"

        return None, ""

    def _parse_multiplier(self, description: str) -> float:
        text = description.lower()

        match = re.search(r"减少\s*(\d+(?:\.\d+)?)\s*%", description)
        if match:
            return round(1 - float(match.group(1)) / 100, 4)

        match = re.search(r"打\s*(\d+(?:\.\d+)?)\s*折", description)
        if match:
            return round(float(match.group(1)) / 10, 4)

        match = re.search(r"增加\s*(\d+(?:\.\d+)?)\s*%", description)
        if match:
            return round(1 + float(match.group(1)) / 100, 4)

        if "8折" in text or "八折" in text:
            return 0.8
        if "20%" in text:
            return 0.8

        return 0.8
