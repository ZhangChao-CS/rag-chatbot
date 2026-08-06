from typing import Optional  # noqa: UP035

from pydantic import BaseModel

from agent.planning.schema import PlanRepair, TaskResult, TaskStatus


class ReflectionResult(BaseModel):
    task_id: int
    status: TaskStatus
    reason: str = ""
    confidence: float = 0.0
    retry: bool = False
    result: Optional[TaskResult] = None  # noqa: UP045
    plan_repair: PlanRepair = PlanRepair()
