from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    TODO = "todo"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class TaskType(str, Enum):
    """Planner 层任务类型，不含具体 Tool。"""

    INFORMATION_COLLECTION = "information_collection"
    COMPUTATION = "computation"


class TaskResult(BaseModel):
    """结构化任务结果 — 各模块唯一共享的数据对象。"""

    status: TaskStatus = TaskStatus.DONE
    summary: str = ""
    output: Dict[str, Any] = Field(default_factory=dict)  # noqa: UP006

    def get(self, key: str, default: Any = None) -> Any:
        return self.output.get(key, default)

    def to_display(self) -> str:
        if not self.output:
            return self.summary or "(空)"
        pairs = ", ".join(f"{k}={v}" for k, v in self.output.items())
        return f"{self.summary} | {pairs}" if self.summary else pairs

    def to_json_line(self) -> str:
        import json
        return json.dumps(
            {"status": self.status.value, "summary": self.summary, "output": self.output},
            ensure_ascii=False,
        )


class Task(BaseModel):
    id: int
    description: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.TODO
    depends_on: List[int] = Field(default_factory=list)  # noqa: UP006
    result: Optional[TaskResult] = None  # noqa: UP045
    error: Optional[str] = None  # noqa: UP045


class PlanRepair(BaseModel):
    """Reflection 触发的计划修复动作。"""

    action: str = "none"  # none | insert_task | retry_task
    task_description: str = ""
    task_type: Optional[TaskType] = None  # noqa: UP045
    insert_before_task_id: Optional[int] = None  # noqa: UP045
    preferred_tool: Optional[str] = None  # noqa: UP045


class Plan(BaseModel):
    tasks: List[Task] = Field(default_factory=list)  # noqa: UP006

    def get_todo_tasks(self) -> List[Task]:  # noqa: UP006
        return [t for t in self.tasks if t.status == TaskStatus.TODO]

    def get_running_tasks(self) -> List[Task]:  # noqa: UP006
        return [t for t in self.tasks if t.status == TaskStatus.RUNNING]

    def get_done_task_ids(self) -> set[int]:
        return {t.id for t in self.tasks if t.status == TaskStatus.DONE}

    def dependencies_met(self, task: Task) -> bool:
        if not task.depends_on:
            return True
        return all(dep_id in self.get_done_task_ids() for dep_id in task.depends_on)

    def get_ready_tasks(self) -> List[Task]:  # noqa: UP006
        return [
            t for t in self.tasks
            if t.status == TaskStatus.TODO and self.dependencies_met(t)
        ]

    def get_next_task(self) -> Optional[Task]:  # noqa: UP045
        ready = self.get_ready_tasks()
        if ready:
            return ready[0]
        running = self.get_running_tasks()
        return running[0] if running else None

    def all_terminal(self) -> bool:
        if not self.tasks:
            return True
        return all(t.status in (TaskStatus.DONE, TaskStatus.FAILED) for t in self.tasks)

    def apply_update(self, task_id: int, status: TaskStatus) -> None:
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                return

    def mark_running(self, task_id: int) -> None:
        self.apply_update(task_id, TaskStatus.RUNNING)

    def mark_done(self, task_id: int) -> None:
        self.apply_update(task_id, TaskStatus.DONE)

    def mark_failed(self, task_id: int) -> None:
        self.apply_update(task_id, TaskStatus.FAILED)

    def get_task(self, task_id: int) -> Optional[Task]:  # noqa: UP045
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_dependency_results(self, task: Task) -> List[Task]:  # noqa: UP006
        if task.depends_on:
            return [
                t for t in self.tasks
                if t.id in task.depends_on and t.status == TaskStatus.DONE
            ]
        return [t for t in self.tasks if t.id < task.id and t.status == TaskStatus.DONE]

    def set_result(self, task_id: int, result: TaskResult) -> None:
        task = self.get_task(task_id)
        if task:
            task.result = result

    def set_error(self, task_id: int, error: str) -> None:
        task = self.get_task(task_id)
        if task:
            task.error = error

    def insert_task(
        self,
        description: str,
        task_type: TaskType,
        depends_on: List[int],  # noqa: UP006
        before_task_id: Optional[int] = None,  # noqa: UP045
    ) -> Task:
        new_id = max(t.id for t in self.tasks) + 1 if self.tasks else 1
        new_task = Task(
            id=new_id,
            description=description,
            task_type=task_type,
            status=TaskStatus.TODO,
            depends_on=depends_on,
        )
        if before_task_id is None:
            self.tasks.append(new_task)
            return new_task

        insert_idx = next(
            (i for i, t in enumerate(self.tasks) if t.id == before_task_id),
            len(self.tasks),
        )
        self.tasks.insert(insert_idx, new_task)
        before = self.get_task(before_task_id)
        if before and before_task_id not in depends_on:
            before.depends_on = [new_id] + [
                d for d in before.depends_on if d != new_id
            ]
        return new_task

    def structured_chain(self) -> str:
        lines = []
        for task in self.tasks:
            if task.status == TaskStatus.FAILED:
                lines.append(
                    f"Task #{task.id} [FAILED]: {task.description}\n"
                    f"  error: {task.error or '执行失败'}"
                )
            elif task.status == TaskStatus.DONE and task.result:
                lines.append(f"Task #{task.id} [DONE]: {task.description}")
                lines.append(f"  output: {task.result.output}")
                if task.result.summary:
                    lines.append(f"  summary: {task.result.summary}")
            elif task.status == TaskStatus.DONE:
                lines.append(f"Task #{task.id} [DONE]: {task.description} (无 result)")
        return "\n".join(lines) if lines else "无任务结果"

    def collect_outputs(self) -> List[dict]:  # noqa: UP006
        outputs = []
        for task in self.tasks:
            if task.status == TaskStatus.DONE and task.result:
                outputs.append({
                    "task_id": task.id,
                    "goal": task.description,
                    "output": task.result.output,
                    "summary": task.result.summary,
                })
        return outputs

    def to_display(self) -> str:
        lines = []
        for task in self.tasks:
            dep = f" ← depends {task.depends_on}" if task.depends_on else ""
            lines.append(
                f"  #{task.id} [{task.status.value}] "
                f"({task.task_type.value}) {task.description}{dep}"
            )
        return "\n".join(lines)
