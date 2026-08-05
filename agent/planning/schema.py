from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class TaskStatus(str, Enum):
    TODO = "todo"

    RUNNING = "running"

    DONE = "done"

    FAILED = "failed"


class Task(BaseModel):
    id: int

    description: str

    status: TaskStatus = TaskStatus.TODO

    tool: Optional[str] = None  # noqa: UP045


class Plan(BaseModel):
    tasks: List[Task]  # noqa: UP006
