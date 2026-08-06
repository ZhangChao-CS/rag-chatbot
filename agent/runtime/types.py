from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RouteResult:
    """Router 输出：Task → Tool 映射结果。"""

    task_id: int
    tool: str
    arguments: dict
    thought: str = ""


@dataclass
class ExecutionResult:
    """Executor 输出：工具执行结果。"""

    success: bool
    task_id: int
    tool: str
    arguments: dict = field(default_factory=dict)
    observation: str = ""
    raw: Any = None
    duration: float = 0.0
    error: str = ""
