from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from agent.planning.schema import Plan
from agent.runtime.observation_buffer import ObservationBuffer
from agent.trace import AgentTrace

if TYPE_CHECKING:
    from agent.reflection.schema import ReflectionResult


@dataclass
class AgentState:
    """Task-Oriented Agent 运行时状态。"""

    question: str
    history_text: str
    local_kb_available: bool = False
    system_context: str = ""

    # Planning
    plan: Optional[Plan] = None  # noqa: UP045
    current_task_id: Optional[int] = None  # noqa: UP045

    # Runtime
    observation_buffer: ObservationBuffer = field(default_factory=ObservationBuffer)
    tool_hints: dict = field(default_factory=dict)  # task_id → preferred_tool
    reflection: Optional["ReflectionResult"] = None  # noqa: UP045, F821
    retrieved_docs: list = field(default_factory=list)

    # Final
    answer: str = ""

    # Control
    finished: bool = False
    step_count: int = 0
    max_steps: int = 20

    # Trace
    trace: AgentTrace = field(default_factory=AgentTrace)

    def next_step(self) -> None:
        self.step_count += 1

    def can_continue(self) -> bool:
        return not self.finished and self.step_count < self.max_steps
