from dataclasses import dataclass, field
from typing import Optional

from agent.planning.schema import Plan
from agent.reflection.schema import ReflectionResult
from agent.trace import AgentTrace


@dataclass
class AgentState:
    question: str

    history_text: str

    # ==================
    # Agent运行状态
    # ==================

    thought: str = ""

    action: object = None

    observation: list[str] = field(default_factory=list)

    reflection: Optional[ReflectionResult] = None  # noqa: UP045

    plan: Optional[Plan] = None  # noqa: UP045

    answer: str = ""

    # ==================
    # ReAct状态
    # ==================

    step_count: int = 0

    max_steps: int = 5

    finished: bool = False

    # ==================
    # Trace
    # ==================

    trace: AgentTrace = field(default_factory=AgentTrace)

    def next_step(self):

        self.step_count += 1

    def can_continue(self):

        return self.step_count < self.max_steps and not self.finished

    def get_observation_text(self):

        if not self.observation:
            return ""

        return "\n\n".join(self.observation)
