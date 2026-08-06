import time

from agent.planning.planner import Planner
from agent.runtime.routing_utils import default_system_context
from agent.state import AgentState
from memory.base_memory import BaseMemory


class Initializer:
    """Agent 运行初始化：Memory → State → Plan。"""

    def __init__(self, memory: BaseMemory, planner: Planner):
        self.memory = memory
        self.planner = planner

    def initialize(
        self,
        question: str,
        local_kb_available: bool = False,
        system_context: str = "",
    ) -> AgentState:
        history_text = self.memory.get_context()
        state = AgentState(
            question=question,
            history_text=history_text,
            local_kb_available=local_kb_available,
            system_context=system_context or default_system_context(local_kb_available),
        )
        state.trace.question = question
        state.trace.add_step(stage="memory_retrieve", context=history_text)
        self.memory.add_user_message(question)

        start = time.time()
        try:
            state.plan = self.planner.create_plan(state)
        except Exception as e:  # noqa: BLE001
            state.trace.add_step(stage="planning_error", error=str(e))
            raise

        planning_time = time.time() - start
        state.trace.add_step(
            stage="planning",
            duration=planning_time,
            plan=state.plan.to_display(),
        )
        print("[Plan]\n", state.plan.to_display())
        return state
