from agent.planning.planner import Planner
from agent.reflection.reflection import Reflection
from agent.runtime.executor import Executor
from agent.runtime.final_generator import FinalGenerator
from agent.runtime.initializer import Initializer
from agent.runtime.router import Router
from agent.runtime.runtime import AgentRuntime
from agent.runtime.summarizer import ObservationSummarizer
from memory.base_memory import BaseMemory
from tools.registry import ToolRegistry


class SimpleAgent:
    """Agent Framework v1.2 入口。"""

    def __init__(
        self,
        tools,
        memory: BaseMemory,
        local_kb_available: bool = False,
        system_context: str = "",
    ):
        self.local_kb_available = local_kb_available
        self.system_context = system_context
        self.registry = ToolRegistry()
        for tool in tools:
            self.registry.register(tool)

        self.memory = memory
        self.planner = Planner()

        self.runtime = AgentRuntime(
            registry=self.registry,
            memory=self.memory,
            initializer=Initializer(memory=self.memory, planner=self.planner),
            router=Router(registry=self.registry),
            executor=Executor(registry=self.registry),
            reflection=Reflection(),
            final_generator=FinalGenerator(),
            summarizer=ObservationSummarizer(),
        )

    def run(self, question: str) -> tuple[str, list]:
        return self.runtime.run(
            question,
            local_kb_available=self.local_kb_available,
            system_context=self.system_context,
        )
