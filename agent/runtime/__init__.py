"""Task-Oriented Agent Runtime v1.2"""

__all__ = [
    "AgentRuntime",
    "Executor",
    "Router",
    "Initializer",
    "FinalGenerator",
    "ObservationBuffer",
    "RouteResult",
    "ExecutionResult",
]


def __getattr__(name: str):
    if name == "AgentRuntime":
        from agent.runtime.runtime import AgentRuntime
        return AgentRuntime
    if name == "Executor":
        from agent.runtime.executor import Executor
        return Executor
    if name == "Router":
        from agent.runtime.router import Router
        return Router
    if name == "Initializer":
        from agent.runtime.initializer import Initializer
        return Initializer
    if name == "FinalGenerator":
        from agent.runtime.final_generator import FinalGenerator
        return FinalGenerator
    if name == "ObservationBuffer":
        from agent.runtime.observation_buffer import ObservationBuffer
        return ObservationBuffer
    if name == "RouteResult":
        from agent.runtime.types import RouteResult
        return RouteResult
    if name == "ExecutionResult":
        from agent.runtime.types import ExecutionResult
        return ExecutionResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
