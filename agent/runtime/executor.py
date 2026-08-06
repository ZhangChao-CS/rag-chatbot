import time

from agent.runtime.retry_utils import is_transient_error
from agent.runtime.types import ExecutionResult, RouteResult
from tools.registry import ToolRegistry

MAX_EXECUTE_ATTEMPTS = 2
EXECUTE_RETRY_DELAY = 1.0


class Executor:
    """封装 Tool 的参数校验、执行与异常处理。"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, route: RouteResult) -> ExecutionResult:
        start = time.time()
        last_error = ""

        for attempt in range(1, MAX_EXECUTE_ATTEMPTS + 1):
            try:
                arguments = self._normalize_arguments(route.arguments)
                validated = self._validate(route.tool, arguments)
                tool = self.registry.get(route.tool)
                result = tool.run(**validated.model_dump())
                duration = time.time() - start

                return ExecutionResult(
                    success=True,
                    task_id=route.task_id,
                    tool=route.tool,
                    arguments=validated.model_dump(),
                    observation=result.get("observation", ""),
                    raw=result.get("raw"),
                    duration=duration,
                )

            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                if attempt < MAX_EXECUTE_ATTEMPTS and is_transient_error(last_error):
                    print(f"[Executor] 瞬态错误，重试 ({attempt}/{MAX_EXECUTE_ATTEMPTS})")
                    time.sleep(EXECUTE_RETRY_DELAY * attempt)
                    continue

        duration = time.time() - start
        return ExecutionResult(
            success=False,
            task_id=route.task_id,
            tool=route.tool,
            arguments=route.arguments,
            observation="",
            duration=duration,
            error=last_error,
        )

    def _normalize_arguments(self, arguments: dict) -> dict:
        normalized = {}
        for key, value in arguments.items():
            if isinstance(value, dict) and len(value) == 1:
                value = next(iter(value.values()))
            normalized[key] = value
        return normalized

    def _validate(self, tool_name: str, arguments: dict):
        tool = self.registry.get(tool_name)
        return tool.args_schema(**arguments)
