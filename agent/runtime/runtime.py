import time
from typing import Optional

from agent.planning.schema import TaskResult, TaskStatus, TaskType
from agent.reflection.reflection import Reflection
from agent.runtime.executor import Executor
from agent.runtime.final_generator import FinalGenerator
from agent.runtime.initializer import Initializer
from agent.runtime.router import Router
from agent.runtime.price_utils import normalize_price_field
from agent.runtime.retry_utils import is_transient_error
from agent.runtime.summarizer import ObservationSummarizer
from agent.state import AgentState
from agent.trace import TaskTraceBlock
from memory.base_memory import BaseMemory
from tools.registry import ToolRegistry


class AgentRuntime:
    """
    Task-Oriented Agent Runtime v1.2

    Task → Router → Tool → Raw Observation → Summary
         → Reflection → TaskResult + Plan Repair → Final
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        registry: ToolRegistry,
        memory: BaseMemory,
        initializer: Initializer,
        router: Router,
        executor: Executor,
        reflection: Reflection,
        final_generator: FinalGenerator,
        summarizer: Optional[ObservationSummarizer] = None,  # noqa: UP045
    ):
        self.registry = registry
        self.memory = memory
        self.initializer = initializer
        self.router = router
        self.executor = executor
        self.reflection = reflection
        self.final_generator = final_generator
        self.summarizer = summarizer or ObservationSummarizer()
        self._retry_counts: dict[int, int] = {}

    def run(
        self,
        question: str,
        local_kb_available: bool = False,
        system_context: str = "",
    ) -> tuple[str, list]:
        try:
            state = self.initializer.initialize(
                question, local_kb_available, system_context
            )
        except Exception:  # noqa: BLE001
            state = AgentState(question=question, history_text="")
            state.trace.question = question
            state.answer = self.final_generator.generate(state)
            self._finalize(state)
            return state.answer, state.retrieved_docs

        while state.can_continue():
            task = state.plan.get_next_task()
            if task is None:
                break

            block = TaskTraceBlock(
                task_id=task.id,
                description=task.description,
                task_type=task.task_type.value,
                depends_on=list(task.depends_on),
            )
            task_start = time.time()

            state.current_task_id = task.id
            state.plan.mark_running(task.id)

            try:
                route = self.router.route(task, state)
            except Exception as e:  # noqa: BLE001
                self._fail_task(state, task.id, str(e), block, task_start)
                continue

            block.router_tool = route.tool
            block.router_thought = route.thought
            block.arguments = route.arguments

            execution = self.executor.execute(route)

            if not execution.success and self._can_retry(task.id, execution.error):
                self._retry_counts[task.id] = self._retry_counts.get(task.id, 0) + 1
                state.plan.apply_update(task.id, TaskStatus.TODO)
                if execution.tool == "web_search":
                    state.tool_hints[task.id] = "retrieval"
                print(f"[Runtime] Task #{task.id} 重试 ({self._retry_counts[task.id]}/{self.MAX_RETRIES})")
                state.next_step()
                continue

            if not execution.success:
                self._fail_task(state, task.id, execution.error, block, task_start)
                continue

            raw_obs = execution.observation or ""
            summary = self.summarizer.summarize(raw_obs)

            state.observation_buffer.add(
                task_id=task.id,
                tool=execution.tool,
                content=raw_obs,
                summary=summary,
                success=True,
                raw=execution.raw,
            )

            if execution.tool == "retrieval" and execution.raw:
                state.retrieved_docs.extend(execution.raw)

            try:
                reflection = self.reflection.evaluate(
                    task=task,
                    execution=execution,
                    plan=state.plan,
                    question=state.question,
                    summary=summary,
                )
            except Exception as e:  # noqa: BLE001
                state.plan.mark_done(task.id)
                state.plan.set_result(
                    task.id,
                    TaskResult(summary=summary[:80], output={"text": summary[:200]}),
                )
                block.error = str(e)
                block.duration = time.time() - task_start
                state.trace.add_task_block(block)
                state.next_step()
                continue

            state.reflection = reflection
            self._apply_reflection(state, task.id, reflection)

            block.observation_summary = summary[:200]
            if reflection.result:
                block.structured_result = reflection.result.to_display()
            block.reflection_status = reflection.status.value
            block.reflection_reason = reflection.reason
            block.confidence = reflection.confidence
            block.duration = time.time() - task_start
            state.trace.add_task_block(block)

            print(f"[Reflection] Task #{task.id} → {reflection.status.value}")
            if reflection.plan_repair.action != "none":
                print(f"[Plan Repair] {reflection.plan_repair.action}")
            print("[Plan]\n", state.plan.to_display())

            state.next_step()
            if state.plan.all_terminal():
                break

        state.answer = self.final_generator.generate(state)
        self._finalize(state)
        return state.answer, state.retrieved_docs

    def _apply_reflection(self, state: AgentState, task_id: int, reflection) -> None:
        repair = reflection.plan_repair

        if reflection.status == TaskStatus.DONE and reflection.result:
            normalize_price_field(reflection.result.output)
            state.plan.mark_done(task_id)
            state.plan.set_result(task_id, reflection.result)
            state.tool_hints.pop(task_id, None)
            return

        if repair.action == "insert_task" and repair.task_description:
            new_task = state.plan.insert_task(
                description=repair.task_description,
                task_type=repair.task_type or TaskType.INFORMATION_COLLECTION,
                depends_on=list(state.plan.get_task(task_id).depends_on) if state.plan.get_task(task_id) else [],
                before_task_id=repair.insert_before_task_id or task_id,
            )
            state.plan.mark_failed(task_id)
            print(f"[Plan Repair] 插入 Task #{new_task.id}: {new_task.description}")
            return

        if repair.action == "retry_task" or reflection.retry:
            if self._can_retry(task_id, reflection.reason):
                self._retry_counts[task_id] = self._retry_counts.get(task_id, 0) + 1
                state.plan.apply_update(task_id, TaskStatus.TODO)
                if repair.preferred_tool:
                    state.tool_hints[task_id] = repair.preferred_tool
                return

        if reflection.status == TaskStatus.FAILED or repair.action == "retry_task":
            state.plan.mark_failed(task_id)
            state.plan.set_error(task_id, reflection.reason or "任务失败")
        elif reflection.status == TaskStatus.RUNNING:
            state.plan.apply_update(task_id, TaskStatus.TODO)

    def _fail_task(
        self, state: AgentState, task_id: int, error: str,
        block: TaskTraceBlock, task_start: float,
    ) -> None:
        block.error = error
        block.duration = time.time() - task_start
        state.plan.mark_failed(task_id)
        state.plan.set_error(task_id, error)
        state.trace.add_task_block(block)
        state.next_step()

    def _can_retry(self, task_id: int, error: str) -> bool:
        if self._retry_counts.get(task_id, 0) >= self.MAX_RETRIES:
            return False
        if not error:
            return True
        return is_transient_error(error) or "validation" in error.lower()

    def _finalize(self, state: AgentState) -> None:
        state.finished = True
        state.trace.add_step(stage="final_answer", answer=state.answer)
        self.memory.add_ai_message(state.answer)
        state.trace.add_step(stage="memory_save", message=state.answer)
        state.trace.answer = state.answer
        print(state.trace)
