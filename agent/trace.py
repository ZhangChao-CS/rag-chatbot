from dataclasses import dataclass, field


@dataclass
class TraceStep:
    stage: str
    content: dict = field(default_factory=dict)
    duration: float = 0.0
    error: str = ""
    task_id: int = 0


@dataclass
class TaskTraceBlock:
    """单个 Task 的完整执行块。"""

    task_id: int
    description: str
    task_type: str = ""
    depends_on: list = field(default_factory=list)
    router_tool: str = ""
    router_thought: str = ""
    arguments: dict = field(default_factory=dict)
    observation_summary: str = ""
    structured_result: str = ""
    reflection_status: str = ""
    reflection_reason: str = ""
    confidence: float = 0.0
    duration: float = 0.0
    error: str = ""


@dataclass
class AgentTrace:
    question: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    task_blocks: list[TaskTraceBlock] = field(default_factory=list)
    answer: str = ""

    def add_step(
        self,
        stage: str,
        duration: float = 0.0,
        error: str = "",
        task_id: int = 0,
        **kwargs,
    ):
        self.steps.append(
            TraceStep(
                stage=stage,
                content=kwargs,
                duration=duration,
                error=error,
                task_id=task_id,
            )
        )

    def add_task_block(self, block: TaskTraceBlock) -> None:
        self.task_blocks.append(block)

    def __str__(self) -> str:
        lines = [
            "\n================ Agent Trace ================\n",
            "Question",
            "--------------------------------",
            self.question,
            "",
        ]

        for block in self.task_blocks:
            lines.extend(self._format_task_block(block))

        for step in self.steps:
            if step.stage in ("memory_retrieve", "planning", "final_answer", "memory_save"):
                lines.extend(self._format_meta_step(step))

        lines.extend([
            "Final Answer",
            "--------------------------------",
            self.answer,
            "\n=============================================\n",
        ])
        return "\n".join(lines)

    def _format_task_block(self, b: TaskTraceBlock) -> list[str]:
        dep = f" (depends: {b.depends_on})" if b.depends_on else ""
        text = [
            "====================",
            f"Task #{b.task_id}",
            "====================",
            f"Description: {b.description}{dep}",
            f"Type: {b.task_type}",
            f"Router: {b.router_tool}",
        ]
        if b.router_thought:
            text.append(f"Thought: {b.router_thought}")
        if b.arguments:
            text.append(f"Arguments: {b.arguments}")
        if b.observation_summary:
            text.append(f"Observation (summary): {b.observation_summary}")
        if b.structured_result:
            text.append(f"TaskResult: {b.structured_result}")
        text.append(f"Reflection: {b.reflection_status}")
        if b.reflection_reason:
            text.append(f"Reason: {b.reflection_reason}")
        if b.confidence:
            text.append(f"Confidence: {b.confidence:.2f}")
        if b.duration:
            text.append(f"Duration: {b.duration:.3f}s")
        if b.error:
            text.append(f"Error: {b.error}")
        text.append("")
        return text

    def _format_meta_step(self, step: TraceStep) -> list[str]:
        text = [
            f"--- {step.stage} ---",
        ]
        for key, value in step.content.items():
            text.append(f"{key}: {value}")
        if step.duration:
            text.append(f"Duration: {step.duration:.3f}s")
        if step.error:
            text.append(f"Error: {step.error}")
        text.append("")
        return text
