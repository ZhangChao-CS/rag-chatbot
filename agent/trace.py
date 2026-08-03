from dataclasses import dataclass, field


@dataclass
class TraceStep:
    """
    单个 Agent 执行步骤
    """

    stage: str

    content: dict = field(default_factory=dict)

    duration: float = 0.0

    error: str = ""


@dataclass
class AgentTrace:
    """
    一次 Agent 执行完整轨迹
    """

    question: str = ""

    steps: list[TraceStep] = field(default_factory=list)

    answer: str = ""

    def add_step(self, stage: str, duration: float = 0.0, error: str = "", **kwargs):

        self.steps.append(
            TraceStep(stage=stage, content=kwargs, duration=duration, error=error)
        )

    def summarize_observation(self, observation: str, max_length: int = 500):

        if not observation:
            return ""

        if len(observation) <= max_length:
            return observation

        return observation[:max_length] + "\n......(truncated)"

    def __str__(self):

        text = []

        text.append("\n================ Agent Trace ================\n")

        text.append("Question")
        text.append("--------------------------------")
        text.append(self.question)
        text.append("")

        for idx, step in enumerate(self.steps, start=1):
            text.append(f"Step {idx} : {step.stage}")

            text.append("--------------------------------")

            for key, value in step.content.items():
                text.append(f"{key}:")

                text.append(str(value))

                text.append("")

            if step.duration:
                text.append(f"Duration: {step.duration:.3f}s")

                text.append("")

            if step.error:
                text.append("Error:")

                text.append(step.error)

                text.append("")

        text.append("Final Answer")

        text.append("--------------------------------")

        text.append(self.answer)

        text.append("\n=============================================\n")

        return "\n".join(text)
