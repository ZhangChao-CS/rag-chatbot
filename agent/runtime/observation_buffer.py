from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ObservationEntry:
    task_id: int
    tool: str
    content: str
    summary: str = ""
    success: bool = True
    raw: Any = None


@dataclass
class ObservationBuffer:
    """结构化 Observation 缓冲区：完整 Observation + Summary 分离存储。"""

    entries: list[ObservationEntry] = field(default_factory=list)

    def add(
        self,
        task_id: int,
        tool: str,
        content: str,
        summary: str = "",
        success: bool = True,
        raw: Any = None,
    ) -> None:
        self.entries.append(
            ObservationEntry(
                task_id=task_id,
                tool=tool,
                content=content,
                summary=summary or content[:500],
                success=success,
                raw=raw,
            )
        )

    def get_by_task(self, task_id: int) -> list[ObservationEntry]:
        return [e for e in self.entries if e.task_id == task_id]

    def get_latest(self) -> Optional[ObservationEntry]:
        if not self.entries:
            return None
        return self.entries[-1]

    def get_summary_for_task(self, task_id: int) -> str:
        entries = self.get_by_task(task_id)
        if entries:
            return entries[-1].summary
        return ""

    def to_text(self) -> str:
        if not self.entries:
            return ""
        return "\n".join(
            f"[Task #{e.task_id}] ({e.tool}) {e.summary}" for e in self.entries
        )
