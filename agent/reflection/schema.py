from typing import List, Optional  # noqa: UP035

from pydantic import BaseModel


class ReflectionResult(BaseModel):
    # 是否完成

    sufficient: bool

    # 可信度

    confidence: float

    # 当前判断

    reason: str

    # 缺少信息

    missing_information: List[str] = []  # noqa: UP006

    # 推荐工具

    suggested_tool: Optional[str] = None  # noqa: UP045
