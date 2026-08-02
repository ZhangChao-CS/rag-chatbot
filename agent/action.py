from dataclasses import dataclass


@dataclass
class ToolAction:

    tool: str

    arguments: dict
    