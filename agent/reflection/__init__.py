"""Reflection module — task status evaluation."""

__all__ = ["Reflection", "ReflectionResult"]


def __getattr__(name: str):
    if name == "Reflection":
        from agent.reflection.reflection import Reflection
        return Reflection
    if name == "ReflectionResult":
        from agent.reflection.schema import ReflectionResult
        return ReflectionResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
