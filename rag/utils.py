import gc
import json
import platform
import re


def parse_llm_json(text: str, error_prefix: str = "LLM") -> dict:
    """解析 LLM 返回的 JSON，兼容 markdown 代码块包裹。"""
    text = text.replace("```json", "").replace("```", "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError(f"{error_prefix} 输出不是 JSON：\n{text}")
    return json.loads(match.group())


def doc_id(document) -> str:
    """用文档前100字符作为唯一标识"""
    return document.page_content[:100]


def cleanup_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def get_memory_info() -> dict:
    try:
        import psutil

        mem = psutil.virtual_memory()
        proc = psutil.Process()
        return {
            "available_mb": mem.available / 1024 / 1024,
            "used_percent": mem.percent,
            "process_mb": proc.memory_info().rss / 1024 / 1024,
        }
    except ImportError:
        return {"available_mb": float("inf"), "used_percent": 0, "process_mb": 0}


def check_memory_available(min_mb: float) -> tuple[bool, str]:
    info = get_memory_info()
    available = info["available_mb"]
    if available < min_mb:
        return False, (
            f"系统可用内存不足（剩余 {available:.0f} MB，"
            f"需要至少 {min_mb:.0f} MB）。"
            "请关闭其他应用后重试，或在侧边栏关闭「重排序」。"
        )
    return True, ""
