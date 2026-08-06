"""Router 泛用选型辅助：基于信息来源类型，而非具体领域关键词。"""

# 任务文本中出现这些信号时，更可能需要实时外部信息
REALTIME_SIGNALS = (
    "最新",
    "实时",
    "今天",
    "当前",
    "新闻",
    "价格",
    "多少钱",
    "多少元",
    "月费",
    "年费",
    "售价",
    "股价",
    "汇率",
    "刚刚",
    "今年",
    "2025",
    "2026",
)


def default_system_context(local_kb_available: bool) -> str:
    if local_kb_available:
        return (
            "用户已上传 PDF 文档，本地知识库已就绪。"
            "retrieval 可检索文档内的概念、方法、术语、模型对比与论文内容。"
        )
    return "未加载本地知识库，information_collection 任务应使用 web_search。"


def build_routing_guidance(
    question: str,
    task_description: str,
    local_kb_available: bool,
) -> str:
    if not local_kb_available:
        return "本地知识库不可用，information_collection 应使用 web_search。"

    text = f"{question} {task_description}"
    if any(signal in text for signal in REALTIME_SIGNALS):
        return (
            "任务涉及实时或外部公开信息（如价格、新闻）。"
            "若用户上传的文档中也可能包含相关内容，仍应优先 retrieval；"
            "仅当明确需要联网获取最新数据时使用 web_search。"
        )

    return (
        "本地知识库已就绪，且任务未明确要求实时外部信息。"
        "应优先使用 retrieval 查询用户上传的文档。"
        "禁止因某个术语听起来“很新”或“很专业”就选择 web_search。"
    )
