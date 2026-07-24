import gc


def doc_id(document) -> str:
    """用文档前100字符作为唯一标识"""
    return document.page_content[:100]


def cleanup_memory() -> None:
    gc.collect()
