from typing import Union

TRANSIENT_KEYWORDS = (
    "ssl",
    "connection",
    "timeout",
    "timed out",
    "max retries",
    "eof occurred",
    "503",
    "502",
    "429",
    "网络搜索失败",
    "network",
)


def is_transient_error(error: Union[str, Exception]) -> bool:
    msg = str(error).lower()
    return any(k in msg for k in TRANSIENT_KEYWORDS)
