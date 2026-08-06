import re
from typing import Any, Optional

from agent.planning.schema import Task


def extract_numeric(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").replace("USD", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    if isinstance(value, dict):
        if "value" in value:
            return extract_numeric(value["value"])
        if "price" in value:
            return extract_numeric(value["price"])
        for v in value.values():
            n = extract_numeric(v)
            if n is not None:
                return n
    if isinstance(value, list):
        for item in value:
            n = extract_numeric(item)
            if n is not None:
                return n
    return None


def _resolve_preferred_model(data: dict, preferred_model: Optional[str] = None) -> Optional[str]:
    if preferred_model:
        return preferred_model

    for key in ("model", "latest_model", "primary_model"):
        val = data.get(key)
        if val and isinstance(val, str):
            return val

    for key in ("latest_model_names", "latest_models", "models"):
        val = data.get(key)
        if isinstance(val, list) and val:
            return str(val[0])

    return None


def _price_from_map(price_map: dict, model_name: Optional[str]) -> Optional[float]:
    if not price_map or not isinstance(price_map, dict):
        return None

    if model_name:
        for key, val in price_map.items():
            key_str = str(key).lower()
            model_lower = model_name.lower()
            if model_lower in key_str or key_str in model_lower:
                n = extract_numeric(val)
                if n is not None:
                    return n

    for val in price_map.values():
        n = extract_numeric(val)
        if n is not None:
            return n

    return None


def extract_price_from_summary(summary: str) -> Optional[float]:
    """从 TaskResult.summary 文本中提取首个可用月费数值。"""
    if not summary:
        return None

    patterns = [
        r"每月\s*(\d+(?:\.\d+)?)\s*美元",
        r"(\d+(?:\.\d+)?)\s*美元\s*/?\s*月",
        r"价格(?:分别为)?(?:每月)?\s*(\d+(?:\.\d+)?)\s*美元",
        r"月费\s*(\d+(?:\.\d+)?)",
        r"\$\s*(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, summary)
        if match:
            return float(match.group(1))

    return None


def extract_price_from_data(
    data: dict,
    preferred_model: Optional[str] = None,
) -> Optional[float]:
    if not data:
        return None

    model = _resolve_preferred_model(data, preferred_model)

    price_field = data.get("price")
    if price_field is not None:
        if isinstance(price_field, dict):
            matched = _price_from_map(price_field, model)
            if matched is not None:
                return matched
        n = extract_numeric(price_field)
        if n is not None:
            return n

    for key in ("monthly_prices", "prices", "price_map"):
        price_map = data.get(key)
        if isinstance(price_map, dict):
            n = _price_from_map(price_map, model)
            if n is not None:
                return n

    entities = data.get("entities")
    if isinstance(entities, list):
        for ent in entities:
            if isinstance(ent, dict):
                if model and ent.get("name"):
                    if model.lower() in str(ent["name"]).lower():
                        n = extract_numeric(ent.get("price") or ent.get("monthly_price"))
                        if n is not None:
                            return n
                n = extract_numeric(ent.get("price") or ent.get("monthly_price"))
                if n is not None:
                    return n

    text = data.get("text")
    if isinstance(text, str):
        n = extract_price_from_summary(text)
        if n is not None:
            return n

    for key in ("value", "amount", "monthly_price", "monthly_fee"):
        n = extract_numeric(data.get(key))
        if n is not None:
            return n

    facts = data.get("facts")
    if isinstance(facts, dict):
        return extract_price_from_data(facts, model)

    return None


def extract_price_from_task(task: Task, preferred_model: Optional[str] = None) -> Optional[float]:
    """从 Task 的 TaskResult（output + summary）提取价格。"""
    if not task.result:
        return None

    output = task.result.output
    normalize_price_field(output, preferred_model)

    price = extract_price_from_data(output, preferred_model)
    if price is not None:
        return price

    return extract_price_from_summary(task.result.summary)


def normalize_price_field(data: dict, preferred_model: Optional[str] = None) -> None:
    model = _resolve_preferred_model(data, preferred_model)
    price = extract_price_from_data(data, model)
    if price is None and data.get("text"):
        price = extract_price_from_summary(str(data["text"]))
    if price is not None:
        data["price"] = price
    if model and "model" not in data:
        data["model"] = model


def get_model_from_plan(plan, before_task_id: int) -> Optional[str]:
    for t in plan.tasks:
        if t.id >= before_task_id or not t.result:
            continue
        model = _resolve_preferred_model(t.result.output, None)
        if model:
            return model
    return None


def infer_currency(data: dict, summary: str = "") -> Optional[str]:
    for key in ("currency", "price_currency", "unit"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    text = " ".join(
        str(data.get(k, "")) for k in ("text", "summary") if data.get(k)
    )
    text = f"{text} {summary}"
    if re.search(r"美元|USD|\$", text, re.I):
        return "美元"
    if re.search(r"人民币|CNY|RMB|¥", text, re.I):
        return "人民币"
    if re.search(r"欧元|EUR|€", text, re.I):
        return "欧元"
    return None


def infer_price_period(data: dict, summary: str = "") -> Optional[str]:
    for key in ("price_period", "period", "billing_period"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    text = " ".join(
        str(data.get(k, "")) for k in ("text", "summary") if data.get(k)
    )
    text = f"{text} {summary}"
    if re.search(r"每月|/月|per month|monthly|月费", text, re.I):
        return "月"
    if re.search(r"每年|/年|per year|yearly|年费", text, re.I):
        return "年"
    return None


def extract_price_metadata(output: dict, summary: str = "") -> dict:
    """从 information_collection 的 output 提取价格上下文，供计算与最终答案使用。"""
    if not output:
        output = {}

    currency = infer_currency(output, summary)
    price_period = infer_price_period(output, summary)
    model = _resolve_preferred_model(output, None)
    original_price = output.get("price")
    if original_price is None:
        original_price = extract_price_from_data(output, model)

    meta = {}
    if model:
        meta["model"] = model
    if original_price is not None:
        meta["original_price"] = original_price
    if currency:
        meta["currency"] = currency
    if price_period:
        meta["price_period"] = price_period
    return meta


def format_price_unit(currency: Optional[str], price_period: Optional[str]) -> str:
    if currency and price_period:
        return f"{currency}/{price_period}"
    return currency or price_period or ""

