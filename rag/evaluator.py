import json
import re

from zhipuai import ZhipuAI

import config
from rag.prompts import (
    ANSWER_RELEVANCY_PROMPT,
    CONTEXT_UTILIZATION_PROMPT,
    FAITHFULNESS_PROMPT,
)

client = ZhipuAI(api_key=config.API_KEY)


def _parse_json(text: str) -> dict:
    """解析 LLM 返回的 JSON，兼容 markdown 代码块包裹"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def judge(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content
    result = _parse_json(text)
    result["score"] = float(result["score"])
    return result


def evaluate_faithfulness(question: str, answer: str, contexts: list) -> dict:
    prompt = FAITHFULNESS_PROMPT.format(
        question=question,
        answer=answer,
        context="\n\n".join(contexts),
    )
    return judge(prompt)


def evaluate_answer_relevancy(question: str, answer: str) -> dict:
    prompt = ANSWER_RELEVANCY_PROMPT.format(
        question=question,
        answer=answer,
    )
    return judge(prompt)


def evaluate_context_utilization(
    question: str, answer: str, contexts: list
) -> dict:
    prompt = CONTEXT_UTILIZATION_PROMPT.format(
        question=question,
        answer=answer,
        context="\n\n".join(contexts),
    )
    return judge(prompt)


def _overall_label(score: float) -> str:
    if score >= 0.9:
        return "回答质量优秀"
    if score >= 0.75:
        return "回答质量良好"
    if score >= 0.6:
        return "回答质量一般，仍有改进空间"
    return "回答质量较差，建议优化检索或生成策略"


def _compute_overall(faith: dict, relevancy: dict, utilization: dict) -> float:
    weights = config.EVAL_WEIGHTS
    return (
        weights["faithfulness"] * faith["score"]
        + weights["answer_relevancy"] * relevancy["score"]
        + weights["context_utilization"] * utilization["score"]
    )


def evaluate_rag(questions, answers, contexts):
    if len(questions) == 0:
        return None

    question = questions[-1]
    answer = answers[-1]
    context = contexts[-1]

    faith = evaluate_faithfulness(question, answer, context)
    relevancy = evaluate_answer_relevancy(question, answer)
    utilization = evaluate_context_utilization(question, answer, context)

    overall = _compute_overall(faith, relevancy, utilization)

    return {
        "faithfulness": faith,
        "answer_relevancy": relevancy,
        "context_utilization": utilization,
        "overall_score": round(overall, 3),
        "overall_label": _overall_label(overall),
    }


def get_metric_scores(result: dict) -> dict:
    return {
        "faithfulness": result["faithfulness"]["score"],
        "answer_relevancy": result["answer_relevancy"]["score"],
        "context_utilization": result["context_utilization"]["score"],
        "overall_score": result["overall_score"],
    }


def build_evaluation_table(result: dict) -> list:
    """构建企业风格评估表格数据"""
    return [
        {
            "指标": "Faithfulness",
            "分数": f"{result['faithfulness']['score']:.2f}",
            "评价": result["faithfulness"]["reason"],
        },
        {
            "指标": "Answer Relevancy",
            "分数": f"{result['answer_relevancy']['score']:.2f}",
            "评价": result["answer_relevancy"]["reason"],
        },
        {
            "指标": "Context Utilization",
            "分数": f"{result['context_utilization']['score']:.2f}",
            "评价": result["context_utilization"]["reason"],
        },
        {
            "指标": "Overall Score",
            "分数": f"{result['overall_score']:.2f}",
            "评价": result["overall_label"],
        },
    ]
