FAITHFULNESS_PROMPT = """
你是一名RAG评测专家。
请根据下面提供的Context判断Answer是否完全依据Context生成，没有产生幻觉或编造内容。

Context:
{context}

Question:
{question}

Answer:
{answer}

输出JSON示例（按照以下格式输出，但不是就是输出以下内容）：
{{
    "score": 0.91,
    "reason": "回答完全依据Context，没有产生幻觉。"
}}

score 取值范围 0~1，越高表示越忠实于检索内容。
不要输出其它内容。
""".strip()

ANSWER_RELEVANCY_PROMPT = """
你是一名RAG评测专家。

Question:
{question}

Answer:
{answer}

请判断回答是否真正回答了问题，是否覆盖了问题的核心内容。

输出JSON示例（按照以下格式输出，但不是就是输出以下内容）：
{{
    "score": 0.93,
    "reason": "回答覆盖了问题所有内容。"
}}

score 取值范围 0~1，越高表示回答与问题越相关。
不要输出其它内容。
""".strip()

CONTEXT_UTILIZATION_PROMPT = """
你是一名RAG评测专家。
请评估Answer对检索Context的利用程度：是否充分使用了Context中的关键信息，是否存在遗漏重要内容的情况。

Context:
{context}

Question:
{question}

Answer:
{answer}

输出JSON示例（按照以下格式输出，但不是就是输出以下内容）：
{{
    "score": 0.89,
    "reason": "检索内容利用充分，关键信息覆盖较好。"
}}

score 取值范围 0~1，越高表示对检索内容的利用越充分。
不要输出其它内容。
""".strip()
