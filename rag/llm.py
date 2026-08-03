from zhipuai import ZhipuAI

import config

_client = ZhipuAI(api_key=config.API_KEY)


def ask_llm(prompt: str) -> str:
    try:
        response = _client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "请基于资料回答问题，不要胡编"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"抱歉，生成回答时出现错误：{e}"
