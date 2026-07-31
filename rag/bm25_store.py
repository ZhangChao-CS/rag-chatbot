import re
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

import config


def tokenize_text(text):
    # 检测是否包含中文字符
    if re.search(r'[\u4e00-\u9fff]', text):
        return list(jieba.cut(text))
    else:
        # 英文：去标点 + 小写 + 空格分词
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower().split()


def build_bm25_index(chunks):
    tokenized = [tokenize_text(chunk.page_content) for chunk in chunks]
    return BM25Okapi(tokenized)


def bm25_retrieve(query: str, bm25_index, chunks, k=None):
    k = k or config.BM25_K
    tokenized_query = list(jieba.cut(query))
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-k:][::-1]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append(
                {"document": chunks[idx], "score": scores[idx], "index": idx}
            )
    return results
