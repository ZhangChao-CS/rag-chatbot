import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
EMBEDDING_MODEL_PATH = MODELS_DIR / "embedding" / "bge-m3"
RERANKER_MODEL_PATH = MODELS_DIR / "reranker" / "bge-reranker-base"
FAISS_DB_DIR = BASE_DIR / "faiss_db"

# 本地模型不存在时回退到 HuggingFace
EMBEDDING_MODEL = (
    str(EMBEDDING_MODEL_PATH) if EMBEDDING_MODEL_PATH.exists() else "BAAI/bge-m3"
)
RERANKER_MODEL = (
    str(RERANKER_MODEL_PATH) if RERANKER_MODEL_PATH.exists() else "BAAI/bge-reranker-base"
)

API_KEY = os.getenv("ZHIPU_API_KEY", "aca83aa24bf24542a898dcac616fb9ec.D3PVknZeuwQ1upWh")
LLM_MODEL = "glm-4-flash"

# 文档处理
MAX_PAGES = 50
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MAX_CHUNKS = 100
# 检索参数
VECTOR_K = 5
BM25_K = 5
RRF_K = 60
FUSED_TOP_K = 10
RERANK_TOP_K = 5
FINAL_TOP_K = 3

# LLM-as-a-Judge 评估权重
EVAL_WEIGHTS = {
    "faithfulness": 0.4,
    "answer_relevancy": 0.35,
    "context_utilization": 0.25,
}

# 内存优化
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
