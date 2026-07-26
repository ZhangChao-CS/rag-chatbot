import gc
import hashlib
import tempfile

import pandas as pd
import streamlit as st

import config
from rag.bm25_store import build_bm25_index
from rag.evaluator import build_evaluation_table, evaluate_rag
from rag.loader import load_and_chunk
from rag.retrieval_service import RetrievalService
from rag.utils import check_memory_available, cleanup_memory, get_memory_info
from rag.vector_store import build_vector_store, save_vector_store

from tools.retrieval_tool import RetrievalTool
from tools.calculator_tool import CalculatorTool

from agent.simple_agent import SimpleAgent

# ===== session_state =====
for key, default in [
    ("history", []),
    ("eval_questions", []),
    ("eval_answers", []),
    ("eval_contexts", []),
    ("db_loaded", False),
    ("doc_hash", None),
    ("db", None),
    ("chunks", None),
    ("bm25_index", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _clear_document_state():
    st.session_state.db = None
    st.session_state.chunks = None
    st.session_state.bm25_index = None
    st.session_state.db_loaded = False
    st.session_state.doc_hash = None
    cleanup_memory()
    gc.collect()


def load_document(file_bytes: bytes, file_path: str):
    """加载文档到 session_state，同一文件不重复索引。"""
    doc_hash = _file_hash(file_bytes)
    if st.session_state.doc_hash == doc_hash and st.session_state.db is not None:
        return

    ok, msg = check_memory_available(config.MIN_AVAILABLE_MEMORY_MB)
    if not ok:
        raise MemoryError(msg)

    _clear_document_state()

    chunks = load_and_chunk(file_path)
    ok, msg = check_memory_available(config.MIN_AVAILABLE_MEMORY_MB)
    if not ok:
        raise MemoryError(msg)

    db = build_vector_store(chunks)
    save_vector_store(db)
    bm25_index = build_bm25_index(chunks)
    cleanup_memory()

    st.session_state.db = db
    st.session_state.chunks = chunks
    st.session_state.bm25_index = bm25_index
    st.session_state.doc_hash = doc_hash
    st.session_state.db_loaded = True

st.set_page_config(page_title="RAG问答系统", page_icon="📚", layout="wide")
st.title("📚 我的RAG问答系统")

# ===== 侧边栏 =====
st.sidebar.markdown("## ⚙️ 系统配置")
mem_info = get_memory_info()
if mem_info["available_mb"] != float("inf"):
    st.sidebar.metric("系统可用内存", f"{mem_info['available_mb']:.0f} MB")
    st.sidebar.metric("进程内存", f"{mem_info['process_mb']:.1f} MB")

eval_mode = st.sidebar.checkbox("🔬 开启LLM评估模式", value=False)
use_rerank = st.sidebar.checkbox(
    "📊 启用重排序",
    value=False,
    help="重排序会额外加载一个模型（约 500MB+），内存紧张时请关闭",
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 检索策略")
st.sidebar.markdown("✅ BM25关键词检索\n✅ 稠密向量检索\n✅ RRF融合算法")
if use_rerank:
    st.sidebar.markdown("✅ 重排序优化")

# ===== 文档上传 =====
uploaded_file = st.file_uploader("上传你的PDF文件", type="pdf")
db, chunks, bm25_index = None, None, None

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        try:
            with st.spinner("正在加载文档（首次加载可能需要30秒）..."):
                load_document(file_bytes, tmp.name)
            st.success("✅ 文档加载完成！")
        except MemoryError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"文档加载失败：{e}")
else:
    st.warning("请先上传你的PDF文件")

db = st.session_state.db
chunks = st.session_state.chunks
bm25_index = st.session_state.bm25_index

# ===== 问答 =====
question = st.text_input("请输入你的问题：")
if st.button("发送") and question and db is not None:
    history_text = "".join(
        f"用户：{q}\nAI：{a}\n" for q, a in st.session_state.history[-3:]
    )
    with st.spinner("正在检索和生成回答..."):
        retrieval_service = RetrievalService(db, chunks, bm25_index)
        retrieval_tool = RetrievalTool(retrieval_service)
        calculator_tool = CalculatorTool()

        agent = SimpleAgent([
            retrieval_tool,
            calculator_tool
        ])

        answer, docs = agent.run(question, history_text, use_rerank=use_rerank)

    st.session_state.history.append((question, answer))
    if eval_mode:
        st.session_state.eval_questions.append(question)
        st.session_state.eval_answers.append(answer)
        st.session_state.eval_contexts.append([doc.page_content for doc in docs])

    st.write("## 💬 对话记录")
    for q, a in st.session_state.history[-5:]:
        st.write("🙋‍♂️", q)
        st.write("🤖", a)

    cleanup_memory()

# ===== LLM-as-a-Judge 评估 =====
if eval_mode and st.session_state.eval_questions:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 RAG 评估报告")
    if st.sidebar.button("🚀 运行评估（LLM-as-a-Judge）"):
        with st.spinner("正在评估（约需数秒）..."):
            result = evaluate_rag(
                st.session_state.eval_questions,
                st.session_state.eval_answers,
                st.session_state.eval_contexts,
            )
        if result is None:
            st.sidebar.error("评估失败，请稍后重试")
        else:
            st.sidebar.metric(
                "Overall Score",
                f"{result['overall_score']:.2f}",
                help=result["overall_label"],
            )
            table = build_evaluation_table(result)
            st.sidebar.dataframe(
                pd.DataFrame(table),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("查看详细评估理由"):
                st.markdown(f"**Faithfulness**：{result['faithfulness']['reason']}")
                st.markdown(
                    f"**Answer Relevancy**：{result['answer_relevancy']['reason']}"
                )
                st.markdown(
                    f"**Context Utilization**：{result['context_utilization']['reason']}"
                )
                st.markdown(f"**综合结论**：{result['overall_label']}")

# ===== 系统状态 =====
if st.session_state.db_loaded:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ✅ 系统状态")
    st.sidebar.success("✅ 文档已加载")
    if chunks:
        st.sidebar.info(f"📄 文档块数: {len(chunks)}")
    st.sidebar.info(f"💬 对话轮数: {len(st.session_state.history)}")
    if st.sidebar.button("🧹 清理内存"):
        _clear_document_state()
        st.session_state.history.clear()
        st.sidebar.success("内存已清理！")
        st.rerun()
