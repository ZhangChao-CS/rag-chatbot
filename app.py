import gc
import tempfile

import pandas as pd
import streamlit as st

from rag.bm25_store import build_bm25_index
from rag.evaluator import build_evaluation_table, evaluate_rag
from rag.llm import ask_llm
from rag.loader import load_and_chunk
from rag.retrieval_service import RetrievalService
from rag.utils import cleanup_memory
from rag.vector_store import build_vector_store, save_vector_store

from tools.retrieval_tool import RetrievalTool

from agent.simple_agent import SimpleAgent

# ===== session_state =====
for key, default in [
    ("history", []),
    ("eval_questions", []),
    ("eval_answers", []),
    ("eval_contexts", []),
    ("db_loaded", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_resource
def load_db(file_path: str):
    chunks = load_and_chunk(file_path)
    db = build_vector_store(chunks)
    save_vector_store(db)
    bm25_index = build_bm25_index(chunks)
    cleanup_memory()
    st.session_state.db_loaded = True
    return db, chunks, bm25_index


st.set_page_config(page_title="RAG问答系统", page_icon="📚", layout="wide")
st.title("📚 我的RAG问答系统")

# ===== 侧边栏 =====
st.sidebar.markdown("## ⚙️ 系统配置")
if st.sidebar.checkbox("显示内存使用情况", value=False):
    try:
        import psutil
        proc = psutil.Process()
        st.sidebar.metric("内存使用", f"{proc.memory_info().rss / 1024 / 1024:.1f} MB")
        st.sidebar.metric("CPU使用率", f"{psutil.cpu_percent():.1f}%")
    except ImportError:
        pass

eval_mode = st.sidebar.checkbox("🔬 开启LLM评估模式", value=False)
use_rerank = st.sidebar.checkbox("📊 启用重排序", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 检索策略")
st.sidebar.markdown("✅ BM25关键词检索\n✅ 稠密向量检索\n✅ RRF融合算法")
if use_rerank:
    st.sidebar.markdown("✅ 重排序优化")

# ===== 文档上传 =====
uploaded_file = st.file_uploader("上传你的PDF文件", type="pdf")
db, chunks, bm25_index = None, None, None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        with st.spinner("正在加载文档（首次加载可能需要30秒）..."):
            db, chunks, bm25_index = load_db(tmp.name)
        st.success("✅ 文档加载完成！")
else:
    st.warning("请先上传你的PDF文件")

# ===== 问答 =====
question = st.text_input("请输入你的问题：")
if st.button("发送") and question and db is not None:
    history_text = "".join(
        f"用户：{q}\nAI：{a}\n" for q, a in st.session_state.history[-3:]
    )
    with st.spinner("正在检索和生成回答..."):
        retrieval_service = RetrievalService(db, chunks, bm25_index)
        retrieval_tool = RetrievalTool(retrieval_service)
        agent = SimpleAgent(retrieval_tool)

        answer, docs = agent.run(question, history_text)

    st.session_state.history.append((question, answer))
    if eval_mode:
        st.session_state.eval_questions.append(question)
        st.session_state.eval_answers.append(answer)
        st.session_state.eval_contexts.append([doc.page_content for doc in docs])

    st.write("## 💬 对话记录")
    for q, a in st.session_state.history[-5:]:
        st.write("🙋‍♂️", q)
        st.write("🤖", a)

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
        cleanup_memory()
        gc.collect()
        st.sidebar.success("内存已清理！")
        st.rerun()
