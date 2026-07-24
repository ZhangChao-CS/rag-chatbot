from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS

import config
from rag.embedding import get_embeddings


def build_vector_store(chunks) -> FAISS:
    return FAISS.from_documents(chunks, get_embeddings())


def save_vector_store(db: FAISS, path: Optional[Path] = None) -> Path:
    save_path = path or config.FAISS_DB_DIR
    save_path.mkdir(parents=True, exist_ok=True)
    db.save_local(str(save_path))
    return save_path


def load_vector_store(path: Optional[Path] = None) -> FAISS:
    load_path = path or config.FAISS_DB_DIR
    return FAISS.load_local(
        str(load_path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
