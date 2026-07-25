import fitz

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

def extract_paragraphs(pdf_path):

    doc = fitz.open(pdf_path)
    paragraphs = []

    for page_id, page in enumerate(doc):
        text = page.get_text("text")
        if not text:
            continue
        # 按空行分段
        paras = text.split("\n\n")

        for p in paras:
            p = p.strip()

            if len(p) < 50:
                continue

            paragraphs.append(
                {
                    "text": p,
                    "page": page_id + 1
                }
            )

    return paragraphs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    return chunks[: config.MAX_CHUNKS]


def load_and_chunk(pdf_path):

    paragraphs = extract_paragraphs(pdf_path)
    documents=[]

    for item in paragraphs:
        documents.append(
            Document(
                page_content=item["text"],
                metadata={
                    "page":item["page"]
                }
            )
        )

    # 超长段落继续切

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ". ",
            " "
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks[:config.MAX_CHUNKS]
