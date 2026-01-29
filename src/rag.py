from __future__ import annotations
from typing import List, Dict, Any
import hashlib

from .pdf_ingest import extract_pdf_pages
from .text_chunking import chunk_text
from .vectorstore import get_collection, upsert_chunks, query_similar
from .llm import chat
from .prompts import RAG_TEMPLATE


def _id_for(file_name: str, page: int, chunk_i: int, text: str) -> str:
    s = f"{file_name}|{page}|{chunk_i}|{text}"
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def ingest_pdfs_to_chroma(
    uploaded_files,
    collection_name: str,
    embed_model: str,
    persist_dir: str,
) -> Dict[str, Any]:
    collection = get_collection(persist_dir, collection_name)

    total_pages = 0
    total_chunks = 0

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for uf in uploaded_files:
        file_name = uf.name
        pdf_bytes = uf.read()
        pages = extract_pdf_pages(pdf_bytes)
        total_pages += len(pages)

        for page_num, page_text in pages:
            chunks = chunk_text(page_text, chunk_size=900, overlap=180)
            for i, ch in enumerate(chunks):
                cid = _id_for(file_name, page_num, i, ch)
                ids.append(cid)
                docs.append(ch)
                metas.append({"file_name": file_name, "page": page_num})
                total_chunks += 1

    if total_chunks > 0:
        upsert_chunks(collection, ids, docs, metas, embed_model)

    return {
        "files": len(uploaded_files),
        "pages": total_pages,
        "chunks": total_chunks,
        "collection": collection_name,
    }


def answer_question_with_citations(
    question: str,
    collection_name: str,
    chat_model: str,
    embed_model: str,
    persist_dir: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    collection = get_collection(persist_dir, collection_name)
    res = query_similar(collection, question, embed_model, top_k=top_k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    sources: List[Dict[str, Any]] = []
    sources_text_lines: List[str] = []

    for idx, (d, m) in enumerate(zip(docs, metas), start=1):
        file_name = m.get("file_name", "?")
        page = m.get("page", "?")
        sources.append({"text": d, "file_name": file_name, "page": page})
        sources_text_lines.append(f"[S{idx}] {file_name} (page {page}):\n{d}")

    sources_block = "\n\n".join(sources_text_lines) if sources_text_lines else "No sources found."
    prompt = RAG_TEMPLATE.format(question=question, sources=sources_block)
    answer = chat(chat_model, prompt)

    return {"answer": answer, "sources": sources}
