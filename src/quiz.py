from __future__ import annotations
from .vectorstore import get_collection, query_similar
from .llm import chat
from .prompts import QUIZ_TEMPLATE

def generate_quiz_from_query(
    query: str,
    collection_name: str,
    chat_model: str,
    embed_model: str,
    persist_dir: str,
    num_questions: int = 10
) -> str:
    collection = get_collection(persist_dir, collection_name)
    res = query_similar(collection, query, embed_model, top_k=6)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    sources_lines = []
    for idx, (d, m) in enumerate(zip(docs, metas), start=1):
        file_name = m.get("file_name", "?")
        page = m.get("page", "?")
        sources_lines.append(f"[S{idx}] {file_name} (page {page}):\n{d}")

    sources_block = "\n\n".join(sources_lines) if sources_lines else "No sources found."
    prompt = QUIZ_TEMPLATE.format(n=num_questions, query=query, sources=sources_block)
    return chat(chat_model, prompt)
