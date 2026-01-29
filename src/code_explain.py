from __future__ import annotations
from .llm import chat
from .prompts import CODE_EXPLAIN_TEMPLATE

def explain_code(code: str, language: str, chat_model: str) -> str:
    prompt = CODE_EXPLAIN_TEMPLATE.format(code=code, language=language)
    return chat(chat_model, prompt)
