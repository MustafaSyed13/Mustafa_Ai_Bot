from __future__ import annotations
import ollama
from .prompts import SYSTEM_RAG

def chat(model: str, prompt: str) -> str:
    r = ollama.chat(
        model=model,
        messages=[
            {'role':'system', 'content': SYSTEM_RAG},
            {'role':'user', 'content': prompt},
        ],
    )
    return r['message']['content']
