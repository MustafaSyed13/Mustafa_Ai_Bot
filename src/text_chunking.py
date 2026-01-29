from __future__ import annotations
from typing import List

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 180) -> List[str]:
    text = (text or '').replace('\x00', ' ')
    text = ' '.join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks
