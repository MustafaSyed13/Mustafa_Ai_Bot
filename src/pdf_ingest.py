from __future__ import annotations
from typing import List, Tuple
from pypdf import PdfReader
import io

def extract_pdf_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[Tuple[int, str]] = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ''
        pages.append((idx + 1, text))
    return pages
