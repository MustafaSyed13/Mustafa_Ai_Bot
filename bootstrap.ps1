$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path ".\src",".\data\chroma_db",".\sample_docs" | Out-Null
New-Item -ItemType File -Force -Path ".\src\__init__.py" | Out-Null

Set-Content -Encoding UTF8 -Path ".\requirements.txt" -Value @"
streamlit
chromadb
pypdf
ollama
"@

Set-Content -Encoding UTF8 -Path ".\.gitignore" -Value @"
venv/
__pycache__/
*.pyc
data/chroma_db/
.DS_Store
"@

Set-Content -Encoding UTF8 -Path ".\src\text_chunking.py" -Value @"
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
"@

Set-Content -Encoding UTF8 -Path ".\src\pdf_ingest.py" -Value @"
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
"@

Set-Content -Encoding UTF8 -Path ".\src\vectorstore.py" -Value @"
from __future__ import annotations
from typing import List, Dict, Any
import chromadb
import ollama

def get_collection(persist_dir: str, name: str):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=name)

def embed_texts(texts: List[str], embed_model: str) -> List[List[float]]:
    vecs: List[List[float]] = []
    for t in texts:
        r = ollama.embeddings(model=embed_model, prompt=t)
        vecs.append(r['embedding'])
    return vecs

def upsert_chunks(collection, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]], embed_model: str):
    embeddings = embed_texts(documents, embed_model)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

def query_similar(collection, query: str, embed_model: str, top_k: int = 5) -> Dict[str, Any]:
    q_emb = embed_texts([query], embed_model)[0]
    return collection.query(query_embeddings=[q_emb], n_results=top_k, include=['documents','metadatas','distances'])
"@

Set-Content -Encoding UTF8 -Path ".\src\prompts.py" -Value @"
SYSTEM_RAG = '''You are Campus Copilot, an offline assistant for engineering students.

Rules:
- Use ONLY the provided SOURCES to answer.
- If the answer is not supported by the sources, say you don't know and suggest what to look up.
- Always include citations like [S1], [S2] matching the numbered sources provided.
- Do NOT invent page numbers or filenames.
'''

RAG_TEMPLATE = '''Question:
{question}

SOURCES:
{sources}

Write a helpful answer using ONLY the SOURCES. Add citations like [S1] where used.
'''

QUIZ_TEMPLATE = '''Create a quiz using ONLY the SOURCES.
Requirements:
- {n} questions
- Mix multiple choice + short answer
- Provide the answer + 1-2 sentence explanation
- Include citations [S#]

Topic/query: {query}

SOURCES:
{sources}
'''

FLASHCARDS_TEMPLATE = '''Create {n} flashcards using ONLY the SOURCES.
Return in CSV format with header: Front,Back
- Front: question/term
- Back: short answer/definition (with citation like [S#])

Topic/query: {query}

SOURCES:
{sources}
'''

CODE_EXPLAIN_TEMPLATE = '''You are a code tutor for engineering students.
Explain the code clearly and practically.

Language: {language}

Tasks:
1) Brief summary of what it does
2) Line-by-line explanation (group lines if long)
3) Bugs / edge cases
4) Improvements (keep it course-style, not over-engineered)

Code:
{code}
'''
"@

Set-Content -Encoding UTF8 -Path ".\src\llm.py" -Value @"
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
"@

Set-Content -Encoding UTF8 -Path ".\src\rag.py" -Value @"
from __future__ import annotations
from typing import List, Dict, Any
import hashlib

from .pdf_ingest import extract_pdf_pages
from .text_chunking import chunk_text
from .vectorstore import get_collection, upsert_chunks, query_similar
from .llm import chat
from .prompts import RAG_TEMPLATE

def _id_for(file_name: str, page: int, chunk_i: int, text: str) -> str:
    s = f'{file_name}|{page}|{chunk_i}|{text}'
    return hashlib.sha1(s.encode('utf-8', errors='ignore')).hexdigest()

def ingest_pdfs_to_chroma(uploaded_files, collection_name: str, embed_model: str, persist_dir: str) -> Dict[str, Any]:
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
                metas.append({'file_name': file_name, 'page': page_num})
                total_chunks += 1

    if total_chunks > 0:
        upsert_chunks(collection, ids, docs, metas, embed_model)

    return {'files': len(uploaded_files), 'pages': total_pages, 'chunks': total_chunks, 'collection': collection_name}

def answer_question_with_citations(question: str, collection_name: str, chat_model: str, embed_model: str, persist_dir: str, top_k: int = 5) -> Dict[str, Any]:
    collection = get_collection(persist_dir, collection_name)
    res = query_similar(collection, question, embed_model, top_k=top_k)

    docs = res.get('documents', [[]])[0]
    metas = res.get('metadatas', [[]])[0]

    sources = []
    sources_text_lines = []
    for idx, (d, m) in enumerate(zip(docs, metas), start=1):
        sources.append({'text': d, 'file_name': m.get('file_name','?'), 'page': m.get('page','?')})
        sources_text_lines.append(f'[S{idx}] {m.get(\"file_name\",\"?\")} (page {m.get(\"page\",\"?\")}):\\n{d}')

    sources_block = '\\n\\n'.join(sources_text_lines) if sources_text_lines else 'No sources found.'
    prompt = RAG_TEMPLATE.format(question=question, sources=sources_block)
    answer = chat(chat_model, prompt)

    return {'answer': answer, 'sources': sources}
"@

Set-Content -Encoding UTF8 -Path ".\src\quiz.py" -Value @"
from __future__ import annotations
from .vectorstore import get_collection, query_similar
from .llm import chat
from .prompts import QUIZ_TEMPLATE

def generate_quiz_from_query(query: str, collection_name: str, chat_model: str, embed_model: str, persist_dir: str, num_questions: int = 10) -> str:
    collection = get_collection(persist_dir, collection_name)
    res = query_similar(collection, query, embed_model, top_k=6)

    docs = res.get('documents', [[]])[0]
    metas = res.get('metadatas', [[]])[0]

    sources_lines = []
    for idx, (d, m) in enumerate(zip(docs, metas), start=1):
        sources_lines.append(f'[S{idx}] {m.get(\"file_name\",\"?\")} (page {m.get(\"page\",\"?\")}):\\n{d}')
    sources_block = '\\n\\n'.join(sources_lines) if sources_lines else 'No sources found.'

    prompt = QUIZ_TEMPLATE.format(n=num_questions, query=query, sources=sources_block)
    return chat(chat_model, prompt)
"@

Set-Content -Encoding UTF8 -Path ".\src\flashcards.py" -Value @"
from __future__ import annotations
from .vectorstore import get_collection, query_similar
from .llm import chat
from .prompts import FLASHCARDS_TEMPLATE

def generate_flashcards_from_query(query: str, collection_name: str, chat_model: str, embed_model: str, persist_dir: str, num_cards: int = 15) -> str:
    collection = get_collection(persist_dir, collection_name)
    res = query_similar(collection, query, embed_model, top_k=6)

    docs = res.get('documents', [[]])[0]
    metas = res.get('metadatas', [[]])[0]

    sources_lines = []
    for idx, (d, m) in enumerate(zip(docs, metas), start=1):
        sources_lines.append(f'[S{idx}] {m.get(\"file_name\",\"?\")} (page {m.get(\"page\",\"?\")}):\\n{d}')
    sources_block = '\\n\\n'.join(sources_lines) if sources_lines else 'No sources found.'

    prompt = FLASHCARDS_TEMPLATE.format(n=num_cards, query=query, sources=sources_block)
    return chat(chat_model, prompt)
"@

Set-Content -Encoding UTF8 -Path ".\src\code_explain.py" -Value @"
from __future__ import annotations
from .llm import chat
from .prompts import CODE_EXPLAIN_TEMPLATE

def explain_code(code: str, language: str, chat_model: str) -> str:
    prompt = CODE_EXPLAIN_TEMPLATE.format(code=code, language=language)
    return chat(chat_model, prompt)
"@

Set-Content -Encoding UTF8 -Path ".\app.py" -Value @"
import streamlit as st

from src.rag import ingest_pdfs_to_chroma, answer_question_with_citations
from src.quiz import generate_quiz_from_query
from src.flashcards import generate_flashcards_from_query
from src.code_explain import explain_code

st.set_page_config(page_title='Campus Copilot (Offline)', layout='wide')
st.title('📚 Campus Copilot — Offline RAG + Study Tools')

with st.sidebar:
    st.header('Settings')
    chat_model = st.text_input('Chat model (Ollama)', value='llama3.1:8b')
    embed_model = st.text_input('Embedding model (Ollama)', value='nomic-embed-text')
    collection_name = st.text_input('Course collection (separate per course)', value='default_course')

tab1, tab2, tab3, tab4 = st.tabs(['📄 Ingest PDFs', '❓ Ask (Citations)', '📝 Quiz/Flashcards', '💻 Code Explainer'])

with tab1:
    st.subheader('Upload PDFs → build local vector DB')
    files = st.file_uploader('Upload one or more PDFs', type=['pdf'], accept_multiple_files=True)
    if st.button('Ingest PDFs', type='primary', disabled=not files):
        with st.spinner('Extracting, chunking, embedding, storing...'):
            stats = ingest_pdfs_to_chroma(
                uploaded_files=files,
                collection_name=collection_name,
                embed_model=embed_model,
                persist_dir='data/chroma_db',
            )
        st.success('Ingest complete.')
        st.json(stats)

with tab2:
    st.subheader('Ask questions about your files (with citations)')
    q = st.text_input('Question', placeholder='e.g., Explain recursion base case from Lab 3.')
    k = st.slider('Top-K source chunks', 2, 10, 5)
    if st.button('Answer with citations', type='primary', disabled=not q):
        with st.spinner('Retrieving sources + generating answer...'):
            result = answer_question_with_citations(
                question=q,
                collection_name=collection_name,
                chat_model=chat_model,
                embed_model=embed_model,
                persist_dir='data/chroma_db',
                top_k=k,
            )
        st.markdown('### ✅ Answer')
        st.write(result['answer'])
        st.markdown('### 📌 Sources')
        for i, s in enumerate(result['sources'], start=1):
            with st.expander(f\"Source {i}: {s['file_name']} (page {s['page']})\"):
                st.write(s['text'])

with tab3:
    st.subheader('Generate study material (grounded in your sources)')
    topic = st.text_input('Topic/query for quiz', placeholder='e.g., pointers, FSM, pipelining')
    n_q = st.slider('Quiz questions', 5, 30, 10)
    if st.button('Generate quiz', type='primary', disabled=not topic):
        with st.spinner('Generating quiz...'):
            out = generate_quiz_from_query(
                query=topic,
                collection_name=collection_name,
                chat_model=chat_model,
                embed_model=embed_model,
                persist_dir='data/chroma_db',
                num_questions=n_q,
            )
        st.markdown('### 📝 Quiz')
        st.write(out)

    st.divider()

    topic2 = st.text_input('Topic/query for flashcards', placeholder='e.g., Big-O, recursion, malloc/free', key='fc')
    n_fc = st.slider('Flashcards', 5, 40, 15)
    if st.button('Generate flashcards (CSV)', disabled=not topic2):
        with st.spinner('Generating flashcards...'):
            out = generate_flashcards_from_query(
                query=topic2,
                collection_name=collection_name,
                chat_model=chat_model,
                embed_model=embed_model,
                persist_dir='data/chroma_db',
                num_cards=n_fc,
            )
        st.markdown('### 🗂 Flashcards CSV')
        st.code(out)

with tab4:
    st.subheader('Paste code → explanation + bugs + improvements')
    lang = st.selectbox('Language', ['C','Java','VHDL','Python','Other'], index=0)
    code = st.text_area('Code', height=300, placeholder='Paste your code here...')
    if st.button('Explain code', type='primary', disabled=not code.strip()):
        with st.spinner('Analyzing code...'):
            out = explain_code(code=code, language=lang, chat_model=chat_model)
        st.markdown('### 💡 Explanation')
        st.write(out)
"@

Write-Host "Created files. Next step: run .\bootstrap.ps1" -ForegroundColor Green
