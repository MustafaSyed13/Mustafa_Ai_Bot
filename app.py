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
            with st.expander(f"Source {i}: {s['file_name']} (page {s['page']})"):
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
