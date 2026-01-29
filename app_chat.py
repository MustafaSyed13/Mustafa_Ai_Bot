import streamlit as st
from src.rag import ingest_pdfs_to_chroma, answer_question_with_citations

st.set_page_config(page_title="Mustafa AI", layout="wide")
st.title("🤖 Mustafa AI")


# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_ingest" not in st.session_state:
    st.session_state.last_ingest = None

# ---------- sidebar (settings + upload) ----------
with st.sidebar:
    st.header("Settings")
    chat_model = st.text_input("Chat model (Ollama)", value="llama3.1:8b")
    embed_model = st.text_input("Embedding model (Ollama)", value="nomic-embed-text")
    collection_name = st.text_input("Course collection", value="default_course")

    st.divider()
    st.subheader("Upload PDFs")
    files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Ingest PDFs", type="primary", disabled=not files):
        with st.spinner("Extracting → chunking → embedding → storing..."):
            stats = ingest_pdfs_to_chroma(
                uploaded_files=files,
                collection_name=collection_name,
                embed_model=embed_model,
                persist_dir="data/chroma_db",
            )
        st.session_state.last_ingest = stats
        st.success("Ingest complete.")
        st.json(stats)

    if st.button("Clear chat"):
        st.session_state.messages = []

# ---------- render chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(msg["sources"], start=1):
                    st.markdown(f"**[S{i}] {s['file_name']} (page {s['page']})**")
                    st.write(s["text"])

# ---------- chat input (single type bar at bottom) ----------
user_text = st.chat_input("Ask a question about your PDFs…")

if user_text:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources + generating answer..."):
            result = answer_question_with_citations(
                question=user_text,
                collection_name=collection_name,
                chat_model=chat_model,
                embed_model=embed_model,
                persist_dir="data/chroma_db",
                top_k=5,
            )

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for i, s in enumerate(sources, start=1):
                    st.markdown(f"**[S{i}] {s['file_name']} (page {s['page']})**")
                    st.write(s["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
