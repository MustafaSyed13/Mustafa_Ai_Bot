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
