"""High-level semantic retriever connecting EmbeddingService and FAISSVectorStore."""

from typing import Any

import streamlit as st

from src.config import EMBEDDING_MODEL, TOP_K
from src.retrieval.embeddings import EmbeddingService, preprocess_text
from src.retrieval.vector_store import FAISSVectorStore, get_default_vector_store


def retrieve_chunks(
    question: str,
    *,
    top_k: int | None = None,
    document_id: str | None = None,
    store: FAISSVectorStore | None = None,
    embedding_service: EmbeddingService | None = None,
) -> list[dict[str, Any]]:
    """Embed a query and retrieve the top-K most relevant PDF chunks."""
    cleaned_query = preprocess_text(question)

    service = embedding_service or EmbeddingService(model_name=EMBEDDING_MODEL)
    active_store = store
    if active_store is None:
        active_store = st.session_state.get("vector_store")
    if active_store is None:
        active_store = get_default_vector_store(embedding_service=service)

    if active_store.total_vectors == 0:
        raise ValueError("No PDF index is available. Upload a PDF first.")

    query_vector = service.embed_query(cleaned_query, normalize=True)
    k = TOP_K if top_k is None else top_k
    return active_store.search(query_vector, top_k=k, document_id=document_id)
