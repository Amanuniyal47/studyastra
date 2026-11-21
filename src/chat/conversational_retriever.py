"""Document-scoped conversational retriever for multi-turn PDF chat.

Stage 10.2 Implementation:
- Scopes semantic search strictly to the active document_id, preventing cross-document vector leakage.
- Integrates with EmbeddingService (MiniLM) and FAISSVectorStore.
- Respects CHAT_TOP_K configuration.
- Validates chunk metadata and formats clean chunk dictionaries.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

import src.config as config
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import retrieve_chunks
from src.retrieval.vector_store import FAISSVectorStore, get_default_vector_store

logger = logging.getLogger(__name__)


def retrieve_chat_context(
    query: str,
    document_id: str,
    *,
    store: FAISSVectorStore | None = None,
    embedding_service: EmbeddingService | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-K chunks scoped strictly to the specified document ID.

    Args:
        query: Search query (typically rewritten standalone query or original question).
        document_id: Target document ID. Must be a non-empty string.
        store: Optional FAISSVectorStore. If None, falls back to session state or default store.
        embedding_service: Optional EmbeddingService for query embedding.
        top_k: Optional number of chunks to retrieve. Defaults to config.CHAT_TOP_K.

    Returns:
        list[dict[str, Any]]: Retrieved chunk dictionaries guaranteed to belong to document_id.

    Raises:
        ValueError: If document_id is empty or missing, or if store is empty.
    """
    if not document_id or not isinstance(document_id, str) or not document_id.strip():
        raise ValueError(
            "document_id must be a non-empty string for document-scoped chat retrieval."
        )

    clean_doc_id = document_id.strip()

    if not query or not isinstance(query, str) or not query.strip():
        return []

    effective_top_k = config.CHAT_TOP_K if top_k is None else int(top_k)
    if effective_top_k <= 0:
        return []

    # Use the existing retrieve_chunks pipeline with document_id scoping
    try:
        chunks = retrieve_chunks(
            question=query.strip(),
            top_k=effective_top_k,
            document_id=clean_doc_id,
            store=store,
            embedding_service=embedding_service,
        )
    except ValueError as exc:
        logger.warning(
            "Retrieval failed for document '%s' with query '%s': %s",
            clean_doc_id,
            query,
            exc,
        )
        raise

    # Defensively verify that all returned chunks belong to clean_doc_id
    scoped_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_doc_id = chunk.get("metadata", {}).get("document_id") or chunk.get(
            "document_id"
        )
        if chunk_doc_id == clean_doc_id:
            scoped_chunks.append(chunk)
        else:
            logger.error(
                "Cross-document leakage detected: retrieved chunk document_id '%s' "
                "does not match requested document_id '%s'. Filtering chunk out.",
                chunk_doc_id,
                clean_doc_id,
            )

    return scoped_chunks
