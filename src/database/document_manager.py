"""Document registry managing indexed PDFs and their owned FAISS vector IDs.

Kept independent of FAISS, Streamlit, and LLM providers so document metadata
and per-document vector ownership remain clean and auditable.
"""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


REQUIRED_DOCUMENT_FIELDS = (
    "document_id",
    "filename",
    "page_count",
    "chunk_count",
    "vector_ids",
    "embedding_model",
    "embedding_dimension",
    "chunk_size_tokens",
    "chunk_overlap_tokens",
    "indexed_at",
)


class DocumentRegistryError(ValueError):
    """Raised when document registry data is invalid or inconsistent."""


class DocumentManager:
    """In-memory document registry serialized transactionally with the vector store."""

    def __init__(self, documents: dict[str, dict[str, Any]] | None = None) -> None:
        self._documents: dict[str, dict[str, Any]] = {}
        if documents is not None:
            self.load_from_dict(documents)

    @staticmethod
    def validate_record(record: dict[str, Any]) -> None:
        if not isinstance(record, dict):
            raise DocumentRegistryError("Document record must be a dictionary.")

        missing = [field for field in REQUIRED_DOCUMENT_FIELDS if field not in record]
        if missing:
            raise DocumentRegistryError(
                f"Document record is missing required field(s): {', '.join(missing)}."
            )

        if not isinstance(record["document_id"], str) or not record["document_id"].strip():
            raise DocumentRegistryError("document_id must be a non-empty string.")
        if not isinstance(record["filename"], str) or not record["filename"].strip():
            raise DocumentRegistryError("filename must be a non-empty string.")
        if not isinstance(record["page_count"], int) or record["page_count"] <= 0:
            raise DocumentRegistryError("page_count must be a positive integer.")
        if not isinstance(record["chunk_count"], int) or record["chunk_count"] <= 0:
            raise DocumentRegistryError("chunk_count must be a positive integer.")

        vector_ids = record["vector_ids"]
        if not isinstance(vector_ids, list) or not vector_ids:
            raise DocumentRegistryError("vector_ids must be a non-empty list of positive integers.")
        if any(not isinstance(vid, int) or isinstance(vid, bool) or vid <= 0 for vid in vector_ids):
            raise DocumentRegistryError("All vector_ids must be positive integers.")
        if len(set(vector_ids)) != len(vector_ids):
            raise DocumentRegistryError(
                f"Document '{record['document_id']}' contains duplicate vector_ids."
            )
        if len(vector_ids) != record["chunk_count"]:
            raise DocumentRegistryError(
                f"Document '{record['document_id']}' chunk_count ({record['chunk_count']}) "
                f"does not match len(vector_ids) ({len(vector_ids)})."
            )

        if not isinstance(record["embedding_model"], str) or not record["embedding_model"].strip():
            raise DocumentRegistryError("embedding_model must be a non-empty string.")
        if not isinstance(record["embedding_dimension"], int) or record["embedding_dimension"] <= 0:
            raise DocumentRegistryError("embedding_dimension must be a positive integer.")
        if not isinstance(record["chunk_size_tokens"], int) or record["chunk_size_tokens"] <= 0:
            raise DocumentRegistryError("chunk_size_tokens must be a positive integer.")
        if not isinstance(record["chunk_overlap_tokens"], int) or record["chunk_overlap_tokens"] < 0:
            raise DocumentRegistryError("chunk_overlap_tokens must be a non-negative integer.")
        if not isinstance(record["indexed_at"], str) or not record["indexed_at"].strip():
            raise DocumentRegistryError("indexed_at must be a non-empty ISO timestamp string.")

    def validate_all(self) -> None:
        seen_vector_ids: set[int] = set()
        for doc_id, record in self._documents.items():
            self.validate_record(record)
            if record["document_id"] != doc_id:
                raise DocumentRegistryError(
                    f"Document key '{doc_id}' does not match record document_id '{record['document_id']}'."
                )
            for vid in record["vector_ids"]:
                if vid in seen_vector_ids:
                    raise DocumentRegistryError(
                        f"Vector ID {vid} is assigned to multiple documents."
                    )
                seen_vector_ids.add(vid)

    def load_from_dict(self, data: dict[str, dict[str, Any]]) -> None:
        if not isinstance(data, dict):
            raise DocumentRegistryError("Documents payload must be a dictionary.")
        candidate = deepcopy(data)
        temp_manager = DocumentManager()
        temp_manager._documents = candidate
        temp_manager.validate_all()
        self._documents = candidate

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._documents)

    def clone(self) -> "DocumentManager":
        return DocumentManager(self.to_dict())

    def is_indexed(self, document_id: str) -> bool:
        return document_id in self._documents

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        record = self._documents.get(document_id)
        return deepcopy(record) if record is not None else None

    def list_documents(self) -> list[dict[str, Any]]:
        docs = [deepcopy(doc) for doc in self._documents.values()]
        docs.sort(key=lambda item: (item["indexed_at"], item["filename"]))
        return docs

    def count(self) -> int:
        return len(self._documents)

    def all_vector_ids(self) -> set[int]:
        ids: set[int] = set()
        for record in self._documents.values():
            ids.update(record["vector_ids"])
        return ids

    def get_vector_ids(self, document_id: str) -> list[int]:
        if document_id not in self._documents:
            raise KeyError(f"Document '{document_id}' is not indexed.")
        return list(self._documents[document_id]["vector_ids"])

    @staticmethod
    def build_record(
        *,
        document_id: str,
        filename: str,
        page_count: int,
        vector_ids: list[int],
        embedding_model: str,
        embedding_dimension: int,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
        indexed_at: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "document_id": document_id,
            "filename": filename,
            "page_count": int(page_count),
            "chunk_count": len(vector_ids),
            "vector_ids": [int(v) for v in vector_ids],
            "embedding_model": embedding_model,
            "embedding_dimension": int(embedding_dimension),
            "chunk_size_tokens": int(chunk_size_tokens),
            "chunk_overlap_tokens": int(chunk_overlap_tokens),
            "indexed_at": indexed_at or datetime.now(timezone.utc).isoformat(),
        }
        DocumentManager.validate_record(record)
        return record

    def register_document(self, record: dict[str, Any]) -> dict[str, Any]:
        self.validate_record(record)
        doc_id = record["document_id"]
        if doc_id in self._documents:
            raise DocumentRegistryError(f"Document '{doc_id}' is already registered.")

        existing_ids = self.all_vector_ids()
        overlap = existing_ids.intersection(record["vector_ids"])
        if overlap:
            raise DocumentRegistryError(
                f"Cannot register document '{doc_id}': vector ID(s) {sorted(overlap)} already exist."
            )

        stored = deepcopy(record)
        self._documents[doc_id] = stored
        return deepcopy(stored)

    def remove_document(self, document_id: str) -> dict[str, Any]:
        if document_id not in self._documents:
            raise KeyError(f"Document '{document_id}' is not indexed.")
        removed = self._documents.pop(document_id)
        return deepcopy(removed)
