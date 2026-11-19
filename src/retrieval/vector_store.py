"""Persistent multi-PDF FAISS vector database with transactional commits.

Architecture:
- Uses `faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))` on L2-normalized
  embeddings so inner-product scores equal exact cosine similarity.
- Assigns stable, monotonically increasing `int64` vector IDs that are NEVER
  reused after document deletion.
- Persists each update as a complete generation (`generations/gen_<N>/`)
  containing `index.faiss`, `metadata.json`, and `documents.json`, then flips
  `manifest.json` atomically via `os.replace()`. Interrupted writes cannot
  corrupt the previously committed generation.
- Completely independent of Groq or any LLM provider.
"""

from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any

import faiss
import numpy as np
import streamlit as st

from src.config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    EMBEDDING_MODEL,
    TOP_K,
    VECTOR_STORE_DIR,
)
from src.database.document_manager import DocumentManager, DocumentRegistryError
from src.ingestion.chunker import chunk_text
from src.ingestion.pdf_loader import extract_pdf_text
from src.retrieval.embeddings import EmbeddingService
from src.utils.file_utils import file_hash

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
GENERATIONS_DIRNAME = "generations"
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"
DOCUMENTS_FILENAME = "documents.json"

REQUIRED_CHUNK_FIELDS = (
    "chunk_id",
    "vector_id",
    "document_id",
    "filename",
    "page",
    "page_range",
    "chunk_index",
    "text",
    "token_count",
)


class VectorStoreError(RuntimeError):
    """Base exception for FAISS vector store failures."""


class VectorStoreCompatibilityError(VectorStoreError):
    """Raised when the persisted index uses an incompatible embedding model or dimension."""


class VectorStoreCorruptionError(VectorStoreError):
    """Raised when persisted index, metadata, or manifest files are missing or corrupted."""


class VectorStorePersistenceError(VectorStoreError):
    """Raised when writing or committing the vector store to disk fails."""


class DuplicateDocumentError(VectorStoreError):
    """Raised when indexing an already-indexed document with skip_duplicates=False."""


def _write_json_fsync(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise VectorStoreCorruptionError(f"Missing {label} file: {path.name}.")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.exception("Failed to parse %s at %s", label, path)
        raise VectorStoreCorruptionError(
            f"Corrupted {label} file ({path.name}): unable to parse JSON."
        ) from exc

    if not isinstance(data, dict):
        raise VectorStoreCorruptionError(
            f"Corrupted {label} file ({path.name}): expected a JSON object."
        )
    return data


def _create_empty_index(dimension: int) -> faiss.IndexIDMap2:
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("embedding_dimension must be a positive integer.")
    return faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))


def _extract_faiss_ids(index: faiss.Index) -> list[int]:
    if not hasattr(index, "id_map"):
        raise VectorStoreCorruptionError(
            "Loaded FAISS index is not an ID-mapped index (expected IndexIDMap2)."
        )
    raw_ids = faiss.vector_to_array(index.id_map)
    return [int(v) for v in raw_ids.tolist()]


class FAISSVectorStore:
    """Persistent multi-PDF vector store backed by FAISS IndexIDMap2."""

    def __init__(
        self,
        store_dir: str | Path | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.store_dir = Path(VECTOR_STORE_DIR if store_dir is None else store_dir)

        if embedding_service is not None:
            resolved_model = embedding_model or embedding_service.model_name
            resolved_dim = (
                embedding_dimension
                if embedding_dimension is not None
                else embedding_service.get_dimension()
            )
        else:
            resolved_model = embedding_model or EMBEDDING_MODEL
            resolved_dim = embedding_dimension

        if not isinstance(resolved_model, str) or not resolved_model.strip():
            raise ValueError("embedding_model must be a non-empty string.")
        self.embedding_model = resolved_model.strip()

        self._embedding_service = embedding_service
        self.embedding_dimension: int | None = resolved_dim
        self._active_generation: int = 0
        self._next_vector_id: int = 1
        self._index: faiss.IndexIDMap2 | None = None
        self._chunks: dict[int, dict[str, Any]] = {}
        self._document_manager = DocumentManager()

        self.load_or_initialize()

    @property
    def manifest_path(self) -> Path:
        return self.store_dir / MANIFEST_FILENAME

    @property
    def generations_dir(self) -> Path:
        return self.store_dir / GENERATIONS_DIRNAME

    @property
    def total_vectors(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    @property
    def total_documents(self) -> int:
        return self._document_manager.count()

    @property
    def next_vector_id(self) -> int:
        return self._next_vector_id

    @property
    def active_generation(self) -> int:
        return self._active_generation

    def _resolve_dimension(self) -> int:
        if self.embedding_dimension is not None:
            return self.embedding_dimension
        service = self._embedding_service or EmbeddingService(model_name=self.embedding_model)
        self._embedding_service = service
        self.embedding_dimension = service.get_dimension()
        return self.embedding_dimension

    def _ensure_index_initialized(self) -> faiss.IndexIDMap2:
        if self._index is None:
            dim = self._resolve_dimension()
            self._index = _create_empty_index(dim)
        return self._index

    def _has_untracked_store_artifacts(self) -> bool:
        if not self.store_dir.exists():
            return False
        suspicious = [
            self.store_dir / INDEX_FILENAME,
            self.store_dir / METADATA_FILENAME,
            self.store_dir / DOCUMENTS_FILENAME,
        ]
        if any(path.exists() for path in suspicious):
            return True
        if self.generations_dir.is_dir():
            for child in self.generations_dir.iterdir():
                if child.is_dir() and not child.name.endswith(".staging"):
                    return True
        return False

    @staticmethod
    def _validate_consistency(
        *,
        index: faiss.Index,
        chunks: dict[int, dict[str, Any]],
        document_manager: DocumentManager,
        expected_dimension: int,
        next_vector_id: int,
    ) -> None:
        if index.d != expected_dimension:
            raise VectorStoreCompatibilityError(
                f"FAISS index dimension ({index.d}) does not match expected "
                f"embedding dimension ({expected_dimension})."
            )

        faiss_ids = _extract_faiss_ids(index)
        if len(faiss_ids) != int(index.ntotal):
            raise VectorStoreCorruptionError(
                f"FAISS id_map count ({len(faiss_ids)}) does not match index.ntotal ({index.ntotal})."
            )

        faiss_id_set = set(faiss_ids)
        if len(faiss_id_set) != len(faiss_ids):
            raise VectorStoreCorruptionError("FAISS index contains duplicate vector IDs.")

        if any(vid <= 0 for vid in faiss_ids):
            raise VectorStoreCorruptionError("FAISS index contains non-positive vector IDs.")

        if len(chunks) != int(index.ntotal):
            raise VectorStoreCorruptionError(
                f"Metadata chunk count ({len(chunks)}) does not match FAISS vector count ({index.ntotal})."
            )

        chunk_id_set = set(chunks.keys())
        if chunk_id_set != faiss_id_set:
            raise VectorStoreCorruptionError(
                "Metadata vector IDs do not match FAISS index vector IDs."
            )

        try:
            document_manager.validate_all()
        except DocumentRegistryError as exc:
            raise VectorStoreCorruptionError(f"Document registry validation failed: {exc}") from exc

        doc_vector_ids = document_manager.all_vector_ids()
        if doc_vector_ids != faiss_id_set:
            raise VectorStoreCorruptionError(
                "Document registry vector IDs do not match FAISS index vector IDs."
            )

        seen_chunk_strings: set[str] = set()
        per_doc_chunk_vids: dict[str, set[int]] = {}
        for vid, chunk in chunks.items():
            if not isinstance(chunk, dict):
                raise VectorStoreCorruptionError(f"Chunk metadata for vector {vid} is not a dict.")
            missing = [f for f in REQUIRED_CHUNK_FIELDS if f not in chunk]
            if missing:
                raise VectorStoreCorruptionError(
                    f"Chunk metadata for vector {vid} is missing field(s): {', '.join(missing)}."
                )
            if chunk["vector_id"] != vid:
                raise VectorStoreCorruptionError(
                    f"Chunk metadata vector_id ({chunk['vector_id']}) does not match key ({vid})."
                )
            chunk_str_id = chunk["chunk_id"]
            if not isinstance(chunk_str_id, str) or not chunk_str_id.strip():
                raise VectorStoreCorruptionError(f"Chunk {vid} has invalid chunk_id.")
            if chunk_str_id in seen_chunk_strings:
                raise VectorStoreCorruptionError(f"Duplicate chunk_id detected: '{chunk_str_id}'.")
            seen_chunk_strings.add(chunk_str_id)

            doc_id = chunk["document_id"]
            if not document_manager.is_indexed(doc_id):
                raise VectorStoreCorruptionError(
                    f"Chunk '{chunk_str_id}' references unregistered document '{doc_id}'."
                )
            per_doc_chunk_vids.setdefault(doc_id, set()).add(vid)

        for doc in document_manager.list_documents():
            doc_id = doc["document_id"]
            expected_vids = set(doc["vector_ids"])
            actual_vids = per_doc_chunk_vids.get(doc_id, set())
            if actual_vids != expected_vids:
                raise VectorStoreCorruptionError(
                    f"Chunk ownership mismatch for document '{doc_id}'."
                )

        max_id = max(faiss_ids, default=0)
        if not isinstance(next_vector_id, int) or next_vector_id <= max_id:
            raise VectorStoreCorruptionError(
                f"Invalid next_vector_id ({next_vector_id}); must be > max stored ID ({max_id})."
            )

    def load_or_initialize(self) -> None:
        """Load the active persisted generation or initialize a fresh empty store."""
        if not self.manifest_path.exists():
            if self._has_untracked_store_artifacts():
                raise VectorStoreCorruptionError(
                    "Vector store manifest.json is missing while store files exist. "
                    "The store may be corrupted or incomplete."
                )
            self._active_generation = 0
            self._next_vector_id = 1
            self._chunks = {}
            self._document_manager = DocumentManager()
            self._index = (
                _create_empty_index(self.embedding_dimension)
                if self.embedding_dimension is not None
                else None
            )
            return

        manifest = _read_json(self.manifest_path, "manifest")
        required_manifest = (
            "schema_version",
            "active_generation",
            "embedding_model",
            "dimension",
            "next_vector_id",
            "index_file",
            "metadata_file",
            "documents_file",
        )
        missing = [k for k in required_manifest if k not in manifest]
        if missing:
            raise VectorStoreCorruptionError(
                f"Corrupted manifest.json: missing field(s): {', '.join(missing)}."
            )

        stored_model = manifest["embedding_model"]
        stored_dim = manifest["dimension"]
        if not isinstance(stored_model, str) or not stored_model.strip():
            raise VectorStoreCorruptionError("Corrupted manifest.json: invalid embedding_model.")
        if not isinstance(stored_dim, int) or stored_dim <= 0:
            raise VectorStoreCorruptionError("Corrupted manifest.json: invalid dimension.")

        if stored_model != self.embedding_model:
            raise VectorStoreCompatibilityError(
                f"Incompatible embedding model: persisted index was built with "
                f"'{stored_model}', but current configuration uses '{self.embedding_model}'."
            )

        if self.embedding_dimension is not None and stored_dim != self.embedding_dimension:
            raise VectorStoreCompatibilityError(
                f"Incompatible embedding dimension: persisted index has dimension "
                f"{stored_dim}, but current model '{self.embedding_model}' has dimension "
                f"{self.embedding_dimension}."
            )

        index_path = self.store_dir / manifest["index_file"]
        metadata_path = self.store_dir / manifest["metadata_file"]
        documents_path = self.store_dir / manifest["documents_file"]

        if not index_path.is_file():
            raise VectorStoreCorruptionError(f"Missing FAISS index file: {manifest['index_file']}.")

        try:
            loaded_index = faiss.read_index(str(index_path))
        except Exception as exc:
            logger.exception("Failed to read FAISS index from %s", index_path)
            raise VectorStoreCorruptionError(
                f"Corrupted FAISS index file ({index_path.name}): unable to load."
            ) from exc

        metadata_payload = _read_json(metadata_path, "metadata")
        documents_payload = _read_json(documents_path, "documents")

        if metadata_payload.get("embedding_model") != stored_model:
            raise VectorStoreCorruptionError(
                "Metadata embedding_model does not match manifest embedding_model."
            )
        if metadata_payload.get("dimension") != stored_dim:
            raise VectorStoreCorruptionError(
                "Metadata dimension does not match manifest dimension."
            )

        raw_chunks = metadata_payload.get("chunks")
        if not isinstance(raw_chunks, dict):
            raise VectorStoreCorruptionError("Corrupted metadata.json: 'chunks' must be an object.")

        parsed_chunks: dict[int, dict[str, Any]] = {}
        for raw_key, chunk_record in raw_chunks.items():
            try:
                vid = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise VectorStoreCorruptionError(
                    f"Corrupted metadata.json: non-integer vector key '{raw_key}'."
                ) from exc
            parsed_chunks[vid] = chunk_record

        raw_docs = documents_payload.get("documents")
        if not isinstance(raw_docs, dict):
            raise VectorStoreCorruptionError(
                "Corrupted documents.json: 'documents' must be an object."
            )

        doc_manager = DocumentManager()
        try:
            doc_manager.load_from_dict(raw_docs)
        except DocumentRegistryError as exc:
            raise VectorStoreCorruptionError(f"Corrupted documents.json: {exc}") from exc

        next_vid = manifest["next_vector_id"]
        if metadata_payload.get("next_vector_id") != next_vid:
            raise VectorStoreCorruptionError(
                "Metadata next_vector_id does not match manifest next_vector_id."
            )

        self._validate_consistency(
            index=loaded_index,
            chunks=parsed_chunks,
            document_manager=doc_manager,
            expected_dimension=stored_dim,
            next_vector_id=next_vid,
        )

        self.embedding_dimension = stored_dim
        self._active_generation = int(manifest["active_generation"])
        self._next_vector_id = int(next_vid)
        self._index = loaded_index
        self._chunks = parsed_chunks
        self._document_manager = doc_manager

    def _cleanup_old_generations(self, keep_generation: int) -> None:
        if not self.generations_dir.is_dir():
            return
        keep_name = f"gen_{keep_generation}"
        for entry in self.generations_dir.iterdir():
            if entry.name == keep_name:
                continue
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except OSError:
                logger.warning("Could not clean up stale generation artifact: %s", entry)

    def _commit_transaction(
        self,
        *,
        next_index: faiss.IndexIDMap2,
        next_chunks: dict[int, dict[str, Any]],
        next_doc_manager: DocumentManager,
        next_vector_id: int,
    ) -> None:
        """Persist a full generation snapshot and atomically flip manifest.json."""
        dim = self._resolve_dimension()
        self._validate_consistency(
            index=next_index,
            chunks=next_chunks,
            document_manager=next_doc_manager,
            expected_dimension=dim,
            next_vector_id=next_vector_id,
        )

        new_generation = self._active_generation + 1
        gen_name = f"gen_{new_generation}"
        staging_dir = self.generations_dir / f"{gen_name}.staging"
        final_gen_dir = self.generations_dir / gen_name
        tmp_manifest = self.store_dir / f"{MANIFEST_FILENAME}.tmp"

        try:
            self.generations_dir.mkdir(parents=True, exist_ok=True)
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            if final_gen_dir.exists():
                shutil.rmtree(final_gen_dir)
            staging_dir.mkdir(parents=True, exist_ok=False)

            index_file = staging_dir / INDEX_FILENAME
            metadata_file = staging_dir / METADATA_FILENAME
            documents_file = staging_dir / DOCUMENTS_FILENAME

            faiss.write_index(next_index, str(index_file))
            if not index_file.is_file() or index_file.stat().st_size == 0:
                raise OSError("FAISS index file was not written cleanly.")

            metadata_payload = {
                "schema_version": SCHEMA_VERSION,
                "embedding_model": self.embedding_model,
                "dimension": dim,
                "next_vector_id": next_vector_id,
                "chunks": {str(vid): chunk for vid, chunk in next_chunks.items()},
            }
            _write_json_fsync(metadata_file, metadata_payload)

            documents_payload = {
                "schema_version": SCHEMA_VERSION,
                "documents": next_doc_manager.to_dict(),
            }
            _write_json_fsync(documents_file, documents_payload)

            os.replace(staging_dir, final_gen_dir)

            manifest_payload = {
                "schema_version": SCHEMA_VERSION,
                "active_generation": new_generation,
                "embedding_model": self.embedding_model,
                "dimension": dim,
                "next_vector_id": next_vector_id,
                "index_file": f"{GENERATIONS_DIRNAME}/{gen_name}/{INDEX_FILENAME}",
                "metadata_file": f"{GENERATIONS_DIRNAME}/{gen_name}/{METADATA_FILENAME}",
                "documents_file": f"{GENERATIONS_DIRNAME}/{gen_name}/{DOCUMENTS_FILENAME}",
            }
            _write_json_fsync(tmp_manifest, manifest_payload)
            os.replace(tmp_manifest, self.manifest_path)

        except Exception as exc:
            logger.exception("Failed to persist vector store generation %d", new_generation)
            for leftover in (staging_dir, final_gen_dir):
                if leftover.exists():
                    shutil.rmtree(leftover, ignore_errors=True)
            if tmp_manifest.exists():
                try:
                    tmp_manifest.unlink()
                except OSError:
                    pass
            raise VectorStorePersistenceError(
                f"Failed to save vector database to '{self.store_dir}': {exc}"
            ) from exc

        # Commit in-memory state ONLY after manifest flip succeeded
        self._index = next_index
        self._chunks = next_chunks
        self._document_manager = next_doc_manager
        self._next_vector_id = next_vector_id
        self._active_generation = new_generation
        self._cleanup_old_generations(keep_generation=new_generation)

    def is_document_indexed(self, document_id: str) -> bool:
        return self._document_manager.is_indexed(document_id)

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        return self._document_manager.get_document(document_id)

    def list_documents(self) -> list[dict[str, Any]]:
        return self._document_manager.list_documents()

    def get_document_chunks(
        self,
        document_id: str,
        *,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return deep-copied chunks strictly for one document in reading order."""
        if not self.is_document_indexed(document_id):
            raise KeyError(f"Document '{document_id}' is not indexed.")

        if start_page is not None and (
            not isinstance(start_page, int) or isinstance(start_page, bool) or start_page <= 0
        ):
            raise ValueError("start_page must be a positive integer.")
        if end_page is not None and (
            not isinstance(end_page, int) or isinstance(end_page, bool) or end_page <= 0
        ):
            raise ValueError("end_page must be a positive integer.")
        if start_page is not None and end_page is not None and start_page > end_page:
            raise ValueError("start_page cannot be greater than end_page.")

        vector_ids = self._document_manager.get_vector_ids(document_id)
        doc_chunks: list[dict[str, Any]] = []
        for vid in vector_ids:
            chunk = self._chunks.get(vid)
            if chunk is None:
                raise VectorStoreCorruptionError(
                    f"Document '{document_id}' references missing chunk metadata for vector {vid}."
                )
            if chunk["document_id"] != document_id:
                raise VectorStoreCorruptionError(
                    f"Vector {vid} belongs to '{chunk['document_id']}', not '{document_id}'."
                )
            page = int(chunk["page"])
            if start_page is not None and page < start_page:
                continue
            if end_page is not None and page > end_page:
                continue
            doc_chunks.append(deepcopy(chunk))

        doc_chunks.sort(
            key=lambda item: (int(item["page"]), int(item["chunk_index"]), int(item["vector_id"]))
        )
        return doc_chunks

    def get_document_page_numbers(self, document_id: str) -> list[int]:
        """Return sorted unique page numbers actually indexed for one document."""
        chunks = self.get_document_chunks(document_id)
        return sorted({int(chunk["page"]) for chunk in chunks})

    def add_document_chunks(
        self,
        *,
        document_id: str,
        filename: str,
        page_count: int,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
        chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
        skip_duplicates: bool = True,
    ) -> dict[str, Any]:
        """Add chunks and embeddings for one document atomically."""
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string.")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string.")

        if self.is_document_indexed(document_id):
            existing = self.get_document(document_id)
            if not skip_duplicates:
                raise DuplicateDocumentError(
                    f"Document '{filename}' (id={document_id[:12]}) is already indexed."
                )
            return {
                "status": "duplicate",
                "already_indexed": True,
                "document_id": document_id,
                "filename": existing["filename"],
                "page_count": existing["page_count"],
                "chunks_generated": existing["chunk_count"],
                "embeddings_generated": 0,
                "embedding_dimension": existing["embedding_dimension"],
                "total_vectors": self.total_vectors,
                "total_documents": self.total_documents,
                "vector_ids": list(existing["vector_ids"]),
            }

        if not chunks:
            raise ValueError("Cannot index a document with zero chunks.")

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Embeddings count ({vectors.shape[0] if vectors.ndim == 2 else 'invalid'}) "
                f"must match chunks count ({len(chunks)})."
            )
        if not np.all(np.isfinite(vectors)):
            raise ValueError("Embeddings contain NaN or Inf values.")

        if self.embedding_dimension is None:
            self.embedding_dimension = int(vectors.shape[1])
        expected_dim = self._resolve_dimension()
        if vectors.shape[1] != expected_dim:
            raise VectorStoreCompatibilityError(
                f"Embedding dimension mismatch: store expects {expected_dim}, "
                f"received {vectors.shape[1]}."
            )

        current_index = self._ensure_index_initialized()
        staged_index = faiss.clone_index(current_index)
        staged_chunks = deepcopy(self._chunks)
        staged_docs = self._document_manager.clone()

        allocated_ids = list(range(self._next_vector_id, self._next_vector_id + len(chunks)))
        next_vector_id = self._next_vector_id + len(chunks)

        for vid, chunk in zip(allocated_ids, chunks):
            text = chunk.get("text", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Cannot index an empty chunk.")

            chunk_index = int(chunk.get("chunk_index", 0))
            if chunk_index <= 0:
                raise ValueError("Chunk metadata must include a positive chunk_index.")

            page = int(chunk["page"])
            chunk_id = chunk.get("chunk_id") or f"{document_id[:12]}_p{page}_c{chunk_index}"

            staged_chunks[vid] = {
                "chunk_id": str(chunk_id),
                "vector_id": int(vid),
                "document_id": document_id,
                "filename": filename,
                "page": page,
                "page_range": str(chunk.get("page_range", page)),
                "chunk_index": chunk_index,
                "text": text,
                "token_count": int(chunk.get("token_count", len(text.split()))),
            }

        doc_record = DocumentManager.build_record(
            document_id=document_id,
            filename=filename,
            page_count=page_count,
            vector_ids=allocated_ids,
            embedding_model=self.embedding_model,
            embedding_dimension=expected_dim,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        )
        staged_docs.register_document(doc_record)

        id_array = np.array(allocated_ids, dtype=np.int64)
        staged_index.add_with_ids(vectors, id_array)

        self._commit_transaction(
            next_index=staged_index,
            next_chunks=staged_chunks,
            next_doc_manager=staged_docs,
            next_vector_id=next_vector_id,
        )

        return {
            "status": "indexed",
            "already_indexed": False,
            "document_id": document_id,
            "filename": filename,
            "page_count": page_count,
            "chunks_generated": len(chunks),
            "embeddings_generated": len(chunks),
            "embedding_dimension": expected_dim,
            "total_vectors": self.total_vectors,
            "total_documents": self.total_documents,
            "vector_ids": allocated_ids,
        }

    def add_pdf(
        self,
        pdf_file,
        *,
        filename: str | None = None,
        embedding_service: EmbeddingService | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        skip_duplicates: bool = True,
    ) -> dict[str, Any]:
        """Extract, chunk, embed, and persistently index an uploaded PDF."""
        doc_id = file_hash(pdf_file)
        resolved_filename = (
            filename
            or getattr(pdf_file, "name", None)
            or f"document_{doc_id[:8]}.pdf"
        )

        if self.is_document_indexed(doc_id):
            existing = self.get_document(doc_id)
            if not skip_duplicates:
                raise DuplicateDocumentError(
                    f"Document '{resolved_filename}' is already indexed."
                )
            return {
                "status": "duplicate",
                "already_indexed": True,
                "document_id": doc_id,
                "filename": existing["filename"],
                "page_count": existing["page_count"],
                "chunks_generated": existing["chunk_count"],
                "embeddings_generated": 0,
                "embedding_dimension": existing["embedding_dimension"],
                "total_vectors": self.total_vectors,
                "total_documents": self.total_documents,
                "vector_ids": list(existing["vector_ids"]),
            }

        text_by_page = extract_pdf_text(pdf_file)
        if not text_by_page:
            raise ValueError("No readable text was found in this PDF.")

        used_chunk_size = CHUNK_SIZE_TOKENS if chunk_size is None else chunk_size
        used_chunk_overlap = CHUNK_OVERLAP_TOKENS if chunk_overlap is None else chunk_overlap

        chunks = chunk_text(
            text_by_page,
            document_id=doc_id,
            filename=resolved_filename,
            chunk_size=used_chunk_size,
            chunk_overlap=used_chunk_overlap,
        )
        if not chunks:
            raise ValueError("No readable text was found in this PDF.")

        service = (
            embedding_service
            or self._embedding_service
            or EmbeddingService(model_name=self.embedding_model)
        )
        if service.model_name != self.embedding_model:
            raise VectorStoreCompatibilityError(
                f"EmbeddingService model '{service.model_name}' does not match "
                f"store model '{self.embedding_model}'."
            )

        texts = [chunk["text"] for chunk in chunks]
        embeddings = service.embed_texts(texts, normalize=True)

        return self.add_document_chunks(
            document_id=doc_id,
            filename=resolved_filename,
            page_count=len(text_by_page),
            chunks=chunks,
            embeddings=embeddings,
            chunk_size_tokens=used_chunk_size,
            chunk_overlap_tokens=used_chunk_overlap,
            skip_duplicates=skip_duplicates,
        )

    def delete_document(self, document_id: str) -> dict[str, Any]:
        """Remove a single document's vectors and metadata without affecting others."""
        if not self.is_document_indexed(document_id):
            raise KeyError(f"Document '{document_id}' is not indexed.")

        current_index = self._ensure_index_initialized()
        staged_index = faiss.clone_index(current_index)
        staged_chunks = deepcopy(self._chunks)
        staged_docs = self._document_manager.clone()

        removed_doc = staged_docs.remove_document(document_id)
        vector_ids = removed_doc["vector_ids"]

        removed_count = int(staged_index.remove_ids(np.array(vector_ids, dtype=np.int64)))
        if removed_count != len(vector_ids):
            raise VectorStoreCorruptionError(
                f"Expected to remove {len(vector_ids)} vectors for document "
                f"'{document_id}', but FAISS removed {removed_count}."
            )

        for vid in vector_ids:
            staged_chunks.pop(vid, None)

        # Monotonic next_vector_id is intentionally preserved (never decremented/reused)
        self._commit_transaction(
            next_index=staged_index,
            next_chunks=staged_chunks,
            next_doc_manager=staged_docs,
            next_vector_id=self._next_vector_id,
        )

        return {
            "status": "deleted",
            "document_id": document_id,
            "filename": removed_doc["filename"],
            "removed_vectors": removed_count,
            "total_vectors": self.total_vectors,
            "total_documents": self.total_documents,
        }

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search FAISS for top-K matching chunks by cosine similarity."""
        if self._index is None or self.total_vectors == 0:
            raise ValueError("No PDF index is available. Upload a PDF first.")

        k = TOP_K if top_k is None else top_k
        if not isinstance(k, int) or k <= 0:
            raise ValueError("top_k must be a positive integer.")

        qvec = np.ascontiguousarray(query_vector, dtype=np.float32)
        if qvec.ndim == 1:
            qvec = np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32)

        expected_dim = self._resolve_dimension()
        if qvec.ndim != 2 or qvec.shape != (1, expected_dim):
            raise VectorStoreCompatibilityError(
                f"Query vector shape {qvec.shape} does not match index dimension (1, {expected_dim})."
            )
        if not np.all(np.isfinite(qvec)):
            raise ValueError("Query vector contains non-finite values.")

        if document_id is not None and not self.is_document_indexed(document_id):
            return []

        search_k = self.total_vectors if document_id is not None else min(k, self.total_vectors)
        scores, ids = self._index.search(qvec, search_k)

        results: list[dict[str, Any]] = []
        for score, raw_vid in zip(scores[0], ids[0]):
            vid = int(raw_vid)
            if vid == -1:
                continue
            chunk = self._chunks.get(vid)
            if chunk is None:
                raise VectorStoreCorruptionError(
                    f"FAISS returned vector ID {vid} that is missing from metadata."
                )
            if document_id is not None and chunk["document_id"] != document_id:
                continue

            similarity = float(score)
            results.append(
                {
                    "text": chunk["text"],
                    "score": similarity,
                    "distance": float(1.0 - similarity),
                    "metadata": {
                        "chunk_id": chunk["chunk_id"],
                        "vector_id": chunk["vector_id"],
                        "document_id": chunk["document_id"],
                        "filename": chunk["filename"],
                        "page": chunk["page"],
                        "page_range": chunk["page_range"],
                        "chunk_index": chunk["chunk_index"],
                        "token_count": chunk["token_count"],
                    },
                }
            )
            if len(results) >= k:
                break

        return results

    def reset_store(self) -> None:
        """Delete persisted files and reset to an empty store."""
        if self.store_dir.exists():
            shutil.rmtree(self.store_dir)
        self._active_generation = 0
        self._next_vector_id = 1
        self._chunks = {}
        self._document_manager = DocumentManager()
        self._index = (
            _create_empty_index(self.embedding_dimension)
            if self.embedding_dimension is not None
            else None
        )


def get_default_vector_store(
    *,
    store_dir: str | Path | None = None,
    embedding_service: EmbeddingService | None = None,
) -> FAISSVectorStore:
    """Return a persistent FAISSVectorStore instance for the configured store directory."""
    service = embedding_service or EmbeddingService(model_name=EMBEDDING_MODEL)
    return FAISSVectorStore(
        store_dir=store_dir or VECTOR_STORE_DIR,
        embedding_service=service,
    )


def index_pdf(
    pdf_file,
    *,
    store: FAISSVectorStore | None = None,
    embedding_service: EmbeddingService | None = None,
) -> int:
    """Backwards-compatible wrapper used by existing app.py until Stage 4.

    Indexes `pdf_file` into the persistent vector store and returns its chunk count.
    """
    service = embedding_service or EmbeddingService(model_name=EMBEDDING_MODEL)
    active_store = store or st.session_state.get("vector_store")
    if active_store is None:
        active_store = get_default_vector_store(embedding_service=service)

    result = active_store.add_pdf(pdf_file, embedding_service=service, skip_duplicates=True)
    st.session_state["vector_store"] = active_store
    return int(result["chunks_generated"])
