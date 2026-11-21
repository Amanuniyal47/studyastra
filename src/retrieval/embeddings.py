"""Local embedding service independent of any LLM provider.

Uses SentenceTransformers (default: `all-MiniLM-L6-v2`) for both document chunks
and search queries so preprocessing, dimensions, and vector normalization stay
strictly aligned and decoupled from Groq or future local LLMs (e.g., Ollama).
"""

import os
# Enforce offline operation for Hugging Face Hub and Transformers before library imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from collections.abc import Callable, Sequence
import logging

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment, misc]
import streamlit as st

from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

REQUIRED_CACHE_FILES = [
    "modules.json",
    "config.json",
    "model.safetensors",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def verify_offline_cache(model_name: str | None = None) -> tuple[bool, list[str]]:
    """Verify the presence of all 6 mandatory files for offline model loading.

    Uses huggingface_hub's cache inspection without importing torch or initializing
    heavy neural network weights.

    Returns:
        tuple[bool, list[str]]: (is_complete, missing_files)
    """
    target_model = (EMBEDDING_MODEL if model_name is None else model_name).strip()
    missing: list[str] = []

    try:
        from huggingface_hub import try_to_load_from_cache

        repo_id = target_model if "/" in target_model else f"sentence-transformers/{target_model}"
        for filename in REQUIRED_CACHE_FILES:
            res = try_to_load_from_cache(repo_id, filename)
            if not isinstance(res, str) or not os.path.isfile(res):
                missing.append(filename)
    except Exception as exc:
        logger.warning("Cache verification encountered exception: %s", exc)
        missing.append(f"Cache inspection error: {exc}")

    return (len(missing) == 0, missing)


class EmbeddingError(RuntimeError):
    """Base exception for embedding-related failures."""


class EmbeddingModelError(EmbeddingError):
    """Raised when the configured embedding model cannot be initialized."""


class EmbeddingGenerationError(EmbeddingError):
    """Raised when vector generation fails or produces invalid output."""


def preprocess_text(text: str, index: int | None = None) -> str:
    """Validate and normalize text identically for document chunks and queries."""
    label = f"Text at index {index}" if index is not None else "Input text"
    if not isinstance(text, str):
        raise ValueError(f"{label} must be a string.")

    cleaned = " ".join(text.split())
    if not cleaned:
        raise ValueError(f"{label} cannot be empty or whitespace.")

    return cleaned


@st.cache_resource(show_spinner=False)
def _load_cached_embedding_model(model_name: str) -> SentenceTransformer:
    if SentenceTransformer is None:
        raise EmbeddingModelError(
            "The 'sentence-transformers' package is not installed. "
            "Please install it using 'pip install sentence-transformers'."
        )
    try:
        logger.info("Loading embedding model '%s' from offline cache", model_name)
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        logger.exception("Failed to initialize embedding model '%s' offline", model_name)
        raise EmbeddingModelError(
            f"Unable to load embedding model '{model_name}' from local offline cache. "
            "Check that the model files are cached or run 'python scripts/prepare_model.py' to download them."
        ) from exc


def load_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model across Streamlit reruns.

    Resolves the default model name before cache lookup so calling
    `load_embedding_model()` and `load_embedding_model(config.EMBEDDING_MODEL)`
    always share the exact same cached model instance.
    """
    resolved_name = (EMBEDDING_MODEL if model_name is None else model_name).strip()
    if not resolved_name:
        raise EmbeddingModelError(
            "Embedding model name is empty. Set EMBEDDING_MODEL in config.py or .env."
        )
    return _load_cached_embedding_model(resolved_name)


load_embedding_model.clear = _load_cached_embedding_model.clear  # type: ignore[attr-defined]


class EmbeddingService:
    """Configurable local embedding generator for chunks and queries."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        resolved_name = (EMBEDDING_MODEL if model_name is None else model_name).strip()
        if not resolved_name:
            raise ValueError("model_name cannot be empty.")

        resolved_batch_size = EMBEDDING_BATCH_SIZE if batch_size is None else batch_size
        if not isinstance(resolved_batch_size, int) or resolved_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.model_name = resolved_name
        self.batch_size = resolved_batch_size
        self._dimension: int | None = None

    def _get_model(self) -> SentenceTransformer:
        return load_embedding_model(self.model_name)

    def get_dimension(self) -> int:
        """Return the embedding vector dimension for the configured model."""
        if self._dimension is not None:
            return self._dimension

        model = self._get_model()
        dim = None
        dim_getter = getattr(model, "get_embedding_dimension", None) or getattr(
            model, "get_sentence_embedding_dimension", None
        )
        if callable(dim_getter):
            try:
                dim = dim_getter()
            except Exception as exc:
                logger.exception(
                    "Failed to query embedding dimension for model '%s'",
                    self.model_name,
                )
                raise EmbeddingModelError(
                    f"Unable to determine embedding dimension for '{self.model_name}'."
                ) from exc

        if dim is None:
            probe = self._encode_raw(["dimension probe"], batch_size=1)
            dim = int(probe.shape[1])

        if not isinstance(dim, int) or dim <= 0:
            raise EmbeddingModelError(
                f"Embedding model '{self.model_name}' reported invalid dimension: {dim}."
            )

        self._dimension = dim
        return self._dimension

    def _encode_raw(self, cleaned_texts: list[str], batch_size: int) -> np.ndarray:
        model = self._get_model()
        try:
            raw = model.encode(
                cleaned_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.exception(
                "Embedding generation failed for %d text(s) with model '%s'",
                len(cleaned_texts),
                self.model_name,
            )
            raise EmbeddingGenerationError(
                f"Failed to generate embeddings using '{self.model_name}'."
            ) from exc

        vectors = np.ascontiguousarray(raw, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = np.ascontiguousarray(vectors.reshape(1, -1), dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[0] != len(cleaned_texts) or vectors.shape[1] <= 0:
            raise EmbeddingGenerationError(
                f"Unexpected embedding output shape {vectors.shape} for {len(cleaned_texts)} input(s)."
            )

        if not np.all(np.isfinite(vectors)):
            raise EmbeddingGenerationError(
                "Embedding model produced non-finite values (NaN or Inf)."
            )

        return vectors

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise EmbeddingGenerationError(
                "Embedding model produced a zero-norm vector that cannot be L2-normalized."
            )
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        normalize: bool = True,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> np.ndarray:
        """Generate validated embeddings for a non-empty sequence of texts.

        Rejects empty collections and blank strings rather than skipping them so
        output row indices remain strictly aligned with input chunk metadata.
        Emits incremental progress updates per batch when progress_callback is provided.
        """
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("texts must be a non-empty sequence of strings.")
        if len(texts) == 0:
            raise ValueError("texts cannot be empty.")

        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if not isinstance(effective_batch_size, int) or effective_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        cleaned_texts = [
            preprocess_text(text, index=idx) for idx, text in enumerate(texts)
        ]

        expected_dim = self.get_dimension()
        total_texts = len(cleaned_texts)

        if progress_callback is None:
            vectors = self._encode_raw(cleaned_texts, batch_size=effective_batch_size)
        else:
            # Batch-by-batch execution for large documents to bound memory and emit progress
            batch_vectors: list[np.ndarray] = []
            completed = 0
            for start_idx in range(0, total_texts, effective_batch_size):
                end_idx = min(start_idx + effective_batch_size, total_texts)
                batch_slice = cleaned_texts[start_idx:end_idx]
                batch_vecs = self._encode_raw(batch_slice, batch_size=len(batch_slice))
                batch_vectors.append(batch_vecs)
                completed += len(batch_slice)
                pct = min(100.0, (completed / max(1, total_texts)) * 100.0)
                progress_callback(
                    completed,
                    total_texts,
                    f"Generated embeddings for {completed}/{total_texts} chunks ({pct:.0f}%)...",
                )
            vectors = np.vstack(batch_vectors)

        if vectors.shape[1] != expected_dim:
            raise EmbeddingGenerationError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {vectors.shape[1]}."
            )

        if normalize:
            vectors = self._l2_normalize(vectors)

        return vectors

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Generate a validated (1, dim) embedding for a search query."""
        cleaned_query = preprocess_text(query)
        return self.embed_texts([cleaned_query], batch_size=1, normalize=normalize)
