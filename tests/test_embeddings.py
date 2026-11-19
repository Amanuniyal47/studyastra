import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.retrieval import embeddings
from src.retrieval.embeddings import (
    EmbeddingGenerationError,
    EmbeddingModelError,
    EmbeddingService,
    load_embedding_model,
    preprocess_text,
)


class _DummySentenceTransformer:
    def __init__(self, dim: int = 384):
        self._dim = dim
        self.encode_calls: list[dict] = []

    def get_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ):
        self.encode_calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        rows = []
        for idx, text in enumerate(texts):
            vec = np.zeros(self._dim, dtype=np.float32)
            vec[0] = float(len(text))
            vec[1] = float(idx + 1)
            vec[2] = 1.0
            rows.append(vec)
        return np.array(rows, dtype=np.float32)


class TestEmbeddingServiceUnit(unittest.TestCase):
    def test_preprocess_normalizes_whitespace_consistently(self):
        self.assertEqual(preprocess_text("  Hello\n\t world  "), "Hello world")

    def test_preprocess_rejects_empty_and_non_string(self):
        with self.assertRaises(ValueError):
            preprocess_text("")
        with self.assertRaises(ValueError):
            preprocess_text("   \n\t ")
        with self.assertRaises(ValueError):
            preprocess_text(None)  # type: ignore[arg-type]

    def test_embed_texts_batch_shape_dtype_and_l2_norm(self):
        dummy = _DummySentenceTransformer(dim=384)
        service = EmbeddingService(model_name="all-MiniLM-L6-v2", batch_size=2)

        with patch("src.retrieval.embeddings.load_embedding_model", return_value=dummy):
            vectors = service.embed_texts(
                ["First chunk", "Second  chunk\nwith newline", "Third chunk"]
            )

        self.assertEqual(vectors.shape, (3, 384))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertTrue(vectors.flags["C_CONTIGUOUS"])
        self.assertEqual(service.get_dimension(), 384)

        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones(3, dtype=np.float32), atol=1e-5)

        self.assertEqual(len(dummy.encode_calls), 1)
        self.assertEqual(dummy.encode_calls[0]["batch_size"], 2)
        self.assertEqual(
            dummy.encode_calls[0]["texts"],
            ["First chunk", "Second chunk with newline", "Third chunk"],
        )

    def test_embed_query_uses_same_preprocessing_and_shape(self):
        dummy = _DummySentenceTransformer(dim=384)
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")

        with patch("src.retrieval.embeddings.load_embedding_model", return_value=dummy):
            query_vec = service.embed_query("  What is   FAISS? \n")

        self.assertEqual(query_vec.shape, (1, 384))
        self.assertEqual(query_vec.dtype, np.float32)
        self.assertEqual(dummy.encode_calls[0]["texts"], ["What is FAISS?"])
        np.testing.assert_allclose(np.linalg.norm(query_vec, axis=1), [1.0], atol=1e-5)

    def test_rejects_empty_collection_and_blank_elements_without_skipping(self):
        dummy = _DummySentenceTransformer(dim=384)
        service = EmbeddingService(model_name="all-MiniLM-L6-v2")

        with patch("src.retrieval.embeddings.load_embedding_model", return_value=dummy):
            with self.assertRaises(ValueError):
                service.embed_texts([])
            with self.assertRaises(ValueError):
                service.embed_texts("not a list")  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                service.embed_texts(["valid chunk", "   ", "another valid chunk"])
            with self.assertRaises(ValueError):
                service.embed_query("   ")

        # Never invoked model.encode when validation failed
        self.assertEqual(len(dummy.encode_calls), 0)

    def test_invalid_init_arguments_raise(self):
        with self.assertRaises(ValueError):
            EmbeddingService(model_name="   ")
        with self.assertRaises(ValueError):
            EmbeddingService(batch_size=0)

    def test_model_initialization_failure_raises_embedding_model_error(self):
        load_embedding_model.clear()
        with patch(
            "src.retrieval.embeddings.SentenceTransformer",
            side_effect=OSError("Model files missing"),
        ):
            with self.assertRaises(EmbeddingModelError):
                load_embedding_model("missing/nonexistent-model")
        load_embedding_model.clear()

    def test_load_embedding_model_caches_single_instance(self):
        load_embedding_model.clear()
        sentinel = _DummySentenceTransformer(dim=384)
        with patch(
            "src.retrieval.embeddings.SentenceTransformer",
            return_value=sentinel,
        ) as mock_cls:
            first = load_embedding_model()
            second = load_embedding_model(embeddings.EMBEDDING_MODEL)

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        mock_cls.assert_called_once_with(embeddings.EMBEDDING_MODEL)
        load_embedding_model.clear()

    def test_encoding_failure_raises_embedding_generation_error(self):
        failing_model = MagicMock()
        failing_model.get_embedding_dimension.return_value = 384
        failing_model.encode.side_effect = RuntimeError("CUDA/CPU OOM")

        service = EmbeddingService()
        with patch("src.retrieval.embeddings.load_embedding_model", return_value=failing_model):
            with self.assertRaises(EmbeddingGenerationError):
                service.embed_texts(["Sample text"])

    def test_dimension_mismatch_raises_embedding_generation_error(self):
        bad_model = MagicMock()
        bad_model.get_embedding_dimension.return_value = 384
        bad_model.encode.return_value = np.ones((2, 128), dtype=np.float32)

        service = EmbeddingService()
        with patch("src.retrieval.embeddings.load_embedding_model", return_value=bad_model):
            with self.assertRaises(EmbeddingGenerationError):
                service.embed_texts(["a", "b"])

    def test_non_finite_or_zero_norm_vector_raises(self):
        nan_model = MagicMock()
        nan_model.get_embedding_dimension.return_value = 4
        nan_model.encode.return_value = np.array([[np.nan, 1.0, 0.0, 0.0]], dtype=np.float32)

        zero_model = MagicMock()
        zero_model.get_embedding_dimension.return_value = 4
        zero_model.encode.return_value = np.zeros((1, 4), dtype=np.float32)

        service = EmbeddingService()
        with patch("src.retrieval.embeddings.load_embedding_model", return_value=nan_model):
            with self.assertRaises(EmbeddingGenerationError):
                service.embed_texts(["nan check"])

        service2 = EmbeddingService()
        with patch("src.retrieval.embeddings.load_embedding_model", return_value=zero_model):
            with self.assertRaises(EmbeddingGenerationError):
                service2.embed_texts(["zero check"])


@unittest.skipUnless(
    os.getenv("RUN_REAL_EMBEDDING_TESTS") == "1",
    "Set RUN_REAL_EMBEDDING_TESTS=1 to run the real all-MiniLM-L6-v2 smoke test.",
)
class TestRealEmbeddingModelSmoke(unittest.TestCase):
    def test_real_minilm_batch_query_and_cache(self):
        load_embedding_model.clear()
        service = EmbeddingService(batch_size=2)

        dim = service.get_dimension()
        self.assertEqual(dim, 384)

        texts = [
            "FAISS performs fast similarity search over dense vectors.",
            "Mitochondria generate ATP during cellular respiration.",
            "Streamlit builds interactive Python data applications.",
        ]
        chunk_vecs = service.embed_texts(texts, batch_size=2)
        query_vec = service.embed_query("How does vector similarity search work?")

        self.assertEqual(chunk_vecs.shape, (3, 384))
        self.assertEqual(query_vec.shape, (1, 384))
        self.assertEqual(chunk_vecs.dtype, np.float32)
        self.assertEqual(query_vec.dtype, np.float32)

        # Cosine similarity (dot product of normalized vectors) should rank FAISS chunk highest
        scores = (chunk_vecs @ query_vec.T).reshape(-1)
        self.assertEqual(int(np.argmax(scores)), 0)

        # Confirm model cache returns the same object without reloading
        m1 = load_embedding_model()
        m2 = load_embedding_model()
        self.assertIs(m1, m2)


if __name__ == "__main__":
    unittest.main()
