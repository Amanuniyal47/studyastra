"""Targeted tests for batched embedding generation and incremental progress (Stage 11.3 Work D)."""

import io
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pymupdf

from src.retrieval.embeddings import EmbeddingService, EmbeddingGenerationError
from src.retrieval.vector_store import FAISSVectorStore, VectorStorePersistenceError


class TestEmbeddingBatchingAndProgress(unittest.TestCase):
    """Test chunked embedding generation, progress tracking, and ingestion integration."""

    def test_01_embed_texts_batches_and_emits_progress(self):
        """Verify embed_texts chunks large inputs into batch_size and emits progress."""
        service = EmbeddingService(batch_size=4)
        texts = [f"Academic statement number {i} regarding machine learning." for i in range(10)]

        progress_records = []
        def _on_progress(completed, total, msg):
            progress_records.append((completed, total, msg))

        # Mock _encode_raw to return unit vectors without running heavy PyTorch
        def _mock_encode(batch_texts, batch_size):
            n = len(batch_texts)
            # return shape (n, 384)
            vecs = np.ones((n, 384), dtype=np.float32)
            return vecs

        with patch.object(service, "get_dimension", return_value=384):
            with patch.object(service, "_encode_raw", side_effect=_mock_encode) as mock_enc:
                embeddings = service.embed_texts(
                    texts,
                    batch_size=4,
                    normalize=True,
                    progress_callback=_on_progress,
                )

        self.assertEqual(embeddings.shape, (10, 384))
        # 10 texts with batch_size 4 -> 3 batches (4, 4, 2)
        self.assertEqual(mock_enc.call_count, 3)
        self.assertEqual(len(progress_records), 3)
        self.assertEqual(progress_records[0][0], 4)
        self.assertEqual(progress_records[1][0], 8)
        self.assertEqual(progress_records[2][0], 10)
        self.assertEqual(progress_records[2][1], 10)

        # Verify L2 normalization
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_02_embedding_failure_rolls_back_in_add_pdf(self):
        """Verify that if embedding generation fails, FAISSVectorStore.add_pdf does not commit."""
        doc = pymupdf.open()
        p = doc.new_page()
        p.insert_text((50, 50), "Sample content that will generate chunks.")
        pdf_bytes = doc.write()
        doc.close()

        store = FAISSVectorStore()
        initial_total = store.total_vectors

        failing_service = MagicMock()
        failing_service.model_name = store.embedding_model
        failing_service.embed_texts.side_effect = EmbeddingGenerationError("Simulated OOM in embedding")

        with self.assertRaises(EmbeddingGenerationError):
            store.add_pdf(
                io.BytesIO(pdf_bytes),
                filename="failing_doc.pdf",
                embedding_service=failing_service,
            )

        # Verify no vectors or documents were partially added
        self.assertEqual(store.total_vectors, initial_total)
        self.assertFalse(store.is_document_indexed("failing_doc"))

    def test_03_embedding_failure_mid_batch_preserves_initial_state(self):
        """Verify that failure mid-way through multiple batches raises and leaves no partial state."""
        service = EmbeddingService(batch_size=3)
        texts = [f"Text item {i}" for i in range(8)]
        call_count = 0

        def _fail_on_batch_2(batch_texts, batch_size):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise EmbeddingGenerationError("Simulated OOM on batch 2")
            return np.ones((len(batch_texts), 384), dtype=np.float32)

        with patch.object(service, "get_dimension", return_value=384):
            with patch.object(service, "_encode_raw", side_effect=_fail_on_batch_2):
                with self.assertRaises(EmbeddingGenerationError):
                    service.embed_texts(
                        texts,
                        batch_size=3,
                        progress_callback=lambda c, t, m: None,
                    )
        self.assertEqual(call_count, 2)

    def test_04_failure_before_commit_preserves_active_generation(self):
        """Verify that failure immediately before transaction commit preserves active generation."""
        doc = pymupdf.open()
        p = doc.new_page()
        p.insert_text((50, 50), "Content for pre-commit failure test.")
        pdf_bytes = doc.write()
        doc.close()

        store = FAISSVectorStore()
        gen_before = store.active_generation
        vectors_before = store.total_vectors

        mock_service = MagicMock()
        mock_service.model_name = store.embedding_model
        mock_service.embed_texts.return_value = np.ones((1, 384), dtype=np.float32)

        # Fail during _commit_transaction
        with patch.object(store, "_commit_transaction", side_effect=VectorStorePersistenceError("Disk write failed")):
            with self.assertRaises(VectorStorePersistenceError):
                store.add_pdf(
                    io.BytesIO(pdf_bytes),
                    filename="fail_precommit.pdf",
                    embedding_service=mock_service,
                )

        self.assertEqual(store.active_generation, gen_before)
        self.assertEqual(store.total_vectors, vectors_before)


if __name__ == "__main__":
    unittest.main()
