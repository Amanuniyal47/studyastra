"""Targeted tests for Vector Store Transaction Memory Optimization (Stage 11.2 Work Area B)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import faiss

from src.retrieval.vector_store import (
    FAISSVectorStore,
    _clone_chunk_metadata,
    VectorStorePersistenceError,
)
from src.retrieval.embeddings import EmbeddingService


class TestVectorStoreTransactionOptimization(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.store_dir = Path(self.tmp_dir.name)
        self.mock_embed = MagicMock()
        self.mock_embed.model_name = "all-MiniLM-L6-v2"
        self.mock_embed.get_dimension.return_value = 384
        self.store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.mock_embed,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _generate_test_chunks(self, doc_id: str, count: int):
        chunks = []
        for i in range(1, count + 1):
            chunks.append({
                "chunk_id": f"{doc_id}_c{i}",
                "chunk_index": i,
                "page": (i % 10) + 1,
                "page_range": str((i % 10) + 1),
                "text": f"Sample academic content for {doc_id} chunk {i}.",
                "token_count": 25,
                "metadata": {"nested_key": f"val_{i}"},
            })
        vectors = np.ones((count, 384), dtype=np.float32)
        # L2-normalize
        faiss.normalize_L2(vectors)
        return chunks, vectors

    def test_01_clone_chunk_metadata_nested_isolation(self):
        original = {
            "chunk_id": "c1",
            "text": "original text",
            "metadata": {"tags": ["biology", "botany"], "author": {"name": "Mendel"}},
        }
        cloned = _clone_chunk_metadata(original)

        # Mutate top-level and nested structures on clone
        cloned["text"] = "mutated text"
        cloned["metadata"]["tags"].append("genetics")
        cloned["metadata"]["author"]["name"] = "Darwin"

        # Verify original remains completely untouched
        self.assertEqual(original["text"], "original text")
        self.assertEqual(original["metadata"]["tags"], ["biology", "botany"])
        self.assertEqual(original["metadata"]["author"]["name"], "Mendel")

    def test_02_successful_document_addition(self):
        chunks, vectors = self._generate_test_chunks("doc_a", 10)
        result = self.store.add_document_chunks(
            document_id="doc_a",
            filename="doc_a.pdf",
            page_count=10,
            chunks=chunks,
            embeddings=vectors,
        )
        self.assertEqual(result["document_id"], "doc_a")
        self.assertEqual(result["chunks_generated"], 10)
        self.assertEqual(self.store.total_vectors, 10)

        retrieved = self.store.get_document_chunks("doc_a")
        self.assertEqual(len(retrieved), 10)
        self.assertEqual(retrieved[0]["document_id"], "doc_a")

    def test_03_failed_transaction_rollback_preserves_state(self):
        chunks_a, vectors_a = self._generate_test_chunks("doc_a", 5)
        self.store.add_document_chunks(
            document_id="doc_a",
            filename="doc_a.pdf",
            page_count=5,
            chunks=chunks_a,
            embeddings=vectors_a,
        )
        self.assertEqual(self.store.total_vectors, 5)
        initial_chunks = {k: dict(v) for k, v in self.store._chunks.items()}

        # Attempt to add doc_b but force a failure during write_index
        chunks_b, vectors_b = self._generate_test_chunks("doc_b", 5)
        with patch("faiss.write_index", side_effect=OSError("Disk write error")):
            with self.assertRaises(VectorStorePersistenceError):
                self.store.add_document_chunks(
                    document_id="doc_b",
                    filename="doc_b.pdf",
                    page_count=5,
                    chunks=chunks_b,
                    embeddings=vectors_b,
                )

        # Vector count and chunks must be strictly rolled back
        self.assertEqual(self.store.total_vectors, 5)
        self.assertFalse(self.store.is_document_indexed("doc_b"))
        self.assertEqual(len(self.store._chunks), 5)
        self.assertEqual(self.store._chunks.keys(), initial_chunks.keys())

    def test_04_caller_mutations_isolated_from_store_chunks(self):
        chunks, vectors = self._generate_test_chunks("doc_iso", 3)
        self.store.add_document_chunks(
            document_id="doc_iso",
            filename="doc_iso.pdf",
            page_count=3,
            chunks=chunks,
            embeddings=vectors,
        )

        retrieved = self.store.get_document_chunks("doc_iso")
        # Mutate retrieved chunk
        retrieved[0]["text"] = "MUTATED CALLER TEXT"

        # Check internal store chunk
        internal_chunks = self.store.get_document_chunks("doc_iso")
        self.assertNotEqual(internal_chunks[0]["text"], "MUTATED CALLER TEXT")
        self.assertIn("Sample academic content", internal_chunks[0]["text"])

    def test_05_multiple_documents_and_deletion_isolation(self):
        # Index doc 1, 2, 3
        for name in ["doc_1", "doc_2", "doc_3"]:
            c, v = self._generate_test_chunks(name, 5)
            self.store.add_document_chunks(
                document_id=name,
                filename=f"{name}.pdf",
                page_count=5,
                chunks=c,
                embeddings=v,
            )

        self.assertEqual(self.store.total_vectors, 15)
        self.assertTrue(self.store.is_document_indexed("doc_1"))
        self.assertTrue(self.store.is_document_indexed("doc_2"))
        self.assertTrue(self.store.is_document_indexed("doc_3"))

        # Delete doc 2
        del_res = self.store.delete_document("doc_2")
        self.assertEqual(del_res["document_id"], "doc_2")
        self.assertEqual(self.store.total_vectors, 10)
        self.assertFalse(self.store.is_document_indexed("doc_2"))
        self.assertTrue(self.store.is_document_indexed("doc_1"))
        self.assertTrue(self.store.is_document_indexed("doc_3"))

        # Verify doc 1 and doc 3 chunks are completely intact
        self.assertEqual(len(self.store.get_document_chunks("doc_1")), 5)
        self.assertEqual(len(self.store.get_document_chunks("doc_3")), 5)

    def test_06_large_collection_staging_memory_and_integrity(self):
        # Index 100 chunks and verify all are intact
        chunks, vectors = self._generate_test_chunks("doc_large", 100)
        self.store.add_document_chunks(
            document_id="doc_large",
            filename="doc_large.pdf",
            page_count=10,
            chunks=chunks,
            embeddings=vectors,
        )
        self.assertEqual(self.store.total_vectors, 100)
        self.assertEqual(len(self.store.get_document_chunks("doc_large")), 100)

    def test_07_failure_during_metadata_write_rolls_back_cleanly(self):
        """Test rollback when failure occurs at the metadata JSON serialization stage."""
        chunks, vectors = self._generate_test_chunks("doc_base", 4)
        self.store.add_document_chunks(
            document_id="doc_base",
            filename="doc_base.pdf",
            page_count=2,
            chunks=chunks,
            embeddings=vectors,
        )
        self.assertEqual(self.store.total_vectors, 4)

        new_chunks, new_vectors = self._generate_test_chunks("doc_fail", 3)
        with patch("src.retrieval.vector_store._write_json_fsync", side_effect=OSError("Disk full")):
            with self.assertRaises(VectorStorePersistenceError):
                self.store.add_document_chunks(
                    document_id="doc_fail",
                    filename="doc_fail.pdf",
                    page_count=2,
                    chunks=new_chunks,
                    embeddings=new_vectors,
                )

        self.assertEqual(self.store.total_vectors, 4)
        self.assertFalse(self.store.is_document_indexed("doc_fail"))
        self.assertTrue(self.store.is_document_indexed("doc_base"))

    def test_08_failed_deletion_preserves_existing_document(self):
        """Test rollback when failure occurs during document deletion commit."""
        chunks, vectors = self._generate_test_chunks("doc_keep", 4)
        self.store.add_document_chunks(
            document_id="doc_keep",
            filename="doc_keep.pdf",
            page_count=2,
            chunks=chunks,
            embeddings=vectors,
        )
        self.assertEqual(self.store.total_vectors, 4)

        with patch("faiss.write_index", side_effect=OSError("Read-only filesystem")):
            with self.assertRaises(VectorStorePersistenceError):
                self.store.delete_document("doc_keep")

        # Document must remain fully indexed and retrievable
        self.assertEqual(self.store.total_vectors, 4)
        self.assertTrue(self.store.is_document_indexed("doc_keep"))
        self.assertEqual(len(self.store.get_document_chunks("doc_keep")), 4)


if __name__ == "__main__":
    unittest.main()
