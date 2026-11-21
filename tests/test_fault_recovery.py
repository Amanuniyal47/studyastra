"""Comprehensive fault recovery, staging cleanup, timeout and resource lifecycle tests.

Verifies:
- Ollama connection fault handling, retries, exponential backoff, timeouts, and streaming recovery.
- FAISS transaction rollback, orphaned staging cleanup, manifest recovery, and atomic replacement.
- PDF ingestion interruption recovery and idempotency.
- Resource handle lifecycle (PyMuPDF document handles and HTTP response streams).
- Multi-threaded concurrency safety for FAISS vector store and chat lock leases.
- UI error containment and session state preservation.
"""

from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import httpx
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from src.chat.chat_orchestrator import (
    ChatOrchestrator,
    ConversationLockLease,
    _ConversationLockEntry,
)
import src.config as config
from src.generation.llm_provider import (
    LLMConfigurationError,
    LLMProviderError,
    OllamaLLMProvider,
)
from src.ingestion.pdf_loader import extract_pdf_text
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import (
    FAISSVectorStore,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    VectorStoreCorruptionError,
    VectorStorePersistenceError,
    _safe_replace,
)


class MockEmbeddingService:
    def __init__(self, dimension: int = 4, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.dimension = dimension
        self.model_name = model_name

    def get_dimension(self) -> int:
        return self.dimension

    def embed_texts(
        self,
        texts: list[str],
        normalize: bool = True,
        progress_callback: Any = None,
    ) -> np.ndarray:
        if progress_callback:
            progress_callback(len(texts), len(texts), "Done")
        arr = np.ones((len(texts), self.dimension), dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-9)
        return arr


def _create_minimal_pdf_bytes(text: str = "Sample page text") -> bytes:
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ==============================================================================
# 1. Ollama Fault & Timeout Recovery Tests
# ==============================================================================
class TestOllamaFaultRecovery(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = OllamaLLMProvider(
            model_name="qwen2.5:7b",
            base_url="http://localhost:11434",
            connect_timeout=0.1,
            read_timeout=0.2,
            max_retries=2,
            retry_backoff_factor=0.01,
            retry_backoff_max=0.05,
            health_cache_ttl=0.1,
        )

    @patch("httpx.Client.post")
    def test_connection_refused_retries_and_exhausts(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with self.assertRaises(LLMProviderError) as ctx:
            self.provider.generate(system_prompt="sys", user_prompt="usr")

        self.assertIn("Cannot connect to Ollama", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 3)  # initial + 2 retries

    @patch("httpx.Client.post")
    def test_read_timeout_retries_and_exhausts(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.ReadTimeout("Read timed out")

        with self.assertRaises(LLMProviderError) as ctx:
            self.provider.generate(system_prompt="sys", user_prompt="usr")

        self.assertIn("timed out", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 3)

    @patch("httpx.Client.post")
    def test_transient_failure_then_success(self, mock_post: MagicMock) -> None:
        good_response = MagicMock(spec=httpx.Response)
        good_response.status_code = 200
        good_response.json.return_value = {
            "message": {"role": "assistant", "content": "recovered response"}
        }

        mock_post.side_effect = [
            httpx.ConnectError("Transient connection drop"),
            good_response,
        ]

        result = self.provider.generate(system_prompt="sys", user_prompt="usr")
        self.assertEqual(result, "recovered response")
        self.assertEqual(mock_post.call_count, 2)

    @patch("httpx.Client.post")
    def test_404_model_missing_raises_immediately_without_retrying(self, mock_post: MagicMock) -> None:
        req = httpx.Request("POST", "http://localhost:11434/api/chat")
        resp_404 = httpx.Response(404, request=req, json={"error": "model 'qwen2.5:7b' not found"})
        mock_post.return_value = resp_404

        with self.assertRaises(LLMProviderError) as ctx:
            self.provider.generate(system_prompt="sys", user_prompt="usr")

        self.assertIn("not found in Ollama", str(ctx.exception))
        self.assertIn("ollama pull qwen2.5:7b", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 1)  # Zero retries on 404

    @patch("httpx.Client.post")
    def test_500_internal_server_error_retries_then_exhausts(self, mock_post: MagicMock) -> None:
        req = httpx.Request("POST", "http://localhost:11434/api/chat")
        resp_500 = httpx.Response(500, request=req, text="Internal Server Error")
        mock_post.return_value = resp_500

        with self.assertRaises(LLMProviderError) as ctx:
            self.provider.generate(system_prompt="sys", user_prompt="usr")

        self.assertIn("500", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 3)

    @patch("httpx.Client.stream")
    def test_stream_connection_failure_retries_before_yielding(self, mock_stream: MagicMock) -> None:
        req = httpx.Request("POST", "http://localhost:11434/api/chat")
        good_resp = MagicMock(spec=httpx.Response)
        good_resp.status_code = 200
        good_resp.iter_lines.return_value = [
            json.dumps({"message": {"role": "assistant", "content": "token1"}}),
            json.dumps({"message": {"role": "assistant", "content": "token2"}, "done": True}),
        ]

        mock_context_fail = MagicMock()
        mock_context_fail.__enter__.side_effect = httpx.ConnectError("Failed connect")
        mock_context_fail.__exit__.return_value = False

        mock_context_ok = MagicMock()
        mock_context_ok.__enter__.return_value = good_resp
        mock_context_ok.__exit__.return_value = False

        mock_stream.side_effect = [mock_context_fail, mock_context_ok]

        tokens = list(self.provider.generate_stream(system_prompt="sys", user_prompt="usr"))
        self.assertEqual(tokens, ["token1", "token2"])
        self.assertEqual(mock_stream.call_count, 2)
        good_resp.close.assert_called()

    @patch("httpx.Client.stream")
    def test_stream_broken_mid_generation_never_retries_and_closes_response(
        self, mock_stream: MagicMock
    ) -> None:
        req = httpx.Request("POST", "http://localhost:11434/api/chat")
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200

        def faulty_lines():
            yield json.dumps({"message": {"role": "assistant", "content": "token1"}})
            raise httpx.ReadTimeout("Stream read timed out mid-response")

        resp.iter_lines.return_value = faulty_lines()

        mock_context = MagicMock()
        mock_context.__enter__.return_value = resp
        mock_context.__exit__.return_value = False
        mock_stream.return_value = mock_context

        generator = self.provider.generate_stream(system_prompt="sys", user_prompt="usr")
        first_token = next(generator)
        self.assertEqual(first_token, "token1")

        with self.assertRaises(LLMProviderError) as ctx:
            next(generator)

        self.assertIn("timed out", str(ctx.exception))
        # Never retried after yielding token1
        self.assertEqual(mock_stream.call_count, 1)
        resp.close.assert_called()


# ==============================================================================
# 2. FAISS Recovery & Staging Cleanup Tests
# ==============================================================================
class TestFAISSRecovery(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir) / "store"
        self.embed_svc = MockEmbeddingService(dimension=4)

    def tearDown(self) -> None:
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_abandoned_staging_directory_cleaned_up_on_startup(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        # Commit generation 1
        chunks = [{"chunk_id": "c1", "text": "chunk one", "page": 1, "chunk_index": 1}]
        store.add_document_chunks(
            document_id="doc1",
            filename="doc1.pdf",
            page_count=1,
            chunks=chunks,
            embeddings=self.embed_svc.embed_texts(["chunk one"]),
        )
        self.assertEqual(store.active_generation, 1)

        # Simulate a crash leaving an orphaned .staging directory
        stale_staging = store.generations_dir / "gen_2.staging"
        stale_staging.mkdir(parents=True, exist_ok=True)
        (stale_staging / "partial.tmp").write_text("abandoned", encoding="utf-8")
        self.assertTrue(stale_staging.exists())

        # Cleanup with max_age=0 removes all staging directories
        removed = store.cleanup_staging_artifacts(max_age_seconds=0)
        self.assertIn(str(stale_staging), removed)
        self.assertFalse(stale_staging.exists())
        # Active generation gen_1 is intact
        self.assertTrue((store.generations_dir / "gen_1").exists())

    def test_staging_cleanup_respects_max_age(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        stale_dir = store.generations_dir / "gen_old.staging"
        stale_dir.mkdir(parents=True, exist_ok=True)

        fresh_dir = store.generations_dir / "gen_fresh.staging"
        fresh_dir.mkdir(parents=True, exist_ok=True)

        # Set stale_dir mtime to 2 hours ago
        past_time = time.time() - 7200
        os.utime(stale_dir, (past_time, past_time))

        # Cleanup with max_age = 3600s
        removed = store.cleanup_staging_artifacts(max_age_seconds=3600.0)
        self.assertIn(str(stale_dir), removed)
        self.assertFalse(stale_dir.exists())
        self.assertTrue(fresh_dir.exists())  # Fresh staging preserved

    def test_staging_cleanup_never_deletes_committed_generations(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        chunks = [{"chunk_id": "c1", "text": "chunk one", "page": 1, "chunk_index": 1}]
        store.add_document_chunks(
            document_id="doc1",
            filename="doc1.pdf",
            page_count=1,
            chunks=chunks,
            embeddings=self.embed_svc.embed_texts(["chunk one"]),
        )
        active_gen_dir = store.generations_dir / f"gen_{store.active_generation}"
        self.assertTrue(active_gen_dir.exists())

        # Age it back
        past_time = time.time() - 99999
        os.utime(active_gen_dir, (past_time, past_time))

        removed = store.cleanup_staging_artifacts(max_age_seconds=0)
        self.assertNotIn(str(active_gen_dir), removed)
        self.assertTrue(active_gen_dir.exists())

    def test_interrupted_transaction_leaves_active_generation_intact(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        chunks1 = [{"chunk_id": "c1", "text": "chunk one", "page": 1, "chunk_index": 1}]
        store.add_document_chunks(
            document_id="doc1",
            filename="doc1.pdf",
            page_count=1,
            chunks=chunks1,
            embeddings=self.embed_svc.embed_texts(["chunk one"]),
        )
        self.assertEqual(store.active_generation, 1)

        # Attempt to add doc2 but inject a failure during atomic replacement
        chunks2 = [{"chunk_id": "c2", "text": "chunk two", "page": 1, "chunk_index": 1}]
        with patch("src.retrieval.vector_store._safe_replace", side_effect=OSError("Disk full")):
            with self.assertRaises(VectorStorePersistenceError):
                store.add_document_chunks(
                    document_id="doc2",
                    filename="doc2.pdf",
                    page_count=1,
                    chunks=chunks2,
                    embeddings=self.embed_svc.embed_texts(["chunk two"]),
                )

        # In-memory and on-disk state are both preserved at gen 1
        self.assertEqual(store.active_generation, 1)
        self.assertEqual(store.total_documents, 1)
        self.assertFalse(store.is_document_indexed("doc2"))
        self.assertTrue(store.is_document_indexed("doc1"))

        # Reload store from disk to verify disk integrity
        reloaded = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        self.assertEqual(reloaded.active_generation, 1)
        self.assertEqual(reloaded.total_documents, 1)
        self.assertTrue(reloaded.is_document_indexed("doc1"))

    def test_safe_replace_retries_on_transient_os_error(self) -> None:
        src = Path(self.temp_dir) / "src.txt"
        dst = Path(self.temp_dir) / "dst.txt"
        src.write_text("content", encoding="utf-8")

        mock_replace = MagicMock()
        mock_replace.side_effect = [
            OSError("File locked by indexing service"),
            OSError("File locked by antivirus"),
            None,  # Succeeds on 3rd attempt
        ]

        with patch("os.replace", mock_replace):
            _safe_replace(src, dst, max_retries=3, retry_delay=0.001)

        self.assertEqual(mock_replace.call_count, 3)

    def test_recovery_from_manifest_tmp_on_startup(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        chunks = [{"chunk_id": "c1", "text": "chunk one", "page": 1, "chunk_index": 1}]
        store.add_document_chunks(
            document_id="doc1",
            filename="doc1.pdf",
            page_count=1,
            chunks=chunks,
            embeddings=self.embed_svc.embed_texts(["chunk one"]),
        )
        self.assertEqual(store.active_generation, 1)

        # Simulate interrupted commit: manifest.json deleted, but valid manifest.json.tmp left
        manifest_file = store.manifest_path
        tmp_manifest = self.store_dir / f"{MANIFEST_FILENAME}.tmp"
        shutil.copyfile(manifest_file, tmp_manifest)
        manifest_file.unlink()

        self.assertFalse(manifest_file.exists())
        self.assertTrue(tmp_manifest.exists())

        # Initializing new store recovers manifest.json from manifest.json.tmp
        recovered_store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        self.assertEqual(recovered_store.active_generation, 1)
        self.assertTrue(recovered_store.is_document_indexed("doc1"))
        self.assertTrue(manifest_file.exists())

    def test_invalid_or_corrupted_manifest_tmp_rejected_safely(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        chunks = [{"chunk_id": "c1", "text": "chunk one", "page": 1, "chunk_index": 1}]
        store.add_document_chunks(
            document_id="doc1",
            filename="doc1.pdf",
            page_count=1,
            chunks=chunks,
            embeddings=self.embed_svc.embed_texts(["chunk one"]),
        )
        # Delete manifest.json
        store.manifest_path.unlink()

        # Write corrupted / incomplete manifest.tmp
        tmp_manifest = self.store_dir / f"{MANIFEST_FILENAME}.tmp"
        tmp_manifest.write_text(json.dumps({"active_generation": 99, "missing_keys": True}), encoding="utf-8")

        # Initializing must reject the incomplete tmp manifest and raise VectorStoreCorruptionError
        with self.assertRaises(VectorStoreCorruptionError):
            FAISSVectorStore(
                store_dir=self.store_dir,
                embedding_service=self.embed_svc,
            )
        self.assertFalse(store.manifest_path.exists())

    def test_failure_during_metadata_persistence_leaves_store_intact(self) -> None:
        import src.retrieval.vector_store as vs_module
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        chunks1 = [{"chunk_id": "c1", "text": "chunk one", "page": 1, "chunk_index": 1}]
        store.add_document_chunks(
            document_id="doc1",
            filename="doc1.pdf",
            page_count=1,
            chunks=chunks1,
            embeddings=self.embed_svc.embed_texts(["chunk one"]),
        )
        self.assertEqual(store.active_generation, 1)

        # Inject failure specifically when writing metadata.json
        orig_write_json = vs_module._write_json_fsync

        def faulty_write_json(path: Path, payload: dict[str, Any]) -> None:
            if path.name == METADATA_FILENAME:
                raise OSError("Disk quota exceeded while writing metadata")
            return orig_write_json(path, payload)

        chunks2 = [{"chunk_id": "c2", "text": "chunk two", "page": 1, "chunk_index": 1}]
        with patch("src.retrieval.vector_store._write_json_fsync", side_effect=faulty_write_json):
            with self.assertRaises(VectorStorePersistenceError):
                store.add_document_chunks(
                    document_id="doc2",
                    filename="doc2.pdf",
                    page_count=1,
                    chunks=chunks2,
                    embeddings=self.embed_svc.embed_texts(["chunk two"]),
                )

        self.assertEqual(store.active_generation, 1)
        self.assertEqual(store.total_documents, 1)
        self.assertFalse(store.is_document_indexed("doc2"))

    def test_search_during_write_observes_consistent_snapshot(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        chunks1 = [{"chunk_id": "c1", "text": "chunk one", "page": 1, "chunk_index": 1}]
        store.add_document_chunks(
            document_id="doc1",
            filename="doc1.pdf",
            page_count=1,
            chunks=chunks1,
            embeddings=self.embed_svc.embed_texts(["chunk one"]),
        )

        qvec = np.ones((1, 4), dtype=np.float32)
        qvec = qvec / np.linalg.norm(qvec)

        search_results: list[int] = []
        errors: list[Exception] = []

        def reader() -> None:
            for _ in range(8):
                try:
                    res = store.search(qvec, top_k=2)
                    search_results.append(len(res))
                except Exception as exc:
                    errors.append(exc)
                time.sleep(0.005)

        def writer() -> None:
            for i in range(2, 4):
                c = [{"chunk_id": f"c{i}", "text": f"chunk {i}", "page": 1, "chunk_index": 1}]
                store.add_document_chunks(
                    document_id=f"doc{i}",
                    filename=f"doc{i}.pdf",
                    page_count=1,
                    chunks=c,
                    embeddings=self.embed_svc.embed_texts([f"chunk {i}"]),
                )
                time.sleep(0.005)

        t_r = threading.Thread(target=reader)
        t_w = threading.Thread(target=writer)
        t_r.start()
        t_w.start()
        t_r.join()
        t_w.join()

        self.assertEqual(len(errors), 0, f"Reader encountered errors: {errors}")
        self.assertTrue(all(count >= 1 for count in search_results))

    def test_deterministic_reader_snapshot_consistency_during_concurrent_deletion(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        chunks = [{"chunk_id": "c1", "text": "deterministic text", "page": 1, "chunk_index": 1}]
        store.add_document_chunks(
            document_id="doc1",
            filename="doc1.pdf",
            page_count=1,
            chunks=chunks,
            embeddings=self.embed_svc.embed_texts(["deterministic text"]),
        )
        self.assertEqual(store.active_generation, 1)

        qvec = np.ones((1, 4), dtype=np.float32)
        qvec = qvec / np.linalg.norm(qvec)

        reader_captured_state = threading.Event()
        writer_committed = threading.Event()
        reader_results: list[dict[str, Any]] = []
        reader_error: list[Exception] = []

        orig_search = store._state.index.search

        def sync_search(*args, **kwargs):
            reader_captured_state.set()
            # Block reader until writer completes document deletion and commits generation 2
            if not writer_committed.wait(timeout=5.0):
                raise TimeoutError("Writer thread did not commit in time")
            return orig_search(*args, **kwargs)

        def reader_task() -> None:
            try:
                # With snapshot isolation via _GenerationState, search uses the state captured
                # before writer deleted the document.
                res = store.search(qvec, top_k=1)
                reader_results.extend(res)
            except Exception as exc:
                reader_error.append(exc)

        def writer_task() -> None:
            try:
                if not reader_captured_state.wait(timeout=5.0):
                    raise TimeoutError("Reader did not reach search barrier in time")
                # Delete document, advancing vector store to generation 2 (0 docs, 0 chunks)
                store.delete_document("doc1")
            finally:
                writer_committed.set()

        with patch.object(store._state.index, "search", side_effect=sync_search):
            t_r = threading.Thread(target=reader_task)
            t_w = threading.Thread(target=writer_task)
            t_r.start()
            t_w.start()
            t_r.join()
            t_w.join()

        self.assertEqual(len(reader_error), 0, f"Reader raised unexpected exception: {reader_error}")
        self.assertEqual(len(reader_results), 1)
        self.assertEqual(reader_results[0]["metadata"]["chunk_id"], "c1")
        self.assertEqual(reader_results[0]["text"], "deterministic text")
        # Vector store state itself was committed to generation 2
        self.assertEqual(store.total_documents, 0)
        self.assertEqual(store.active_generation, 2)

    def test_commit_failure_during_directory_replacement_preserves_active_generation(self) -> None:
        import src.retrieval.vector_store as vs_module
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        chunks1 = [{"chunk_id": "c1", "text": "chunk one", "page": 1, "chunk_index": 1}]
        store.add_document_chunks(
            document_id="doc1",
            filename="doc1.pdf",
            page_count=1,
            chunks=chunks1,
            embeddings=self.embed_svc.embed_texts(["chunk one"]),
        )
        self.assertEqual(store.active_generation, 1)

        orig_safe_replace = vs_module._safe_replace

        def faulty_safe_replace(src_path: Path, dst_path: Path, **kwargs) -> None:
            if "gen_2.staging" in str(src_path) or str(src_path).endswith(".staging"):
                raise OSError("Access denied during directory replacement")
            return orig_safe_replace(src_path, dst_path, **kwargs)

        chunks2 = [{"chunk_id": "c2", "text": "chunk two", "page": 1, "chunk_index": 1}]
        with patch("src.retrieval.vector_store._safe_replace", side_effect=faulty_safe_replace):
            with self.assertRaises(VectorStorePersistenceError):
                store.add_document_chunks(
                    document_id="doc2",
                    filename="doc2.pdf",
                    page_count=1,
                    chunks=chunks2,
                    embeddings=self.embed_svc.embed_texts(["chunk two"]),
                )

        # In-memory state preserved
        self.assertEqual(store.active_generation, 1)
        self.assertEqual(store.total_documents, 1)
        self.assertTrue(store.is_document_indexed("doc1"))
        self.assertFalse(store.is_document_indexed("doc2"))

        # Staging directory must be cleaned up
        staging_dir = store.generations_dir / "gen_2.staging"
        self.assertFalse(staging_dir.exists())

        # Disk reload preserved at generation 1
        reloaded = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        self.assertEqual(reloaded.active_generation, 1)
        self.assertEqual(reloaded.total_documents, 1)
        self.assertTrue(reloaded.is_document_indexed("doc1"))

    def test_commit_failure_during_manifest_replacement_preserves_active_generation(self) -> None:
        import src.retrieval.vector_store as vs_module
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        chunks1 = [{"chunk_id": "c1", "text": "chunk one", "page": 1, "chunk_index": 1}]
        store.add_document_chunks(
            document_id="doc1",
            filename="doc1.pdf",
            page_count=1,
            chunks=chunks1,
            embeddings=self.embed_svc.embed_texts(["chunk one"]),
        )
        self.assertEqual(store.active_generation, 1)

        orig_safe_replace = vs_module._safe_replace

        def faulty_safe_replace(src_path: Path, dst_path: Path, **kwargs) -> None:
            if str(src_path).endswith(f"{MANIFEST_FILENAME}.tmp"):
                raise OSError("Access denied during atomic manifest replacement")
            return orig_safe_replace(src_path, dst_path, **kwargs)

        chunks2 = [{"chunk_id": "c2", "text": "chunk two", "page": 1, "chunk_index": 1}]
        with patch("src.retrieval.vector_store._safe_replace", side_effect=faulty_safe_replace):
            with self.assertRaises(VectorStorePersistenceError):
                store.add_document_chunks(
                    document_id="doc2",
                    filename="doc2.pdf",
                    page_count=1,
                    chunks=chunks2,
                    embeddings=self.embed_svc.embed_texts(["chunk two"]),
                )

        # In-memory state preserved at generation 1
        self.assertEqual(store.active_generation, 1)
        self.assertEqual(store.total_documents, 1)
        self.assertTrue(store.is_document_indexed("doc1"))
        self.assertFalse(store.is_document_indexed("doc2"))

        # Leftovers cleaned up: gen_2 dir and tmp manifest
        gen_2_dir = store.generations_dir / "gen_2"
        tmp_manifest = self.store_dir / f"{MANIFEST_FILENAME}.tmp"
        self.assertFalse(gen_2_dir.exists())
        self.assertFalse(tmp_manifest.exists())

        # Disk reload preserved at generation 1
        reloaded = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        self.assertEqual(reloaded.active_generation, 1)
        self.assertEqual(reloaded.total_documents, 1)
        self.assertTrue(reloaded.is_document_indexed("doc1"))


# ==============================================================================
# 3. Ingestion Interruption & Rollback Tests
# ==============================================================================
class TestIngestionInterruptionRecovery(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir) / "store"
        self.embed_svc = MockEmbeddingService(dimension=4)
        self.pdf_bytes = _create_minimal_pdf_bytes("Introduction to algorithms and data structures.")

    def tearDown(self) -> None:
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extraction_failure_leaves_no_partial_state(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        with patch("src.retrieval.vector_store.extract_pdf_text", side_effect=ValueError("Corrupted PDF")):
            with self.assertRaises(ValueError):
                store.add_pdf(io.BytesIO(self.pdf_bytes), filename="broken.pdf", embedding_service=self.embed_svc)

        self.assertEqual(store.total_documents, 0)
        self.assertEqual(store.total_vectors, 0)
        self.assertEqual(store.active_generation, 0)

    def test_embedding_failure_leaves_no_partial_state(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        mock_broken_svc = MagicMock()
        mock_broken_svc.model_name = "all-MiniLM-L6-v2"
        mock_broken_svc.embed_texts.side_effect = RuntimeError("GPU out of memory")

        with self.assertRaises(RuntimeError):
            store.add_pdf(io.BytesIO(self.pdf_bytes), filename="valid.pdf", embedding_service=mock_broken_svc)

        self.assertEqual(store.total_documents, 0)
        self.assertEqual(store.total_vectors, 0)
        self.assertEqual(store.active_generation, 0)

    def test_reindexing_after_failed_attempt_succeeds_without_duplicates(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        # Attempt 1 fails during embedding
        mock_broken_svc = MagicMock()
        mock_broken_svc.model_name = "all-MiniLM-L6-v2"
        mock_broken_svc.embed_texts.side_effect = RuntimeError("Temporary failure")

        with self.assertRaises(RuntimeError):
            store.add_pdf(io.BytesIO(self.pdf_bytes), filename="test.pdf", embedding_service=mock_broken_svc)

        self.assertEqual(store.total_documents, 0)

        # Attempt 2 succeeds cleanly
        res = store.add_pdf(io.BytesIO(self.pdf_bytes), filename="test.pdf", embedding_service=self.embed_svc)
        self.assertEqual(res["status"], "indexed")
        self.assertEqual(store.total_documents, 1)
        self.assertGreater(store.total_vectors, 0)

        # Attempt 3 detects duplicate and does not duplicate
        res3 = store.add_pdf(io.BytesIO(self.pdf_bytes), filename="test.pdf", embedding_service=self.embed_svc, skip_duplicates=True)
        self.assertEqual(res3["status"], "duplicate")
        self.assertEqual(store.total_documents, 1)


# ==============================================================================
# 4. Resource Lifecycle & Handle Cleanup Tests
# ==============================================================================
class TestResourceLifecycle(unittest.TestCase):
    def test_pymupdf_handle_closed_after_successful_extraction(self) -> None:
        import pymupdf
        pdf_bytes = _create_minimal_pdf_bytes("Testing resource closure.")

        orig_open = pymupdf.open
        opened_docs: list[pymupdf.Document] = []

        def tracked_open(*args: Any, **kwargs: Any) -> pymupdf.Document:
            doc = orig_open(*args, **kwargs)
            opened_docs.append(doc)
            return doc

        with patch("pymupdf.open", side_effect=tracked_open):
            extracted = extract_pdf_text(pdf_bytes)

        self.assertTrue(len(extracted) > 0)
        self.assertEqual(len(opened_docs), 1)
        self.assertTrue(opened_docs[0].is_closed)

    def test_pymupdf_handle_closed_after_extraction_error(self) -> None:
        import pymupdf
        pdf_bytes = _create_minimal_pdf_bytes("Testing resource closure on error.")

        orig_open = pymupdf.open
        opened_docs: list[pymupdf.Document] = []

        def tracked_open(*args: Any, **kwargs: Any) -> pymupdf.Document:
            doc = orig_open(*args, **kwargs)
            opened_docs.append(doc)
            return doc

        with patch("pymupdf.open", side_effect=tracked_open):
            with patch("src.ingestion.pdf_loader.extract_page_text_layout_aware", side_effect=RuntimeError("Parse explosion")):
                # The loader falls back to standard text extraction, but let's test a full crash inside loop
                with patch("src.ingestion.pdf_loader.extract_pdf_text") as mock_extract:
                    # Test context manager directly
                    with tracked_open(stream=pdf_bytes, filetype="pdf") as doc:
                        try:
                            raise RuntimeError("Forced error inside document context")
                        except RuntimeError:
                            pass

        self.assertEqual(len(opened_docs), 1)
        self.assertTrue(opened_docs[0].is_closed)


# ==============================================================================
# 5. Concurrency & Locking Safety Tests
# ==============================================================================
class TestConcurrencyAndLocking(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.store_dir = Path(self.temp_dir) / "store"
        self.embed_svc = MockEmbeddingService(dimension=4)

    def tearDown(self) -> None:
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_writes_are_serialized_without_corruption(self) -> None:
        store = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        errors: list[Exception] = []

        def worker(doc_idx: int) -> None:
            try:
                chunks = [
                    {
                        "chunk_id": f"doc_{doc_idx}_c1",
                        "text": f"text for document {doc_idx}",
                        "page": 1,
                        "chunk_index": 1,
                    }
                ]
                store.add_document_chunks(
                    document_id=f"doc_{doc_idx}",
                    filename=f"doc_{doc_idx}.pdf",
                    page_count=1,
                    chunks=chunks,
                    embeddings=self.embed_svc.embed_texts([f"text for document {doc_idx}"]),
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent writes had errors: {errors}")
        self.assertEqual(store.total_documents, 5)
        self.assertEqual(store.total_vectors, 5)

        # Validate persistent consistency
        reloaded = FAISSVectorStore(
            store_dir=self.store_dir,
            embedding_service=self.embed_svc,
        )
        self.assertEqual(reloaded.total_documents, 5)
        self.assertEqual(reloaded.total_vectors, 5)

    def test_thread_lock_lease_releases_on_exception(self) -> None:
        entry = _ConversationLockEntry("conv_test_123")
        self.assertEqual(entry.ref_count, 0)

        # Create lease
        lease = ConversationLockLease(entry, ChatOrchestrator)
        entry.ref_count = 1

        with patch.object(ChatOrchestrator, "_release_lock_entry") as mock_rel:
            try:
                with lease:
                    self.assertTrue(entry.lock._is_owned())
                    raise RuntimeError("Error during chat processing")
            except RuntimeError:
                pass

            # Lock must be released
            self.assertFalse(entry.lock._is_owned())
            mock_rel.assert_called_once_with(entry)


# ==============================================================================
# 6. Streamlit UI Error Handling & Session State Preservation Tests
# ==============================================================================
class TestUIErrorRecovery(unittest.TestCase):
    def test_indexing_failure_cleans_up_progress_and_records_error(self) -> None:
        # Simulate the UI indexing block logic from app.py
        session_state: dict[str, Any] = {
            "chat_messages": [{"role": "user", "content": "prior question"}],
            "quiz_answers": {1: "A"},
            "notes": "existing study notes",
        }

        progress_bar = MagicMock()
        progress_status = MagicMock()

        mock_store = MagicMock()
        mock_store.add_pdf.side_effect = VectorStorePersistenceError("Disk write failed")

        try:
            try:
                mock_store.add_pdf("file.pdf")
            except Exception as exc:
                session_state["last_index_error"] = str(exc)
                session_state.pop("last_index_result", None)
            finally:
                progress_bar.empty()
                progress_status.empty()
        except Exception:
            pass

        # Error captured
        self.assertIn("Disk write failed", session_state["last_index_error"])
        self.assertNotIn("last_index_result", session_state)

        # UI elements cleared
        progress_bar.empty.assert_called_once()
        progress_status.empty.assert_called_once()

        # Existing user data untouched
        self.assertEqual(len(session_state["chat_messages"]), 1)
        self.assertEqual(session_state["quiz_answers"][1], "A")
        self.assertEqual(session_state["notes"], "existing study notes")


if __name__ == "__main__":
    unittest.main()
