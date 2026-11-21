"""Tests for Stage 11.5-C: Application Lifecycle, Single-Instance Locking & Startup Validation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from streamlit.testing.v1 import AppTest

from src import config
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import FAISSVectorStore
from src.utils.app_lock import (
    ApplicationLock,
    ApplicationLockError,
    get_application_lock,
    reset_global_application_lock,
)

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class _MockEmbeddingService:
    def __init__(self, dim: int = 4):
        self.dim = dim
        self.model_name = "all-MiniLM-L6-v2"

    def get_dimension(self) -> int:
        return self.dim

    def embed_texts(self, texts, batch_size=32):
        return np.ones((len(texts), self.dim), dtype=np.float32)

    def embed_query(self, query):
        return np.ones((1, self.dim), dtype=np.float32)


class TestAppLifecycle(unittest.TestCase):
    def setUp(self):
        reset_global_application_lock()

    def tearDown(self):
        reset_global_application_lock()

    def test_01_successful_lock_acquisition(self):
        """Requirement 1: Verify successful acquisition, state tracking, and cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = ApplicationLock(lock_dir=tmpdir)
            self.assertFalse(lock.is_held)

            ok, info, err = lock.acquire()
            self.assertTrue(ok)
            self.assertIsNone(err)
            self.assertTrue(lock.is_held)
            self.assertIsNotNone(info)
            self.assertEqual(info["pid"], os.getpid())
            self.assertTrue(lock.lock_path.exists())
            self.assertTrue(lock.info_path.exists())

            # Read info file directly
            saved_info = json.loads(lock.info_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_info["pid"], os.getpid())

            # Re-entrant acquisition by same process succeeds
            ok2, info2, err2 = lock.acquire()
            self.assertTrue(ok2)
            self.assertIsNone(err2)
            self.assertEqual(info2["pid"], os.getpid())

            # Clean release
            lock.release()
            self.assertFalse(lock.is_held)
            # Permanent lock anchor remains on disk
            self.assertTrue(lock.lock_path.exists())
            # Companion metadata is cleanly removed
            self.assertFalse(lock.info_path.exists())

    def test_02_rejection_of_second_concurrent_process(self):
        """Requirement 2: Verify cross-process rejection using a genuine Python subprocess."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = ApplicationLock(lock_dir=tmpdir)
            ok, info, _ = lock.acquire()
            self.assertTrue(ok)

            # Spawn a separate Python process to attempt acquiring the same lock
            sub_code = f"""
import sys
from pathlib import Path
sys.path.insert(0, r"{Path(__file__).resolve().parent.parent}")
from src.utils.app_lock import ApplicationLock

lock2 = ApplicationLock(lock_dir=r"{tmpdir}")
ok, info, err = lock2.acquire()
if not ok and info is not None and info.get("pid") == {os.getpid()}:
    print("SUBPROCESS_REJECTED_AS_EXPECTED")
    sys.exit(0)
else:
    print(f"UNEXPECTED_ACQUIRE: ok={{ok}}, err={{err}}")
    sys.exit(1)
"""
            result = subprocess.run(
                [sys.executable, "-c", sub_code],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, f"Subprocess failed:\n{result.stderr}")
            self.assertIn("SUBPROCESS_REJECTED_AS_EXPECTED", result.stdout)

            lock.release()

    def test_03_lock_release_and_subsequent_reacquisition(self):
        """Requirement 3: Verify that after release, a separate process can acquire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = ApplicationLock(lock_dir=tmpdir)
            ok, _, _ = lock.acquire()
            self.assertTrue(ok)
            lock.release()
            self.assertFalse(lock.is_held)

            # Subprocess attempts to acquire now that process 1 has released
            sub_code = f"""
import sys
from pathlib import Path
sys.path.insert(0, r"{Path(__file__).resolve().parent.parent}")
from src.utils.app_lock import ApplicationLock

lock2 = ApplicationLock(lock_dir=r"{tmpdir}")
ok, info, err = lock2.acquire()
if ok:
    print("SUBPROCESS_ACQUIRED_AFTER_RELEASE")
    lock2.release()
    sys.exit(0)
else:
    print(f"FAILED_TO_ACQUIRE: {{err}}")
    sys.exit(1)
"""
            result = subprocess.run(
                [sys.executable, "-c", sub_code],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, f"Subprocess failed:\n{result.stderr}")
            self.assertIn("SUBPROCESS_ACQUIRED_AFTER_RELEASE", result.stdout)

    def test_04_stale_lock_recovery(self):
        """Requirement 4: Verify recovery from an abandoned or crashed lock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stale_lock_path = Path(tmpdir) / ".app.lock"
            stale_info_path = Path(tmpdir) / ".app.lock.info"

            # Create an unheld lock file and info file with a dead PID
            stale_lock_path.write_bytes(b"stale lock content\n")
            stale_info_path.write_text(
                json.dumps({"pid": 99999999, "started_at": "2026-01-01T00:00:00"}),
                encoding="utf-8",
            )

            # Process acquires lock despite stale files
            lock = ApplicationLock(lock_dir=tmpdir)
            ok, info, err = lock.acquire()
            self.assertTrue(ok, f"Failed to acquire stale lock: {err}")
            self.assertEqual(info["pid"], os.getpid())

            # Verify info file was overwritten with current PID
            current_info = json.loads(stale_info_path.read_text(encoding="utf-8"))
            self.assertEqual(current_info["pid"], os.getpid())

            lock.release()

    def test_05_startup_config_validation_success(self):
        """Requirement 5: Verify normal Streamlit startup when config is valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            isolated_lock = ApplicationLock(lock_dir=tmpdir)
            try:
                with (
                    patch.object(config, "VECTOR_STORE_DIR", store_dir),
                    patch("src.utils.app_lock.get_application_lock", return_value=isolated_lock),
                ):
                    at = AppTest.from_file(APP_PATH).run(timeout=15)

                self.assertFalse(at.exception)
                self.assertEqual(at.title[0].value, "PDF Chat - RAG Question Answering System")
                # Verify no blocking configuration error was displayed
                config_errors = [
                    e.value for e in at.error if "Production Configuration Error" in e.value
                ]
                self.assertEqual(len(config_errors), 0)
            finally:
                isolated_lock.release()

    def test_06_startup_config_validation_failure(self):
        """Requirement 6: Verify Streamlit startup halts when config is critically invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            isolated_lock = ApplicationLock(lock_dir=tmpdir)
            try:
                # OLLAMA_MAX_TOKENS must be >= 64; 16 triggers fatal validation error
                with (
                    patch.object(config, "VECTOR_STORE_DIR", store_dir),
                    patch("src.utils.app_lock.get_application_lock", return_value=isolated_lock),
                    patch.object(config, "OLLAMA_MAX_TOKENS", 16),
                    patch.dict(os.environ, {"OLLAMA_MAX_TOKENS": "16"}),
                ):
                    at = AppTest.from_file(APP_PATH).run(timeout=15)

                # Error banner rendered and execution halted before vector store load
                config_errors = [
                    e.value for e in at.error if "Production Configuration Error" in e.value
                ]
                self.assertTrue(len(config_errors) > 0)
                self.assertIn("OLLAMA_MAX_TOKENS", config_errors[0])
                self.assertNotIn("vector_store", at.session_state)
            finally:
                isolated_lock.release()

    def test_07_startup_exception_cleanup(self):
        """Requirement 7: Verify context manager releases lock when exceptions occur."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = ApplicationLock(lock_dir=tmpdir)
            try:
                with lock:
                    self.assertTrue(lock.is_held)
                    raise RuntimeError("Simulated startup failure")
            except RuntimeError:
                pass

            self.assertFalse(lock.is_held)

            # Another lock instance can immediately acquire
            lock2 = ApplicationLock(lock_dir=tmpdir)
            ok, _, _ = lock2.acquire()
            self.assertTrue(ok)
            lock2.release()

    def test_08_streamlit_reruns_preserve_lock(self):
        """Requirement 8: Verify Streamlit reruns do not release or fail to re-acquire the lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = get_application_lock(lock_dir=tmpdir)
            ok, info, _ = lock.acquire()
            self.assertTrue(ok)

            # Simulate 5 consecutive Streamlit reruns
            for i in range(5):
                re_ok, re_info, re_err = lock.acquire()
                self.assertTrue(re_ok, f"Rerun {i+1} failed to retain lock: {re_err}")
                self.assertEqual(re_info["pid"], os.getpid())
                self.assertTrue(lock.is_held)

            lock.release()

    def test_09_faiss_transaction_and_rollback_remain_intact(self):
        """Requirement 9: Verify FAISS generation rollback and consistency with application lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Acquire application lock for this storage environment
            app_lock = ApplicationLock(lock_dir=tmpdir)
            ok, _, _ = app_lock.acquire()
            self.assertTrue(ok)

            try:
                store_dir = Path(tmpdir) / "vector_store"

                # 2. Initialize store and add first document (generation 1)
                store = FAISSVectorStore(store_dir=store_dir, embedding_dimension=4)
                chunks = [
                    {
                        "text": "Sample text for chunk 1",
                        "page": 1,
                        "page_number": 1,
                        "chunk_index": 1,
                        "metadata": {"page": 1, "page_number": 1, "chunk_index": 1},
                    }
                ]
                emb = np.ones((1, 4), dtype=np.float32)
                res = store.add_document_chunks(
                    document_id="doc1",
                    filename="doc1.pdf",
                    page_count=1,
                    chunks=chunks,
                    embeddings=emb,
                )
                self.assertEqual(res["document_id"], "doc1")
                self.assertEqual(store.total_documents, 1)
                self.assertEqual(store.active_generation, 1)

                # 3. Simulate failure during next document addition (verify rollback)
                with patch.object(store, "_commit_transaction", side_effect=IOError("Disk commit failed")):
                    with self.assertRaises(IOError):
                        store.add_document_chunks(
                            document_id="doc2",
                            filename="doc2.pdf",
                            page_count=1,
                            chunks=[
                                {
                                    "text": "Sample 2",
                                    "page": 1,
                                    "page_number": 1,
                                    "chunk_index": 1,
                                    "metadata": {"page": 1, "page_number": 1, "chunk_index": 1},
                                }
                            ],
                            embeddings=np.ones((1, 4), dtype=np.float32),
                        )

                # 4. Verify generation 1 is intact and active
                store_reloaded = FAISSVectorStore(store_dir=store_dir, embedding_dimension=4)
                self.assertEqual(store_reloaded.active_generation, 1)
                self.assertEqual(store_reloaded.total_documents, 1)
            finally:
                app_lock.release()

    def test_10_app_ui_lock_conflict_halts_startup(self):
        """Requirement 10: Verify that when another process holds the lock, app startup renders conflict error and halts."""
        mock_lock = MagicMock()
        mock_lock.acquire.return_value = (
            False,
            {"pid": 42424, "started_at": "2026-11-21T00:00:00Z"},
            "Another StudyAstra process holds the lock",
        )
        with patch("src.utils.app_lock.get_application_lock", return_value=mock_lock):
            at = AppTest.from_file(APP_PATH).run(timeout=15)

        errors = [e.value for e in at.error if "Application Lock Conflict" in e.value]
        self.assertTrue(len(errors) > 0)
        self.assertIn("42424", errors[0])
        self.assertNotIn("vector_store", at.session_state)

    def test_11_metadata_removed_before_kernel_lock_release(self):
        """Requirement 11: Verify companion metadata is unlinked strictly before releasing OS kernel lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = ApplicationLock(lock_dir=tmpdir)
            ok, _, _ = lock.acquire()
            self.assertTrue(ok)
            self.assertTrue(lock.info_path.exists())

            call_order: list[str] = []
            real_info_path = lock._info_path

            class TrackingInfoPath:
                def __init__(self, target: Path):
                    self._target = target

                def exists(self):
                    return self._target.exists()

                def unlink(self, missing_ok=False):
                    call_order.append("unlink_info")
                    # Crucial assertion: OS file handle must still be open and locked
                    assert lock._file_handle is not None, "File handle was closed before unlinking metadata!"
                    return self._target.unlink(missing_ok=missing_ok)

                def read_text(self, *args, **kwargs):
                    return self._target.read_text(*args, **kwargs)

            lock._info_path = TrackingInfoPath(real_info_path)

            if sys.platform == "win32":
                import msvcrt

                orig_locking = msvcrt.locking

                def tracked_locking(fd, mode, nbytes):
                    if mode == msvcrt.LK_UNLCK:
                        call_order.append("os_unlock")
                    return orig_locking(fd, mode, nbytes)

                with patch("msvcrt.locking", side_effect=tracked_locking):
                    lock.release()
            else:
                import fcntl

                orig_flock = fcntl.flock

                def tracked_flock(fd, op):
                    if op == fcntl.LOCK_UN:
                        call_order.append("os_unlock")
                    return orig_flock(fd, op)

                with patch("fcntl.flock", side_effect=tracked_flock):
                    lock.release()

            self.assertEqual(call_order, ["unlink_info", "os_unlock"])
            self.assertFalse(real_info_path.exists())
            self.assertTrue(lock.lock_path.exists())

    def test_12_second_process_metadata_not_deleted_by_first_process(self):
        """Requirement 12: Verify second process's metadata is never removed by the first process."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Process 1 acquires and releases
            lock1 = ApplicationLock(lock_dir=tmpdir)
            ok1, _, _ = lock1.acquire()
            self.assertTrue(ok1)
            lock1.release()
            self.assertFalse(lock1.is_held)
            self.assertTrue(lock1.lock_path.exists())

            # Spawn real separate subprocess (Process 2) to acquire and hold the lock
            sub_code = f"""
import sys, time
from pathlib import Path
sys.path.insert(0, r"{Path(__file__).resolve().parent.parent}")
from src.utils.app_lock import ApplicationLock

lock2 = ApplicationLock(lock_dir=r"{tmpdir}")
ok, info, err = lock2.acquire()
if not ok:
    sys.exit(2)
print(f"P2_ACQUIRED:{{info['pid']}}", flush=True)
sys.stdin.readline()
lock2.release()
sys.exit(0)
"""
            proc = subprocess.Popen(
                [sys.executable, "-u", "-c", sub_code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                line = proc.stdout.readline().strip()
                self.assertIn("P2_ACQUIRED", line)
                p2_pid = int(line.split(":")[1])

                # Verify Process 2 metadata exists and has P2's PID
                info_path = Path(tmpdir) / ".app.lock.info"
                self.assertTrue(info_path.exists())
                info_data = json.loads(info_path.read_text(encoding="utf-8"))
                self.assertEqual(info_data["pid"], p2_pid)

                # Process 1 attempts to release() again (e.g. spurious or late call)
                lock1.release()

                # Process 2 metadata must remain intact!
                self.assertTrue(info_path.exists())
                info_after = json.loads(info_path.read_text(encoding="utf-8"))
                self.assertEqual(info_after["pid"], p2_pid)

            finally:
                try:
                    proc.communicate(input="\n", timeout=5)
                except Exception:
                    proc.kill()

    def test_13_metadata_cleanup_exception_does_not_strand_kernel_lock(self):
        """Requirement 13: Verify that an exception during metadata unlink does not prevent OS lock release."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock1 = ApplicationLock(lock_dir=tmpdir)
            ok, _, _ = lock1.acquire()
            self.assertTrue(ok)

            real_info_path = lock1._info_path

            class FailingInfoPath:
                def __init__(self, target: Path):
                    self._target = target

                def exists(self):
                    return True

                def unlink(self, missing_ok=False):
                    raise PermissionError("Simulated metadata unlink permission failure")

                def read_text(self, *args, **kwargs):
                    return self._target.read_text(*args, **kwargs)

            lock1._info_path = FailingInfoPath(real_info_path)

            # Release should catch the metadata error and still release kernel lock & close handle
            lock1.release()
            self.assertFalse(lock1.is_held)
            self.assertIsNone(lock1._file_handle)

            # Another lock instance can immediately acquire the OS lock
            lock2 = ApplicationLock(lock_dir=tmpdir)
            ok2, _, err2 = lock2.acquire()
            self.assertTrue(ok2, f"Failed to acquire after metadata exception: {err2}")
            lock2.release()

    def test_14_repeated_release_calls_remain_safe_no_ops(self):
        """Requirement 14: Verify repeated release() calls are safe, idempotent no-ops."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = ApplicationLock(lock_dir=tmpdir)
            ok, _, _ = lock.acquire()
            self.assertTrue(ok)

            # First release
            lock.release()
            self.assertFalse(lock.is_held)
            self.assertTrue(lock.lock_path.exists())
            self.assertFalse(lock.info_path.exists())

            # Repeated releases
            for _ in range(5):
                lock.release()
                self.assertFalse(lock.is_held)
                self.assertTrue(lock.lock_path.exists())
                self.assertFalse(lock.info_path.exists())

    def test_15_unacquired_instance_cannot_release_active_lock(self):
        """Requirement 15: Verify an unacquired lock instance cannot release another instance's lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_holder = ApplicationLock(lock_dir=tmpdir)
            ok, info, _ = lock_holder.acquire()
            self.assertTrue(ok)
            self.assertTrue(lock_holder.is_held)
            self.assertTrue(lock_holder.info_path.exists())

            # Unacquired instance
            rogue_lock = ApplicationLock(lock_dir=tmpdir)
            self.assertFalse(rogue_lock.is_held)
            rogue_lock.release()

            # Active holder is completely unaffected
            self.assertTrue(lock_holder.is_held)
            self.assertIsNotNone(lock_holder._file_handle)
            self.assertTrue(lock_holder.info_path.exists())
            holder_info = json.loads(lock_holder.info_path.read_text(encoding="utf-8"))
            self.assertEqual(holder_info["pid"], os.getpid())

            lock_holder.release()


if __name__ == "__main__":
    unittest.main()
