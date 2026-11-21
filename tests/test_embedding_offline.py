import os
import socket
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from src.retrieval import embeddings
from src.retrieval.embeddings import (
    EmbeddingModelError,
    REQUIRED_CACHE_FILES,
    load_embedding_model,
    verify_offline_cache,
)


class TestEmbeddingOffline(unittest.TestCase):
    def test_offline_environment_variables_enforced(self):
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

    def test_verify_offline_cache_success_when_all_files_present(self):
        with patch("huggingface_hub.try_to_load_from_cache", return_value="C:/fake/path/file.json"):
            with patch("os.path.isfile", return_value=True):
                is_complete, missing = verify_offline_cache("all-MiniLM-L6-v2")
                self.assertTrue(is_complete)
                self.assertEqual(len(missing), 0)

    def test_verify_offline_cache_detects_missing_file(self):
        def fake_load(repo_id, filename):
            if filename == "model.safetensors":
                return None
            return f"C:/fake/{filename}"

        with patch("huggingface_hub.try_to_load_from_cache", side_effect=fake_load):
            with patch("os.path.isfile", return_value=True):
                is_complete, missing = verify_offline_cache("all-MiniLM-L6-v2")
                self.assertFalse(is_complete)
                self.assertIn("model.safetensors", missing)
                self.assertEqual(len(missing), 1)

    def test_sentence_transformer_called_with_local_files_only(self):
        mock_model = MagicMock()
        with patch("src.retrieval.embeddings.SentenceTransformer", return_value=mock_model) as mock_st_cls:
            load_embedding_model.clear()
            m = load_embedding_model("test-model")
            self.assertIs(m, mock_model)
            mock_st_cls.assert_called_once_with("test-model", local_files_only=True)
            load_embedding_model.clear()

    def test_missing_offline_model_raises_actionable_error(self):
        with patch(
            "src.retrieval.embeddings.SentenceTransformer",
            side_effect=OSError("Offline model files not found"),
        ):
            load_embedding_model.clear()
            with self.assertRaises(EmbeddingModelError) as ctx:
                load_embedding_model("missing-offline-model")
            self.assertIn("prepare_model.py", str(ctx.exception))
            load_embedding_model.clear()

    def test_subprocess_offline_loading_with_network_blocked(self):
        """Spawns a fresh Python subprocess with outbound network access blocked at the socket level."""
        code = """
import socket
import os
import sys

# Monkeypatch socket.socket.connect to simulate strict host firewall / blocked internet
def fail_connect(self, address):
    raise OSError("Network connection strictly blocked by test harness.")

socket.socket.connect = fail_connect

from src.retrieval.embeddings import load_embedding_model, verify_offline_cache

# 1. Verify cache check works without network
is_complete, missing = verify_offline_cache()
assert is_complete, f"Cache verification failed: {missing}"

# 2. Verify model loads with local_files_only=True without network
model = load_embedding_model()
dim = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension", None)
assert dim() == 384, f"Dimension mismatch: {dim()}"

print("SUBPROCESS_OFFLINE_OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Subprocess failed with code {result.returncode}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        )
        self.assertIn("SUBPROCESS_OFFLINE_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
