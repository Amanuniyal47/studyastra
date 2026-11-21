"""Targeted tests for Ollama Health-Check TTL Caching (Stage 11.2 Work Area A)."""

import time
import unittest
from unittest.mock import MagicMock, patch
import threading

import httpx

from src.generation.llm_provider import (
    OllamaLLMProvider,
    LLMProviderError,
)


class TestOllamaHealthCheckTTLCache(unittest.TestCase):
    def setUp(self):
        OllamaLLMProvider.invalidate_health_cache()
        self.provider = OllamaLLMProvider(
            base_url="http://localhost:11434",
            model_name="qwen2.5:7b",
            timeout=5.0,
        )

    def tearDown(self):
        OllamaLLMProvider.invalidate_health_cache()

    def test_01_first_check_executes_http_request(self):
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            ok, msg = self.provider.check_connection()
            self.assertTrue(ok)
            self.assertEqual(mock_get.call_count, 1)

    def test_02_repeated_check_within_ttl_uses_cached_result(self):
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            ok1, msg1 = self.provider.check_connection(ttl=10.0)
            self.assertTrue(ok1)
            self.assertEqual(mock_get.call_count, 1)

            # Second call within TTL must NOT invoke httpx.get
            ok2, msg2 = self.provider.check_connection(ttl=10.0)
            self.assertTrue(ok2)
            self.assertEqual(mock_get.call_count, 1)
            self.assertEqual(msg1, msg2)

    def test_03_expired_ttl_triggers_fresh_request(self):
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            # Check with a very short TTL
            ok1, _ = self.provider.check_connection(ttl=0.05)
            self.assertTrue(ok1)
            self.assertEqual(mock_get.call_count, 1)

            # Wait for TTL to expire
            time.sleep(0.06)

            # Second call should trigger fresh HTTP request
            ok2, _ = self.provider.check_connection(ttl=0.05)
            self.assertTrue(ok2)
            self.assertEqual(mock_get.call_count, 2)

    def test_04_force_flag_bypasses_cache(self):
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            ok1, _ = self.provider.check_connection(ttl=20.0)
            self.assertTrue(ok1)
            self.assertEqual(mock_get.call_count, 1)

            # force=True must issue new request even within TTL
            ok2, _ = self.provider.check_connection(force=True, ttl=20.0)
            self.assertTrue(ok2)
            self.assertEqual(mock_get.call_count, 2)

    def test_05_failed_connection_cached_and_recovers(self):
        request = httpx.Request("GET", "http://localhost:11434/api/tags")
        with patch("httpx.get", side_effect=httpx.ConnectError("Daemon down", request=request)) as mock_get:
            ok1, msg1 = self.provider.check_connection(ttl=10.0)
            self.assertFalse(ok1)
            self.assertIn("Cannot connect to Ollama", msg1)
            self.assertEqual(mock_get.call_count, 1)

            # Subsequent call within TTL returns cached failure without calling HTTP
            ok2, msg2 = self.provider.check_connection(ttl=10.0)
            self.assertFalse(ok2)
            self.assertEqual(mock_get.call_count, 1)

        # Invalidate cache, and verify recovery when daemon comes back up
        OllamaLLMProvider.invalidate_health_cache()
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            ok3, msg3 = self.provider.check_connection()
            self.assertTrue(ok3)
            self.assertIn("reachable", msg3)
            self.assertEqual(mock_get.call_count, 1)

    def test_06_model_availability_cached_per_model_and_endpoint(self):
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "models": [{"name": "qwen2.5:7b"}, {"name": "llama3:latest"}]
        }
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            # Check model 1
            ok1, _ = self.provider.check_model_available("qwen2.5:7b", ttl=10.0)
            self.assertTrue(ok1)
            self.assertEqual(mock_get.call_count, 1)

            # Check model 1 again -> uses cache
            ok1_cached, _ = self.provider.check_model_available("qwen2.5:7b", ttl=10.0)
            self.assertTrue(ok1_cached)
            self.assertEqual(mock_get.call_count, 1)

            # Check model 2 -> triggers new request because cache key differs
            ok2, _ = self.provider.check_model_available("llama3:latest", ttl=10.0)
            self.assertTrue(ok2)
            self.assertEqual(mock_get.call_count, 2)

            # Check unknown model -> triggers new request and caches missing
            ok3, _ = self.provider.check_model_available("mistral:7b", ttl=10.0)
            self.assertFalse(ok3)
            self.assertEqual(mock_get.call_count, 3)

            # Check unknown model again -> uses cached missing
            ok3_cached, _ = self.provider.check_model_available("mistral:7b", ttl=10.0)
            self.assertFalse(ok3_cached)
            self.assertEqual(mock_get.call_count, 3)

    def test_07_changing_endpoint_uses_distinct_cache(self):
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            p1 = OllamaLLMProvider(base_url="http://localhost:11434")
            p2 = OllamaLLMProvider(base_url="http://otherhost:11434")

            p1.check_connection(ttl=10.0)
            self.assertEqual(mock_get.call_count, 1)

            # p2 has different base_url, must not reuse p1's cache
            p2.check_connection(ttl=10.0)
            self.assertEqual(mock_get.call_count, 2)

    def test_08_generation_failure_invalidates_and_marks_health_cache(self):
        # Seed positive cache
        mock_resp_health = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp_health):
            ok, _ = self.provider.check_connection(ttl=10.0)
            self.assertTrue(ok)

        # Trigger generation ConnectError
        request = httpx.Request("POST", "http://localhost:11434/api/chat")
        with patch("httpx.Client.post", side_effect=httpx.ConnectError("Abrupt drop", request=request)):
            with self.assertRaises(LLMProviderError):
                self.provider.generate("Sys", "User")

        # Now health check must report failure immediately without making a new HTTP request
        with patch("httpx.get") as mock_get:
            ok_after, msg_after = self.provider.check_connection(ttl=10.0)
            self.assertFalse(ok_after)
            self.assertIn("Cannot connect to Ollama", msg_after)
            # mock_get should NOT have been called because failure state was updated in cache
            self.assertEqual(mock_get.call_count, 0)

    def test_09_concurrent_checks_thread_safe(self):
        mock_resp = MagicMock(status_code=200)
        with patch("httpx.get", return_value=mock_resp) as mock_get:
            results = []

            def worker():
                ok, msg = self.provider.check_connection(ttl=5.0)
                results.append(ok)

            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(results), 10)
            self.assertTrue(all(results))
            # With cache lock and immediate caching, all threads receive success cleanly
            self.assertLessEqual(mock_get.call_count, 2)

    def test_10_generation_404_updates_model_health_cache(self):
        """Verify HTTP 404 during generation sets model cache to False and bypasses HTTP on subsequent checks."""
        mock_resp_404 = MagicMock(status_code=404, text='{"error": "model not found"}')
        mock_resp_404.json.return_value = {"error": "model not found"}

        with patch("httpx.Client.post", return_value=mock_resp_404):
            with self.assertRaises(LLMProviderError):
                self.provider.generate("Sys", "User")

        # Now check_model_available should report False from cache without calling httpx.get
        with patch("httpx.get") as mock_get:
            ok, msg = self.provider.check_model_available("qwen2.5:7b", ttl=10.0)
            self.assertFalse(ok)
            self.assertIn("not found in Ollama", msg)
            self.assertEqual(mock_get.call_count, 0)

    def test_11_http_500_status_cached_as_failure(self):
        """Verify HTTP 500 error responses are captured and cached as failure."""
        mock_resp_500 = MagicMock(status_code=500, text="Internal Server Error")
        with patch("httpx.get", return_value=mock_resp_500) as mock_get:
            ok, msg = self.provider.check_connection(ttl=10.0)
            self.assertFalse(ok)
            self.assertIn("500", msg)
            self.assertEqual(mock_get.call_count, 1)

            # Subsequent check within TTL uses cached failure
            ok2, msg2 = self.provider.check_connection(ttl=10.0)
            self.assertFalse(ok2)
            self.assertEqual(mock_get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
