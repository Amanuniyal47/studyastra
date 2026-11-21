from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from scripts import check_env
from src import config


class TestCheckEnv(unittest.TestCase):
    def test_run_all_checks_passes_on_valid_environment(self):
        mock_resp_version = MagicMock(status_code=200)
        mock_resp_version.json.return_value = {"version": "0.5.1"}

        mock_resp_tags = MagicMock(status_code=200)
        mock_resp_tags.json.return_value = {"models": [{"name": "qwen2.5:7b"}]}

        def fake_get(url, timeout=2.0):
            if "version" in url:
                return mock_resp_version
            if "tags" in url:
                return mock_resp_tags
            return MagicMock(status_code=404)

        with (
            patch("httpx.get", side_effect=fake_get),
            patch("src.retrieval.embeddings.verify_offline_cache", return_value=(True, [])),
        ):
            results = check_env.run_all_checks()
            failures = [r for r in results if r.status == "FAIL"]
            self.assertEqual(len(failures), 0, f"Unexpected failures: {[f.message for f in failures]}")

    def test_check_env_warns_when_ollama_unavailable(self):
        with (
            patch("httpx.get", side_effect=ConnectionError("Refused")),
            patch("src.retrieval.embeddings.verify_offline_cache", return_value=(True, [])),
        ):
            results = check_env.run_all_checks()
            warnings = [r for r in results if r.status == "WARN"]
            self.assertTrue(any("Ollama" in w.name for w in warnings))
            # Critical failures should still be 0 if only Ollama is unreachable
            failures = [r for r in results if r.status == "FAIL"]
            self.assertEqual(len(failures), 0)

    def test_check_env_warns_when_qwen_model_missing(self):
        mock_resp_version = MagicMock(status_code=200)
        mock_resp_version.json.return_value = {"version": "0.5.1"}

        mock_resp_tags = MagicMock(status_code=200)
        mock_resp_tags.json.return_value = {"models": [{"name": "llama3:latest"}]}

        def fake_get(url, timeout=2.0):
            if "version" in url:
                return mock_resp_version
            if "tags" in url:
                return mock_resp_tags
            return MagicMock(status_code=404)

        with (
            patch("httpx.get", side_effect=fake_get),
            patch("src.retrieval.embeddings.verify_offline_cache", return_value=(True, [])),
        ):
            results = check_env.run_all_checks()
            warnings = [r for r in results if r.status == "WARN" and "Model" in r.name]
            self.assertEqual(len(warnings), 1)
            self.assertIn("not found", warnings[0].message)

    def test_check_env_fails_when_cache_incomplete(self):
        with patch("src.retrieval.embeddings.verify_offline_cache", return_value=(False, ["model.safetensors"])):
            results = check_env.run_all_checks()
            failures = [r for r in results if r.status == "FAIL" and "Cache" in r.name]
            self.assertEqual(len(failures), 1)
            self.assertIn("prepare_model.py", failures[0].message)

    def test_check_env_fails_when_config_invalid(self):
        with patch.object(config, "LLM_PROVIDER", "groq"):
            results = check_env.run_all_checks()
            failures = [r for r in results if r.status == "FAIL" and "Configuration" in r.name]
            self.assertTrue(len(failures) >= 1)
            self.assertTrue(any("groq" in f.message.lower() for f in failures))

    def test_check_env_never_modifies_data_dir(self):
        data_dir = config.PROJECT_ROOT / "data"
        if data_dir.exists():
            files_before = set(data_dir.rglob("*"))
        else:
            files_before = set()

        check_env.run_all_checks()

        if data_dir.exists():
            files_after = set(data_dir.rglob("*"))
        else:
            files_after = set()

        self.assertEqual(files_before, files_after, "check_env modified files inside data/")


if __name__ == "__main__":
    unittest.main()
