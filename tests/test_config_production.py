import os
from pathlib import Path
import unittest
from unittest.mock import patch

from src import config
from src.config import validate_production_config, _safe_int, _safe_float, _safe_bool


class TestConfigProduction(unittest.TestCase):
    def test_default_config_is_production_ready(self):
        is_valid, errors, _ = validate_production_config()
        self.assertTrue(is_valid, f"Default config failed validation: {errors}")
        self.assertEqual(len(errors), 0)

    def test_groq_provider_rejected_in_production(self):
        with patch.object(config, "LLM_PROVIDER", "groq"):
            is_valid, errors, _ = validate_production_config()
            self.assertFalse(is_valid)
            self.assertTrue(
                any("groq" in err.lower() and "prohibited" in err.lower() for err in errors),
                f"Expected Groq prohibition error, got: {errors}",
            )

    def test_unsupported_provider_rejected(self):
        with patch.object(config, "LLM_PROVIDER", "anthropic_claude"):
            is_valid, errors, _ = validate_production_config()
            self.assertFalse(is_valid)
            self.assertTrue(any("unsupported" in err.lower() for err in errors))

    def test_invalid_temperature_bounds(self):
        # Temperature must be between 0.0 and 2.0
        with patch.object(config, "OLLAMA_TEMPERATURE", 2.5):
            with patch.dict(os.environ, {"OLLAMA_TEMPERATURE": "2.5"}):
                is_valid, errors, _ = validate_production_config()
                self.assertFalse(is_valid)
                self.assertTrue(any("OLLAMA_TEMPERATURE" in err for err in errors))

        with patch.object(config, "OLLAMA_TEMPERATURE", -0.5):
            with patch.dict(os.environ, {"OLLAMA_TEMPERATURE": "-0.5"}):
                is_valid, errors, _ = validate_production_config()
                self.assertFalse(is_valid)
                self.assertTrue(any("OLLAMA_TEMPERATURE" in err for err in errors))

    def test_invalid_max_tokens_bounds(self):
        # Max tokens must be >= 64
        with patch.object(config, "OLLAMA_MAX_TOKENS", 32):
            with patch.dict(os.environ, {"OLLAMA_MAX_TOKENS": "32"}):
                is_valid, errors, _ = validate_production_config()
                self.assertFalse(is_valid)
                self.assertTrue(any("OLLAMA_MAX_TOKENS" in err for err in errors))

    def test_invalid_chunk_overlap_constraints(self):
        # Overlap must be < chunk_size and >= 0
        with patch.object(config, "CHUNK_SIZE_TOKENS", 200), patch.object(config, "CHUNK_OVERLAP_TOKENS", 250):
            is_valid, errors, _ = validate_production_config()
            self.assertFalse(is_valid)
            self.assertTrue(any("CHUNK_OVERLAP_TOKENS" in err for err in errors))

        with patch.object(config, "CHUNK_SIZE_TOKENS", 200), patch.object(config, "CHUNK_OVERLAP_TOKENS", -10):
            is_valid, errors, _ = validate_production_config()
            self.assertFalse(is_valid)
            self.assertTrue(any("CHUNK_OVERLAP_TOKENS" in err for err in errors))

        with patch.object(config, "CHUNK_SIZE_TOKENS", 30):
            is_valid, errors, _ = validate_production_config()
            self.assertFalse(is_valid)
            self.assertTrue(any("CHUNK_SIZE_TOKENS" in err for err in errors))

    def test_invalid_ollama_url(self):
        with patch.object(config, "OLLAMA_BASE_URL", "ftp://localhost:11434"):
            is_valid, errors, _ = validate_production_config()
            self.assertFalse(is_valid)
            self.assertTrue(any("OLLAMA_BASE_URL" in err for err in errors))

        with patch.object(config, "OLLAMA_BASE_URL", ""):
            is_valid, errors, _ = validate_production_config()
            self.assertFalse(is_valid)
            self.assertTrue(any("OLLAMA_BASE_URL" in err for err in errors))

    def test_safe_casting_helpers(self):
        self.assertEqual(_safe_int("42", 10), 42)
        self.assertEqual(_safe_int("invalid_int", 10), 10)
        self.assertEqual(_safe_int(None, 10), 10)
        self.assertEqual(_safe_int("", 10), 10)

        self.assertEqual(_safe_float("3.14", 1.0), 3.14)
        self.assertEqual(_safe_float("invalid_float", 1.0), 1.0)
        self.assertEqual(_safe_float(None, 1.0), 1.0)

        self.assertTrue(_safe_bool("true", False))
        self.assertTrue(_safe_bool("1", False))
        self.assertTrue(_safe_bool("YES", False))
        self.assertFalse(_safe_bool("false", True))
        self.assertFalse(_safe_bool("0", True))
        self.assertFalse(_safe_bool(None, False))

    def test_storage_unwritable_directory_detected(self):
        with patch("os.access", return_value=False):
            is_valid, errors, _ = validate_production_config()
            self.assertFalse(is_valid)
            self.assertTrue(any("not writable" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
