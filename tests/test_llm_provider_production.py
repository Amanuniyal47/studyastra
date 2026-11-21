import sys
import unittest
from unittest.mock import MagicMock, patch

from src import config
from src.generation.answer_generator import generate_answer
from src.generation.llm_provider import (
    BaseLLMProvider,
    GroqLLMProvider,
    LLMConfigurationError,
    LLMProviderError,
    OllamaLLMProvider,
    get_llm_provider,
)


class TestLLMProviderProduction(unittest.TestCase):
    def test_imports_succeed_when_groq_package_uninstalled(self):
        """Simulate groq package being completely unavailable in sys.modules."""
        with patch.dict(sys.modules, {"groq": None}):
            # Verify module attributes and classes load cleanly without error
            from src.generation import llm_provider, answer_generator

            self.assertTrue(hasattr(llm_provider, "OllamaLLMProvider"))
            self.assertTrue(hasattr(llm_provider, "GroqLLMProvider"))
            self.assertTrue(hasattr(answer_generator, "generate_answer"))

    def test_instantiating_groq_without_client_or_package_raises(self):
        with patch("src.generation.llm_provider.Groq", None):
            with self.assertRaises(LLMConfigurationError) as ctx:
                GroqLLMProvider(api_key="test-key", client_cls=None)
            self.assertIn("not installed", str(ctx.exception))

    def test_get_llm_provider_rejects_groq_default(self):
        """When LLM_PROVIDER is set to groq, get_llm_provider() raises LLMConfigurationError."""
        with patch.object(config, "LLM_PROVIDER", "groq"):
            with self.assertRaises(LLMConfigurationError) as ctx:
                get_llm_provider()
            self.assertIn("prohibited in production", str(ctx.exception))

    def test_generate_answer_rejects_groq_in_production(self):
        """Calling generate_answer with LLM_PROVIDER=groq without mock client raises error."""
        chunks = [{"text": "FAISS vector data.", "metadata": {"page": 1, "chunk_index": 1}}]
        # Real Groq class or unpatched Groq
        with patch.object(config, "LLM_PROVIDER", "groq"):
            # Ensure Groq is not a MagicMock
            real_dummy_cls = type("DummyRealGroq", (), {})
            with patch("src.generation.answer_generator.Groq", real_dummy_cls):
                with self.assertRaises(LLMConfigurationError) as ctx:
                    generate_answer("What is FAISS?", chunks)
                self.assertIn("disabled in production", str(ctx.exception))

    def test_ollama_failure_never_calls_groq(self):
        """Verify that when Ollama raises an error, zero attempts are made to call Groq."""
        chunks = [{"text": "FAISS vector data.", "metadata": {"page": 1, "chunk_index": 1}}]
        mock_ollama = MagicMock(spec=OllamaLLMProvider)
        mock_ollama.generate.side_effect = LLMProviderError("Ollama daemon down.")

        mock_groq_cls = MagicMock()
        with (
            patch.object(config, "LLM_PROVIDER", "ollama"),
            patch("src.generation.answer_generator.get_llm_provider", return_value=mock_ollama),
            patch("src.generation.answer_generator.Groq", mock_groq_cls),
        ):
            with self.assertRaises(LLMProviderError):
                generate_answer("Question?", chunks)

            mock_groq_cls.assert_not_called()

    def test_injected_provider_used_directly(self):
        class _TestCustomProvider(BaseLLMProvider):
            def is_configured(self) -> bool:
                return True

            def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
                return "Injected provider answer."

        custom = _TestCustomProvider("custom_model")
        chunks = [{"text": "Context chunk", "metadata": {"page": 1, "chunk_index": 1}}]
        result = generate_answer("Question?", chunks, provider=custom)
        self.assertEqual(result, "Injected provider answer.")


if __name__ == "__main__":
    unittest.main()
