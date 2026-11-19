import unittest
from unittest.mock import MagicMock, patch

import groq
import httpx

import src.config as config
from src.generation.answer_generator import generate_answer
from src.generation.llm_provider import (
    BaseLLMProvider,
    GroqLLMProvider,
    LLMConfigurationError,
    LLMProviderError,
    _PROVIDER_REGISTRY,
    get_llm_provider,
    list_registered_providers,
    register_llm_provider,
)


class _MockCustomProvider(BaseLLMProvider):
    provider_name = "mock_local"

    def __init__(self, model_name: str = "qwen2.5:3b"):
        super().__init__(model_name=model_name)
        self.calls: list[dict] = []

    def is_configured(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return "Mock local response."


def _dummy_httpx_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=request)


class TestStage61aLLMProvider(unittest.TestCase):
    def test_config_defaults_present(self):
        self.assertEqual(config.LLM_PROVIDER, "groq")
        self.assertGreater(config.SUMMARY_MAX_BATCH_TOKENS, 500)
        self.assertGreater(config.SUMMARY_TOPIC_TOP_K, 0)
        self.assertGreater(config.LLM_TIMEOUT_SECONDS, 0)

    def test_groq_provider_success_and_max_tokens(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="  Summary output.  "))]
        mock_client_cls = MagicMock()
        mock_client_cls.return_value.chat.completions.create.return_value = mock_response

        provider = GroqLLMProvider(
            api_key="gsk_test_123",
            model_name="openai/gpt-oss-120b",
            timeout=30.0,
            client_cls=mock_client_cls,
        )
        self.assertTrue(provider.is_configured())
        result = provider.generate(
            "You are a helper.",
            "Summarize this text.",
            temperature=0.3,
            max_tokens=400,
        )
        self.assertEqual(result, "Summary output.")
        mock_client_cls.assert_called_once_with(api_key="gsk_test_123", timeout=30.0)
        kwargs = mock_client_cls.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/gpt-oss-120b")
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertEqual(kwargs["max_tokens"], 400)

    def test_groq_provider_missing_key_and_invalid_inputs(self):
        provider = GroqLLMProvider(api_key="", model_name="openai/gpt-oss-120b")
        self.assertFalse(provider.is_configured())
        with self.assertRaises(LLMConfigurationError):
            provider.generate("sys", "user")

        configured = GroqLLMProvider(api_key="test-key", model_name="openai/gpt-oss-120b")
        with self.assertRaises(ValueError):
            configured.generate("   ", "user")
        with self.assertRaises(ValueError):
            configured.generate("sys", "")
        with self.assertRaises(ValueError):
            configured.generate("sys", "user", max_tokens=0)

    def test_groq_provider_handles_api_errors_cleanly(self):
        error_cases = [
            groq.AuthenticationError("invalid key", response=_dummy_httpx_response(401), body=None),
            groq.RateLimitError("rate limited", response=_dummy_httpx_response(429), body=None),
            groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com")),
            groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com")),
            groq.InternalServerError("server err", response=_dummy_httpx_response(500), body=None),
            RuntimeError("unexpected failure"),
        ]

        for exc in error_cases:
            mock_client_cls = MagicMock()
            mock_client_cls.return_value.chat.completions.create.side_effect = exc
            provider = GroqLLMProvider(
                api_key="test-key",
                model_name="openai/gpt-oss-120b",
                client_cls=mock_client_cls,
            )
            with self.assertRaises(LLMProviderError):
                provider.generate("sys", "user")

    def test_groq_provider_rejects_empty_completion(self):
        for bad_response in (
            MagicMock(choices=[]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="   "))]),
        ):
            mock_client_cls = MagicMock()
            mock_client_cls.return_value.chat.completions.create.return_value = bad_response
            provider = GroqLLMProvider(
                api_key="test-key",
                model_name="openai/gpt-oss-120b",
                client_cls=mock_client_cls,
            )
            with self.assertRaises(LLMProviderError):
                provider.generate("sys", "user")

    def test_provider_factory_and_future_ollama_extensibility(self):
        provider = get_llm_provider("groq", api_key="test-key")
        self.assertIsInstance(provider, GroqLLMProvider)
        self.assertIn("groq", list_registered_providers())

        # Ollama is intentionally not implemented yet in Stage 6.1
        with self.assertRaises(LLMConfigurationError) as ctx:
            get_llm_provider("ollama")
        self.assertIn("not implemented yet", str(ctx.exception).lower())

        with self.assertRaises(LLMConfigurationError):
            get_llm_provider("unknown_provider")

        # Registering a future provider works cleanly via registry
        try:
            register_llm_provider("mock_local", _MockCustomProvider)
            local_provider = get_llm_provider("mock_local", model_name="qwen2.5:3b")
            self.assertEqual(local_provider.model_name, "qwen2.5:3b")
            self.assertEqual(local_provider.generate("sys", "user"), "Mock local response.")
        finally:
            _PROVIDER_REGISTRY.pop("mock_local", None)

    def test_generate_answer_accepts_injected_provider(self):
        custom = _MockCustomProvider()
        chunks = [{"text": "Study fact.", "metadata": {"page": 1, "chunk_index": 1}}]
        answer = generate_answer("What is the fact?", chunks, provider=custom)
        self.assertEqual(answer, "Mock local response.")
        self.assertEqual(len(custom.calls), 1)
        self.assertIn("Study fact.", custom.calls[0]["user_prompt"])


if __name__ == "__main__":
    unittest.main()
