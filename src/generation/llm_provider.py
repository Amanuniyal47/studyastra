"""Reusable LLM provider abstraction decoupled from embeddings and FAISS.

Currently implements `GroqLLMProvider` and exposes a provider registry/factory
(`register_llm_provider`, `get_llm_provider`) so future local providers such as
Ollama + Qwen can be plugged in without rewriting summarization, Q&A, or UI code.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any

import groq
from groq import Groq

import src.config as config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base exception for LLM generation errors."""


class LLMConfigurationError(LLMError):
    """Raised when the configured LLM provider or credentials are invalid."""


class LLMProviderError(LLMError):
    """Raised when an LLM API call fails or returns an invalid response."""


class BaseLLMProvider(ABC):
    """Common interface for all LLM chat-completion providers."""

    provider_name: str = "base"

    def __init__(self, model_name: str) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise LLMConfigurationError("LLM model_name must be a non-empty string.")
        self.model_name = model_name.strip()

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if required provider credentials/settings are present."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text completion from system and user prompts."""


class GroqLLMProvider(BaseLLMProvider):
    """Groq Cloud chat-completion provider."""

    provider_name: str = "groq"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: float | None = None,
        client_cls: type[Any] | None = None,
    ) -> None:
        resolved_model = config.GROQ_MODEL if model_name is None else model_name
        super().__init__(model_name=resolved_model)
        self._api_key = config.GROQ_API_KEY if api_key is None else api_key
        self.timeout = config.LLM_TIMEOUT_SECONDS if timeout is None else float(timeout)
        self._client_cls = Groq if client_cls is None else client_cls

    def is_configured(self) -> bool:
        return bool(isinstance(self._api_key, str) and self._api_key.strip())

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        if not self.is_configured():
            raise LLMConfigurationError(
                "GROQ_API_KEY is missing. Add it to your .env file."
            )
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string.")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string.")
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer.")

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            client = self._client_cls(api_key=self._api_key, timeout=self.timeout)
        except TypeError:
            # Compatibility if a mock or custom client only accepts api_key
            client = self._client_cls(api_key=self._api_key)

        try:
            response = client.chat.completions.create(**kwargs)
        except groq.AuthenticationError as exc:
            logger.exception("Groq authentication failed for model '%s'", self.model_name)
            raise LLMProviderError(
                "Groq authentication failed. Verify your GROQ_API_KEY in .env."
            ) from exc
        except groq.RateLimitError as exc:
            logger.exception("Groq rate limit exceeded for model '%s'", self.model_name)
            raise LLMProviderError(
                "Groq rate limit reached. Please wait a moment and try again, "
                "or reduce SUMMARY_MAX_BATCH_TOKENS."
            ) from exc
        except groq.APITimeoutError as exc:
            logger.exception("Groq request timed out for model '%s'", self.model_name)
            raise LLMProviderError(
                "The Groq API request timed out. Please try again."
            ) from exc
        except groq.APIConnectionError as exc:
            logger.exception("Groq connection error for model '%s'", self.model_name)
            raise LLMProviderError(
                "Could not connect to the Groq API. Check your internet connection."
            ) from exc
        except groq.APIStatusError as exc:
            logger.exception("Groq API error for model '%s'", self.model_name)
            raise LLMProviderError(
                f"Groq API returned an error (HTTP {getattr(exc, 'status_code', 'unknown')}): {exc}"
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected Groq error for model '%s'", self.model_name)
            raise LLMProviderError(f"LLM generation failed: {exc}") from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMProviderError("Groq returned an empty response with no choices.")

        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("Groq returned an empty completion text.")

        return content.strip()


_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "groq": GroqLLMProvider,
}


def register_llm_provider(name: str, provider_cls: type[BaseLLMProvider]) -> None:
    """Register a custom or future LLM provider (e.g., Ollama) by name."""
    normalized = (name or "").strip().lower()
    if not normalized:
        raise ValueError("Provider name must be a non-empty string.")
    if not isinstance(provider_cls, type) or not issubclass(provider_cls, BaseLLMProvider):
        raise TypeError("provider_cls must be a subclass of BaseLLMProvider.")
    _PROVIDER_REGISTRY[normalized] = provider_cls


def list_registered_providers() -> list[str]:
    return sorted(_PROVIDER_REGISTRY.keys())


def get_llm_provider(
    provider_name: str | None = None,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Instantiate the configured LLM provider."""
    raw_name = config.LLM_PROVIDER if provider_name is None else provider_name
    normalized = (raw_name or "").strip().lower()
    if not normalized:
        raise LLMConfigurationError("LLM_PROVIDER is not configured.")

    if normalized in _PROVIDER_REGISTRY:
        return _PROVIDER_REGISTRY[normalized](**kwargs)

    if normalized == "ollama":
        raise LLMConfigurationError(
            "LLM provider 'ollama' is planned for a future phase and is not implemented yet. "
            "Set LLM_PROVIDER=groq in .env for now."
        )

    available = ", ".join(list_registered_providers())
    raise LLMConfigurationError(
        f"Unsupported LLM provider '{normalized}'. Available provider(s): {available}."
    )
