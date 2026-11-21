"""Reusable LLM provider abstraction decoupled from embeddings and FAISS.

Currently implements `GroqLLMProvider` and exposes a provider registry/factory
(`register_llm_provider`, `get_llm_provider`) so future local providers such as
Ollama + Qwen can be plugged in without rewriting summarization, Q&A, or UI code.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
import json
import logging
import threading
import time
from typing import Any

try:
    import groq
    from groq import Groq
    _GROQ_AUTH_ERR = groq.AuthenticationError
    _GROQ_RATE_ERR = groq.RateLimitError
    _GROQ_TIMEOUT_ERR = groq.APITimeoutError
    _GROQ_CONN_ERR = groq.APIConnectionError
    _GROQ_STATUS_ERR = groq.APIStatusError
except ImportError:
    groq = None  # type: ignore[assignment]
    Groq = None  # type: ignore[assignment, misc]
    class _GroqDummyError(Exception):
        pass
    _GROQ_AUTH_ERR = _GroqDummyError
    _GROQ_RATE_ERR = _GroqDummyError
    _GROQ_TIMEOUT_ERR = _GroqDummyError
    _GROQ_CONN_ERR = _GroqDummyError
    _GROQ_STATUS_ERR = _GroqDummyError
import httpx

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

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Generate streaming text chunks. Default implementation yields full completion."""
        full_text = self.generate(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        yield full_text


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
        if Groq is None and client_cls is None:
            raise LLMConfigurationError(
                "The 'groq' package is not installed. Groq is not supported in production."
            )
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
        except _GROQ_AUTH_ERR as exc:
            logger.exception("Groq authentication failed for model '%s'", self.model_name)
            raise LLMProviderError(
                "Groq authentication failed. Verify your GROQ_API_KEY in .env."
            ) from exc
        except _GROQ_RATE_ERR as exc:
            logger.exception("Groq rate limit exceeded for model '%s'", self.model_name)
            raise LLMProviderError(
                "Groq rate limit reached. Please wait a moment and try again, "
                "or reduce SUMMARY_MAX_BATCH_TOKENS."
            ) from exc
        except _GROQ_TIMEOUT_ERR as exc:
            logger.exception("Groq request timed out for model '%s'", self.model_name)
            raise LLMProviderError(
                "The Groq API request timed out. Please try again."
            ) from exc
        except _GROQ_CONN_ERR as exc:
            logger.exception("Groq connection error for model '%s'", self.model_name)
            raise LLMProviderError(
                "Could not connect to the Groq API. Check your internet connection."
            ) from exc
        except _GROQ_STATUS_ERR as exc:
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

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
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
            "stream": True,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            client = self._client_cls(api_key=self._api_key, timeout=self.timeout)
        except TypeError:
            client = self._client_cls(api_key=self._api_key)

        try:
            response = client.chat.completions.create(**kwargs)
        except _GROQ_AUTH_ERR as exc:
            logger.exception("Groq authentication failed for model '%s'", self.model_name)
            raise LLMProviderError(
                "Groq authentication failed. Verify your GROQ_API_KEY in .env."
            ) from exc
        except _GROQ_RATE_ERR as exc:
            logger.exception("Groq rate limit exceeded for model '%s'", self.model_name)
            raise LLMProviderError(
                "Groq rate limit reached. Please wait a moment and try again."
            ) from exc
        except _GROQ_TIMEOUT_ERR as exc:
            logger.exception("Groq request timed out for model '%s'", self.model_name)
            raise LLMProviderError(
                "The Groq API request timed out. Please try again."
            ) from exc
        except _GROQ_CONN_ERR as exc:
            logger.exception("Groq connection error for model '%s'", self.model_name)
            raise LLMProviderError(
                "Could not connect to the Groq API. Check your internet connection."
            ) from exc
        except _GROQ_STATUS_ERR as exc:
            logger.exception("Groq API error for model '%s'", self.model_name)
            raise LLMProviderError(
                f"Groq API returned an error (HTTP {getattr(exc, 'status_code', 'unknown')}): {exc}"
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected Groq error for model '%s'", self.model_name)
            raise LLMProviderError(f"LLM streaming failed: {exc}") from exc

        yielded_any = False
        for chunk in response:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            delta_content = getattr(delta, "content", None)
            if delta_content:
                yielded_any = True
                yield delta_content

        if not yielded_any:
            raise LLMProviderError("Groq returned an empty completion stream.")


class OllamaLLMProvider(BaseLLMProvider):
    """Local Ollama chat-completion provider for models such as Qwen2.5 7B."""

    provider_name: str = "ollama"

    _health_cache_lock = threading.Lock()
    _connection_cache: dict[str, tuple[float, bool, str]] = {}
    _model_cache: dict[tuple[str, str], tuple[float, bool, str]] = {}

    @classmethod
    def invalidate_health_cache(
        cls,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Invalidate cached health check and model availability results."""
        with cls._health_cache_lock:
            if base_url is None:
                cls._connection_cache.clear()
            else:
                cls._connection_cache.pop(base_url.rstrip("/"), None)

            if base_url is None and model_name is None:
                cls._model_cache.clear()
            elif base_url is not None and model_name is not None:
                cls._model_cache.pop((base_url.rstrip("/"), model_name.strip().lower()), None)
            elif base_url is not None:
                keys_to_remove = [k for k in cls._model_cache if k[0] == base_url.rstrip("/")]
                for k in keys_to_remove:
                    cls._model_cache.pop(k, None)
            elif model_name is not None:
                keys_to_remove = [k for k in cls._model_cache if k[1] == model_name.strip().lower()]
                for k in keys_to_remove:
                    cls._model_cache.pop(k, None)

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float | None = None,
        context_window: int | None = None,
        temperature: float | None = None,
        keep_alive: str | None = None,
        client: Any | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff_factor: float | None = None,
        retry_backoff_max: float | None = None,
        stream_idle_timeout: float | None = None,
        health_cache_ttl: float | None = None,
    ) -> None:
        resolved_model = config.OLLAMA_MODEL if model_name is None else model_name
        super().__init__(model_name=resolved_model)

        raw_base_url = config.OLLAMA_BASE_URL if base_url is None else base_url
        if not isinstance(raw_base_url, str) or not raw_base_url.strip():
            raise LLMConfigurationError("Ollama base_url must be a non-empty string.")
        self.base_url = raw_base_url.strip().rstrip("/")

        self.timeout = (
            config.OLLAMA_TIMEOUT_SECONDS if timeout is None else float(timeout)
        )
        self.connect_timeout = (
            float(connect_timeout)
            if connect_timeout is not None
            else float(getattr(config, "OLLAMA_CONNECT_TIMEOUT_SECONDS", 10.0))
        )
        self.read_timeout = (
            float(read_timeout)
            if read_timeout is not None
            else float(getattr(config, "OLLAMA_READ_TIMEOUT_SECONDS", 120.0))
        )
        self.max_retries = (
            int(max_retries)
            if max_retries is not None
            else int(getattr(config, "OLLAMA_MAX_RETRIES", 2))
        )
        self.backoff_factor = (
            float(retry_backoff_factor)
            if retry_backoff_factor is not None
            else float(getattr(config, "OLLAMA_RETRY_BACKOFF_FACTOR", 0.5))
        )
        self.backoff_max = (
            float(retry_backoff_max)
            if retry_backoff_max is not None
            else float(getattr(config, "OLLAMA_RETRY_BACKOFF_MAX_SECONDS", 3.0))
        )
        self.stream_idle_timeout = (
            float(stream_idle_timeout)
            if stream_idle_timeout is not None
            else float(getattr(config, "OLLAMA_STREAM_IDLE_TIMEOUT_SECONDS", 30.0))
        )
        self.context_window = (
            config.OLLAMA_CONTEXT_WINDOW if context_window is None else context_window
        )
        self.default_temperature = (
            config.OLLAMA_TEMPERATURE if temperature is None else float(temperature)
        )
        self.keep_alive = (
            config.OLLAMA_KEEP_ALIVE if keep_alive is None else str(keep_alive)
        )
        self._client = client

    def _build_client_timeout(self, is_stream: bool = False) -> httpx.Timeout:
        """Construct fine-grained connection, read, write, and pool timeouts."""
        read_t = self.stream_idle_timeout if is_stream else self.read_timeout
        return httpx.Timeout(
            timeout=self.timeout,
            connect=self.connect_timeout,
            read=read_t,
            write=self.timeout,
        )

    def is_configured(self) -> bool:
        """Return True if base_url and model_name are set."""
        return bool(self.base_url and self.model_name)

    def check_connection(
        self,
        force: bool = False,
        ttl: float | None = None,
    ) -> tuple[bool, str]:
        """Check if the local Ollama server is running and reachable with TTL caching."""
        effective_ttl = ttl if ttl is not None else getattr(config, "OLLAMA_HEALTH_CHECK_TTL_SECONDS", 20.0)
        url_key = self.base_url.rstrip("/")
        now = time.monotonic()

        if not force and effective_ttl > 0:
            with self._health_cache_lock:
                cached = self._connection_cache.get(url_key)
                if cached is not None:
                    cached_time, ok, msg = cached
                    if now - cached_time < effective_ttl:
                        return ok, msg

        endpoint = f"{self.base_url}/api/tags"
        try:
            if self._client is not None and hasattr(self._client, "get"):
                resp = self._client.get(endpoint, timeout=min(self.timeout, 5.0))
            else:
                resp = httpx.get(endpoint, timeout=min(self.timeout, 5.0))
            if resp.status_code == 200:
                is_ok, msg = True, f"Ollama is reachable at {self.base_url}."
            else:
                is_ok, msg = False, f"Ollama returned HTTP status {resp.status_code}."
        except httpx.ConnectError:
            is_ok, msg = False, (
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure the Ollama service is running (`ollama serve`)."
            )
        except httpx.TimeoutException:
            is_ok, msg = False, f"Connection to Ollama at {self.base_url} timed out."
        except Exception as exc:
            is_ok, msg = False, f"Failed to connect to Ollama at {self.base_url}: {exc}"

        with self._health_cache_lock:
            self._connection_cache[url_key] = (time.monotonic(), is_ok, msg)
        return is_ok, msg

    def check_model_available(
        self,
        model_name: str | None = None,
        force: bool = False,
        ttl: float | None = None,
    ) -> tuple[bool, str]:
        """Check if the target model is installed/pulled in Ollama with TTL caching."""
        target = (self.model_name if model_name is None else model_name).strip().lower()
        effective_ttl = ttl if ttl is not None else getattr(config, "OLLAMA_HEALTH_CHECK_TTL_SECONDS", 20.0)
        cache_key = (self.base_url.rstrip("/"), target)
        now = time.monotonic()

        if not force and effective_ttl > 0:
            with self._health_cache_lock:
                cached = self._model_cache.get(cache_key)
                if cached is not None:
                    cached_time, ok, msg = cached
                    if now - cached_time < effective_ttl:
                        return ok, msg

        endpoint = f"{self.base_url}/api/tags"
        try:
            if self._client is not None and hasattr(self._client, "get"):
                resp = self._client.get(endpoint, timeout=min(self.timeout, 5.0))
            else:
                resp = httpx.get(endpoint, timeout=min(self.timeout, 5.0))
            if resp.status_code != 200:
                is_ok, msg = False, f"Ollama returned HTTP status {resp.status_code}."
            else:
                data = (
                    resp.json()
                    if callable(getattr(resp, "json", None))
                    else json.loads(resp.text)
                )
                models = data.get("models", [])
                installed: list[str] = []
                for m in models:
                    name = (m.get("name") or m.get("model") or "").lower()
                    if name:
                        installed.append(name)

                matched = any(
                    name == target
                    or name.split(":")[0] == target
                    or (":" not in target and name.startswith(f"{target}:"))
                    for name in installed
                )
                if matched:
                    is_ok, msg = True, f"Model '{target}' is installed and available in Ollama."
                else:
                    avail_str = ", ".join(installed) if installed else "none"
                    is_ok, msg = False, (
                        f"Model '{target}' is not installed in Ollama. "
                        f"Currently installed: [{avail_str}]. Run `ollama pull {target}`."
                    )
        except httpx.ConnectError:
            is_ok, msg = False, (
                f"Cannot connect to Ollama at {self.base_url}. "
                "Ensure the Ollama service is running (`ollama serve`)."
            )
        except httpx.TimeoutException:
            is_ok, msg = False, f"Connection to Ollama at {self.base_url} timed out."
        except Exception as exc:
            is_ok, msg = False, f"Error checking Ollama models: {exc}"

        with self._health_cache_lock:
            self._model_cache[cache_key] = (time.monotonic(), is_ok, msg)
        return is_ok, msg

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int | None,
        stream: bool,
        format: str | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        elif config.OLLAMA_MAX_TOKENS:
            options["num_predict"] = config.OLLAMA_MAX_TOKENS

        if self.context_window is not None:
            options["num_ctx"] = self.context_window

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": stream,
            "options": options,
        }

        # Enable constrained JSON decoding when format is requested or prompt demands valid JSON
        sys_lower = system_prompt.lower()
        if format == "json" or "valid json" in sys_lower or "json schema" in sys_lower:
            payload["format"] = "json"

        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        return payload

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        format: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not self.is_configured():
            raise LLMConfigurationError("Ollama provider is not configured.")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string.")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string.")
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer.")

        payload = self._build_payload(
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            stream=False,
            format=format,
        )
        url = f"{self.base_url}/api/chat"
        client_timeout = self._build_client_timeout(is_stream=False)

        for attempt in range(self.max_retries + 1):
            try:
                if self._client is not None and hasattr(self._client, "post"):
                    response = self._client.post(url, json=payload, timeout=client_timeout)
                else:
                    with httpx.Client(timeout=client_timeout) as client:
                        response = client.post(url, json=payload)
            except httpx.ConnectError as exc:
                if attempt < self.max_retries:
                    delay = min(self.backoff_max, self.backoff_factor * (2 ** attempt))
                    logger.warning(
                        "Ollama connection error (attempt %d/%d); retrying in %.2fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue

                logger.exception("Failed to connect to Ollama at %s after %d attempt(s)", self.base_url, attempt + 1)
                self.invalidate_health_cache(base_url=self.base_url)
                with self._health_cache_lock:
                    self._connection_cache[self.base_url.rstrip("/")] = (
                        time.monotonic(),
                        False,
                        f"Cannot connect to Ollama at {self.base_url}. Ensure the Ollama service is running (`ollama serve`).",
                    )
                raise LLMProviderError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Please ensure Ollama is installed and running (`ollama serve`)."
                ) from exc
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    delay = min(self.backoff_max, self.backoff_factor * (2 ** attempt))
                    logger.warning(
                        "Ollama request timeout (attempt %d/%d) for model '%s'; retrying in %.2fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        self.model_name,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue

                logger.exception("Ollama request timed out for model '%s' after %d attempt(s)", self.model_name, attempt + 1)
                raise LLMProviderError(
                    f"Ollama request timed out after {self.timeout}s for model '{self.model_name}'."
                ) from exc
            except Exception as exc:
                logger.exception("Unexpected error communicating with Ollama at %s", self.base_url)
                raise LLMProviderError(f"Ollama connection error: {exc}") from exc

            # HTTP 404: Model not found is permanent; do NOT retry
            if response.status_code == 404:
                self.invalidate_health_cache(base_url=self.base_url, model_name=self.model_name)
                with self._health_cache_lock:
                    self._model_cache[(self.base_url.rstrip("/"), self.model_name.strip().lower())] = (
                        time.monotonic(),
                        False,
                        f"Model '{self.model_name}' was not found in Ollama. Please pull it first using `ollama pull {self.model_name}`.",
                    )
                raise LLMProviderError(
                    f"Model '{self.model_name}' was not found in Ollama. "
                    f"Please pull it first using `ollama pull {self.model_name}`."
                )

            # HTTP 429, 500, 502, 503, 504: Transient server errors; retry with backoff
            if response.status_code in (429, 500, 502, 503, 504):
                err_text = response.text
                if attempt < self.max_retries:
                    delay = min(self.backoff_max, self.backoff_factor * (2 ** attempt))
                    logger.warning(
                        "Ollama returned transient HTTP %d (attempt %d/%d); retrying in %.2fs: %s",
                        response.status_code,
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        err_text[:200],
                    )
                    time.sleep(delay)
                    continue
                raise LLMProviderError(
                    f"Ollama returned HTTP {response.status_code}: {err_text}"
                )

            # Permanent non-200 errors (e.g. 400 Bad Request)
            if response.status_code != 200:
                err_text = response.text
                try:
                    err_json = (
                        response.json()
                        if callable(getattr(response, "json", None))
                        else json.loads(response.text)
                    )
                    if isinstance(err_json, dict) and "error" in err_json:
                        err_text = err_json["error"]
                except Exception:
                    pass
                raise LLMProviderError(
                    f"Ollama returned HTTP {response.status_code}: {err_text}"
                )

            # Response 200 OK
            try:
                data = (
                    response.json()
                    if callable(getattr(response, "json", None))
                    else json.loads(response.text)
                )
            except Exception as exc:
                raise LLMProviderError(f"Failed to parse Ollama response as JSON: {exc}") from exc

            message = data.get("message")
            if not isinstance(message, dict):
                raise LLMProviderError("Ollama returned a response missing the 'message' field.")

            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise LLMProviderError("Ollama returned an empty completion text.")

            return content.strip()

        raise LLMProviderError("Ollama generation failed after maximum retries.")

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        format: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if not self.is_configured():
            raise LLMConfigurationError("Ollama provider is not configured.")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string.")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string.")
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer.")

        payload = self._build_payload(
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            stream=True,
            format=format,
        )
        url = f"{self.base_url}/api/chat"
        client_timeout = self._build_client_timeout(is_stream=True)

        yielded_any = False

        for attempt in range(self.max_retries + 1):
            try:
                if self._client is not None and hasattr(self._client, "stream"):
                    cm = self._client.stream("POST", url, json=payload, timeout=client_timeout)
                    with cm as response:
                        for token in self._process_stream(response):
                            yielded_any = True
                            yield token
                    return
                else:
                    with httpx.Client(timeout=client_timeout) as client:
                        with client.stream("POST", url, json=payload) as response:
                            for token in self._process_stream(response):
                                yielded_any = True
                                yield token
                    return
            except httpx.ConnectError as exc:
                if yielded_any:
                    logger.error("Connection lost mid-stream from Ollama at %s", self.base_url)
                    raise LLMProviderError(f"Connection lost mid-stream from Ollama: {exc}") from exc
                if attempt < self.max_retries:
                    delay = min(self.backoff_max, self.backoff_factor * (2 ** attempt))
                    logger.warning(
                        "Ollama stream connection error (attempt %d/%d); retrying in %.2fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue

                logger.exception("Failed to connect to Ollama streaming at %s", self.base_url)
                self.invalidate_health_cache(base_url=self.base_url)
                with self._health_cache_lock:
                    self._connection_cache[self.base_url.rstrip("/")] = (
                        time.monotonic(),
                        False,
                        f"Cannot connect to Ollama at {self.base_url}. Ensure the Ollama service is running (`ollama serve`).",
                    )
                raise LLMProviderError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Please ensure Ollama is installed and running (`ollama serve`)."
                ) from exc
            except httpx.TimeoutException as exc:
                if yielded_any:
                    logger.error("Streaming timed out mid-stream for model '%s'", self.model_name)
                    raise LLMProviderError(
                        f"Streaming timed out mid-stream for model '{self.model_name}'."
                    ) from exc
                if attempt < self.max_retries:
                    delay = min(self.backoff_max, self.backoff_factor * (2 ** attempt))
                    logger.warning(
                        "Ollama stream timeout (attempt %d/%d); retrying in %.2fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue

                logger.exception("Ollama streaming timed out for model '%s'", self.model_name)
                raise LLMProviderError(
                    f"Ollama streaming timed out after {self.timeout}s for model '{self.model_name}'."
                ) from exc
            except LLMProviderError:
                raise
            except Exception as exc:
                logger.exception("Unexpected error during Ollama streaming from %s", self.base_url)
                raise LLMProviderError(f"Ollama streaming failed: {exc}") from exc

    def _process_stream(self, response: Any) -> Iterator[str]:
        if response.status_code == 404:
            self.invalidate_health_cache(base_url=self.base_url, model_name=self.model_name)
            with self._health_cache_lock:
                self._model_cache[(self.base_url.rstrip("/"), self.model_name.strip().lower())] = (
                    time.monotonic(),
                    False,
                    f"Model '{self.model_name}' was not found in Ollama. Please pull it first using `ollama pull {self.model_name}`.",
                )
            raise LLMProviderError(
                f"Model '{self.model_name}' was not found in Ollama. "
                f"Please pull it first using `ollama pull {self.model_name}`."
            )
        if response.status_code != 200:
            raise LLMProviderError(
                f"Ollama returned HTTP {response.status_code} during streaming."
            )

        yielded_any = False
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.strip()
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "error" in chunk:
                    raise LLMProviderError(f"Ollama streaming error: {chunk['error']}")

                content = chunk.get("message", {}).get("content", "")
                if content:
                    yielded_any = True
                    yield content

                if chunk.get("done", False):
                    break
        finally:
            if hasattr(response, "close"):
                try:
                    response.close()
                except Exception:
                    pass

        if not yielded_any:
            raise LLMProviderError("Ollama returned an empty completion stream.")


_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "groq": GroqLLMProvider,
    "ollama": OllamaLLMProvider,
}


def register_llm_provider(name: str, provider_cls: type[BaseLLMProvider]) -> None:
    """Register a custom or future LLM provider by name."""
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

    if provider_name is None and normalized == "groq":
        raise LLMConfigurationError(
            "The 'groq' provider is prohibited in production. "
            "StudyAstra operates strictly local-only using Ollama (qwen2.5:7b). Set LLM_PROVIDER=ollama."
        )

    if normalized in _PROVIDER_REGISTRY:
        return _PROVIDER_REGISTRY[normalized](**kwargs)

    available = ", ".join(list_registered_providers())
    raise LLMConfigurationError(
        f"Unsupported LLM provider '{normalized}'. Available provider(s): {available}."
    )
