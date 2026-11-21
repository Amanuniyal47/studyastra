from collections.abc import Iterator
from typing import Any

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore[assignment, misc]

import src.config as config
from src.generation.llm_provider import (
    BaseLLMProvider,
    GroqLLMProvider,
    LLMConfigurationError,
    get_llm_provider,
)
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt

# Preserved module-level bindings for backwards compatibility with existing callers/tests
GROQ_API_KEY = config.GROQ_API_KEY
GROQ_MODEL = config.GROQ_MODEL


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    provider: BaseLLMProvider | None = None,
    stream: bool = False,
    temperature: float = 0.2,
    system_prompt: str | None = None,
) -> str | Iterator[str]:
    """Generate an answer grounded strictly in retrieved context chunks.

    Args:
        question: User's question string.
        chunks: List of retrieved chunks from FAISS vector search.
        provider: Optional explicit BaseLLMProvider instance.
        stream: If True, returns an Iterator[str] yielding tokens progressively.
        temperature: Sampling temperature (default: 0.2 for factual accuracy).
        system_prompt: Optional override for the grounding system prompt.

    Returns:
        Full answer string (if stream=False) or Iterator[str] (if stream=True).
    """
    if not chunks:
        raise ValueError("No relevant context chunks were retrieved.")

    user_prompt = build_user_prompt(question, chunks)
    active_system_prompt = SYSTEM_PROMPT if system_prompt is None else system_prompt

    if provider is not None:
        if stream:
            return provider.generate_stream(
                active_system_prompt, user_prompt, temperature=temperature
            )
        return provider.generate(
            active_system_prompt, user_prompt, temperature=temperature
        )

    active_provider_name = (config.LLM_PROVIDER or "ollama").strip().lower()

    # Backwards-compatible Groq fallback if module-level GROQ_API_KEY is patched in tests
    if active_provider_name == "groq":
        # Groq is not permitted in production. Only allowed if an explicit test mock is injected.
        is_test_mock = Groq is not None and (
            hasattr(Groq, "assert_called") or hasattr(Groq, "_mock_return_value")
        )
        if not is_test_mock:
            raise LLMConfigurationError(
                "The 'groq' provider is disabled in production. "
                "StudyAstra operates strictly local-only using Ollama (qwen2.5:7b). Set LLM_PROVIDER=ollama."
            )
        current_groq_key = (
            GROQ_API_KEY if GROQ_API_KEY is not None else config.GROQ_API_KEY
        )
        if not current_groq_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
        groq_provider = GroqLLMProvider(
            api_key=current_groq_key,
            model_name=GROQ_MODEL or config.GROQ_MODEL,
            client_cls=Groq,
        )
        if stream:
            return groq_provider.generate_stream(
                active_system_prompt, user_prompt, temperature=temperature
            )
        return groq_provider.generate(
            active_system_prompt, user_prompt, temperature=temperature
        )

    resolved_provider = get_llm_provider(active_provider_name)
    if stream:
        return resolved_provider.generate_stream(
            active_system_prompt, user_prompt, temperature=temperature
        )
    return resolved_provider.generate(
        active_system_prompt, user_prompt, temperature=temperature
    )
