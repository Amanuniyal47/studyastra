from groq import Groq

import src.config as config
from src.generation.llm_provider import BaseLLMProvider, GroqLLMProvider, get_llm_provider
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt

# Preserved module-level bindings for backwards compatibility with existing callers/tests
GROQ_API_KEY = config.GROQ_API_KEY
GROQ_MODEL = config.GROQ_MODEL


def generate_answer(question, chunks, *, provider: BaseLLMProvider | None = None):
    if not chunks:
        raise ValueError("No relevant context chunks were retrieved.")

    user_prompt = build_user_prompt(question, chunks)

    if provider is not None:
        return provider.generate(SYSTEM_PROMPT, user_prompt, temperature=0.2)

    active_provider_name = (config.LLM_PROVIDER or "groq").strip().lower()
    if active_provider_name == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")
        groq_provider = GroqLLMProvider(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            client_cls=Groq,
        )
        return groq_provider.generate(SYSTEM_PROMPT, user_prompt, temperature=0.2)

    resolved_provider = get_llm_provider(active_provider_name)
    return resolved_provider.generate(SYSTEM_PROMPT, user_prompt, temperature=0.2)
