"""Document Concept & Excerpt Explanation Generator powered by Qwen2.5 7B.

Provides grounded, multi-level explanations of complex concepts or excerpts strictly based
on retrieved PDF context.
Key guarantees:
- Grounded strictly in retrieved chunks.
- Refuses to fabricate explanations if the concept is missing from the document.
- Supports styles: 'standard', 'simplified' (plain English / ELI5), and 'academic'.
- Supports progressive streaming via generate_stream().
- Centralized LLM access via BaseLLMProvider / OllamaLLMProvider.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import logging
from typing import Any

import src.config as config
from src.generation.llm_provider import BaseLLMProvider, get_llm_provider
from src.generation.prompts import build_context
from src.generation.summarizer import format_page_range
from src.retrieval.retriever import retrieve_chunks
from src.retrieval.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

EXPLANATION_STYLES: dict[str, str] = {
    "standard": (
        "Explain clearly, highlighting key ideas, definitions, and relationships in clean prose."
    ),
    "simplified": (
        "Explain in simple, everyday language (ELI5 / beginner friendly). "
        "Use analogies where helpful, but do not alter the technical facts from the text."
    ),
    "academic": (
        "Provide an in-depth, rigorous academic explanation. Analyze theoretical mechanisms, "
        "underlying assumptions, and technical nuances present in the text."
    ),
}

EXPLANATION_SYSTEM_PROMPT = (
    "You are a master academic explainer. Your job is to explain concepts strictly and "
    "faithfully using the provided document excerpts.\n\n"
    "CRITICAL RULES:\n"
    "1. Base your explanation EXCLUSIVELY on the provided context.\n"
    "2. If the document does NOT contain information to explain the concept, explicitly state: "
    "'I cannot explain this concept because it is not covered in the provided document context.'\n"
    "3. Never introduce outside facts or speculative theories.\n"
    "4. Always cite source pages (e.g. [Page X])."
)


@dataclass
class ExplanationResult:
    explanation_text: str
    concept: str
    style: str
    document_id: str
    filename: str
    pages_covered: list[int]
    page_range_label: str
    source_chunks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanation_text": self.explanation_text,
            "concept": self.concept,
            "style": self.style,
            "document_id": self.document_id,
            "filename": self.filename,
            "pages_covered": list(self.pages_covered),
            "page_range_label": self.page_range_label,
            "source_chunks": self.source_chunks,
        }


def explain_concept(
    concept: str,
    chunks: list[dict[str, Any]],
    *,
    style: str = "standard",
    provider: BaseLLMProvider | None = None,
    stream: bool = False,
    temperature: float = 0.2,
) -> str | Iterator[str]:
    """Explain a concept or excerpt grounded strictly in retrieved context chunks."""
    if not concept or not str(concept).strip():
        raise ValueError("Concept or question cannot be empty.")
    if not chunks:
        raise ValueError("No context chunks provided for explanation.")

    style_key = (style or "standard").strip().lower()
    style_instruction = EXPLANATION_STYLES.get(style_key, EXPLANATION_STYLES["standard"])

    context = build_context(chunks)
    user_prompt = (
        f"Context from Document:\n{context}\n\n"
        f"Concept / Passage to Explain: {concept.strip()}\n\n"
        f"Explanation Style: {style_instruction}\n\n"
        "Instructions: Explain the concept strictly based on the context above. "
        "Cite source page numbers. If the concept is not found in the excerpts, "
        "clearly state that the document does not cover it."
    )

    active_provider = provider if provider is not None else get_llm_provider("ollama")

    if stream:
        return active_provider.generate_stream(
            EXPLANATION_SYSTEM_PROMPT,
            user_prompt,
            temperature=temperature,
            max_tokens=config.EXPLANATION_MAX_TOKENS,
        )

    return active_provider.generate(
        EXPLANATION_SYSTEM_PROMPT,
        user_prompt,
        temperature=temperature,
        max_tokens=config.EXPLANATION_MAX_TOKENS,
    )
