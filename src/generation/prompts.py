"""System prompts and context construction for PDF-grounded RAG generation."""

from typing import Any

# Robust grounded system prompt for local Qwen2.5 and cloud LLMs
GROUNDED_SYSTEM_PROMPT = (
    "You are an expert AI study assistant answering questions strictly based on the provided PDF context.\n\n"
    "CRITICAL GROUNDING RULES:\n"
    "1. Strict Context Adherence: Base your answer EXCLUSIVELY on the facts and information directly "
    "provided in the context below. Do NOT use outside knowledge, prior assumptions, or speculative extrapolation.\n"
    "2. Factual Accuracy: Prioritise factual accuracy and precision over answer completeness. "
    "Never fabricate, hallucinate, or assume facts, statistics, dates, quotations, or references.\n"
    "3. Insufficient Context Handling: If the provided context does NOT contain sufficient information to "
    "answer the question, clearly and explicitly state: 'I cannot answer this question based on the provided "
    "document context.' Do not guess or partially fabricate an answer.\n"
    "4. Explicit vs. Interpretation: Clearly distinguish between information explicitly stated in the text "
    "and reasonable interpretations or inferences.\n"
    "5. Technical Terminology: Preserve technical terminology, mathematical expressions, formulas, and "
    "domain-specific definitions exactly as they appear in the source text.\n"
    "6. Structured Explanations: Organize complex explanations using clear headings, bullet points, or "
    "numbered steps when appropriate for study clarity.\n"
    "7. Source Attribution: Include source references using the source numbers and page references from "
    "the context (e.g., [Source 1, Page 2]).\n"
    "8. Page-Specific Accuracy: For page-specific questions, ONLY state that information appears on a "
    "particular page when the retrieved source metadata explicitly confirms that page number."
)

# Backwards compatibility alias
SYSTEM_PROMPT = GROUNDED_SYSTEM_PROMPT


def build_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a structured context string with source numbering and page metadata."""
    sources = []
    for index, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        page = meta.get("page", chunk.get("page", "unknown"))
        filename = meta.get("filename", chunk.get("filename", ""))
        file_part = f" ({filename})" if filename else ""
        text = chunk.get("text", "")
        sources.append(f"Source {index} (page {page}):{file_part}\n{text}")
    return "\n\n".join(sources)


def build_user_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Construct a user prompt combining formatted context, user question, and grounding instructions."""
    context = build_context(chunks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question.strip()}\n\n"
        f"Instructions: Answer the question using ONLY the provided context above. "
        f"Cite source numbers (e.g. Source 1, page X) for your answers. "
        f"If the answer cannot be found in the context, explicitly state that the document does not contain the answer."
    )
