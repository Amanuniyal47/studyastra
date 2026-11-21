"""Grounded Follow-Up Question Generator powered by Qwen2.5 7B.

Generates intelligent, grounded follow-up study questions following an initial Q&A turn.
Key guarantees:
- Grounded strictly in retrieved PDF excerpts and previous answer.
- Strict JSON output validation with fallback.
- Centralized LLM access via BaseLLMProvider / OllamaLLMProvider.
"""

import json
import logging
import re
from typing import Any

import src.config as config
from src.generation.llm_provider import BaseLLMProvider, get_llm_provider
from src.generation.prompts import build_context

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

FOLLOW_UP_SYSTEM_PROMPT = (
    "You are an active academic tutor. Based on the user's question, the document answer, "
    "and the source context, suggest relevant, inquisitive follow-up questions to help the "
    "student deepen their conceptual mastery.\n\n"
    "CRITICAL RULES:\n"
    "1. Every follow-up question MUST be answerable from or directly related to the provided context.\n"
    "2. Do not suggest questions about outside or unrelated topics.\n"
    "3. Return ONLY valid JSON matching this schema:\n"
    "{\n"
    '  "follow_up_questions": [\n'
    '    "<concise question 1 ending with a question mark?>",\n'
    '    "<concise question 2 ending with a question mark?>"\n'
    '  ]\n'
    "}"
)


def extract_follow_up_json(raw_text: str) -> list[str]:
    """Parse JSON or extract question lines from LLM output."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return []

    stripped = raw_text.strip()
    candidates = [stripped]

    for match in _CODE_FENCE_RE.findall(stripped):
        fenced = match.strip()
        if fenced:
            candidates.insert(0, fenced)

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict) and "follow_up_questions" in data:
                raw_list = data["follow_up_questions"]
                if isinstance(raw_list, list):
                    cleaned = [
                        str(q).strip()
                        for q in raw_list
                        if isinstance(q, str) and len(str(q).strip()) > 5
                    ]
                    if cleaned:
                        return cleaned
        except Exception:
            continue

    # Fallback line extraction if model emitted a numbered/bulleted list
    lines = []
    for line in stripped.splitlines():
        line_clean = re.sub(r"^\s*(?:\d+[\.\)]|\-|\*)\s*", "", line).strip()
        if line_clean.endswith("?") and len(line_clean) > 8:
            lines.append(line_clean)
    return lines


def generate_follow_up_questions(
    question: str,
    answer: str,
    chunks: list[dict[str, Any]],
    *,
    count: int = 3,
    provider: BaseLLMProvider | None = None,
) -> list[str]:
    """Generate 2-3 logical follow-up questions grounded in the Q&A context."""
    if not question or not str(question).strip():
        return []
    if not chunks:
        return []

    context = build_context(chunks)
    target_count = max(1, min(count, 5))

    user_prompt = (
        f"Context from Document:\n{context}\n\n"
        f"Student's Question: {question.strip()}\n\n"
        f"Provided Answer: {answer.strip()[:400]}\n\n"
        f"Task: Generate exactly {target_count} thoughtful follow-up questions grounded in the context. "
        "Return strictly valid JSON with the 'follow_up_questions' array."
    )

    active_provider = provider if provider is not None else get_llm_provider("ollama")

    try:
        raw_output = active_provider.generate(
            FOLLOW_UP_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.3,
            max_tokens=400,
            format="json",
        )
        questions = extract_follow_up_json(raw_output)
        return questions[:target_count]
    except Exception as exc:
        logger.warning("Failed to generate follow-up questions: %s", exc)
        return []
