"""Grounded prompt templates, difficulty rubrics, and JSON schema instructions
for Practice Question Generation (Stage 6.2).

Every source chunk header exposes its exact `chunk_id`, and every schema requires
`source_chunk_ids` so downstream deterministic validation can verify chunk ownership
and derive exact page numbers without trusting LLM-generated page metadata.
"""

from typing import Any

import src.config as config
from src.generation.question_validation import (
    CONCRETE_QUESTION_TYPES,
    validate_difficulty,
    validate_question_type,
)

QUESTION_SYSTEM_PROMPT = (
    "You are an academic study-question generator. "
    "Generate practice questions grounded STRICTLY in the provided PDF source chunks. "
    "Never use outside knowledge or invent facts not supported by the excerpts. "
    "Every question MUST include 'source_chunk_ids' containing ONLY exact Chunk IDs "
    "shown in the [Chunk ID: ...] headers that directly support the question and answer. "
    "Return ONLY valid JSON matching the requested schema—no markdown commentary or extra prose."
)

DIFFICULTY_RUBRICS: dict[str, str] = {
    "easy": (
        "Easy: Test direct factual recall, key definitions, and explicit statements "
        "found directly in the excerpts."
    ),
    "medium": (
        "Medium: Test conceptual understanding, relationships between ideas, comparisons, "
        "and cause-and-effect mechanisms explained in the excerpts."
    ),
    "hard": (
        "Hard: Test synthesis across details, subtle distinctions, edge cases, multi-step "
        "reasoning, or applied analysis strictly supported by the excerpts."
    ),
}

QUESTION_TYPE_SCHEMAS: dict[str, str] = {
    "mcq": (
        'Each item in "questions" must follow:\n'
        "{\n"
        '  "question_type": "mcq",\n'
        '  "difficulty": "<easy|medium|hard>",\n'
        '  "question": "<clear, unambiguous question grounded in the excerpts>",\n'
        '  "options": ["<option 1>", "<option 2>", "<option 3>", "<option 4>"],\n'
        '  "correct_answer": "<exact text of the single correct option>",\n'
        '  "explanation": "<concise explanation grounded in the excerpts>",\n'
        '  "source_chunk_ids": ["<exact chunk_id from headers>"]\n'
        "}\n"
        "Rules for MCQ: Provide exactly 4 distinct, plausible options with exactly 1 correct answer."
    ),
    "short_answer": (
        'Each item in "questions" must follow:\n'
        "{\n"
        '  "question_type": "short_answer",\n'
        '  "difficulty": "<easy|medium|hard>",\n'
        '  "question": "<clear open-ended question grounded in the excerpts>",\n'
        '  "expected_answer": "<concise, accurate expected answer from the excerpts>",\n'
        '  "explanation": "<brief grounded explanation of why the answer is correct>",\n'
        '  "source_chunk_ids": ["<exact chunk_id from headers>"]\n'
        "}"
    ),
    "true_false": (
        'Each item in "questions" must follow:\n'
        "{\n"
        '  "question_type": "true_false",\n'
        '  "difficulty": "<easy|medium|hard>",\n'
        '  "question": "<clear declarative statement that is strictly True or False>",\n'
        '  "correct_answer": "<True or False>",\n'
        '  "correct_statement": "<the accurate, source-supported statement (for False items, rewrite to be true)>",\n'
        '  "explanation": "<concise explanation grounded in the excerpts>",\n'
        '  "source_chunk_ids": ["<exact chunk_id from headers>"]\n'
        "}"
    ),
}


def format_question_chunk_entry(chunk: dict[str, Any]) -> str:
    """Format a source chunk with explicit Chunk ID, filename, and page metadata."""
    chunk_id = chunk["chunk_id"]
    filename = chunk["filename"]
    page_range = chunk.get("page_range", chunk["page"])
    chunk_index = chunk["chunk_index"]
    text = chunk["text"].strip()
    return (
        f"[Chunk ID: {chunk_id} | Source: {filename} | Page {page_range} | Chunk {chunk_index}]\n"
        f"{text}"
    )


def format_previous_stems_block(
    previous_stems: list[str] | None,
    *,
    max_stems: int | None = None,
) -> str:
    """Build a bounded anti-duplication block so previous-question context cannot grow unbounded."""
    if not previous_stems:
        return ""

    limit = config.QUESTION_MAX_PREVIOUS_STEMS if max_stems is None else max_stems
    if limit <= 0:
        return ""

    # Keep the most recent stems up to `limit` and truncate each stem concisely
    bounded = previous_stems[-limit:]
    lines = []
    for idx, stem in enumerate(bounded, start=1):
        compact = " ".join(stem.split())
        if len(compact) > 140:
            compact = compact[:137].rstrip() + "..."
        lines.append(f"- ({idx}) {compact}")

    return (
        "Already generated questions (DO NOT repeat or paraphrase these):\n"
        + "\n".join(lines)
        + "\n\n"
    )


def estimate_output_token_allowance(
    question_count: int,
    question_type: str,
    *,
    max_output_tokens: int | None = None,
) -> int:
    """Reserve a bounded output-token allowance scaled by question count and type."""
    qtype = validate_question_type(question_type)
    cap = config.QUESTION_MAX_OUTPUT_TOKENS if max_output_tokens is None else max_output_tokens
    per_question = {
        "mcq": 190,
        "short_answer": 150,
        "true_false": 160,
        "mixed": 175,
    }[qtype]
    estimated = max(250, question_count * per_question + 120)
    return min(cap, estimated)


def build_question_generation_prompt(
    *,
    filename: str,
    scope_description: str,
    question_type: str,
    difficulty: str,
    question_count: int,
    formatted_chunks: list[str],
    type_counts: dict[str, int] | None = None,
    previous_stems: list[str] | None = None,
    max_previous_stems: int | None = None,
    retry_reminder: bool = False,
) -> str:
    """Build a grounded JSON-schema prompt for one batch of practice questions."""
    qtype = validate_question_type(question_type)
    diff = validate_difficulty(difficulty)

    if qtype == "mixed":
        if not type_counts:
            raise ValueError("type_counts is required when building a 'mixed' prompt.")
        active_types = [t for t in CONCRETE_QUESTION_TYPES if type_counts.get(t, 0) > 0]
        count_summary = ", ".join(
            f"{type_counts[t]} {t}" for t in active_types
        )
        type_instruction = (
            f"Generate exactly {question_count} practice questions distributed as: {count_summary}."
        )
        schema_block = "\n\n".join(QUESTION_TYPE_SCHEMAS[t] for t in active_types)
    else:
        type_instruction = (
            f"Generate exactly {question_count} '{qtype}' practice question(s)."
        )
        schema_block = QUESTION_TYPE_SCHEMAS[qtype]

    previous_block = format_previous_stems_block(
        previous_stems,
        max_stems=max_previous_stems,
    )

    retry_block = ""
    if retry_reminder:
        retry_block = (
            "IMPORTANT RETRY NOTICE: Your previous response could not be parsed or failed validation. "
            "Output ONLY a single valid JSON object {\"questions\": [...]} with exact Chunk IDs "
            "copied verbatim from the [Chunk ID: ...] headers below.\n\n"
        )

    joined_chunks = "\n\n".join(formatted_chunks)
    return (
        f"{retry_block}"
        f"Document: {filename}\n"
        f"Scope: {scope_description}\n"
        f"Task: {type_instruction}\n"
        f"Difficulty: {DIFFICULTY_RUBRICS[diff]}\n\n"
        "Quality & Grounding Rules:\n"
        "- Ground every question, answer, distractor, and explanation strictly in the excerpts below.\n"
        "- Copy exact Chunk ID(s) verbatim from the [Chunk ID: ...] headers into 'source_chunk_ids'.\n"
        "- Do NOT invent facts, chunk IDs, or information absent from the excerpts.\n"
        "- Avoid trivial, ambiguous, or duplicate questions. If the excerpts only support fewer than "
        f"{question_count} strong unique questions, return only the strong supported ones.\n\n"
        "Required JSON Output Format:\n"
        'Return ONLY a JSON object: {"questions": [ ... ]}\n'
        f"{schema_block}\n\n"
        f"{previous_block}"
        f"Source Excerpts:\n{joined_chunks}"
    )
