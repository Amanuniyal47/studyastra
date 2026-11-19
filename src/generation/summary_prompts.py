"""Grounded prompt templates for single-pass and hierarchical PDF summarization."""

from typing import Any

SUMMARY_LENGTH_SPECS: dict[str, dict[str, Any]] = {
    "short": {
        "label": "Short",
        "guidance": (
            "Write a concise executive summary (about 100-150 words, 3-5 focused bullet points). "
            "Highlight only the core thesis, key definitions, and most critical findings."
        ),
        "max_tokens": 350,
    },
    "medium": {
        "label": "Medium",
        "guidance": (
            "Write a balanced structured study summary (about 250-400 words). "
            "Organize by major themes with concise bullet points covering important facts, "
            "key definitions, notable examples, and main takeaways."
        ),
        "max_tokens": 700,
    },
    "detailed": {
        "label": "Detailed",
        "guidance": (
            "Write comprehensive structured study notes (about 500-800 words). "
            "Use clear section headings and preserve important facts, definitions, "
            "mechanisms, formulas or processes, concrete examples, and key caveats."
        ),
        "max_tokens": 1300,
    },
}

SUMMARY_SYSTEM_PROMPT = (
    "You are a careful academic study assistant that summarizes PDF study material. "
    "Use ONLY the provided source excerpts. "
    "Preserve important facts, definitions, key terms, and concrete examples. "
    "Cite page numbers inline using ONLY page numbers explicitly shown in the excerpt headers "
    "(for example, [p. 3] or [pp. 3-4]). "
    "NEVER invent, guess, or extrapolate page numbers or facts not present in the excerpts."
)


def validate_summary_length(length: str) -> str:
    normalized = (length or "").strip().lower()
    if normalized not in SUMMARY_LENGTH_SPECS:
        valid = ", ".join(SUMMARY_LENGTH_SPECS.keys())
        raise ValueError(f"Unsupported summary length '{length}'. Choose one of: {valid}.")
    return normalized


def format_chunk_entry(chunk: dict[str, Any]) -> str:
    """Format a single raw PDF chunk with explicit filename and page attribution."""
    filename = chunk["filename"]
    page_range = chunk.get("page_range", chunk["page"])
    chunk_index = chunk["chunk_index"]
    text = chunk["text"].strip()
    return f"[Source: {filename} | Page {page_range} | Chunk {chunk_index}]\n{text}"


def format_intermediate_summary_entry(
    *,
    filename: str,
    page_range_label: str,
    batch_label: str,
    summary_text: str,
) -> str:
    """Format an intermediate batch summary with verified source page coverage."""
    return (
        f"[Source: {filename} | {page_range_label} | {batch_label}]\n"
        f"{summary_text.strip()}"
    )


def build_single_pass_user_prompt(
    *,
    filename: str,
    page_range_label: str,
    scope_description: str,
    length: str,
    formatted_entries: list[str],
) -> str:
    spec = SUMMARY_LENGTH_SPECS[length]
    joined = "\n\n".join(formatted_entries)
    return (
        f"Document: {filename}\n"
        f"Scope: {scope_description} ({page_range_label})\n"
        f"Target Length: {spec['label']} — {spec['guidance']}\n\n"
        "Instructions:\n"
        "- Summarize strictly from the excerpts below.\n"
        "- Preserve key facts, definitions, examples, and accurate inline page citations "
        "using only the page numbers shown in each [Source: ...] header.\n"
        "- Do not invent any page numbers or outside facts.\n\n"
        f"Excerpts:\n{joined}"
    )


def build_map_batch_user_prompt(
    *,
    filename: str,
    page_range_label: str,
    batch_index: int,
    total_batches: int,
    scope_description: str,
    length: str,
    formatted_entries: list[str],
) -> str:
    spec = SUMMARY_LENGTH_SPECS[length]
    joined = "\n\n".join(formatted_entries)
    return (
        f"Document: {filename}\n"
        f"Scope: {scope_description}\n"
        f"Batch: {batch_index} of {total_batches} ({page_range_label})\n"
        f"Final Target Length: {spec['label']}\n\n"
        "Instructions:\n"
        "- Produce a faithful intermediate study summary of this batch so it can be combined "
        "with other batches from the same document.\n"
        "- Retain important facts, definitions, formulas/mechanisms, and concrete examples "
        "rather than over-compressing.\n"
        "- Preserve exact inline page references (e.g., [p. X]) using ONLY page numbers "
        "explicitly shown in the excerpt headers.\n\n"
        f"Batch Excerpts:\n{joined}"
    )


def build_reduce_user_prompt(
    *,
    filename: str,
    page_range_label: str,
    scope_description: str,
    length: str,
    formatted_entries: list[str],
    is_final: bool,
    reduction_level: int = 1,
) -> str:
    spec = SUMMARY_LENGTH_SPECS[length]
    joined = "\n\n".join(formatted_entries)

    if is_final:
        role_instruction = (
            f"Combine the partial summaries below from '{filename}' into ONE cohesive final summary.\n"
            f"Target Length: {spec['label']} — {spec['guidance']}\n"
            "- Remove repetition across overlapping batches while preserving key facts, "
            "definitions, concrete examples, and accurate inline page citations.\n"
            "- Cite ONLY page numbers explicitly present in the partial summaries/headers."
        )
    else:
        role_instruction = (
            f"Consolidate the partial summaries below from '{filename}' (reduction level {reduction_level}).\n"
            "- Preserve important facts, definitions, examples, and exact page references "
            "from every covered page range so no critical study details are lost before final synthesis.\n"
            "- Cite ONLY page numbers explicitly present in the partial summaries/headers."
        )

    return (
        f"Document: {filename}\n"
        f"Scope: {scope_description} ({page_range_label})\n\n"
        f"Instructions:\n{role_instruction}\n\n"
        f"Partial Summaries:\n{joined}"
    )
