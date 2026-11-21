"""Grounded chat prompts, document evidence budgeting, and citation validation for multi-turn PDF chat.

Stage 10.2 Implementation:
- Grounded system prompt enforcing strict context adherence, mandatory [Source X, Page Y] citations,
  and explicit refusal when information is absent.
- Deterministic document evidence block formatting ([Source 1, Page X] ...) with source_map generation.
- Strict token-budget hierarchy:
  1. Complete User Question & Grounding Framing: 100% byte-for-byte preserved; NEVER silently truncated.
  2. Document Evidence: Authoritative context prioritized first.
  3. Conversation History: Dynamically reduced pair-aware before evidence is touched.
  4. PromptBudgetExceededError: Raised predictably if question + system prompt + output reserve cannot fit.
- Structured GroundedChatPrompt dataclass with prompt assembly and unpackable interface.
- Citation extraction, validation against actual chunk metadata, and hallucination sanitization.

TOKEN-BUDGET & CONTEXT CEILING GUARANTEES (Area 3):
All token budgeting, sliding-window allocations, and hard ceiling limits in Studyastra
are computed and enforced against the project's internal regex-based token estimator
(src.ingestion.chunker.estimate_tokens).

Because local Qwen2.5 7B utilizes a Byte-Pair Encoding (BPE) tokenizer, token counts
for rare technical terms, code, or non-English text may exhibit a ~10-15% variance
compared to regex-based word and punctuation estimates.

The configured `output_reserve_tokens` (default: 1,000) and `CHAT_SAFETY_MARGIN_RATIO`
(default: 0.10) serve as explicit architectural buffers designed to absorb this
estimation variance. Hard ceiling guarantees are strictly enforced against estimated
token metrics, ensuring prompts do not silently overflow model context boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any

from src.chat.conversation_memory import (
    ChatMessage,
    ConversationMemoryConfig,
    ConversationThread,
    SlidingWindowMemoryManager,
)
from src.ingestion.chunker import estimate_tokens

logger = logging.getLogger(__name__)


class PromptBudgetExceededError(ValueError):
    """Raised when the complete user question, system instructions, and output reserve
    cannot fit within the configured context window even after all reductions.
    """


# Grounded system prompt for conversational multi-turn chat
CHAT_GROUNDED_SYSTEM_PROMPT = (
    "You are Studyastra, an expert academic AI study assistant answering questions strictly "
    "based on the provided document excerpts.\n\n"
    "CRITICAL GROUNDING & CITATION RULES:\n"
    "1. Strict Context Adherence: Base your answer EXCLUSIVELY on the provided Document Evidence below. "
    "Do NOT assume, speculate, or use outside knowledge not present in the evidence.\n"
    "2. Mandatory Source Citations: You MUST cite every factual statement, definition, finding, or summary "
    "using exact source notation: [Source X, Page Y], where X is the Source number and Y is the Page number "
    "indicated in the Document Evidence.\n"
    "3. Insufficient Context Handling: If the Document Evidence does NOT contain sufficient information "
    "to answer the question, clearly state: 'I cannot answer this question based on the provided document context.' "
    "Do NOT fabricate, guess, or synthesize unsupported answers.\n"
    "4. Conversation History Grounding: Use conversation history ONLY to resolve conversational references "
    "(such as pronouns or follow-up context). Never treat prior conversation statements as authoritative document "
    "facts unless they are supported by the current Document Evidence.\n"
    "5. Technical Precision: Preserve domain terminology, equations, numbers, and exact definitions as they "
    "appear in the document excerpts."
)

# Regular expression matching bracketed source citations: [Source 1, Page 2], [Source 1, p. 2], [Source 1]
_CITATION_PATTERN = re.compile(
    r"\[Source\s*(\d+)(?:,\s*(?:Page|p\.|p)?\s*([0-9a-zA-Z\-_]+))?\]",
    re.IGNORECASE,
)


@dataclass
class GroundedChatPrompt:
    """Encapsulates system prompt, user prompt, and source metadata for chat generation."""

    system_prompt: str
    user_prompt: str
    source_map: dict[int, dict[str, Any]] = field(default_factory=dict)
    evidence_tokens: int = 0
    history_tokens: int = 0
    history_budget: int = 0
    total_prompt_tokens: int = 0
    output_reserve_tokens: int = 0
    context_window_tokens: int = 0
    excluded_chunks: int = 0
    evidence_truncated: bool = False

    @property
    def full_prompt(self) -> str:
        """Combined prompt string for inspection or unified prompt callers."""
        return f"{self.system_prompt}\n\n{self.user_prompt}"

    def __iter__(self):
        """Allows direct unpacking: sys_prompt, user_prompt, source_map = prompt"""
        return iter((self.system_prompt, self.user_prompt, self.source_map))


class EvidenceBlockResult(tuple):
    """3-tuple subclass (evidence_text, token_count, source_map) with budget metadata.

    Preserves 100% backwards compatibility for callers unpacking as a 3-tuple:
        evidence_text, tokens, source_map = build_document_evidence_block(...)
    while exposing .excluded_count and .evidence_truncated attributes for audit checks.
    """

    def __new__(
        cls,
        evidence_text: str,
        token_count: int,
        source_map: dict[int, dict[str, Any]],
        *,
        excluded_count: int = 0,
        evidence_truncated: bool = False,
    ):
        return super().__new__(cls, (evidence_text, token_count, source_map))

    def __init__(
        self,
        evidence_text: str,
        token_count: int,
        source_map: dict[int, dict[str, Any]],
        *,
        excluded_count: int = 0,
        evidence_truncated: bool = False,
    ):
        self.evidence_text = evidence_text
        self.token_count = token_count
        self.source_map = source_map
        self.excluded_count = excluded_count
        self.evidence_truncated = evidence_truncated


def build_document_evidence_block(
    chunks: list[dict[str, Any]],
    *,
    max_evidence_tokens: int | None = None,
) -> EvidenceBlockResult:
    """Format retrieved chunks into numbered document evidence blocks with budget enforcement.

    Builds deterministic source identifiers [Source 1, Page X] and a mapping
    from source index (1-based) to chunk metadata for validation.

    Budget enforcement rules:
    - Retains as many full chunks as fit within max_evidence_tokens in rank order.
    - If a chunk would exceed the budget, it and all subsequent chunks are excluded.
    - Excluded chunks are omitted from source_map and explicitly noted in the footer.
    - If even Chunk 1 alone exceeds max_evidence_tokens, Chunk 1 is safely truncated
      with an explicit truncation indicator.
    - Strictly guarantees total estimated tokens <= max_evidence_tokens when provided.

    Args:
        chunks: List of chunk dictionaries retrieved from the vector store.
        max_evidence_tokens: Optional token budget cap for the evidence block.

    Returns:
        EvidenceBlockResult: 3-tuple (evidence_block, token_count, source_map) with
        .excluded_count and .evidence_truncated attributes.
    """
    if not chunks:
        empty_text = (
            "=== DOCUMENT EVIDENCE (Authoritative) ===\n"
            "No relevant document excerpts were retrieved for this query.\n"
            "=== END DOCUMENT EVIDENCE ==="
        )
        return EvidenceBlockResult(
            empty_text,
            estimate_tokens(empty_text),
            {},
            excluded_count=0,
            evidence_truncated=False,
        )

    # If evidence budget is too tight to hold even minimal framing, omit evidence cleanly
    if max_evidence_tokens is not None and max_evidence_tokens < 20:
        return EvidenceBlockResult(
            "",
            0,
            {},
            excluded_count=len(chunks),
            evidence_truncated=True,
        )

    # Compact notice when budget is very constrained (< 45 tokens)
    if max_evidence_tokens is not None and max_evidence_tokens < 45:
        compact_notice = (
            "=== DOCUMENT EVIDENCE ===\n"
            f"[Note: {len(chunks)} document excerpt(s) omitted to strictly adhere to context limits.]\n"
            "=== END DOCUMENT EVIDENCE ==="
        )
        return EvidenceBlockResult(
            compact_notice,
            estimate_tokens(compact_notice),
            {},
            excluded_count=len(chunks),
            evidence_truncated=True,
        )

    header = (
        "=== DOCUMENT EVIDENCE (Authoritative) ===\n"
        "The following excerpts are the ONLY authoritative source of truth for answering the question.\n\n"
    )
    footer = "\n=== END DOCUMENT EVIDENCE ==="
    framing_tokens = estimate_tokens(header + footer)

    chunk_budget = (
        max(0, max_evidence_tokens - framing_tokens)
        if max_evidence_tokens is not None
        else None
    )

    sources_text: list[str] = []
    source_map: dict[int, dict[str, Any]] = {}
    excluded_count = 0
    evidence_truncated = False
    accumulated_chunk_tokens = 0

    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        page = meta.get("page", chunk.get("page", "unknown"))
        filename = meta.get("filename", chunk.get("filename", ""))
        chunk_id = meta.get("chunk_id", chunk.get("chunk_id", f"chunk_{idx}"))
        score = chunk.get("score")
        text = chunk.get("text", "").strip()

        file_part = f" (File: {filename})" if filename else ""
        chunk_header = f"[Source {idx}, Page {page}]{file_part}"
        chunk_entry = f"{chunk_header}\n{text}"
        chunk_tokens = estimate_tokens(chunk_entry) + 2

        if chunk_budget is None or accumulated_chunk_tokens + chunk_tokens <= chunk_budget:
            sources_text.append(chunk_entry)
            source_map[idx] = {
                "source_id": idx,
                "page": page,
                "filename": filename,
                "chunk_id": chunk_id,
                "score": score,
                "text": text,
            }
            accumulated_chunk_tokens += chunk_tokens
        else:
            if sources_text:
                excluded_count = len(chunks) - len(sources_text)
                logger.info(
                    "Excluding %d lower-ranked document chunks to respect evidence budget (%d tokens)",
                    excluded_count,
                    max_evidence_tokens,
                )
                break
            else:
                # Even Chunk 1 alone exceeds chunk_budget; safely truncate Chunk 1
                header_tokens = estimate_tokens(chunk_header)
                avail_tokens = max(0, chunk_budget - header_tokens - 18)
                avail_chars = avail_tokens * 4
                if avail_chars > 20:
                    truncated_text = (
                        text[:avail_chars].rsplit(" ", 1)[0]
                        + "\n[... Source 1 excerpt truncated to fit context budget ...]"
                    )
                else:
                    truncated_text = "[... Source 1 excerpt truncated to fit context budget ...]"

                truncated_entry = f"{chunk_header}\n{truncated_text}"
                sources_text.append(truncated_entry)
                source_map[1] = {
                    "source_id": 1,
                    "page": page,
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "score": score,
                    "text": truncated_text,
                    "truncated": True,
                }
                evidence_truncated = True
                excluded_count = len(chunks) - 1
                break

    if excluded_count > 0:
        suffix = "s" if excluded_count > 1 else ""
        sources_text.append(
            f"[Note: {excluded_count} additional lower-ranked document excerpt{suffix} "
            f"omitted to strictly adhere to model context limits.]"
        )

    evidence_body = "\n\n".join(sources_text)
    evidence_block = f"{header}{evidence_body}{footer}"
    total_tokens = estimate_tokens(evidence_block)

    # If adding the exclusion note pushed total tokens over max_evidence_tokens,
    # prune lower-ranked chunks until total_tokens <= max_evidence_tokens
    if max_evidence_tokens is not None and total_tokens > max_evidence_tokens:
        while len(source_map) > 1 and total_tokens > max_evidence_tokens:
            last_id = max(source_map.keys())
            del source_map[last_id]
            sources_text.pop(-2)  # pop chunk immediately preceding the note
            excluded_count = len(chunks) - len(source_map)
            suffix = "s" if excluded_count > 1 else ""
            sources_text[-1] = (
                f"[Note: {excluded_count} additional lower-ranked document excerpt{suffix} "
                f"omitted to strictly adhere to model context limits.]"
            )
            evidence_body = "\n\n".join(sources_text)
            evidence_block = f"{header}{evidence_body}{footer}"
            total_tokens = estimate_tokens(evidence_block)

        if total_tokens > max_evidence_tokens and len(source_map) == 1:
            # Even chunk 1 + note exceeds max_evidence_tokens; safely truncate chunk 1 text
            meta = chunks[0].get("metadata", {})
            page = meta.get("page", chunks[0].get("page", "unknown"))
            filename = meta.get("filename", chunks[0].get("filename", ""))
            chunk_id = meta.get("chunk_id", chunks[0].get("chunk_id", "chunk_1"))
            score = chunks[0].get("score")
            text = chunks[0].get("text", "").strip()
            file_part = f" (File: {filename})" if filename else ""
            chunk_header = f"[Source 1, Page {page}]{file_part}"

            note_tokens = estimate_tokens(sources_text[-1])
            avail_for_c1 = max(0, max_evidence_tokens - framing_tokens - note_tokens - 4)
            header_tokens = estimate_tokens(chunk_header)
            avail_text_tokens = max(0, avail_for_c1 - header_tokens - 18)
            avail_chars = avail_text_tokens * 4
            if avail_chars > 20:
                trunc_text = (
                    text[:avail_chars].rsplit(" ", 1)[0]
                    + "\n[... Source 1 excerpt truncated to fit context budget ...]"
                )
            else:
                trunc_text = "[... Source 1 excerpt truncated to fit context budget ...]"

            truncated_entry = f"{chunk_header}\n{trunc_text}"
            sources_text[0] = truncated_entry
            source_map[1] = {
                "source_id": 1,
                "page": page,
                "filename": filename,
                "chunk_id": chunk_id,
                "score": score,
                "text": trunc_text,
                "truncated": True,
            }
            evidence_truncated = True
            evidence_body = "\n\n".join(sources_text)
            evidence_block = f"{header}{evidence_body}{footer}"
            total_tokens = estimate_tokens(evidence_block)

    return EvidenceBlockResult(
        evidence_block,
        total_tokens,
        source_map,
        excluded_count=excluded_count,
        evidence_truncated=evidence_truncated,
    )


def build_grounded_chat_prompt(
    query: str,
    chunks: list[dict[str, Any]],
    thread: ConversationThread | None = None,
    *,
    memory_manager: SlidingWindowMemoryManager | None = None,
    config_obj: ConversationMemoryConfig | None = None,
    system_prompt: str | None = None,
) -> GroundedChatPrompt:
    """Construct a grounded chat prompt with strict budget enforcement and source tracking.

    Prioritization & Prompt Integrity Guarantees:
    1. Complete User Question: Preserved 100% byte-for-byte; NEVER silently truncated.
    2. Output Reserve: Guaranteed preserved for Qwen generation.
    3. Document Evidence: Authoritative context prioritized first.
    4. Conversation History: Reduced pair-aware (recent turns first, then summary) before
       evidence is touched.
    5. Excluded Evidence: Lower-ranked chunks excluded explicitly and omitted from source_map.
    6. PromptBudgetExceededError: Raised predictably if question + system instructions + output
       reserve cannot fit within the context window even after all reductions.

    Args:
        query: User's current input question (or rewritten standalone query).
        chunks: Retrieved context chunks for the active document.
        thread: Optional ConversationThread containing message history and summary memory.
        memory_manager: Optional SlidingWindowMemoryManager.
        config_obj: Optional ConversationMemoryConfig.
        system_prompt: Optional override for the grounding system prompt.

    Returns:
        GroundedChatPrompt: Structured prompt containing system prompt, user prompt, and source_map.

    Raises:
        PromptBudgetExceededError: If the complete user question, system prompt, and output
        reserve cannot fit within the model context limit even after removing all history and evidence.
    """
    clean_query = (query or "").strip()
    active_sys_prompt = (
        system_prompt if system_prompt is not None else CHAT_GROUNDED_SYSTEM_PROMPT
    )
    sys_tokens = estimate_tokens(active_sys_prompt)

    cfg = config_obj or ConversationMemoryConfig()
    mgr = memory_manager or SlidingWindowMemoryManager(config=cfg)

    # Base instructions block containing the complete, un-truncated user question
    instructions_block = (
        f"Current Question:\n"
        f"User Question: {clean_query}\n\n"
        f"Answer Instructions:\n"
        f"1. Answer the question using ONLY the Document Evidence above.\n"
        f"2. Cite every factual claim using [Source X, Page Y] notation matching the sources above.\n"
        f"3. If the Document Evidence does not contain sufficient information to answer the question, "
        f"explicitly state: 'I cannot answer this question based on the provided document context.'\n"
        f"4. Do NOT make assumptions, invent facts, or cite source numbers that do not appear in the Document Evidence."
    )
    instructions_tokens = estimate_tokens(instructions_block)

    # 1. Minimum irreducible budget requirement:
    # System prompt + complete user question & framing + output reserve
    output_reserve = max(0, min(cfg.output_reserve_tokens, cfg.context_window_tokens))
    max_prompt_capacity = max(0, cfg.context_window_tokens - output_reserve)
    irreducible_overhead = sys_tokens + instructions_tokens

    if irreducible_overhead > max_prompt_capacity:
        raise PromptBudgetExceededError(
            f"User question and system instructions require {irreducible_overhead} estimated tokens, "
            f"which exceeds the maximum allowed prompt capacity of {max_prompt_capacity} tokens "
            f"(context window: {cfg.context_window_tokens}, output reserve: {output_reserve}). "
            f"The complete question cannot fit without truncation."
        )

    # 2. Safe budget calculation with safety margin applied to available dynamic capacity
    safety_margin_ratio = max(0.0, min(0.5, cfg.safety_margin_ratio))
    raw_dynamic_capacity = max(0, max_prompt_capacity - irreducible_overhead)
    safe_dynamic_capacity = int(raw_dynamic_capacity * (1.0 - safety_margin_ratio))
    safe_prompt_budget = irreducible_overhead + safe_dynamic_capacity

    # Available capacity for context (evidence + history)
    max_evidence_budget = safe_dynamic_capacity

    # 3. Format document evidence prioritized and budgeted first
    evidence_res = build_document_evidence_block(
        chunks,
        max_evidence_tokens=max_evidence_budget,
    )
    evidence_text, evidence_tokens, source_map = evidence_res
    excluded_chunks = evidence_res.excluded_count
    evidence_truncated = evidence_res.evidence_truncated

    # 4. Calculate available budget for conversation history
    # Document evidence took precedence:
    # If evidence alone was large enough that lower-ranked chunks had to be excluded
    # or Chunk 1 had to be truncated, zero budget remains for conversation history.
    if excluded_chunks > 0 or evidence_truncated:
        history_budget = 0
    else:
        remaining_for_history = max(
            0, safe_prompt_budget - (irreducible_overhead + evidence_tokens)
        )
        history_budget = min(cfg.max_history_tokens, remaining_for_history)

    summary_memory = ""
    recent_messages: list[ChatMessage] = []
    history_tokens = 0

    if thread is not None and thread.messages and history_budget > 0:
        summary_memory, recent_messages = mgr.build_prompt_history(
            thread,
            available_budget=history_budget,
        )
        history_tokens = (
            estimate_tokens(summary_memory) if summary_memory else 0
        ) + sum(m.token_count for m in recent_messages)

    # 5. Assemble prompt sections cleanly
    prompt_sections: list[str] = [evidence_text]

    if summary_memory and summary_memory.strip():
        prompt_sections.append(
            f"=== PRIOR CONVERSATION SUMMARY (Context Only) ===\n"
            f"{summary_memory.strip()}\n"
            f"=== END PRIOR CONVERSATION SUMMARY ==="
        )

    if recent_messages:
        history_lines: list[str] = []
        for msg in recent_messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{role_label}: {msg.content.strip()}")

        prompt_sections.append(
            f"=== RECENT CONVERSATION HISTORY (Context Only) ===\n"
            f"{chr(10).join(history_lines)}\n"
            f"=== END RECENT CONVERSATION HISTORY ==="
        )

    prompt_sections.append(instructions_block)
    user_prompt = "\n\n".join(prompt_sections)

    # 6. Strict Prompt-Context Ceiling Enforcement:
    # Ensure total_prompt_tokens <= max_prompt_capacity (C - O)
    total_prompt_tokens = sys_tokens + estimate_tokens(user_prompt)

    if total_prompt_tokens > max_prompt_capacity:
        # A. Reduce recent history turns first
        while recent_messages and total_prompt_tokens > max_prompt_capacity:
            recent_messages.pop(0)
            h_lines = [
                f"{'User' if m.role == 'user' else 'Assistant'}: {m.content.strip()}"
                for m in recent_messages
            ]
            sections = [evidence_text]
            if summary_memory and summary_memory.strip():
                sections.append(
                    f"=== PRIOR CONVERSATION SUMMARY (Context Only) ===\n"
                    f"{summary_memory.strip()}\n"
                    f"=== END PRIOR CONVERSATION SUMMARY ==="
                )
            if h_lines:
                sections.append(
                    f"=== RECENT CONVERSATION HISTORY (Context Only) ===\n"
                    f"{chr(10).join(h_lines)}\n"
                    f"=== END RECENT CONVERSATION HISTORY ==="
                )
            sections.append(instructions_block)
            user_prompt = "\n\n".join(sections)
            total_prompt_tokens = sys_tokens + estimate_tokens(user_prompt)

        # B. If summary is still present and total prompt still exceeds, drop summary
        if total_prompt_tokens > max_prompt_capacity and summary_memory:
            summary_memory = ""
            sections = [evidence_text]
            if recent_messages:
                h_lines = [
                    f"{'User' if m.role == 'user' else 'Assistant'}: {m.content.strip()}"
                    for m in recent_messages
                ]
                sections.append(
                    f"=== RECENT CONVERSATION HISTORY (Context Only) ===\n"
                    f"{chr(10).join(h_lines)}\n"
                    f"=== END RECENT CONVERSATION HISTORY ==="
                )
            sections.append(instructions_block)
            user_prompt = "\n\n".join(sections)
            total_prompt_tokens = sys_tokens + estimate_tokens(user_prompt)

        # C. If still exceeding, trim evidence text (NEVER touch instructions_block or clean_query)
        if total_prompt_tokens > max_prompt_capacity:
            max_allowed_evidence_tokens = max_prompt_capacity - (sys_tokens + instructions_tokens + 4)
            if max_allowed_evidence_tokens < 20:
                evidence_text = ""
                source_map = {}
                excluded_chunks = len(chunks)
                evidence_truncated = True
            else:
                evidence_res = build_document_evidence_block(
                    chunks,
                    max_evidence_tokens=max_allowed_evidence_tokens,
                )
                evidence_text = evidence_res.evidence_text
                source_map = evidence_res.source_map
                excluded_chunks = evidence_res.excluded_count
                evidence_truncated = evidence_res.evidence_truncated

            user_prompt = f"{evidence_text}\n\n{instructions_block}".strip() if evidence_text else instructions_block
            total_prompt_tokens = sys_tokens + estimate_tokens(user_prompt)

            # If still exceeding (due to boundary tokens), clear evidence completely
            if total_prompt_tokens > max_prompt_capacity and evidence_text:
                evidence_text = ""
                source_map = {}
                excluded_chunks = len(chunks)
                evidence_truncated = True
                user_prompt = instructions_block
                total_prompt_tokens = sys_tokens + instructions_tokens

            # If even with zero evidence and zero history it still exceeds, raise controlled error
            if total_prompt_tokens > max_prompt_capacity:
                raise PromptBudgetExceededError(
                    f"User question and system instructions require {total_prompt_tokens} estimated tokens, "
                    f"which exceeds the maximum allowed prompt capacity of {max_prompt_capacity} tokens "
                    f"(context window: {cfg.context_window_tokens}, output reserve: {output_reserve}). "
                    f"The complete question cannot fit without truncation."
                )

    # Final assertion: User question must be 100% byte-for-byte intact in user_prompt
    if clean_query and clean_query not in user_prompt:
        raise PromptBudgetExceededError("Invariant violation: user question was corrupted during prompt assembly.")

    return GroundedChatPrompt(
        system_prompt=active_sys_prompt,
        user_prompt=user_prompt,
        source_map=source_map,
        evidence_tokens=evidence_tokens,
        history_tokens=history_tokens,
        history_budget=history_budget,
        total_prompt_tokens=total_prompt_tokens,
        output_reserve_tokens=output_reserve,
        context_window_tokens=cfg.context_window_tokens,
        excluded_chunks=excluded_chunks,
        evidence_truncated=evidence_truncated,
    )


def validate_chat_citations(
    answer: str,
    source_map: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract and validate all source citations in the generated answer against source_map.

    Args:
        answer: Generated LLM response text.
        source_map: Mapping from source index to retrieved chunk metadata.

    Returns:
        list[dict[str, Any]]: List of citation dictionaries with validation status:
            - source_id: int source index
            - page: int | str actual page from source_map (or None if hallucinated)
            - cited_page: str | None page number cited by the model
            - is_valid: bool True if source_id exists in source_map
            - page_valid: bool True if cited page matches true page
            - filename: str filename from chunk metadata
            - chunk_id: str chunk identifier
            - raw_citation: str exact matched string in answer
    """
    if not answer or not isinstance(answer, str):
        return []

    citations: list[dict[str, Any]] = []

    for match in _CITATION_PATTERN.finditer(answer):
        raw_source_idx = match.group(1)
        cited_page = match.group(2)
        raw_text = match.group(0)

        try:
            source_id = int(raw_source_idx)
        except (ValueError, TypeError):
            continue

        if source_id in source_map:
            source_info = source_map[source_id]
            actual_page = source_info.get("page", "unknown")
            page_valid = True
            if cited_page is not None:
                page_valid = str(cited_page).strip().lower() == str(actual_page).strip().lower()

            citations.append(
                {
                    "source_id": source_id,
                    "page": actual_page,
                    "cited_page": cited_page,
                    "is_valid": True,
                    "page_valid": page_valid,
                    "filename": source_info.get("filename", ""),
                    "chunk_id": source_info.get("chunk_id", ""),
                    "raw_citation": raw_text,
                }
            )
        else:
            # Hallucinated source identifier not present in retrieved context
            citations.append(
                {
                    "source_id": source_id,
                    "page": None,
                    "cited_page": cited_page,
                    "is_valid": False,
                    "page_valid": False,
                    "filename": "",
                    "chunk_id": "",
                    "raw_citation": raw_text,
                }
            )

    return citations


def sanitize_and_repair_citations(
    answer: str,
    source_map: dict[int, dict[str, Any]],
) -> str:
    """Sanitize and repair citations in the model's answer.

    - Hallucinated source indices not present in source_map are removed.
    - Valid source indices with mismatched page numbers are corrected to the true page number.
    - Citations without page numbers are upgraded to [Source X, Page Y] when page is known.

    Args:
        answer: Generated LLM response text.
        source_map: Mapping from source index to retrieved chunk metadata.

    Returns:
        str: Sanitized response text with verified, accurate citations.
    """
    if not answer or not isinstance(answer, str) or not answer.strip():
        return ""

    def _replace_citation(match: re.Match) -> str:
        raw_source_idx = match.group(1)
        cited_page = match.group(2)

        try:
            source_id = int(raw_source_idx)
        except (ValueError, TypeError):
            return ""

        if source_id not in source_map:
            # Drop hallucinated source citation
            logger.debug(
                "Stripped hallucinated citation [Source %s] from answer", source_id
            )
            return ""

        actual_page = source_map[source_id].get("page", "unknown")
        if actual_page and str(actual_page).lower() != "unknown":
            return f"[Source {source_id}, Page {actual_page}]"
        return f"[Source {source_id}]"

    # Replace all matches using the repair function
    repaired = _CITATION_PATTERN.sub(_replace_citation, answer)
    # Clean up any leftover double spaces or orphan punctuation caused by removed citations
    repaired = re.sub(r" {2,}", " ", repaired)
    repaired = re.sub(r" \.", ".", repaired)
    repaired = re.sub(r" ,", ",", repaired)
    return repaired.strip()
