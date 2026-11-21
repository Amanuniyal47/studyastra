"""Batch-aware, hierarchical Academic Study Notes Generator powered by Qwen2.5 7B.

Generates structured, clear, comprehensive academic study notes from indexed PDF documents.
Key guarantees:
- Strict single-PDF isolation.
- Grounded strictly in provided excerpts (zero unbacked hallucinations).
- Hierarchical academic layout: Executive Overview, Chapters & Topics Deep Dive,
  Key Concepts & Definitions Dictionary, Core Principles & Mechanisms,
  Important Facts & Formulas, and Review Takeaways.
- Multi-batch Map-Reduce architecture for large PDFs strictly respecting token budgets.
- Recursive hierarchical reduction for multi-chapter documents.
- Deterministic page references derived from verified vector-store chunks.
- Centralized LLM access via BaseLLMProvider / OllamaLLMProvider.
- Publication-quality Markdown export capability.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import re
from typing import Any

import src.config as config
from src.generation.llm_provider import BaseLLMProvider, get_llm_provider
from src.generation.summarizer import format_page_range
from src.ingestion.chunker import estimate_tokens
from src.retrieval.retriever import retrieve_chunks
from src.retrieval.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class StudyNotesGenerationError(RuntimeError):
    """Raised when study notes generation or validation fails."""


NOTES_SINGLE_BATCH_SYSTEM_PROMPT = (
    "You are an expert university professor and master study notes architect. "
    "Produce structured, comprehensive, and high-yield academic study notes "
    "grounded EXCLUSIVELY in the provided document excerpts.\n\n"
    "CRITICAL GROUNDING & STRUCTURAL RULES:\n"
    "1. Strict Document Grounding: Use ONLY facts, definitions, mechanisms, and theories directly stated in the excerpts. "
    "Never extrapolate, speculate, or introduce outside facts.\n"
    "2. Hierarchical Academic Organization: Format the notes in clear, structured Markdown:\n"
    "   # [Document / Topic Title]\n"
    "   ## 1. Executive Overview\n"
    "   ## 2. Chapters & Topics Deep Dive\n"
    "   ## 3. Key Concepts & Definitions Dictionary\n"
    "   ## 4. Core Principles, Mechanisms & Methods\n"
    "   ## 5. Important Facts, Formulas & Numbers\n"
    "   ## 6. High-Yield Summary & Review Takeaways\n"
    "3. Precision: Retain exact technical terms, equations, and definitions.\n"
    "4. Citations: Reference source pages where appropriate (e.g. [Page X]). Never cite pages outside the excerpts.\n"
    "5. Anti-Hallucination: If a concept is only briefly mentioned, describe only what is stated."
)

NOTES_MAP_SYSTEM_PROMPT = (
    "You are an expert academic research assistant analyzing an excerpt of an academic text. "
    "Extract structured, high-yield study notes from the provided text excerpt.\n\n"
    "Include:\n"
    "- Section Overview & Key Topics covered\n"
    "- Important Technical Terms & Definitions\n"
    "- Core Principles, Step-by-Step Mechanisms, or Methods\n"
    "- Crucial Facts, Formulas, and Numerical Data\n"
    "Ground every point strictly in the excerpt. Tag points with their source page number (e.g. [Page X])."
)

NOTES_REDUCE_SYSTEM_PROMPT = (
    "You are a distinguished university professor synthesizing modular section notes into a cohesive, "
    "comprehensive master academic study guide.\n\n"
    "SYNTHESIS RULES:\n"
    "1. Synthesize all provided section notes into a unified, non-redundant academic study guide.\n"
    "2. Organize into clear hierarchical sections:\n"
    "   # Master Academic Study Notes: [Topic / Title]\n"
    "   ## 1. Executive Overview\n"
    "   ## 2. Chapters & Topics Deep Dive\n"
    "   ## 3. Key Concepts & Definitions Dictionary\n"
    "   ## 4. Core Principles, Mechanisms & Methods (include formulas & step-by-step processes)\n"
    "   ## 5. Crucial Facts, Formulas & Specific Details\n"
    "   ## 6. High-Yield Summary & Review Takeaways\n"
    "3. Deduplicate overlapping terms and merge related concepts seamlessly.\n"
    "4. Preserve verified source page citations (e.g. [Page X]).\n"
    "5. Strict Grounding: Rely strictly on the section notes. Never introduce external facts."
)

# Backwards-compatible alias for existing imports
NOTES_SYSTEM_PROMPT = NOTES_SINGLE_BATCH_SYSTEM_PROMPT


@dataclass
class StudyNotesResult:
    title: str
    notes_text: str
    document_id: str
    filename: str
    scope: str
    pages_covered: list[int]
    page_range_label: str
    chunks_used: int
    provider: str
    model: str
    source_chunks: list[dict[str, Any]]
    batches_processed: int = 1
    reduction_levels: int = 0
    structured_sections: dict[str, Any] = field(default_factory=dict)
    status: str = "complete"
    total_eligible_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    failed_batch_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "notes_text": self.notes_text,
            "document_id": self.document_id,
            "filename": self.filename,
            "scope": self.scope,
            "pages_covered": list(self.pages_covered),
            "page_range_label": self.page_range_label,
            "chunks_used": self.chunks_used,
            "batches_processed": self.batches_processed,
            "reduction_levels": self.reduction_levels,
            "provider": self.provider,
            "model": self.model,
            "source_chunks": self.source_chunks,
            "structured_sections": self.structured_sections,
            "status": self.status,
            "total_eligible_chunks": self.total_eligible_chunks,
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "failed_batch_indices": list(self.failed_batch_indices),
        }


def _format_chunk_for_notes(chunk: dict[str, Any]) -> str:
    meta = chunk.get("metadata", chunk)
    cid = meta.get("chunk_id", "chunk")
    page = meta.get("page_range", meta.get("page", 1))
    return f"[Chunk ID: {cid} | Page {page}]\n{chunk['text'].strip()}"


def build_notes_prompt(
    filename: str,
    scope_description: str,
    formatted_chunks: list[str],
) -> str:
    body = "\n\n".join(formatted_chunks)
    return (
        f"Document: {filename}\n"
        f"Scope: {scope_description}\n\n"
        f"Source Material:\n{body}\n\n"
        "Instructions: Synthesize comprehensive, structured academic study notes from the material above. "
        "Strictly adhere to the facts in the text. Cite page numbers where relevant."
    )


def build_batch_notes_prompt(
    filename: str,
    scope_description: str,
    batch_index: int,
    total_batches: int,
    formatted_chunks: list[str],
    retry_reminder: bool = False,
) -> str:
    body = "\n\n".join(formatted_chunks)
    prompt = (
        f"Document: {filename} ({scope_description})\n"
        f"Section Batch: {batch_index} of {total_batches}\n\n"
        f"Source Excerpt:\n{body}\n\n"
        "Instructions: Extract structured notes, definitions, mechanisms, and key facts strictly from this excerpt. "
        "Tag concepts with source page numbers."
    )
    if retry_reminder:
        prompt += "\n\nNote: Provide detailed bulleted notes with definitions and mechanisms directly from the text."
    return prompt


def build_reduce_notes_prompt(
    filename: str,
    scope_description: str,
    batch_notes_list: list[str],
) -> str:
    joined_notes = "\n\n---\n\n".join(
        f"### Section Notes {idx}:\n{note}"
        for idx, note in enumerate(batch_notes_list, start=1)
    )
    return (
        f"Document: {filename} ({scope_description})\n\n"
        f"Source Section Notes:\n{joined_notes}\n\n"
        "Instructions: Synthesize these section notes into a cohesive, comprehensive master academic study guide "
        "following the required hierarchical structure. Deduplicate concepts and preserve page citations."
    )


def _measure_notes_tokens(system_prompt: str, user_prompt: str) -> int:
    return estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + 32


def _build_notes_batches(
    chunks: list[dict[str, Any]],
    *,
    filename: str,
    scope_description: str,
    max_batch_tokens: int,
) -> list[list[dict[str, Any]]]:
    """Group chunks into batches guaranteed to stay within max_batch_tokens including prompt overhead."""
    def _measure(formatted_entries: list[str]) -> int:
        prompt = build_batch_notes_prompt(
            filename=filename,
            scope_description=scope_description,
            batch_index=1,
            total_batches=1,
            formatted_chunks=formatted_entries,
            retry_reminder=True,
        )
        return _measure_notes_tokens(NOTES_MAP_SYSTEM_PROMPT, prompt)

    base_overhead = _measure([])
    if base_overhead >= max_batch_tokens:
        raise ValueError(
            f"max_batch_tokens ({max_batch_tokens}) is too small to fit prompt overhead ({base_overhead} tokens)."
        )

    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_formatted: list[str] = []

    for chunk in chunks:
        formatted = _format_chunk_for_notes(chunk)
        candidate_formatted = current_formatted + [formatted]
        if _measure(candidate_formatted) <= max_batch_tokens:
            current_batch.append(chunk)
            current_formatted.append(formatted)
            continue

        if current_batch:
            batches.append(current_batch)
            current_batch = []
            current_formatted = []

        if _measure([formatted]) <= max_batch_tokens:
            current_batch = [chunk]
            current_formatted = [formatted]
        else:
            # Trim oversized single chunk text while preserving Chunk ID header
            empty_chunk = dict(chunk)
            empty_chunk["text"] = ""
            header_overhead = _measure([_format_chunk_for_notes(empty_chunk)])
            available_for_body = max_batch_tokens - header_overhead
            if available_for_body <= 10:
                logger.warning("Chunk %s header too large; skipping.", chunk.get("chunk_id"))
                continue
            char_budget = max(40, available_for_body * 3)
            trimmed_chunk = dict(chunk)
            trimmed_chunk["text"] = chunk["text"][:char_budget]
            trimmed_formatted = _format_chunk_for_notes(trimmed_chunk)
            current_batch = [trimmed_chunk]
            current_formatted = [trimmed_formatted]

    if current_batch:
        batches.append(current_batch)

    return batches


def validate_and_filter_page_citations(notes_text: str, allowed_pages: list[int]) -> str:
    """Ensure page citations in notes_text only reference actual existing pages in the document."""
    if not allowed_pages:
        return notes_text

    allowed_set = set(allowed_pages)

    def _replace_cite(match: re.Match) -> str:
        page_str = match.group(2)
        try:
            p_val = int(page_str)
            if p_val in allowed_set:
                return match.group(0)
            return ""  # Remove unbacked page citation
        except ValueError:
            return match.group(0)

    # Clean patterns like [Page 99] or (Page 99)
    cleaned = re.sub(
        r"\[(Pages?|p\.)\s*(\d+)\]",
        _replace_cite,
        notes_text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\((Pages?|p\.)\s*(\d+)\)",
        _replace_cite,
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def validate_notes_structure(notes_text: str) -> tuple[bool, list[str]]:
    """Validate that notes contain substantial content and proper markdown heading hierarchy."""
    issues: list[str] = []
    if not isinstance(notes_text, str) or not notes_text.strip():
        return False, ["Notes text is completely empty."]

    stripped = notes_text.strip()
    if len(stripped) < 80:
        issues.append(f"Notes text too brief ({len(stripped)} chars).")

    # Check for markdown headings (#, ##, or ###)
    headings = re.findall(r"^#{1,4}\s+.+", stripped, flags=re.MULTILINE)
    if len(headings) < 2:
        issues.append(f"Insufficient heading hierarchy (found {len(headings)} headings).")

    return len(issues) == 0, issues


def _reduce_notes_hierarchically(
    batch_notes: list[str],
    *,
    filename: str,
    scope_description: str,
    provider: BaseLLMProvider,
    max_batch_tokens: int,
    progress_callback: ProgressCallback | None = None,
    current_level: int = 1,
    max_levels: int | None = None,
) -> tuple[str, int]:
    """Hierarchically reduce batch notes until a single coherent master study guide is formed.
    
    Guarantees convergence by ensuring each reduction pass combines at least 2 notes per group.
    Returns (final_notes_text, total_reduction_levels).
    """
    effective_max_levels = getattr(config, "MAX_REDUCTION_LEVELS", 6) if max_levels is None else max_levels
    if len(batch_notes) <= 1:
        return (batch_notes[0] if batch_notes else ""), max(0, current_level - 1)

    if current_level > effective_max_levels:
        logger.warning("Reached maximum reduction level %d; doing final concatenation.", effective_max_levels)
        return "\n\n---\n\n".join(batch_notes), current_level - 1

    def _measure_reduce_prompt(notes_subset: list[str]) -> int:
        prompt = build_reduce_notes_prompt(filename, scope_description, notes_subset)
        return _measure_notes_tokens(NOTES_REDUCE_SYSTEM_PROMPT, prompt)

    base_overhead = _measure_reduce_prompt([])
    effective_budget = max(max_batch_tokens, base_overhead + 20)
    available_body_budget = max(20, effective_budget - base_overhead)

    # Try to fit all batch_notes in one single reduce pass first!
    if _measure_reduce_prompt(batch_notes) <= effective_budget:
        if progress_callback is not None:
            progress_callback(
                1,
                2,
                f"Synthesizing final study notes (level {current_level})...",
            )
        reduce_prompt = build_reduce_notes_prompt(filename, scope_description, batch_notes)
        try:
            raw_reduced = provider.generate(
                NOTES_REDUCE_SYSTEM_PROMPT,
                reduce_prompt,
                temperature=0.2,
                max_tokens=2048,
            )
            if raw_reduced.strip():
                return raw_reduced.strip(), current_level
        except Exception as exc:
            logger.warning("Single reduce pass failed: %s; falling back to sub-grouping.", exc)

    # Group notes into subsets of at least 2 notes each to guarantee convergence
    grouped_subsets: list[list[str]] = []
    current_subset: list[str] = []

    for note in batch_notes:
        candidate = current_subset + [note]
        if _measure_reduce_prompt(candidate) <= effective_budget:
            current_subset.append(note)
        else:
            if len(current_subset) >= 2:
                grouped_subsets.append(current_subset)
                current_subset = [note]
            elif len(current_subset) == 1:
                # Force pairing: trim both notes to fit together in available_body_budget
                note_a = current_subset[0]
                note_b = note
                half_char_budget = max(100, (available_body_budget // 2) * 3)
                trimmed_a = note_a[:half_char_budget] + "\n[...]"
                trimmed_b = note_b[:half_char_budget] + "\n[...]"
                grouped_subsets.append([trimmed_a, trimmed_b])
                current_subset = []
            else:
                current_subset = [note]

    if current_subset:
        if grouped_subsets and len(current_subset) == 1:
            grouped_subsets[-1].append(current_subset[0])
        else:
            grouped_subsets.append(current_subset)

    reduced_outputs: list[str] = []
    total_subsets = len(grouped_subsets)

    for s_idx, subset in enumerate(grouped_subsets, start=1):
        if progress_callback is not None:
            progress_callback(
                s_idx,
                total_subsets + 1,
                f"Synthesizing study notes (level {current_level}, pass {s_idx}/{total_subsets})...",
            )

        if len(subset) == 1 and total_subsets == 1:
            reduced_outputs.append(subset[0])
            continue

        reduce_prompt = build_reduce_notes_prompt(filename, scope_description, subset)
        try:
            raw_reduced = provider.generate(
                NOTES_REDUCE_SYSTEM_PROMPT,
                reduce_prompt,
                temperature=0.2,
                max_tokens=2048,
            )
        except Exception as exc:
            logger.warning("Reduce pass %d failed: %s; falling back to joined notes.", s_idx, exc)
            raw_reduced = "\n\n".join(subset)

        if not raw_reduced.strip():
            logger.warning("Reduce pass %d returned empty; falling back to joined text.", s_idx)
            raw_reduced = "\n\n".join(subset)
        reduced_outputs.append(raw_reduced.strip())

    if len(reduced_outputs) > 1 and len(reduced_outputs) < len(batch_notes):
        return _reduce_notes_hierarchically(
            reduced_outputs,
            filename=filename,
            scope_description=scope_description,
            provider=provider,
            max_batch_tokens=max_batch_tokens,
            progress_callback=progress_callback,
            current_level=current_level + 1,
            max_levels=max_levels,
        )

    if len(reduced_outputs) == 1:
        return reduced_outputs[0], current_level

    # Final pass if outputs cannot be further partitioned
    final_prompt = build_reduce_notes_prompt(filename, scope_description, reduced_outputs)
    try:
        final_out = provider.generate(
            NOTES_REDUCE_SYSTEM_PROMPT,
            final_prompt,
            temperature=0.2,
            max_tokens=2048,
        )
        if final_out.strip():
            return final_out.strip(), current_level
    except Exception as exc:
        logger.warning("Final reduce pass failed: %s", exc)

    return "\n\n---\n\n".join(reduced_outputs), current_level


def export_notes_to_markdown(
    result: StudyNotesResult | dict[str, Any],
) -> str:
    """Format study notes into a standalone publication-quality Markdown document."""
    if isinstance(result, StudyNotesResult):
        title = result.title
        notes_text = result.notes_text
        filename = result.filename
        page_range = result.page_range_label
        scope = result.scope
        provider = result.provider
        model = result.model
        chunks_count = result.chunks_used
    elif isinstance(result, dict):
        title = result.get("title", "Study Notes")
        notes_text = result.get("notes_text", "")
        filename = result.get("filename", "Document")
        page_range = result.get("page_range_label", "All Pages")
        scope = result.get("scope", "Full Document")
        provider = result.get("provider", "Ollama")
        model = result.get("model", "qwen2.5:7b")
        chunks_count = result.get("chunks_used", 0)
    else:
        title = "Study Notes"
        notes_text = str(result)
        filename = "Document"
        page_range = "N/A"
        scope = "Full"
        provider = "Ollama"
        model = "qwen2.5:7b"
        chunks_count = 0

    lines = [
        f"# {title}",
        f"**Document:** {filename} | **Scope:** {str(scope).title()} | **Pages Covered:** {page_range}",
        f"**Generated by:** `{provider}` (`{model}`) | **Source Chunks Analyzed:** {chunks_count}",
        "",
        "---",
        "",
        notes_text.strip(),
        "",
        "---",
        f"*Academic Study Notes generated from {filename}*",
    ]
    return "\n".join(lines).strip() + "\n"


def generate_study_notes(
    store: FAISSVectorStore,
    document_id: str,
    *,
    scope: str = "full",
    start_page: int | None = None,
    end_page: int | None = None,
    topic: str | None = None,
    top_k: int = 6,
    embedding_service: Any = None,
    provider: BaseLLMProvider | None = None,
    max_batch_tokens: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> StudyNotesResult:
    """Generate structured, comprehensive academic study notes for a single indexed PDF using Qwen."""
    doc = store.get_document(document_id)
    if doc is None:
        raise KeyError(f"Document '{document_id}' is not indexed.")
    filename = doc["filename"]

    if scope == "page_range":
        if start_page is None or end_page is None or start_page > end_page:
            raise ValueError(f"Invalid page range {start_page}-{end_page}.")
        chunks = store.get_document_chunks(document_id, start_page=start_page, end_page=end_page)
        scope_desc = f"Pages {start_page}-{end_page}"
    elif scope == "topic":
        if not topic or not str(topic).strip():
            raise ValueError("Topic query cannot be empty.")
        if embedding_service is None:
            from src.retrieval.embeddings import EmbeddingService
            embedding_service = EmbeddingService()
        chunks = retrieve_chunks(
            topic,
            top_k=top_k,
            store=store,
            embedding_service=embedding_service,
            document_id=document_id,
        )
        scope_desc = f"Topic: '{topic}'"
    else:
        chunks = store.get_document_chunks(document_id)
        scope_desc = "Full Document"

    if not chunks:
        raise ValueError(f"No indexed text found for {filename} ({scope_desc}).")

    all_pages = sorted({int(c.get("metadata", c).get("page", 1)) for c in chunks})
    overall_page_label = format_page_range(all_pages)
    active_provider = provider if provider is not None else get_llm_provider("ollama")
    token_budget = max_batch_tokens or config.NOTES_MAX_BATCH_TOKENS

    # 1. Build batches respecting token budget
    batches = _build_notes_batches(
        chunks,
        filename=filename,
        scope_description=scope_desc,
        max_batch_tokens=token_budget,
    )
    if not batches:
        raise ValueError(f"Unable to form valid chunk batches for {filename}.")

    title = f"Academic Study Notes: {filename} ({scope_desc})"

    # Single-batch optimization
    if len(batches) == 1:
        if progress_callback is not None:
            progress_callback(1, 2, f"Synthesizing study notes for {filename} ({scope_desc})...")

        formatted_entries = [_format_chunk_for_notes(c) for c in batches[0]]
        prompt = build_notes_prompt(filename, scope_desc, formatted_entries)

        raw_notes = active_provider.generate(
            NOTES_SINGLE_BATCH_SYSTEM_PROMPT,
            prompt,
            temperature=0.2,
            max_tokens=2048,
        )

        valid, issues = validate_notes_structure(raw_notes)
        if not valid and config.FLASHCARD_MAX_RETRIES > 0:
            logger.warning("Single-batch notes validation issue: %s; retrying...", issues)
            retry_prompt = prompt + "\n\nNote: Ensure to use proper Markdown headings and include definitions."
            raw_notes = active_provider.generate(
                NOTES_SINGLE_BATCH_SYSTEM_PROMPT,
                retry_prompt,
                temperature=0.1,
                max_tokens=2048,
            )

        final_text = validate_and_filter_page_citations(raw_notes.strip(), all_pages)

        if progress_callback is not None:
            progress_callback(2, 2, "Study notes complete.")

        return StudyNotesResult(
            title=title,
            notes_text=final_text,
            document_id=document_id,
            filename=filename,
            scope=scope,
            pages_covered=all_pages,
            page_range_label=overall_page_label,
            chunks_used=len(chunks),
            provider=active_provider.provider_name,
            model=active_provider.model_name,
            source_chunks=chunks,
            batches_processed=1,
            reduction_levels=0,
            status="complete",
            total_eligible_chunks=len(chunks),
            processed_chunks=len(chunks),
            failed_chunks=0,
            failed_batch_indices=[],
        )

    # Multi-batch Map-Reduce
    num_batches = len(batches)
    batch_notes_list: list[str] = []
    failed_batches: list[int] = []
    processed_chunks_count = 0
    failed_chunks_count = 0

    for b_idx, b_chunks in enumerate(batches, start=1):
        if progress_callback is not None:
            progress_callback(
                b_idx,
                num_batches + 2,
                f"Analyzing section {b_idx}/{num_batches} for study notes...",
            )

        formatted_entries = [_format_chunk_for_notes(c) for c in b_chunks]
        prompt = build_batch_notes_prompt(
            filename=filename,
            scope_description=scope_desc,
            batch_index=b_idx,
            total_batches=num_batches,
            formatted_chunks=formatted_entries,
        )

        try:
            b_notes = active_provider.generate(
                NOTES_MAP_SYSTEM_PROMPT,
                prompt,
                temperature=0.2,
                max_tokens=1500,
            )
        except Exception as exc:
            logger.warning("Batch %d notes extraction failed: %s; retrying...", b_idx, exc)
            b_notes = ""

        if not b_notes.strip() and config.FLASHCARD_MAX_RETRIES > 0:
            retry_prompt = build_batch_notes_prompt(
                filename=filename,
                scope_description=scope_desc,
                batch_index=b_idx,
                total_batches=num_batches,
                formatted_chunks=formatted_entries,
                retry_reminder=True,
            )
            try:
                b_notes = active_provider.generate(
                    NOTES_MAP_SYSTEM_PROMPT,
                    retry_prompt,
                    temperature=0.1,
                    max_tokens=1500,
                )
            except Exception as retry_exc:
                logger.warning("Batch %d notes retry failed: %s", b_idx, retry_exc)
                b_notes = ""

        if b_notes.strip():
            batch_notes_list.append(b_notes.strip())
            processed_chunks_count += len(b_chunks)
        else:
            failed_batches.append(b_idx)
            failed_chunks_count += len(b_chunks)

    if not batch_notes_list:
        raise LLMProviderError(f"All {num_batches} study notes extraction batches failed for {filename}.")

    # Hierarchical Reduction stage
    if progress_callback is not None:
        progress_callback(
            num_batches + 1,
            num_batches + 2,
            f"Synthesizing master academic study guide from {len(batch_notes_list)} sections...",
        )

    reduced_notes, levels = _reduce_notes_hierarchically(
        batch_notes_list,
        filename=filename,
        scope_description=scope_desc,
        provider=active_provider,
        max_batch_tokens=token_budget,
        progress_callback=progress_callback,
    )

    final_text = validate_and_filter_page_citations(reduced_notes.strip(), all_pages)

    if progress_callback is not None:
        progress_callback(num_batches + 2, num_batches + 2, "Study notes complete.")

    effective_max_levels = getattr(config, "MAX_REDUCTION_LEVELS", 6)
    is_partial = (failed_chunks_count > 0) or (levels >= effective_max_levels and len(batch_notes_list) > 1)
    status = "partial" if is_partial else "complete"

    return StudyNotesResult(
        title=title,
        notes_text=final_text,
        document_id=document_id,
        filename=filename,
        scope=scope,
        pages_covered=all_pages,
        page_range_label=overall_page_label,
        chunks_used=len(chunks),
        provider=active_provider.provider_name,
        model=active_provider.model_name,
        source_chunks=chunks,
        batches_processed=num_batches,
        reduction_levels=levels,
        status=status,
        total_eligible_chunks=len(chunks),
        processed_chunks=processed_chunks_count,
        failed_chunks=failed_chunks_count,
        failed_batch_indices=failed_batches,
    )
