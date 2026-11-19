"""Hierarchical single-PDF summarization engine built on BaseLLMProvider.

Supports four strictly single-PDF modes:
- full: full-document summarization
- page_range: explicit start/end page summarization
- section: exact heading match with transparent semantic fallback
- topic: document-scoped FAISS retrieval summarization

Key guarantees:
- Never mixes chunks across different PDFs.
- Accounts for system prompt, user prompt instructions, chunk headers, and text
  when enforcing `SUMMARY_MAX_BATCH_TOKENS`.
- Uses hierarchical map-reduce with bounded reduction levels for large PDFs.
- Computes source filename and page metadata deterministically from actual source
  chunks rather than trusting LLM-generated metadata.
- Never indexes generated summaries back into FAISS.
- Propagates LLM errors cleanly without returning partial summaries as successful.
"""

from collections.abc import Callable
from dataclasses import dataclass
import logging
import re
from typing import Any

import src.config as config
from src.generation.llm_provider import BaseLLMProvider, get_llm_provider
from src.generation.summary_prompts import (
    SUMMARY_LENGTH_SPECS,
    SUMMARY_SYSTEM_PROMPT,
    build_map_batch_user_prompt,
    build_reduce_user_prompt,
    build_single_pass_user_prompt,
    format_chunk_entry,
    format_intermediate_summary_entry,
    validate_summary_length,
)
from src.ingestion.chunker import estimate_tokens
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import retrieve_chunks
from src.retrieval.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

MAX_REDUCTION_LEVELS = 3
DEFAULT_SECTION_MAX_CHUNKS = 12
DEFAULT_SECTION_MIN_SIMILARITY = 0.25
DEFAULT_TOPIC_LOW_SIMILARITY_THRESHOLD = 0.25

ProgressCallback = Callable[[int, int, str], None]

_HEADING_LINE_RE = re.compile(
    r"^(?:chapter\s+\d+|section\s+\d+|part\s+[ivxlcdm\d]+|\d+(?:\.\d+)*\.?\s+[A-Z])",
    re.IGNORECASE,
)


class SummarizationError(RuntimeError):
    """Raised when summarization cannot complete safely within constraints."""


@dataclass
class SummaryResult:
    summary_text: str
    document_id: str
    filename: str
    mode: str
    length: str
    pages_covered: list[int]
    page_range_label: str
    chunks_used: int
    batches_processed: int
    reduction_levels: int
    hierarchical: bool
    provider: str
    model: str
    source_chunks: list[dict[str, Any]]
    match_strategy: str | None = None
    match_notice: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_text": self.summary_text,
            "document_id": self.document_id,
            "filename": self.filename,
            "mode": self.mode,
            "length": self.length,
            "pages_covered": list(self.pages_covered),
            "page_range_label": self.page_range_label,
            "chunks_used": self.chunks_used,
            "batches_processed": self.batches_processed,
            "reduction_levels": self.reduction_levels,
            "hierarchical": self.hierarchical,
            "provider": self.provider,
            "model": self.model,
            "match_strategy": self.match_strategy,
            "match_notice": self.match_notice,
            "source_chunks": self.source_chunks,
        }


def format_page_range(pages: list[int]) -> str:
    """Format sorted page numbers into concise human-readable ranges (e.g., Pages 1-3, 5)."""
    unique_pages = sorted({int(p) for p in pages})
    if not unique_pages:
        return "No pages"
    if len(unique_pages) == 1:
        return f"Page {unique_pages[0]}"

    ranges: list[str] = []
    start = prev = unique_pages[0]
    for page in unique_pages[1:]:
        if page == prev + 1:
            prev = page
        else:
            ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = page
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return f"Pages {', '.join(ranges)}"


def _total_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    return estimate_tokens(system_prompt) + estimate_tokens(user_prompt)


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    words = text.split()
    kept: list[str] = []
    for word in words:
        candidate = " ".join(kept + [word])
        if kept and estimate_tokens(candidate) > max_tokens:
            break
        kept.append(word)
    if kept:
        return " ".join(kept)
    return text[: max(4, max_tokens * 4)]


def _normalize_source_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Normalize raw store chunk or retriever result into a uniform UI-ready dict."""
    if "metadata" in chunk and isinstance(chunk["metadata"], dict):
        meta = chunk["metadata"]
        return {
            "text": chunk["text"],
            "metadata": {
                "chunk_id": meta["chunk_id"],
                "vector_id": meta.get("vector_id"),
                "document_id": meta["document_id"],
                "filename": meta["filename"],
                "page": int(meta["page"]),
                "page_range": str(meta.get("page_range", meta["page"])),
                "chunk_index": int(meta["chunk_index"]),
                "token_count": int(meta.get("token_count", estimate_tokens(chunk["text"]))),
            },
            "score": chunk.get("score"),
        }

    return {
        "text": chunk["text"],
        "metadata": {
            "chunk_id": chunk["chunk_id"],
            "vector_id": chunk.get("vector_id"),
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "page": int(chunk["page"]),
            "page_range": str(chunk.get("page_range", chunk["page"])),
            "chunk_index": int(chunk["chunk_index"]),
            "token_count": int(chunk.get("token_count", estimate_tokens(chunk["text"]))),
        },
        "score": None,
    }


def _pack_items_within_budget(
    items: list[dict[str, Any]],
    *,
    format_item: Callable[[dict[str, Any]], str],
    build_prompt: Callable[[list[dict[str, Any]], list[str]], str],
    system_prompt: str,
    max_batch_tokens: int,
) -> list[list[dict[str, Any]]]:
    """Group items into ordered batches where full system+user prompt <= max_batch_tokens."""
    if not isinstance(max_batch_tokens, int) or max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be a positive integer.")

    # Measure empty-template prompt overhead (system prompt + user wrapper instructions)
    empty_user_prompt = build_prompt([], [])
    overhead_tokens = _total_prompt_tokens(system_prompt, empty_user_prompt)
    if overhead_tokens >= max_batch_tokens:
        raise ValueError(
            f"max_batch_tokens ({max_batch_tokens}) is too small to fit prompt overhead "
            f"({overhead_tokens} tokens)."
        )

    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_formatted: list[str] = []

    for item in items:
        formatted = format_item(item)
        candidate_items = current_batch + [item]
        candidate_formatted = current_formatted + [formatted]
        candidate_prompt = build_prompt(candidate_items, candidate_formatted)
        candidate_tokens = _total_prompt_tokens(system_prompt, candidate_prompt)

        if candidate_tokens <= max_batch_tokens:
            current_batch = candidate_items
            current_formatted = candidate_formatted
            continue

        if current_batch:
            batches.append(current_batch)
            current_batch = []
            current_formatted = []

        # Verify single item fits on its own; trim item text if a single entry exceeds budget
        single_prompt = build_prompt([item], [formatted])
        single_tokens = _total_prompt_tokens(system_prompt, single_prompt)
        if single_tokens <= max_batch_tokens:
            current_batch = [item]
            current_formatted = [formatted]
        else:
            # Trim item's body text while preserving its metadata header
            empty_item = dict(item)
            text_key = "text" if "text" in empty_item else "summary_text"
            empty_item[text_key] = ""
            header_only_prompt = build_prompt([empty_item], [format_item(empty_item)])
            header_overhead = _total_prompt_tokens(system_prompt, header_only_prompt)
            available_for_body = max_batch_tokens - header_overhead
            if available_for_body <= 10:
                raise ValueError(
                    f"max_batch_tokens ({max_batch_tokens}) is too small to fit a single "
                    f"entry header ({header_overhead} tokens)."
                )
            trimmed_item = dict(item)
            trimmed_item[text_key] = _truncate_text_to_tokens(
                str(item[text_key]), available_for_body
            )
            current_batch = [trimmed_item]
            current_formatted = [format_item(trimmed_item)]

    if current_batch:
        batches.append(current_batch)

    return batches


def _verify_single_document(chunks: list[dict[str, Any]], expected_document_id: str) -> None:
    for chunk in chunks:
        doc_id = chunk["document_id"]
        if doc_id != expected_document_id:
            raise SummarizationError(
                f"Cross-document contamination blocked: expected document '{expected_document_id}', "
                f"found chunk from '{doc_id}'."
            )


def _summarize_chunks_hierarchical(
    *,
    chunks: list[dict[str, Any]],
    document_id: str,
    filename: str,
    mode: str,
    scope_description: str,
    length: str,
    provider: BaseLLMProvider,
    max_batch_tokens: int,
    progress_callback: ProgressCallback | None = None,
    match_strategy: str | None = None,
    match_notice: str | None = None,
    source_scores: dict[str, float] | None = None,
) -> SummaryResult:
    if not chunks:
        raise ValueError("No chunks are available to summarize.")

    _verify_single_document(chunks, document_id)
    normalized_length = validate_summary_length(length)
    spec = SUMMARY_LENGTH_SPECS[normalized_length]

    all_pages = sorted({int(c["page"]) for c in chunks})
    overall_page_label = format_page_range(all_pages)

    def _emit_progress(step: int, total: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(step, total, message)

    # First check if ALL chunks fit cleanly into a single prompt within max_batch_tokens
    single_entries = [format_chunk_entry(c) for c in chunks]
    single_prompt = build_single_pass_user_prompt(
        filename=filename,
        page_range_label=overall_page_label,
        scope_description=scope_description,
        length=normalized_length,
        formatted_entries=single_entries,
    )
    if _total_prompt_tokens(SUMMARY_SYSTEM_PROMPT, single_prompt) <= max_batch_tokens:
        _emit_progress(
            1,
            1,
            f"Generating {spec['label'].lower()} summary for {filename} ({overall_page_label})...",
        )
        summary_text = provider.generate(
            SUMMARY_SYSTEM_PROMPT,
            single_prompt,
            temperature=0.2,
            max_tokens=spec["max_tokens"],
        )
        normalized_sources = _build_normalized_sources(chunks, source_scores)
        return SummaryResult(
            summary_text=summary_text,
            document_id=document_id,
            filename=filename,
            mode=mode,
            length=normalized_length,
            pages_covered=all_pages,
            page_range_label=overall_page_label,
            chunks_used=len(chunks),
            batches_processed=1,
            reduction_levels=0,
            hierarchical=False,
            provider=provider.provider_name,
            model=provider.model_name,
            source_chunks=normalized_sources,
            match_strategy=match_strategy,
            match_notice=match_notice,
        )

    # Multi-batch hierarchical map-reduce path
    def _map_prompt_builder(
        batch_chunks: list[dict[str, Any]], formatted_list: list[str]
    ) -> str:
        batch_pages = sorted({int(c["page"]) for c in batch_chunks}) if batch_chunks else all_pages
        return build_map_batch_user_prompt(
            filename=filename,
            page_range_label=format_page_range(batch_pages),
            batch_index=999,
            total_batches=999,
            scope_description=scope_description,
            length=normalized_length,
            formatted_entries=formatted_list,
        )

    map_batches = _pack_items_within_budget(
        chunks,
        format_item=format_chunk_entry,
        build_prompt=_map_prompt_builder,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        max_batch_tokens=max_batch_tokens,
    )

    total_map_batches = len(map_batches)
    # Conservative initial step count: map batches + 1 final reduce step
    total_steps = total_map_batches + 1
    current_step = 0
    intermediate_items: list[dict[str, Any]] = []
    map_max_tokens = min(600, max(300, spec["max_tokens"]))

    for idx, batch in enumerate(map_batches, start=1):
        batch_pages = sorted({int(c["page"]) for c in batch})
        batch_page_label = format_page_range(batch_pages)
        current_step += 1
        _emit_progress(
            current_step,
            total_steps,
            f"Summarizing batch {idx}/{total_map_batches} ({batch_page_label})...",
        )
        map_prompt = build_map_batch_user_prompt(
            filename=filename,
            page_range_label=batch_page_label,
            batch_index=idx,
            total_batches=total_map_batches,
            scope_description=scope_description,
            length=normalized_length,
            formatted_entries=[format_chunk_entry(c) for c in batch],
        )
        batch_summary = provider.generate(
            SUMMARY_SYSTEM_PROMPT,
            map_prompt,
            temperature=0.2,
            max_tokens=map_max_tokens,
        )
        intermediate_items.append(
            {
                "filename": filename,
                "pages": batch_pages,
                "page_range_label": batch_page_label,
                "batch_label": f"Batch {idx}",
                "summary_text": batch_summary,
            }
        )

    # Hierarchical reduce loop (bounded by MAX_REDUCTION_LEVELS to prevent over-compression)
    reduction_level = 0
    total_batches_processed = total_map_batches

    def _format_intermediate(item: dict[str, Any]) -> str:
        return format_intermediate_summary_entry(
            filename=item["filename"],
            page_range_label=item["page_range_label"],
            batch_label=item["batch_label"],
            summary_text=item["summary_text"],
        )

    while True:
        reduction_level += 1
        if reduction_level > MAX_REDUCTION_LEVELS:
            raise SummarizationError(
                f"Document summaries exceeded {MAX_REDUCTION_LEVELS} hierarchical reduction levels "
                f"within SUMMARY_MAX_BATCH_TOKENS={max_batch_tokens}. "
                "Increase SUMMARY_MAX_BATCH_TOKENS or summarize a smaller page range to preserve quality."
            )

        # Check if all current intermediate summaries fit into ONE final reduce call
        all_formatted = [_format_intermediate(item) for item in intermediate_items]
        final_prompt = build_reduce_user_prompt(
            filename=filename,
            page_range_label=overall_page_label,
            scope_description=scope_description,
            length=normalized_length,
            formatted_entries=all_formatted,
            is_final=True,
            reduction_level=reduction_level,
        )
        if _total_prompt_tokens(SUMMARY_SYSTEM_PROMPT, final_prompt) <= max_batch_tokens:
            current_step += 1
            total_steps = max(total_steps, current_step)
            _emit_progress(
                current_step,
                total_steps,
                f"Synthesizing final {spec['label'].lower()} summary ({overall_page_label})...",
            )
            final_summary = provider.generate(
                SUMMARY_SYSTEM_PROMPT,
                final_prompt,
                temperature=0.2,
                max_tokens=spec["max_tokens"],
            )
            total_batches_processed += 1
            break

        # Still too large for a single final reduce call: group into intermediate reduce batches
        def _intermediate_reduce_builder(
            group_items: list[dict[str, Any]], formatted_list: list[str]
        ) -> str:
            grp_pages = (
                sorted({p for it in group_items for p in it["pages"]})
                if group_items
                else all_pages
            )
            return build_reduce_user_prompt(
                filename=filename,
                page_range_label=format_page_range(grp_pages),
                scope_description=scope_description,
                length=normalized_length,
                formatted_entries=formatted_list,
                is_final=False,
                reduction_level=reduction_level,
            )

        reduce_groups = _pack_items_within_budget(
            intermediate_items,
            format_item=_format_intermediate,
            build_prompt=_intermediate_reduce_builder,
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            max_batch_tokens=max_batch_tokens,
        )
        if len(reduce_groups) >= len(intermediate_items):
            raise SummarizationError(
                "Intermediate summaries could not be reduced further within "
                f"SUMMARY_MAX_BATCH_TOKENS={max_batch_tokens}. "
                "Increase SUMMARY_MAX_BATCH_TOKENS or summarize a smaller page range."
            )

        total_steps = current_step + len(reduce_groups) + 1
        next_level_items: list[dict[str, Any]] = []
        for g_idx, group in enumerate(reduce_groups, start=1):
            grp_pages = sorted({p for it in group for p in it["pages"]})
            grp_page_label = format_page_range(grp_pages)
            current_step += 1
            _emit_progress(
                current_step,
                total_steps,
                f"Consolidating level {reduction_level} group {g_idx}/{len(reduce_groups)} ({grp_page_label})...",
            )
            group_prompt = build_reduce_user_prompt(
                filename=filename,
                page_range_label=grp_page_label,
                scope_description=scope_description,
                length=normalized_length,
                formatted_entries=[_format_intermediate(it) for it in group],
                is_final=False,
                reduction_level=reduction_level,
            )
            group_summary = provider.generate(
                SUMMARY_SYSTEM_PROMPT,
                group_prompt,
                temperature=0.2,
                max_tokens=map_max_tokens,
            )
            total_batches_processed += 1
            next_level_items.append(
                {
                    "filename": filename,
                    "pages": grp_pages,
                    "page_range_label": grp_page_label,
                    "batch_label": f"Level {reduction_level} Group {g_idx}",
                    "summary_text": group_summary,
                }
            )
        intermediate_items = next_level_items

    normalized_sources = _build_normalized_sources(chunks, source_scores)
    return SummaryResult(
        summary_text=final_summary,
        document_id=document_id,
        filename=filename,
        mode=mode,
        length=normalized_length,
        pages_covered=all_pages,
        page_range_label=overall_page_label,
        chunks_used=len(chunks),
        batches_processed=total_batches_processed,
        reduction_levels=reduction_level,
        hierarchical=True,
        provider=provider.provider_name,
        model=provider.model_name,
        source_chunks=normalized_sources,
        match_strategy=match_strategy,
        match_notice=match_notice,
    )


def _build_normalized_sources(
    chunks: list[dict[str, Any]],
    source_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    normalized = [_normalize_source_chunk(c) for c in chunks]
    if source_scores:
        for item in normalized:
            cid = item["metadata"]["chunk_id"]
            if cid in source_scores:
                item["score"] = source_scores[cid]
    return normalized


def _resolve_provider(provider: BaseLLMProvider | None) -> BaseLLMProvider:
    return provider if provider is not None else get_llm_provider()


def _resolve_batch_budget(max_batch_tokens: int | None) -> int:
    return config.SUMMARY_MAX_BATCH_TOKENS if max_batch_tokens is None else max_batch_tokens


def summarize_document(
    store: FAISSVectorStore,
    document_id: str,
    *,
    length: str = "medium",
    provider: BaseLLMProvider | None = None,
    max_batch_tokens: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SummaryResult:
    """Summarize an entire indexed PDF strictly scoped to `document_id`."""
    doc = store.get_document(document_id)
    if doc is None:
        raise KeyError(f"Document '{document_id}' is not indexed.")

    chunks = store.get_document_chunks(document_id)
    if not chunks:
        raise ValueError(f"No indexed chunks were found for '{doc['filename']}'.")

    return _summarize_chunks_hierarchical(
        chunks=chunks,
        document_id=document_id,
        filename=doc["filename"],
        mode="full",
        scope_description="Full document",
        length=length,
        provider=_resolve_provider(provider),
        max_batch_tokens=_resolve_batch_budget(max_batch_tokens),
        progress_callback=progress_callback,
    )


def summarize_page_range(
    store: FAISSVectorStore,
    document_id: str,
    start_page: int,
    end_page: int,
    *,
    length: str = "medium",
    provider: BaseLLMProvider | None = None,
    max_batch_tokens: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SummaryResult:
    """Summarize a specific inclusive page range within a single PDF."""
    doc = store.get_document(document_id)
    if doc is None:
        raise KeyError(f"Document '{document_id}' is not indexed.")

    chunks = store.get_document_chunks(
        document_id,
        start_page=start_page,
        end_page=end_page,
    )
    if not chunks:
        available = store.get_document_page_numbers(document_id)
        avail_label = format_page_range(available)
        raise ValueError(
            f"No indexed text was found in '{doc['filename']}' for pages {start_page}-{end_page}. "
            f"Indexed pages available: {avail_label}."
        )

    return _summarize_chunks_hierarchical(
        chunks=chunks,
        document_id=document_id,
        filename=doc["filename"],
        mode="page_range",
        scope_description=f"Requested page range {start_page}-{end_page}",
        length=length,
        provider=_resolve_provider(provider),
        max_batch_tokens=_resolve_batch_budget(max_batch_tokens),
        progress_callback=progress_callback,
    )


def _normalize_heading_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _chunk_matches_heading(chunk_text: str, normalized_heading: str) -> bool:
    for raw_line in chunk_text.splitlines():
        line = _normalize_heading_text(raw_line)
        if not line:
            continue
        if line == normalized_heading or line.startswith(normalized_heading):
            return True
        # Match numbered/prefixed heading line such as "2.1 Photosynthesis" or "Chapter 3: Mitosis"
        if normalized_heading in line and len(line) <= len(normalized_heading) + 45:
            return True
    return False


def _starts_new_major_section(chunk_text: str, normalized_target: str) -> bool:
    for raw_line in chunk_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        norm_line = _normalize_heading_text(stripped)
        if normalized_target in norm_line:
            return False
        if _HEADING_LINE_RE.match(stripped) and len(stripped) <= 90:
            return True
        # Only inspect the opening line of a chunk for a new section heading
        return False
    return False


def _find_exact_section_chunks(
    all_doc_chunks: list[dict[str, Any]],
    section_heading: str,
    max_chunks: int = DEFAULT_SECTION_MAX_CHUNKS,
) -> list[dict[str, Any]]:
    normalized_target = _normalize_heading_text(section_heading)
    if not normalized_target:
        return []

    start_idx: int | None = None
    for idx, chunk in enumerate(all_doc_chunks):
        if _chunk_matches_heading(chunk["text"], normalized_target):
            start_idx = idx
            break

    if start_idx is None:
        return []

    collected = [all_doc_chunks[start_idx]]
    for next_chunk in all_doc_chunks[start_idx + 1 :]:
        if len(collected) >= max_chunks:
            break
        if _starts_new_major_section(next_chunk["text"], normalized_target):
            break
        collected.append(next_chunk)

    return collected


def summarize_section(
    store: FAISSVectorStore,
    document_id: str,
    section_heading: str,
    *,
    length: str = "medium",
    top_k: int | None = None,
    min_similarity: float = DEFAULT_SECTION_MIN_SIMILARITY,
    max_section_chunks: int = DEFAULT_SECTION_MAX_CHUNKS,
    embedding_service: EmbeddingService | None = None,
    provider: BaseLLMProvider | None = None,
    max_batch_tokens: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SummaryResult:
    """Summarize a section within one PDF, distinguishing exact heading match vs semantic fallback."""
    if not isinstance(section_heading, str) or not section_heading.strip():
        raise ValueError("section_heading must be a non-empty string.")

    cleaned_heading = section_heading.strip()
    doc = store.get_document(document_id)
    if doc is None:
        raise KeyError(f"Document '{document_id}' is not indexed.")

    all_doc_chunks = store.get_document_chunks(document_id)
    if not all_doc_chunks:
        raise ValueError(f"No indexed chunks were found for '{doc['filename']}'.")

    exact_chunks = _find_exact_section_chunks(
        all_doc_chunks,
        cleaned_heading,
        max_chunks=max_section_chunks,
    )
    if exact_chunks:
        return _summarize_chunks_hierarchical(
            chunks=exact_chunks,
            document_id=document_id,
            filename=doc["filename"],
            mode="section",
            scope_description=f"Section '{cleaned_heading}' (exact heading match)",
            length=length,
            provider=_resolve_provider(provider),
            max_batch_tokens=_resolve_batch_budget(max_batch_tokens),
            progress_callback=progress_callback,
            match_strategy="exact_heading",
            match_notice=None,
        )

    # No literal heading found: fall back to single-PDF semantic retrieval and inform the user clearly
    k = config.SUMMARY_TOPIC_TOP_K if top_k is None else top_k
    retrieved = retrieve_chunks(
        cleaned_heading,
        top_k=k,
        document_id=document_id,
        store=store,
        embedding_service=embedding_service,
    )
    if not retrieved:
        raise ValueError(
            f"Section '{cleaned_heading}' could not be located in '{doc['filename']}'."
        )

    top_score = max(float(item["score"]) for item in retrieved)
    if top_score < min_similarity:
        raise ValueError(
            f"Section '{cleaned_heading}' could not be reliably identified in '{doc['filename']}' "
            f"(no exact heading match and highest semantic similarity was {top_score:.2f}, "
            f"below threshold {min_similarity:.2f}). Try an exact heading or Page Range mode."
        )

    score_map = {item["metadata"]["chunk_id"]: float(item["score"]) for item in retrieved}
    matched_chunks = [
        {
            "chunk_id": item["metadata"]["chunk_id"],
            "vector_id": item["metadata"]["vector_id"],
            "document_id": item["metadata"]["document_id"],
            "filename": item["metadata"]["filename"],
            "page": item["metadata"]["page"],
            "page_range": item["metadata"]["page_range"],
            "chunk_index": item["metadata"]["chunk_index"],
            "token_count": item["metadata"]["token_count"],
            "text": item["text"],
        }
        for item in retrieved
    ]
    matched_chunks.sort(key=lambda c: (int(c["page"]), int(c["chunk_index"])))
    pages_label = format_page_range([int(c["page"]) for c in matched_chunks])
    notice = (
        f"Exact heading '{cleaned_heading}' was not found in '{doc['filename']}'. "
        f"Used semantic fallback across {len(matched_chunks)} chunk(s) on {pages_label} "
        f"(top cosine similarity: {top_score:.2f})."
    )

    return _summarize_chunks_hierarchical(
        chunks=matched_chunks,
        document_id=document_id,
        filename=doc["filename"],
        mode="section",
        scope_description=f"Section '{cleaned_heading}' (semantic fallback)",
        length=length,
        provider=_resolve_provider(provider),
        max_batch_tokens=_resolve_batch_budget(max_batch_tokens),
        progress_callback=progress_callback,
        match_strategy="semantic_fallback",
        match_notice=notice,
        source_scores=score_map,
    )


def summarize_topic(
    store: FAISSVectorStore,
    document_id: str,
    topic: str,
    *,
    length: str = "medium",
    top_k: int | None = None,
    embedding_service: EmbeddingService | None = None,
    provider: BaseLLMProvider | None = None,
    max_batch_tokens: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SummaryResult:
    """Summarize a specific topic inside one PDF using existing FAISS retrieval."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic must be a non-empty string.")

    cleaned_topic = topic.strip()
    doc = store.get_document(document_id)
    if doc is None:
        raise KeyError(f"Document '{document_id}' is not indexed.")

    k = config.SUMMARY_TOPIC_TOP_K if top_k is None else top_k
    retrieved = retrieve_chunks(
        cleaned_topic,
        top_k=k,
        document_id=document_id,
        store=store,
        embedding_service=embedding_service,
    )
    if not retrieved:
        raise ValueError(
            f"No relevant chunks were retrieved for topic '{cleaned_topic}' in '{doc['filename']}'."
        )

    score_map = {item["metadata"]["chunk_id"]: float(item["score"]) for item in retrieved}
    top_score = max(score_map.values())
    notice = None
    if top_score < DEFAULT_TOPIC_LOW_SIMILARITY_THRESHOLD:
        notice = (
            f"Low semantic match for topic '{cleaned_topic}' in '{doc['filename']}' "
            f"(top cosine similarity: {top_score:.2f}). Results may be only loosely related."
        )

    matched_chunks = [
        {
            "chunk_id": item["metadata"]["chunk_id"],
            "vector_id": item["metadata"]["vector_id"],
            "document_id": item["metadata"]["document_id"],
            "filename": item["metadata"]["filename"],
            "page": item["metadata"]["page"],
            "page_range": item["metadata"]["page_range"],
            "chunk_index": item["metadata"]["chunk_index"],
            "token_count": item["metadata"]["token_count"],
            "text": item["text"],
        }
        for item in retrieved
    ]
    matched_chunks.sort(key=lambda c: (int(c["page"]), int(c["chunk_index"])))

    return _summarize_chunks_hierarchical(
        chunks=matched_chunks,
        document_id=document_id,
        filename=doc["filename"],
        mode="topic",
        scope_description=f"Topic '{cleaned_topic}'",
        length=length,
        provider=_resolve_provider(provider),
        max_batch_tokens=_resolve_batch_budget(max_batch_tokens),
        progress_callback=progress_callback,
        match_strategy="semantic_retrieval",
        match_notice=notice,
        source_scores=score_map,
    )
