"""Batch-aware, chunk-grounded Practice Question Generator (Stage 6.2).

Generates MCQ, Short Answer, True/False, and Mixed practice questions strictly
for a single indexed PDF via `BaseLLMProvider`, reusing existing vector-store
document chunk helpers and document-scoped FAISS retrieval.

Key guarantees:
- Strict single-PDF isolation (never mixes chunks across PDFs).
- Full prompt token-budget accounting (system prompt, user prompt, schema,
  chunk headers, chunk text, bounded previous-stem context, and retry headroom).
- Output-token allowance scaled and bounded via `QUESTION_MAX_OUTPUT_TOKENS`.
- Deterministic chunk-ID grounding (`source_chunk_ids` validated against actual
  source batch chunks; `source_pages`, `filename`, and `document_id` derived
  strictly from store metadata, never trusted from LLM output).
- Bounded retries (`QUESTION_MAX_RETRIES`) and bounded top-up (max 1 call).
- Graceful shortfall reporting when content yields fewer unique valid questions
  than requested, without inventing facts or weakening validation.
- Never returns more questions than requested and never returns partial sets
  after an LLM provider failure.
- Completely decoupled from Streamlit and FAISS indexing.
"""

from collections.abc import Callable
import logging
from typing import Any

import src.config as config
from src.generation.llm_provider import BaseLLMProvider, get_llm_provider
from src.generation.question_prompts import (
    QUESTION_SYSTEM_PROMPT,
    build_question_generation_prompt,
    estimate_output_token_allowance,
    format_question_chunk_entry,
)
from src.generation.question_validation import (
    CONCRETE_QUESTION_TYPES,
    PracticeQuestion,
    QuestionGenerationError,
    QuestionSetResult,
    allocate_mixed_counts,
    compute_type_distribution,
    deduplicate_questions,
    parse_and_validate_questions,
    validate_difficulty,
    validate_question_type,
)
from src.generation.summarizer import format_page_range
from src.ingestion.chunker import estimate_tokens
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import retrieve_chunks
from src.retrieval.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

VALID_SCOPES = ("full", "page_range", "topic")
MAX_TOPUP_CALLS = 1
DEFAULT_TOPIC_LOW_SIMILARITY_THRESHOLD = 0.25


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


def _verify_single_document(chunks: list[dict[str, Any]], expected_document_id: str) -> None:
    for chunk in chunks:
        doc_id = chunk.get("document_id")
        if doc_id != expected_document_id:
            raise QuestionGenerationError(
                f"Cross-document contamination blocked: expected document '{expected_document_id}', "
                f"found chunk from '{doc_id}'."
            )


def _pack_chunks_within_budget(
    chunks: list[dict[str, Any]],
    *,
    filename: str,
    scope_description: str,
    question_type: str,
    difficulty: str,
    max_batch_tokens: int,
) -> list[list[dict[str, Any]]]:
    """Pack ordered chunks into batches strictly <= max_batch_tokens including prompt overhead.

    Measures using `retry_reminder=True` and full schema overhead so retries never
    exceed `max_batch_tokens`.
    """
    if not isinstance(max_batch_tokens, int) or max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be a positive integer.")

    probe_type_counts = allocate_mixed_counts(3) if question_type == "mixed" else None

    def _measure(formatted_entries: list[str]) -> int:
        prompt = build_question_generation_prompt(
            filename=filename,
            scope_description=scope_description,
            question_type=question_type,
            difficulty=difficulty,
            question_count=max(1, config.QUESTION_DEFAULT_COUNT),
            formatted_chunks=formatted_entries,
            type_counts=probe_type_counts,
            previous_stems=None,
            retry_reminder=True,
        )
        return _total_prompt_tokens(QUESTION_SYSTEM_PROMPT, prompt)

    base_overhead = _measure([])
    if base_overhead >= max_batch_tokens:
        raise ValueError(
            f"max_batch_tokens ({max_batch_tokens}) is too small to fit prompt and schema "
            f"overhead ({base_overhead} tokens)."
        )

    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_formatted: list[str] = []

    for chunk in chunks:
        formatted = format_question_chunk_entry(chunk)
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
            # Trim single oversized chunk body while keeping its Chunk ID header intact
            empty_chunk = dict(chunk)
            empty_chunk["text"] = ""
            header_overhead = _measure([format_question_chunk_entry(empty_chunk)])
            available_for_body = max_batch_tokens - header_overhead
            if available_for_body <= 10:
                raise ValueError(
                    f"max_batch_tokens ({max_batch_tokens}) is too small to fit a single "
                    f"chunk header ({header_overhead} tokens)."
                )
            trimmed = dict(chunk)
            trimmed["text"] = _truncate_text_to_tokens(str(chunk["text"]), available_for_body)
            current_batch = [trimmed]
            current_formatted = [format_question_chunk_entry(trimmed)]

    if current_batch:
        batches.append(current_batch)

    return batches


def _select_evenly_spaced_indices(total_batches: int, target_count: int) -> list[int]:
    """Select up to `min(total_batches, target_count)` evenly spaced indices covering start to end."""
    selected_num = min(total_batches, target_count)
    if selected_num <= 1:
        return [0]
    if selected_num == total_batches:
        return list(range(total_batches))
    return [
        round(i * (total_batches - 1) / (selected_num - 1))
        for i in range(selected_num)
    ]


def _distribute_batch_plans(
    *,
    num_batches: int,
    total_count: int,
    question_type: str,
) -> list[tuple[str, int, dict[str, int] | None]]:
    """Build per-batch `(batch_question_type, batch_count, batch_type_counts)` plan."""
    if question_type != "mixed":
        base, rem = divmod(total_count, num_batches)
        plans: list[tuple[str, int, dict[str, int] | None]] = []
        for idx in range(num_batches):
            q_count = base + (1 if idx < rem else 0)
            plans.append((question_type, q_count, None))
        return plans

    # Mixed mode: round-robin balanced distribution matching allocate_mixed_counts(total_count)
    global_counts = allocate_mixed_counts(total_count)
    remaining = dict(global_counts)
    ordered_types: list[str] = []
    while len(ordered_types) < total_count:
        for qtype in CONCRETE_QUESTION_TYPES:
            if remaining[qtype] > 0:
                ordered_types.append(qtype)
                remaining[qtype] -= 1

    base, rem = divmod(total_count, num_batches)
    plans = []
    cursor = 0
    for idx in range(num_batches):
        q_count = base + (1 if idx < rem else 0)
        slice_types = ordered_types[cursor : cursor + q_count]
        cursor += q_count

        per_type = {t: slice_types.count(t) for t in CONCRETE_QUESTION_TYPES}
        active = [t for t, cnt in per_type.items() if cnt > 0]
        if len(active) == 1:
            plans.append((active[0], q_count, None))
        else:
            plans.append(("mixed", q_count, per_type))

    return plans


def _build_budgeted_prompt(
    *,
    filename: str,
    scope_description: str,
    batch_question_type: str,
    difficulty: str,
    batch_count: int,
    batch_chunks: list[dict[str, Any]],
    batch_type_counts: dict[str, int] | None,
    previous_stems: list[str],
    max_batch_tokens: int,
    retry_reminder: bool,
) -> str:
    """Build a batch prompt guaranteed to stay <= max_batch_tokens including previous stems."""
    formatted_chunks = [format_question_chunk_entry(c) for c in batch_chunks]
    max_prev = config.QUESTION_MAX_PREVIOUS_STEMS
    candidate_stems = list(previous_stems[-max_prev:]) if previous_stems and max_prev > 0 else []

    while True:
        prompt = build_question_generation_prompt(
            filename=filename,
            scope_description=scope_description,
            question_type=batch_question_type,
            difficulty=difficulty,
            question_count=batch_count,
            formatted_chunks=formatted_chunks,
            type_counts=batch_type_counts,
            previous_stems=candidate_stems if candidate_stems else None,
            max_previous_stems=len(candidate_stems) if candidate_stems else 0,
            retry_reminder=retry_reminder,
        )
        if _total_prompt_tokens(QUESTION_SYSTEM_PROMPT, prompt) <= max_batch_tokens:
            return prompt
        if candidate_stems:
            # Drop oldest previous stem first so source chunks remain intact
            candidate_stems.pop(0)
            continue
        # Fallback if a very tight budget combined with dynamic scope text exceeded by a few tokens
        if len(formatted_chunks) > 1:
            formatted_chunks.pop()
            continue
        raise ValueError(
            f"max_batch_tokens ({max_batch_tokens}) is too small for the batch prompt."
        )


def _normalize_source_chunks(
    chunks: list[dict[str, Any]],
    source_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for chunk in chunks:
        cid = chunk["chunk_id"]
        normalized.append(
            {
                "text": chunk["text"],
                "metadata": {
                    "chunk_id": cid,
                    "vector_id": chunk.get("vector_id"),
                    "document_id": chunk["document_id"],
                    "filename": chunk["filename"],
                    "page": int(chunk["page"]),
                    "page_range": str(chunk.get("page_range", chunk["page"])),
                    "chunk_index": int(chunk["chunk_index"]),
                    "token_count": int(chunk.get("token_count", estimate_tokens(chunk["text"]))),
                },
                "score": source_scores.get(cid) if source_scores else None,
            }
        )
    return normalized


def _run_batch_with_retry(
    *,
    provider: BaseLLMProvider,
    filename: str,
    scope_description: str,
    batch_question_type: str,
    difficulty: str,
    batch_count: int,
    batch_chunks: list[dict[str, Any]],
    batch_type_counts: dict[str, int] | None,
    document_id: str,
    previous_stems: list[str],
    max_batch_tokens: int,
    max_retries: int,
) -> tuple[list[PracticeQuestion], list[str], int]:
    """Execute one batch with bounded retry on malformed JSON or zero valid items.

    LLM provider/configuration errors propagate immediately without swallowing.
    """
    allowed_by_id = {c["chunk_id"]: c for c in batch_chunks}
    output_tokens = estimate_output_token_allowance(batch_count, batch_question_type)
    calls_made = 0
    last_malformed_error: QuestionGenerationError | None = None
    all_rejections: list[str] = []

    for attempt in range(max_retries + 1):
        prompt = _build_budgeted_prompt(
            filename=filename,
            scope_description=scope_description,
            batch_question_type=batch_question_type,
            difficulty=difficulty,
            batch_count=batch_count,
            batch_chunks=batch_chunks,
            batch_type_counts=batch_type_counts,
            previous_stems=previous_stems,
            max_batch_tokens=max_batch_tokens,
            retry_reminder=(attempt > 0),
        )
        calls_made += 1
        # Provider errors (auth, rate limit, timeout, network) propagate cleanly!
        raw_response = provider.generate(
            QUESTION_SYSTEM_PROMPT,
            prompt,
            temperature=0.3,
            max_tokens=output_tokens,
        )

        try:
            valid, rejections = parse_and_validate_questions(
                raw_response,
                expected_type=batch_question_type,
                expected_difficulty=difficulty,
                allowed_chunks_by_id=allowed_by_id,
                document_id=document_id,
                filename=filename,
                max_questions=batch_count,
            )
            last_malformed_error = None
        except QuestionGenerationError as exc:
            logger.warning(
                "Malformed question JSON on attempt %d/%d: %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )
            last_malformed_error = exc
            continue

        all_rejections.extend(rejections)
        unique_valid = deduplicate_questions(valid, existing_stems=previous_stems)
        if unique_valid:
            return unique_valid, all_rejections, calls_made

        logger.info(
            "Batch yielded 0 unique valid questions on attempt %d/%d (rejections=%d)",
            attempt + 1,
            max_retries + 1,
            len(rejections),
        )

    if last_malformed_error is not None:
        raise last_malformed_error

    return [], all_rejections, calls_made


def _accept_respecting_quotas(
    candidates: list[PracticeQuestion],
    *,
    accepted: list[PracticeQuestion],
    accepted_stems: list[str],
    requested_count: int,
    question_type: str,
    remaining_type_quotas: dict[str, int] | None,
) -> None:
    """Append deduplicated candidates without ever exceeding requested_count or mixed quotas."""
    unique = deduplicate_questions(candidates, existing_stems=accepted_stems)
    for item in unique:
        if len(accepted) >= requested_count:
            break
        if question_type == "mixed" and remaining_type_quotas is not None:
            if remaining_type_quotas.get(item.question_type, 0) <= 0:
                continue
            remaining_type_quotas[item.question_type] -= 1
        accepted.append(item)
        accepted_stems.append(item.question)


def generate_practice_questions(
    store: FAISSVectorStore,
    document_id: str,
    *,
    question_type: str = "mcq",
    difficulty: str = "medium",
    count: int | None = None,
    scope: str = "full",
    start_page: int | None = None,
    end_page: int | None = None,
    topic: str | None = None,
    top_k: int | None = None,
    embedding_service: EmbeddingService | None = None,
    provider: BaseLLMProvider | None = None,
    max_batch_tokens: int | None = None,
    max_retries: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> QuestionSetResult:
    """Generate validated, chunk-grounded practice questions for one indexed PDF."""
    doc = store.get_document(document_id)
    if doc is None:
        raise KeyError(f"Document '{document_id}' is not indexed.")

    qtype = validate_question_type(question_type)
    diff = validate_difficulty(difficulty)

    requested_count = config.QUESTION_DEFAULT_COUNT if count is None else count
    if (
        not isinstance(requested_count, int)
        or isinstance(requested_count, bool)
        or requested_count <= 0
    ):
        raise ValueError("count must be a positive integer.")
    if requested_count > config.QUESTION_MAX_COUNT:
        raise ValueError(
            f"count ({requested_count}) exceeds QUESTION_MAX_COUNT ({config.QUESTION_MAX_COUNT})."
        )

    normalized_scope = (scope or "").strip().lower()
    if normalized_scope not in VALID_SCOPES:
        raise ValueError(f"Unsupported scope '{scope}'. Choose one of: {', '.join(VALID_SCOPES)}.")

    budget = config.QUESTION_MAX_BATCH_TOKENS if max_batch_tokens is None else max_batch_tokens
    retries = config.QUESTION_MAX_RETRIES if max_retries is None else max_retries
    if not isinstance(retries, int) or retries < 0:
        raise ValueError("max_retries must be a non-negative integer.")

    filename = doc["filename"]
    source_scores: dict[str, float] | None = None
    notices: list[str] = []

    # 1. Select strictly single-PDF chunks for the chosen scope
    if normalized_scope == "full":
        selected_chunks = store.get_document_chunks(document_id)
        if not selected_chunks:
            raise ValueError(f"No indexed chunks were found for '{filename}'.")
        scope_description = "Full document"
    elif normalized_scope == "page_range":
        if start_page is None or end_page is None:
            raise ValueError("start_page and end_page are required for page_range scope.")
        selected_chunks = store.get_document_chunks(
            document_id,
            start_page=start_page,
            end_page=end_page,
        )
        if not selected_chunks:
            available = store.get_document_page_numbers(document_id)
            raise ValueError(
                f"No indexed text was found in '{filename}' for pages {start_page}-{end_page}. "
                f"Indexed pages available: {format_page_range(available)}."
            )
        scope_description = f"Page range {start_page}-{end_page}"
    else:  # topic
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be a non-empty string for topic scope.")
        cleaned_topic = topic.strip()
        k = config.QUESTION_TOPIC_TOP_K if top_k is None else top_k
        retrieved = retrieve_chunks(
            cleaned_topic,
            top_k=k,
            document_id=document_id,
            store=store,
            embedding_service=embedding_service,
        )
        if not retrieved:
            raise ValueError(
                f"No relevant chunks were retrieved for topic '{cleaned_topic}' in '{filename}'."
            )
        source_scores = {
            item["metadata"]["chunk_id"]: float(item["score"]) for item in retrieved
        }
        top_score = max(source_scores.values())
        if top_score < DEFAULT_TOPIC_LOW_SIMILARITY_THRESHOLD:
            notices.append(
                f"Low semantic match for topic '{cleaned_topic}' in '{filename}' "
                f"(top cosine similarity: {top_score:.2f})."
            )
        selected_chunks = [
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
        selected_chunks.sort(key=lambda c: (int(c["page"]), int(c["chunk_index"])))
        scope_description = f"Topic '{cleaned_topic}'"

    _verify_single_document(selected_chunks, document_id)

    # 2. Pack chunks into token-budgeted batches
    all_batches = _pack_chunks_within_budget(
        selected_chunks,
        filename=filename,
        scope_description=scope_description,
        question_type=qtype,
        difficulty=diff,
        max_batch_tokens=budget,
    )

    selected_indices = _select_evenly_spaced_indices(len(all_batches), requested_count)
    batch_plans = _distribute_batch_plans(
        num_batches=len(selected_indices),
        total_count=requested_count,
        question_type=qtype,
    )

    active_provider = provider if provider is not None else get_llm_provider()
    remaining_type_quotas = allocate_mixed_counts(requested_count) if qtype == "mixed" else None

    accepted: list[PracticeQuestion] = []
    accepted_stems: list[str] = []
    batches_processed = 0
    chunks_by_id = {c["chunk_id"]: c for c in selected_chunks}

    def _emit_progress(step: int, total: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(step, total, message)

    total_steps = len(selected_indices)
    for step_idx, (batch_idx, (b_qtype, b_count, b_type_counts)) in enumerate(
        zip(selected_indices, batch_plans), start=1
    ):
        if len(accepted) >= requested_count:
            break
        batch_chunks = all_batches[batch_idx]
        batch_pages = format_page_range([int(c["page"]) for c in batch_chunks])
        _emit_progress(
            step_idx,
            total_steps,
            f"Generating batch {step_idx}/{total_steps} ({batch_pages})...",
        )
        candidates, _, _ = _run_batch_with_retry(
            provider=active_provider,
            filename=filename,
            scope_description=scope_description,
            batch_question_type=b_qtype,
            difficulty=diff,
            batch_count=b_count,
            batch_chunks=batch_chunks,
            batch_type_counts=b_type_counts,
            document_id=document_id,
            previous_stems=accepted_stems,
            max_batch_tokens=budget,
            max_retries=retries,
        )
        batches_processed += 1
        _accept_respecting_quotas(
            candidates,
            accepted=accepted,
            accepted_stems=accepted_stems,
            requested_count=requested_count,
            question_type=qtype,
            remaining_type_quotas=remaining_type_quotas,
        )

    # 3. Bounded top-up (at most MAX_TOPUP_CALLS=1) if shortfall remains
    if len(accepted) < requested_count and MAX_TOPUP_CALLS > 0:
        deficit = requested_count - len(accepted)
        unused_indices = [i for i in range(len(all_batches)) if i not in selected_indices]
        topup_batch_idx = unused_indices[0] if unused_indices else max(
            range(len(all_batches)),
            key=lambda i: sum(estimate_tokens(c["text"]) for c in all_batches[i]),
        )
        topup_chunks = all_batches[topup_batch_idx]
        topup_pages = format_page_range([int(c["page"]) for c in topup_chunks])

        if qtype == "mixed" and remaining_type_quotas is not None:
            active_deficit_types = [t for t, rem in remaining_type_quotas.items() if rem > 0]
            if len(active_deficit_types) == 1:
                topup_qtype = active_deficit_types[0]
                topup_type_counts = None
            else:
                topup_qtype = "mixed"
                topup_type_counts = dict(remaining_type_quotas)
        else:
            topup_qtype = qtype
            topup_type_counts = None

        _emit_progress(
            total_steps + 1,
            total_steps + 1,
            f"Generating top-up questions for remaining deficit ({topup_pages})...",
        )
        topup_candidates, _, _ = _run_batch_with_retry(
            provider=active_provider,
            filename=filename,
            scope_description=scope_description,
            batch_question_type=topup_qtype,
            difficulty=diff,
            batch_count=deficit,
            batch_chunks=topup_chunks,
            batch_type_counts=topup_type_counts,
            document_id=document_id,
            previous_stems=accepted_stems,
            max_batch_tokens=budget,
            max_retries=retries,
        )
        batches_processed += 1
        _accept_respecting_quotas(
            topup_candidates,
            accepted=accepted,
            accepted_stems=accepted_stems,
            requested_count=requested_count,
            question_type=qtype,
            remaining_type_quotas=remaining_type_quotas,
        )

    if not accepted:
        raise QuestionGenerationError(
            f"Could not generate any valid, source-grounded practice questions from '{filename}'. "
            "Try a broader page range, different topic, or check the document text."
        )

    # Re-assign clean sequential IDs q1..qK
    for idx, question in enumerate(accepted, start=1):
        question.question_id = f"q{idx}"

    # Shortfall notice (Adjustment #1: never weaken validation or invent facts)
    if len(accepted) < requested_count:
        notices.append(
            f"Requested {requested_count} unique grounded question(s), but the selected PDF "
            f"content yielded {len(accepted)} valid question(s) without duplicating or "
            f"inventing unsupported material."
        )

    referenced_chunk_ids = {cid for q in accepted for cid in q.source_chunk_ids}
    referenced_chunks = [
        chunks_by_id[cid] for cid in referenced_chunk_ids if cid in chunks_by_id
    ]
    referenced_chunks.sort(key=lambda c: (int(c["page"]), int(c["chunk_index"])))

    pages_covered = sorted({p for q in accepted for p in q.source_pages})
    page_range_label = format_page_range(pages_covered)
    type_distribution = compute_type_distribution(accepted)

    return QuestionSetResult(
        questions=accepted,
        document_id=document_id,
        filename=filename,
        scope=normalized_scope,
        question_type=qtype,
        difficulty=diff,
        requested_count=requested_count,
        generated_count=len(accepted),
        type_distribution=type_distribution,
        pages_covered=pages_covered,
        page_range_label=page_range_label,
        chunks_used=len(referenced_chunks),
        batches_processed=batches_processed,
        provider=active_provider.provider_name,
        model=active_provider.model_name,
        source_chunks=_normalize_source_chunks(referenced_chunks, source_scores),
        notice=" ".join(notices) if notices else None,
    )


def generate_document_questions(
    store: FAISSVectorStore,
    document_id: str,
    *,
    question_type: str = "mcq",
    difficulty: str = "medium",
    count: int | None = None,
    provider: BaseLLMProvider | None = None,
    max_batch_tokens: int | None = None,
    max_retries: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> QuestionSetResult:
    """Convenience wrapper for full-document practice question generation."""
    return generate_practice_questions(
        store,
        document_id,
        question_type=question_type,
        difficulty=difficulty,
        count=count,
        scope="full",
        provider=provider,
        max_batch_tokens=max_batch_tokens,
        max_retries=max_retries,
        progress_callback=progress_callback,
    )


def generate_page_range_questions(
    store: FAISSVectorStore,
    document_id: str,
    start_page: int,
    end_page: int,
    *,
    question_type: str = "mcq",
    difficulty: str = "medium",
    count: int | None = None,
    provider: BaseLLMProvider | None = None,
    max_batch_tokens: int | None = None,
    max_retries: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> QuestionSetResult:
    """Convenience wrapper for page-range practice question generation."""
    return generate_practice_questions(
        store,
        document_id,
        question_type=question_type,
        difficulty=difficulty,
        count=count,
        scope="page_range",
        start_page=start_page,
        end_page=end_page,
        provider=provider,
        max_batch_tokens=max_batch_tokens,
        max_retries=max_retries,
        progress_callback=progress_callback,
    )


def generate_topic_questions(
    store: FAISSVectorStore,
    document_id: str,
    topic: str,
    *,
    question_type: str = "mcq",
    difficulty: str = "medium",
    count: int | None = None,
    top_k: int | None = None,
    embedding_service: EmbeddingService | None = None,
    provider: BaseLLMProvider | None = None,
    max_batch_tokens: int | None = None,
    max_retries: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> QuestionSetResult:
    """Convenience wrapper for document-scoped topic practice question generation."""
    return generate_practice_questions(
        store,
        document_id,
        question_type=question_type,
        difficulty=difficulty,
        count=count,
        scope="topic",
        topic=topic,
        top_k=top_k,
        embedding_service=embedding_service,
        provider=provider,
        max_batch_tokens=max_batch_tokens,
        max_retries=max_retries,
        progress_callback=progress_callback,
    )
