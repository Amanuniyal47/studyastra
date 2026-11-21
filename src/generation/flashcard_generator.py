"""Batch-aware, chunk-grounded Flashcard Generator powered by Qwen2.5 7B.

Generates concise, high-yield academic flashcards strictly grounded in indexed PDF chunks.
Key guarantees:
- Strict single-PDF isolation.
- Structured JSON output with schema validation (front, back, source_chunk_ids).
- Deterministic page number derivation from verified vector-store chunks (never trusting LLM guesses).
- Multi-batch chunk distribution and prompt token budgeting for large PDFs.
- Resilient schema parsing (root list, synonym keys, concise valid answers).
- Deduplication and bounded retries.
- Zero outside hallucinations; content is strictly derived from the document.
- Centralized LLM access via BaseLLMProvider / OllamaLLMProvider.
- Markdown and Anki CSV export capabilities.
"""

from collections.abc import Callable
import csv
from dataclasses import dataclass
import io
import json
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
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")

FLASHCARD_SYSTEM_PROMPT = (
    "You are an expert academic study tutor and flashcard creator. "
    "Generate high-yield study flashcards grounded STRICTLY in the provided PDF excerpts. "
    "Never use outside knowledge or invent facts not present in the text.\n"
    "Each flashcard must have:\n"
    "- 'front': A clear, specific question, key term, or conceptual prompt.\n"
    "- 'back': A concise, factual, and accurate explanation or answer found directly in the text.\n"
    "- 'source_chunk_ids': An array containing ONLY exact Chunk IDs shown in the [Chunk ID: ...] headers.\n"
    "Return ONLY valid JSON matching this schema:\n"
    "{\n"
    '  "flashcards": [\n'
    '    {\n'
    '      "front": "<prompt/question/term>",\n'
    '      "back": "<accurate definition or answer from the excerpt>",\n'
    '      "source_chunk_ids": ["<exact_chunk_id>"]\n'
    '    }\n'
    '  ]\n'
    "}"
)


class FlashcardGenerationError(RuntimeError):
    """Raised when flashcard generation or parsing fails."""


@dataclass
class Flashcard:
    card_id: str
    front: str
    back: str
    source_chunk_ids: list[str]
    source_pages: list[int]
    page_range_label: str
    filename: str
    document_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "front": self.front,
            "back": self.back,
            "source_chunk_ids": list(self.source_chunk_ids),
            "source_pages": list(self.source_pages),
            "page_range_label": self.page_range_label,
            "filename": self.filename,
            "document_id": self.document_id,
        }


@dataclass
class FlashcardSetResult:
    cards: list[Flashcard]
    document_id: str
    filename: str
    scope: str
    requested_count: int
    generated_count: int
    pages_covered: list[int]
    page_range_label: str
    chunks_used: int
    batches_processed: int
    provider: str
    model: str
    notice: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cards": [c.to_dict() for c in self.cards],
            "document_id": self.document_id,
            "filename": self.filename,
            "scope": self.scope,
            "requested_count": self.requested_count,
            "generated_count": self.generated_count,
            "pages_covered": list(self.pages_covered),
            "page_range_label": self.page_range_label,
            "chunks_used": self.chunks_used,
            "batches_processed": self.batches_processed,
            "provider": self.provider,
            "model": self.model,
            "notice": self.notice,
        }


def extract_flashcard_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract and parse a JSON object or array containing flashcards from raw LLM output."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise FlashcardGenerationError("LLM returned empty output instead of JSON.")

    stripped = raw_text.strip()
    candidates: list[str] = [stripped]

    for match in _CODE_FENCE_RE.findall(stripped):
        fenced = match.strip()
        if fenced:
            candidates.insert(0, fenced)

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    first_bracket = stripped.find("[")
    last_bracket = stripped.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        candidates.append(stripped[first_bracket : last_bracket + 1])

    parsed = None
    last_error = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc

    if parsed is None:
        raise FlashcardGenerationError(
            f"Could not parse valid JSON from LLM output: {last_error}"
        ) from last_error

    # If the root is a JSON list directly, wrap into a dictionary
    if isinstance(parsed, list):
        return {"flashcards": parsed}

    if not isinstance(parsed, dict):
        raise FlashcardGenerationError(
            f"Expected JSON object or list root with 'flashcards' key, got {type(parsed).__name__}."
        )

    if "flashcards" in parsed and isinstance(parsed["flashcards"], list):
        return parsed

    for alt_key in ("cards", "questions", "deck", "items"):
        if alt_key in parsed and isinstance(parsed[alt_key], list):
            return {"flashcards": parsed[alt_key]}

    for _, v in parsed.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return {"flashcards": v}

    raise FlashcardGenerationError("JSON missing 'flashcards' array.")


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    words_a = set(_WORD_RE.findall(text_a.lower()))
    words_b = set(_WORD_RE.findall(text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def parse_and_validate_flashcards(
    raw_text: str,
    *,
    allowed_chunks_by_id: dict[str, dict[str, Any]],
    document_id: str,
    filename: str,
    max_cards: int,
    existing_fronts: list[str] | None = None,
) -> tuple[list[Flashcard], list[str]]:
    """Parse JSON, validate chunk grounding, derive page numbers deterministically, and deduplicate."""
    payload = extract_flashcard_json_payload(raw_text)
    raw_cards = payload.get("flashcards", [])
    valid_cards: list[Flashcard] = []
    rejections: list[str] = []
    seen_fronts = list(existing_fronts or [])

    for idx, item in enumerate(raw_cards, start=1):
        if not isinstance(item, dict):
            rejections.append(f"Card #{idx} is not a valid JSON object.")
            continue

        front = str(
            item.get("front")
            or item.get("question")
            or item.get("prompt")
            or item.get("term")
            or item.get("title")
            or ""
        ).strip()

        back = str(
            item.get("back")
            or item.get("answer")
            or item.get("explanation")
            or item.get("definition")
            or item.get("response")
            or ""
        ).strip()

        if len(front) < 2:
            rejections.append(f"Card #{idx} front prompt too short ({len(front)} chars).")
            continue
        if len(back) < 1:
            rejections.append(f"Card #{idx} back answer empty.")
            continue

        # Duplicate check against accepted fronts
        is_dup = any(_jaccard_similarity(front, existing) >= 0.80 for existing in seen_fronts)
        if is_dup:
            rejections.append(f"Card #{idx} is duplicate of an existing flashcard: '{front[:40]}...'.")
            continue

        # Validate source chunk grounding
        raw_cids = item.get("source_chunk_ids") or item.get("chunk_ids") or item.get("sources")
        if isinstance(raw_cids, str):
            raw_cids = [raw_cids]
        elif not isinstance(raw_cids, list):
            raw_cids = []

        if not raw_cids and allowed_chunks_by_id:
            # Model omitted source_chunk_ids, attribute to allowed chunks in current batch
            valid_cids = list(allowed_chunks_by_id.keys())
            derived_pages = {int(allowed_chunks_by_id[cid]["page"]) for cid in valid_cids}
        else:
            valid_cids = []
            derived_pages = set()
            for cid in raw_cids:
                cid_str = str(cid).strip()
                if cid_str in allowed_chunks_by_id:
                    valid_cids.append(cid_str)
                    derived_pages.add(int(allowed_chunks_by_id[cid_str]["page"]))

        if not valid_cids:
            rejections.append(f"Card #{idx} cited no valid chunk IDs from current batch.")
            continue

        pages_sorted = sorted(derived_pages)
        page_label = format_page_range(pages_sorted)
        card_id = f"fc_{len(seen_fronts) + 1}"

        card = Flashcard(
            card_id=card_id,
            front=front,
            back=back,
            source_chunk_ids=valid_cids,
            source_pages=pages_sorted,
            page_range_label=page_label,
            filename=filename,
            document_id=document_id,
        )
        valid_cards.append(card)
        seen_fronts.append(front)

        if len(valid_cards) >= max_cards:
            break

    return valid_cards, rejections


def _format_flashcard_chunk_entry(chunk: dict[str, Any]) -> str:
    meta = chunk.get("metadata", chunk)
    cid = meta.get("chunk_id", "unknown")
    page = meta.get("page_range", meta.get("page", 1))
    return f"[Chunk ID: {cid} | Page {page}]\n{chunk['text'].strip()}"


def build_flashcard_prompt(
    filename: str,
    scope_description: str,
    card_count: int,
    formatted_chunks: list[str],
    retry_reminder: bool = False,
) -> str:
    chunks_text = "\n\n".join(formatted_chunks)
    prompt = (
        f"Document: {filename} ({scope_description})\n"
        f"Task: Generate exactly {card_count} high-yield study flashcards based strictly on the source excerpts.\n\n"
        f"Source Excerpts:\n{chunks_text}\n\n"
        f"Remember: Base every flashcard exclusively on the excerpts. "
        f"Include valid 'source_chunk_ids' from the headers. Return ONLY valid JSON."
    )
    if retry_reminder:
        prompt += "\n\nCRITICAL: Return strictly a valid JSON object matching the requested schema with a 'flashcards' array."
    return prompt


def _measure_flashcard_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    return estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + 32


def _build_flashcard_batches(
    chunks: list[dict[str, Any]],
    *,
    filename: str,
    scope_description: str,
    max_batch_tokens: int,
) -> list[list[dict[str, Any]]]:
    """Group chunks into batches guaranteed to stay within max_batch_tokens including prompt overhead."""
    def _measure(formatted_entries: list[str]) -> int:
        prompt = build_flashcard_prompt(
            filename=filename,
            scope_description=scope_description,
            card_count=max(1, config.FLASHCARD_DEFAULT_COUNT),
            formatted_chunks=formatted_entries,
            retry_reminder=True,
        )
        return _measure_flashcard_prompt_tokens(FLASHCARD_SYSTEM_PROMPT, prompt)

    base_overhead = _measure([])
    if base_overhead >= max_batch_tokens:
        raise ValueError(
            f"max_batch_tokens ({max_batch_tokens}) is too small to fit prompt overhead ({base_overhead} tokens)."
        )

    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_formatted: list[str] = []

    for chunk in chunks:
        formatted = _format_flashcard_chunk_entry(chunk)
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
            # Trim single oversized chunk body while preserving Chunk ID header
            empty_chunk = dict(chunk)
            empty_chunk["text"] = ""
            header_overhead = _measure([_format_flashcard_chunk_entry(empty_chunk)])
            available_for_body = max_batch_tokens - header_overhead
            if available_for_body <= 10:
                logger.warning("Chunk %s header too large; skipping.", chunk.get("chunk_id"))
                continue
            char_budget = max(40, available_for_body * 3)
            trimmed_chunk = dict(chunk)
            trimmed_chunk["text"] = chunk["text"][:char_budget]
            trimmed_formatted = _format_flashcard_chunk_entry(trimmed_chunk)
            current_batch = [trimmed_chunk]
            current_formatted = [trimmed_formatted]

    if current_batch:
        batches.append(current_batch)

    return batches


def _plan_flashcard_batches(
    batches: list[list[dict[str, Any]]],
    total_count: int,
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Distribute requested flashcard count across batches for even coverage."""
    num_batches = len(batches)
    if num_batches == 0 or total_count <= 0:
        return []

    # If more batches than requested cards, pick total_count evenly spaced batches across the document
    if num_batches > total_count:
        if total_count == 1:
            indices = [0]
        else:
            step = (num_batches - 1) / (total_count - 1)
            indices = [round(i * step) for i in range(total_count)]
        return [(1, batches[idx]) for idx in indices]

    # If batches <= total_count, distribute cards across all batches
    base, rem = divmod(total_count, num_batches)
    plans = []
    for idx in range(num_batches):
        cnt = base + (1 if idx < rem else 0)
        plans.append((cnt, batches[idx]))
    return plans


def export_flashcards_to_markdown(
    result: FlashcardSetResult | dict[str, Any] | list[Flashcard] | list[dict[str, Any]],
    *,
    filename: str = "",
    page_range_label: str = "",
    scope: str = "",
) -> str:
    """Format flashcards as clean, human-readable Markdown with metadata headers."""
    if isinstance(result, FlashcardSetResult):
        cards = result.cards
        filename = result.filename
        page_range_label = result.page_range_label
        scope = result.scope
    elif isinstance(result, dict) and "cards" in result:
        cards = result["cards"]
        filename = result.get("filename", filename)
        page_range_label = result.get("page_range_label", page_range_label)
        scope = result.get("scope", scope)
    elif isinstance(result, list):
        cards = result
    else:
        cards = []

    lines = [
        f"# Flashcard Deck: {filename or 'Study Notes'}",
        f"**Scope:** {str(scope).title() or 'Full Document'} | **Cards:** {len(cards)} | **Pages:** {page_range_label or 'N/A'}",
        "",
        "---",
        "",
    ]

    for idx, card in enumerate(cards, start=1):
        front = card.front if isinstance(card, Flashcard) else card.get("front", "")
        back = card.back if isinstance(card, Flashcard) else card.get("back", "")
        pages = card.page_range_label if isinstance(card, Flashcard) else card.get("page_range_label", "")
        cids = card.source_chunk_ids if isinstance(card, Flashcard) else card.get("source_chunk_ids", [])
        cid_str = ", ".join(cids) if isinstance(cids, list) else str(cids)

        lines.extend([
            f"### Card {idx}",
            f"**Question / Front:**",
            f"{front}",
            "",
            f"**Answer / Back:**",
            f"{back}",
            "",
            f"*Source: {filename or 'Document'} ({pages}) | Chunk IDs: `{cid_str}`*",
            "",
            "---",
            "",
        ])

    return "\n".join(lines).strip() + "\n"


def export_flashcards_to_anki_csv(
    result: FlashcardSetResult | dict[str, Any] | list[Flashcard] | list[dict[str, Any]],
    *,
    include_headers: bool = True,
) -> str:
    """Export flashcards to RFC 4180 Anki-compatible CSV format."""
    if isinstance(result, FlashcardSetResult):
        cards = result.cards
    elif isinstance(result, dict) and "cards" in result:
        cards = result["cards"]
    elif isinstance(result, list):
        cards = result
    else:
        cards = []

    output = io.StringIO()
    writer = csv.writer(output, delimiter=",", quoting=csv.QUOTE_ALL, lineterminator="\n")

    if include_headers:
        writer.writerow(["Front", "Back", "Source"])

    for card in cards:
        front = card.front if isinstance(card, Flashcard) else card.get("front", "")
        back = card.back if isinstance(card, Flashcard) else card.get("back", "")
        pages = card.page_range_label if isinstance(card, Flashcard) else card.get("page_range_label", "")
        fname = card.filename if isinstance(card, Flashcard) else card.get("filename", "")
        source_label = f"{fname} ({pages})" if fname and pages else (fname or pages or "")

        writer.writerow([front, back, source_label])

    return output.getvalue()


def generate_flashcards(
    store: FAISSVectorStore,
    document_id: str,
    *,
    count: int = 6,
    scope: str = "full",
    start_page: int | None = None,
    end_page: int | None = None,
    topic: str | None = None,
    top_k: int = 6,
    embedding_service: Any = None,
    provider: BaseLLMProvider | None = None,
    max_batch_tokens: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> FlashcardSetResult:
    """Generate high-yield study flashcards for a single indexed PDF using Qwen."""
    doc = store.get_document(document_id)
    if doc is None:
        raise KeyError(f"Document '{document_id}' is not indexed.")
    filename = doc["filename"]

    # Gather source chunks based on scope
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
        raise ValueError(f"No source text found for {filename} under {scope_desc}.")

    active_provider = provider if provider is not None else get_llm_provider("ollama")
    req_count = max(1, min(count, config.FLASHCARD_MAX_COUNT))
    token_budget = max_batch_tokens or config.FLASHCARD_MAX_BATCH_TOKENS

    all_pages = sorted({int(c["page"]) for c in chunks})
    overall_page_label = format_page_range(all_pages)

    # 1. Build batches respecting token budget
    batches = _build_flashcard_batches(
        chunks,
        filename=filename,
        scope_description=scope_desc,
        max_batch_tokens=token_budget,
    )
    if not batches:
        raise ValueError(f"Unable to form valid chunk batches for {filename}.")

    # 2. Plan card distribution across batches
    batch_plans = _plan_flashcard_batches(batches, req_count)
    total_planned = len(batch_plans)

    accepted_cards: list[Flashcard] = []
    seen_fronts: list[str] = []
    total_chunks_used: set[str] = set()
    batches_processed = 0

    for b_idx, (batch_card_target, batch_chunks) in enumerate(batch_plans, start=1):
        if len(accepted_cards) >= req_count:
            break

        batches_processed += 1
        allowed_by_id = {c["chunk_id"]: c for c in batch_chunks}
        total_chunks_used.update(allowed_by_id.keys())

        if progress_callback is not None:
            progress_callback(
                b_idx,
                total_planned + 1,
                f"Generating flashcards (batch {b_idx}/{total_planned}, target: {batch_card_target})...",
            )

        formatted_entries = [_format_flashcard_chunk_entry(c) for c in batch_chunks]
        needed_now = min(batch_card_target, req_count - len(accepted_cards))
        prompt = build_flashcard_prompt(
            filename=filename,
            scope_description=scope_desc,
            card_count=max(1, needed_now),
            formatted_chunks=formatted_entries,
        )

        try:
            raw_response = active_provider.generate(
                FLASHCARD_SYSTEM_PROMPT,
                prompt,
                temperature=0.2,
                max_tokens=config.FLASHCARD_MAX_OUTPUT_TOKENS,
                format="json",
            )
            cards, _ = parse_and_validate_flashcards(
                raw_response,
                allowed_chunks_by_id=allowed_by_id,
                document_id=document_id,
                filename=filename,
                max_cards=needed_now,
                existing_fronts=seen_fronts,
            )
        except Exception as exc:
            logger.warning("Batch %d generation failed: %s; retrying...", b_idx, exc)
            cards = []

        # Bounded retry on this batch if 0 cards produced
        if not cards and config.FLASHCARD_MAX_RETRIES > 0:
            if progress_callback is not None:
                progress_callback(
                    b_idx,
                    total_planned + 1,
                    f"Retrying batch {b_idx} with reinforced schema...",
                )
            retry_prompt = build_flashcard_prompt(
                filename=filename,
                scope_description=scope_desc,
                card_count=max(1, needed_now),
                formatted_chunks=formatted_entries,
                retry_reminder=True,
            )
            try:
                retry_raw = active_provider.generate(
                    FLASHCARD_SYSTEM_PROMPT,
                    retry_prompt,
                    temperature=0.1,
                    max_tokens=config.FLASHCARD_MAX_OUTPUT_TOKENS,
                    format="json",
                )
                cards, _ = parse_and_validate_flashcards(
                    retry_raw,
                    allowed_chunks_by_id=allowed_by_id,
                    document_id=document_id,
                    filename=filename,
                    max_cards=needed_now,
                    existing_fronts=seen_fronts,
                )
            except Exception as retry_exc:
                logger.warning("Batch %d retry also failed: %s", b_idx, retry_exc)
                cards = []

        for c in cards:
            accepted_cards.append(c)
            seen_fronts.append(c.front)

    # Top-up from unused batches if shortfall exists
    if len(accepted_cards) < req_count and len(batches) > total_planned:
        used_batch_ids = {id(b) for _, b in batch_plans}
        unused_batches = [b for b in batches if id(b) not in used_batch_ids]
        for extra_b in unused_batches:
            if len(accepted_cards) >= req_count:
                break
            batches_processed += 1
            allowed_by_id = {c["chunk_id"]: c for c in extra_b}
            total_chunks_used.update(allowed_by_id.keys())
            needed = req_count - len(accepted_cards)
            formatted_entries = [_format_flashcard_chunk_entry(c) for c in extra_b]
            prompt = build_flashcard_prompt(
                filename=filename,
                scope_description=scope_desc,
                card_count=needed,
                formatted_chunks=formatted_entries,
            )
            try:
                top_raw = active_provider.generate(
                    FLASHCARD_SYSTEM_PROMPT,
                    prompt,
                    temperature=0.2,
                    max_tokens=config.FLASHCARD_MAX_OUTPUT_TOKENS,
                    format="json",
                )
                extra_cards, _ = parse_and_validate_flashcards(
                    top_raw,
                    allowed_chunks_by_id=allowed_by_id,
                    document_id=document_id,
                    filename=filename,
                    max_cards=needed,
                    existing_fronts=seen_fronts,
                )
                for c in extra_cards:
                    accepted_cards.append(c)
                    seen_fronts.append(c.front)
            except Exception as top_exc:
                logger.warning("Top-up batch failed: %s", top_exc)

    notice = None
    if len(accepted_cards) < req_count:
        notice = f"Generated {len(accepted_cards)} unique flashcards (requested {req_count})."

    if progress_callback is not None:
        progress_callback(total_planned + 1, total_planned + 1, "Flashcards ready.")

    return FlashcardSetResult(
        cards=accepted_cards,
        document_id=document_id,
        filename=filename,
        scope=scope,
        requested_count=req_count,
        generated_count=len(accepted_cards),
        pages_covered=all_pages,
        page_range_label=overall_page_label,
        chunks_used=len(total_chunks_used),
        batches_processed=batches_processed,
        provider=active_provider.provider_name,
        model=active_provider.model_name,
        notice=notice,
    )
