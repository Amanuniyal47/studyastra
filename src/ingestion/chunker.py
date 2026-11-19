"""Paragraph- and sentence-aware text chunking for extracted PDF pages.

Limitations:
- Uses a lightweight regex-based token estimator rather than loading a full
  model-specific tokenizer during ingestion. WordPiece/BPE tokenizers may split
  rare technical terms into slightly more subword tokens (~10-15% variance).
- Default settings (220 tokens, 40-token overlap) leave headroom below
  `all-MiniLM-L6-v2`'s 256-token max sequence length to minimize truncation.
- By default, chunks do not cross page boundaries so source page attribution
  remains exact (particularly useful for multi-topic study slides and notes).
"""

import hashlib
import math
import re

from src.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Estimate token count closely matching WordPiece/BPE tokenization."""
    if not text or not text.strip():
        return 0

    total = 0
    for token in _TOKEN_RE.findall(text):
        if len(token) > 16:
            total += math.ceil(len(token) / 4)
        else:
            total += 1
    return total


def _normalize_for_hash(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _split_long_word(word: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Slice an unbroken string (e.g., long URL or raw text) safely."""
    max_chars = max(4, max_tokens * 4)
    overlap_chars = min(max_chars - 4, max(0, overlap_tokens * 4)) if max_chars > 4 else 0
    step = max(1, max_chars - overlap_chars)

    parts = []
    start = 0
    length = len(word)
    while start < length:
        end = start + max_chars
        part = word[start:end].strip()
        if part:
            parts.append(part)
        if end >= length:
            break
        start += step
    return parts


def _split_oversized_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split an oversized paragraph by sentences, then words, then characters."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    units: list[str] = []

    for sentence in sentences:
        if estimate_tokens(sentence) <= max_tokens:
            units.append(sentence)
            continue

        # Sentence itself exceeds max_tokens: split on whitespace into word groups
        words = sentence.split()
        current_words: list[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = estimate_tokens(word)
            if word_tokens > max_tokens:
                if current_words:
                    units.append(" ".join(current_words))
                    current_words = []
                    current_tokens = 0
                units.extend(_split_long_word(word, max_tokens, overlap_tokens))
                continue

            if current_words and current_tokens + word_tokens > max_tokens:
                units.append(" ".join(current_words))
                current_words = [word]
                current_tokens = word_tokens
            else:
                current_words.append(word)
                current_tokens += word_tokens

        if current_words:
            units.append(" ".join(current_words))

    return units


def _extract_page_units(page_text: str, max_tokens: int, overlap_tokens: int) -> list[tuple[str, bool]]:
    """Return list of (unit_text, starts_new_paragraph) for a single page."""
    normalized = page_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    raw_paragraphs = [
        _WHITESPACE_RE.sub(" ", p).strip()
        for p in _PARAGRAPH_SPLIT_RE.split(normalized)
    ]
    paragraphs = [p for p in raw_paragraphs if p]

    units: list[tuple[str, bool]] = []
    for paragraph in paragraphs:
        if estimate_tokens(paragraph) <= max_tokens:
            units.append((paragraph, True))
        else:
            sub_units = _split_oversized_text(paragraph, max_tokens, overlap_tokens)
            for idx, sub_unit in enumerate(sub_units):
                units.append((sub_unit, idx == 0))

    return units


def _join_units(units: list[tuple[str, bool]]) -> str:
    """Join units while preserving paragraph breaks between distinct paragraphs."""
    if not units:
        return ""

    parts = [units[0][0]]
    for text, starts_paragraph in units[1:]:
        separator = "\n\n" if starts_paragraph else " "
        parts.append(separator + text)
    return "".join(parts).strip()


def _select_overlap_units(
    units: list[tuple[str, bool]], overlap_tokens: int
) -> list[tuple[str, bool]]:
    """Select trailing units (or trailing words) up to overlap_tokens."""
    if overlap_tokens <= 0 or not units:
        return []

    selected: list[tuple[str, bool]] = []
    total_tokens = 0

    for text, starts_paragraph in reversed(units):
        unit_tokens = estimate_tokens(text)
        if selected and total_tokens + unit_tokens > overlap_tokens:
            break
        if unit_tokens <= overlap_tokens:
            selected.append((text, starts_paragraph))
            total_tokens += unit_tokens
        else:
            # Take trailing words from this single large unit
            words = text.split()
            tail_words: list[str] = []
            tail_tokens = 0
            for word in reversed(words):
                w_tokens = estimate_tokens(word)
                if tail_words and tail_tokens + w_tokens > overlap_tokens:
                    break
                tail_words.append(word)
                tail_tokens += w_tokens
            if tail_words:
                tail_words.reverse()
                selected.append((" ".join(tail_words), False))
            break

    selected.reverse()
    return selected


def chunk_text(
    text_by_page,
    document_id: str = "default",
    filename: str = "unknown.pdf",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    deduplicate: bool = True,
) -> list[dict]:
    """Divide extracted PDF pages into structured, metadata-rich text chunks.

    Args:
        text_by_page: Iterable of `(page_number, page_text)` tuples.
        document_id: Unique identifier for the source document (e.g. SHA-256 hash).
        filename: Original PDF filename.
        chunk_size: Maximum target tokens per chunk (defaults to config.CHUNK_SIZE_TOKENS).
        chunk_overlap: Overlap tokens carried between adjacent chunks on the same page
            (defaults to config.CHUNK_OVERLAP_TOKENS).
        deduplicate: When True, skip duplicate chunks with identical normalized text
            within the same document.

    Returns:
        List of chunk dictionaries containing `chunk_id`, `document_id`, `filename`,
        `page`, `page_range`, `chunk_index`, `text`, and `token_count`.
    """
    target_size = CHUNK_SIZE_TOKENS if chunk_size is None else chunk_size
    target_overlap = CHUNK_OVERLAP_TOKENS if chunk_overlap is None else chunk_overlap

    if target_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if target_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater.")
    if target_overlap >= target_size:
        raise ValueError("chunk_size must be greater than chunk_overlap.")

    if not text_by_page:
        return []

    doc_prefix = (document_id or "default")[:12]
    chunks: list[dict] = []
    seen_hashes: set[str] = set()
    chunk_index = 0

    def _emit_chunk(page_number: int, unit_buffer: list[tuple[str, bool]]) -> None:
        nonlocal chunk_index
        text = _join_units(unit_buffer)
        if not text:
            return

        if deduplicate:
            norm_hash = hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()
            if norm_hash in seen_hashes:
                return
            seen_hashes.add(norm_hash)

        chunk_index += 1
        chunks.append(
            {
                "chunk_id": f"{doc_prefix}_p{page_number}_c{chunk_index}",
                "document_id": document_id,
                "filename": filename,
                "page": page_number,
                "page_range": str(page_number),
                "chunk_index": chunk_index,
                "text": text,
                "token_count": estimate_tokens(text),
            }
        )

    for page_number, page_text in text_by_page:
        if not isinstance(page_text, str) or not page_text.strip():
            continue

        units = _extract_page_units(page_text, target_size, target_overlap)
        if not units:
            continue

        buffer: list[tuple[str, bool]] = []
        buffer_tokens = 0

        for unit in units:
            unit_tokens = estimate_tokens(unit[0])

            if buffer and buffer_tokens + unit_tokens > target_size:
                _emit_chunk(page_number, buffer)

                # Build overlap prefix for next chunk on this page
                overlap_units = _select_overlap_units(buffer, target_overlap)
                overlap_tokens = sum(estimate_tokens(u[0]) for u in overlap_units)

                # Ensure overlap + new unit fits; otherwise start clean with new unit
                if overlap_units and overlap_tokens + unit_tokens <= target_size:
                    buffer = overlap_units + [unit]
                    buffer_tokens = overlap_tokens + unit_tokens
                else:
                    buffer = [unit]
                    buffer_tokens = unit_tokens
            else:
                buffer.append(unit)
                buffer_tokens += unit_tokens

        if buffer:
            _emit_chunk(page_number, buffer)

    return chunks
