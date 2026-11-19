"""Data structures, JSON extraction, deterministic validation, and deduplication
for Practice Question Generation (Stage 6.2).

Strict source-grounding rules:
- Every question MUST cite at least one valid `source_chunk_id` belonging to the
  selected document and current source batch.
- Questions citing unknown, empty, or cross-document chunk IDs are rejected.
- `source_pages`, `page_range_label`, `filename`, and `document_id` are always
  derived deterministically from verified vector-store chunks—never trusted from
  LLM output.
"""

from dataclasses import dataclass
import json
import re
from typing import Any

from src.generation.summarizer import format_page_range

CONCRETE_QUESTION_TYPES = ("mcq", "short_answer", "true_false")
VALID_QUESTION_TYPES = ("mcq", "short_answer", "true_false", "mixed")
VALID_DIFFICULTIES = ("easy", "medium", "hard")

_MCQ_LETTER_TO_INDEX = {"a": 0, "b": 1, "c": 2, "d": 3}
_OPTION_PREFIX_RE = re.compile(r"^\s*(?:[A-Da-d][\)\.\:\-]\s*)")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")

MIN_QUESTION_CHARS = 10
MIN_EXPLANATION_CHARS = 8
DEFAULT_JACCARD_DUPLICATE_THRESHOLD = 0.80


class QuestionGenerationError(RuntimeError):
    """Raised when question generation fails or LLM output cannot be validated."""


class QuestionValidationError(ValueError):
    """Raised when an individual generated question fails deterministic validation."""


@dataclass
class PracticeQuestion:
    question_id: str
    question_type: str
    difficulty: str
    question: str
    options: list[str] | None
    correct_answer: str
    correct_statement: str | None
    explanation: str
    source_chunk_ids: list[str]
    source_pages: list[int]
    page_range_label: str
    filename: str
    document_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "question": self.question,
            "options": list(self.options) if self.options is not None else None,
            "correct_answer": self.correct_answer,
            "correct_statement": self.correct_statement,
            "explanation": self.explanation,
            "source_chunk_ids": list(self.source_chunk_ids),
            "source_pages": list(self.source_pages),
            "page_range_label": self.page_range_label,
            "filename": self.filename,
            "document_id": self.document_id,
        }


@dataclass
class QuestionSetResult:
    questions: list[PracticeQuestion]
    document_id: str
    filename: str
    scope: str
    question_type: str
    difficulty: str
    requested_count: int
    generated_count: int
    type_distribution: dict[str, int]
    pages_covered: list[int]
    page_range_label: str
    chunks_used: int
    batches_processed: int
    provider: str
    model: str
    source_chunks: list[dict[str, Any]]
    notice: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "questions": [q.to_dict() for q in self.questions],
            "document_id": self.document_id,
            "filename": self.filename,
            "scope": self.scope,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "requested_count": self.requested_count,
            "generated_count": self.generated_count,
            "type_distribution": dict(self.type_distribution),
            "pages_covered": list(self.pages_covered),
            "page_range_label": self.page_range_label,
            "chunks_used": self.chunks_used,
            "batches_processed": self.batches_processed,
            "provider": self.provider,
            "model": self.model,
            "notice": self.notice,
            "source_chunks": self.source_chunks,
        }


def validate_question_type(question_type: str) -> str:
    normalized = re.sub(r"[\s/\-]+", "_", (question_type or "").strip().lower()).strip("_")
    alias_map = {
        "multiple_choice": "mcq",
        "multiple_choice_mcq": "mcq",
        "mcqs": "mcq",
        "short": "short_answer",
        "tf": "true_false",
        "true_or_false": "true_false",
    }
    resolved = alias_map.get(normalized, normalized)
    if resolved not in VALID_QUESTION_TYPES:
        valid = ", ".join(VALID_QUESTION_TYPES)
        raise ValueError(f"Unsupported question_type '{question_type}'. Choose one of: {valid}.")
    return resolved


def validate_difficulty(difficulty: str) -> str:
    normalized = (difficulty or "").strip().lower()
    if normalized not in VALID_DIFFICULTIES:
        valid = ", ".join(VALID_DIFFICULTIES)
        raise ValueError(f"Unsupported difficulty '{difficulty}'. Choose one of: {valid}.")
    return normalized


def allocate_mixed_counts(total_count: int) -> dict[str, int]:
    """Distribute `total_count` as evenly as possible across MCQ, Short Answer, True/False.

    Remainder items are assigned in order: mcq -> short_answer -> true_false.
    Example: 10 -> {"mcq": 4, "short_answer": 3, "true_false": 3}.
    """
    if not isinstance(total_count, int) or isinstance(total_count, bool) or total_count <= 0:
        raise ValueError("total_count must be a positive integer.")

    base, remainder = divmod(total_count, len(CONCRETE_QUESTION_TYPES))
    allocation: dict[str, int] = {}
    for idx, qtype in enumerate(CONCRETE_QUESTION_TYPES):
        allocation[qtype] = base + (1 if idx < remainder else 0)
    return allocation


def compute_type_distribution(questions: list[PracticeQuestion]) -> dict[str, int]:
    counts = {qtype: 0 for qtype in CONCRETE_QUESTION_TYPES}
    for question in questions:
        if question.question_type in counts:
            counts[question.question_type] += 1
    return counts


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract and parse a JSON object containing a `questions` list from raw LLM output."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise QuestionGenerationError("LLM returned empty output instead of JSON.")

    candidates: list[str] = []
    stripped = raw_text.strip()
    candidates.append(stripped)

    # Check markdown fenced blocks first if present
    for match in _CODE_FENCE_RE.findall(stripped):
        fenced = match.strip()
        if fenced:
            candidates.insert(0, fenced)

    # Also extract outermost {...} substring as fallback for minor preamble/postscript text
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    parsed: Any = None
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc

    if parsed is None:
        raise QuestionGenerationError(
            f"Malformed LLM output: could not parse valid JSON ({last_error})."
        ) from last_error

    if isinstance(parsed, list):
        parsed = {"questions": parsed}

    if not isinstance(parsed, dict) or not isinstance(parsed.get("questions"), list):
        raise QuestionGenerationError(
            "Malformed LLM JSON: expected a top-level object with a 'questions' list."
        )

    return parsed


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _normalize_option_text(option: str) -> str:
    stripped = _OPTION_PREFIX_RE.sub("", option).strip()
    return " ".join(stripped.split())


def _normalize_for_compare(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _validate_source_chunk_ids(
    raw_chunk_ids: Any,
    *,
    allowed_chunks_by_id: dict[str, dict[str, Any]],
    expected_document_id: str,
) -> tuple[list[str], list[int]]:
    """Verify every cited chunk_id exists in the current batch & selected document.

    Derives `source_pages` strictly from verified chunks—never from LLM output.
    """
    if isinstance(raw_chunk_ids, str):
        raw_chunk_ids = [raw_chunk_ids]

    if not isinstance(raw_chunk_ids, list) or not raw_chunk_ids:
        raise QuestionValidationError("Question is missing required non-empty 'source_chunk_ids'.")

    verified_ids: list[str] = []
    seen: set[str] = set()
    derived_pages: set[int] = set()

    for raw_cid in raw_chunk_ids:
        if not isinstance(raw_cid, str) or not raw_cid.strip():
            raise QuestionValidationError("source_chunk_ids must contain non-empty strings.")
        cid = raw_cid.strip()
        chunk = allowed_chunks_by_id.get(cid)
        if chunk is None:
            raise QuestionValidationError(
                f"Question references unknown or out-of-batch chunk ID '{cid}'."
            )
        if chunk.get("document_id") != expected_document_id:
            raise QuestionValidationError(
                f"Question references cross-document chunk ID '{cid}'."
            )
        if cid not in seen:
            seen.add(cid)
            verified_ids.append(cid)
            derived_pages.add(int(chunk["page"]))

    if not verified_ids or not derived_pages:
        raise QuestionValidationError("Question has no verified supporting source chunks.")

    return verified_ids, sorted(derived_pages)


def _resolve_mcq_answer(raw_answer: Any, cleaned_options: list[str]) -> str:
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raise QuestionValidationError("MCQ is missing 'correct_answer'.")

    ans = raw_answer.strip()
    # 1. Exact/case-insensitive match against cleaned options
    ans_cleaned = _normalize_option_text(ans)
    matches = [opt for opt in cleaned_options if opt.lower() == ans_cleaned.lower()]
    if len(matches) == 1:
        return matches[0]

    # 2. Normalized alphanumeric match
    ans_norm = _normalize_for_compare(ans_cleaned)
    norm_matches = [opt for opt in cleaned_options if _normalize_for_compare(opt) == ans_norm]
    if len(norm_matches) == 1:
        return norm_matches[0]

    # 3. Single letter label (A / B / C / D or "Option B" / "B.")
    letter_match = re.fullmatch(r"(?:option\s+)?([a-d])[\)\.\:]?", ans.lower())
    if letter_match:
        idx = _MCQ_LETTER_TO_INDEX[letter_match.group(1)]
        return cleaned_options[idx]

    raise QuestionValidationError(
        f"MCQ correct_answer '{ans}' does not unambiguously map to any of the 4 options."
    )


def _resolve_true_false_answer(raw_answer: Any) -> str:
    if isinstance(raw_answer, bool):
        return "True" if raw_answer else "False"
    if isinstance(raw_answer, str):
        lowered = raw_answer.strip().lower().rstrip(".")
        if lowered == "true":
            return "True"
        if lowered == "false":
            return "False"
    raise QuestionValidationError(
        f"True/False correct_answer must be 'True' or 'False', got '{raw_answer}'."
    )


def validate_question_item(
    raw_item: Any,
    *,
    expected_type: str,
    expected_difficulty: str,
    allowed_chunks_by_id: dict[str, dict[str, Any]],
    document_id: str,
    filename: str,
    question_id: str = "q1",
) -> PracticeQuestion:
    """Validate one parsed LLM question dict against strict schema and chunk grounding."""
    if not isinstance(raw_item, dict):
        raise QuestionValidationError("Question entry must be a JSON object.")

    normalized_expected_type = validate_question_type(expected_type)
    normalized_difficulty = validate_difficulty(expected_difficulty)

    # Determine concrete type for this item
    raw_item_type = raw_item.get("question_type") or raw_item.get("type")
    if normalized_expected_type == "mixed":
        if not raw_item_type:
            raise QuestionValidationError("Mixed-mode question must specify 'question_type'.")
        concrete_type = validate_question_type(str(raw_item_type))
        if concrete_type not in CONCRETE_QUESTION_TYPES:
            raise QuestionValidationError(
                f"Invalid concrete question_type '{raw_item_type}' inside mixed output."
            )
    else:
        if raw_item_type is not None:
            parsed_item_type = validate_question_type(str(raw_item_type))
            if parsed_item_type != normalized_expected_type:
                raise QuestionValidationError(
                    f"Expected question_type '{normalized_expected_type}', got '{parsed_item_type}'."
                )
        concrete_type = normalized_expected_type

    # Optional difficulty check if the LLM echoed it back
    raw_diff = raw_item.get("difficulty")
    if raw_diff is not None and str(raw_diff).strip():
        parsed_diff = validate_difficulty(str(raw_diff))
        if parsed_diff != normalized_difficulty:
            raise QuestionValidationError(
                f"Expected difficulty '{normalized_difficulty}', got '{parsed_diff}'."
            )

    # Validate question stem & explanation
    stem = _clean_text(raw_item.get("question") or raw_item.get("statement"))
    if len(stem) < MIN_QUESTION_CHARS:
        raise QuestionValidationError("Question text is missing or too short.")

    explanation = _clean_text(raw_item.get("explanation"))
    if len(explanation) < MIN_EXPLANATION_CHARS:
        raise QuestionValidationError("Question explanation is missing or too short.")

    # Strict source chunk verification (derives pages deterministically; ignores LLM page guesses)
    verified_chunk_ids, verified_pages = _validate_source_chunk_ids(
        raw_item.get("source_chunk_ids"),
        allowed_chunks_by_id=allowed_chunks_by_id,
        expected_document_id=document_id,
    )

    options: list[str] | None = None
    correct_answer: str
    correct_statement: str | None = None

    if concrete_type == "mcq":
        raw_options = raw_item.get("options")
        if not isinstance(raw_options, list) or len(raw_options) != 4:
            raise QuestionValidationError("MCQ must contain exactly 4 options.")

        cleaned_options: list[str] = []
        seen_norm_options: set[str] = set()
        for opt in raw_options:
            if not isinstance(opt, str):
                raise QuestionValidationError("All MCQ options must be strings.")
            cleaned_opt = _normalize_option_text(opt)
            if not cleaned_opt:
                raise QuestionValidationError("MCQ options cannot be empty.")
            norm_key = _normalize_for_compare(cleaned_opt)
            if not norm_key or norm_key in seen_norm_options:
                raise QuestionValidationError("MCQ options must be 4 distinct non-empty choices.")
            seen_norm_options.add(norm_key)
            cleaned_options.append(cleaned_opt)

        options = cleaned_options
        correct_answer = _resolve_mcq_answer(raw_item.get("correct_answer"), cleaned_options)

    elif concrete_type == "short_answer":
        expected_ans = _clean_text(
            raw_item.get("expected_answer") or raw_item.get("correct_answer")
        )
        if not expected_ans:
            raise QuestionValidationError("Short-answer question is missing 'expected_answer'.")
        correct_answer = expected_ans

    elif concrete_type == "true_false":
        correct_answer = _resolve_true_false_answer(raw_item.get("correct_answer"))
        raw_corr_stmt = _clean_text(raw_item.get("correct_statement"))
        if correct_answer == "False":
            if len(raw_corr_stmt) < MIN_QUESTION_CHARS:
                raise QuestionValidationError(
                    "False True/False question must include a grounded 'correct_statement'."
                )
            if _normalize_for_compare(raw_corr_stmt) == _normalize_for_compare(stem):
                raise QuestionValidationError(
                    "False True/False question's 'correct_statement' must correct the false statement."
                )
            correct_statement = raw_corr_stmt
        else:
            # For True statements, preserve provided correct_statement or default to the verified true stem
            correct_statement = raw_corr_stmt if len(raw_corr_stmt) >= MIN_QUESTION_CHARS else stem

    else:  # pragma: no cover
        raise QuestionValidationError(f"Unsupported question type: {concrete_type}")

    return PracticeQuestion(
        question_id=question_id,
        question_type=concrete_type,
        difficulty=normalized_difficulty,
        question=stem,
        options=options,
        correct_answer=correct_answer,
        correct_statement=correct_statement,
        explanation=explanation,
        source_chunk_ids=verified_chunk_ids,
        source_pages=verified_pages,
        page_range_label=format_page_range(verified_pages),
        filename=filename,
        document_id=document_id,
    )


def parse_and_validate_questions(
    raw_text: str,
    *,
    expected_type: str,
    expected_difficulty: str,
    allowed_chunks_by_id: dict[str, dict[str, Any]],
    document_id: str,
    filename: str,
    max_questions: int | None = None,
) -> tuple[list[PracticeQuestion], list[str]]:
    """Parse raw LLM JSON and validate each question deterministically.

    Returns:
        (valid_questions, rejection_reasons)
    """
    payload = extract_json_payload(raw_text)
    raw_questions = payload["questions"]

    valid_questions: list[PracticeQuestion] = []
    rejections: list[str] = []

    for idx, raw_item in enumerate(raw_questions, start=1):
        if max_questions is not None and len(valid_questions) >= max_questions:
            break
        try:
            validated = validate_question_item(
                raw_item,
                expected_type=expected_type,
                expected_difficulty=expected_difficulty,
                allowed_chunks_by_id=allowed_chunks_by_id,
                document_id=document_id,
                filename=filename,
                question_id=f"q{len(valid_questions) + 1}",
            )
            valid_questions.append(validated)
        except (QuestionValidationError, ValueError) as exc:
            rejections.append(f"Item {idx}: {exc}")

    return valid_questions, rejections


def normalize_stem(text: str) -> str:
    return _normalize_for_compare(text)


def stem_token_jaccard(stem_a: str, stem_b: str) -> float:
    tokens_a = set(_WORD_RE.findall(stem_a.lower()))
    tokens_b = set(_WORD_RE.findall(stem_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union)


def is_duplicate_question(
    candidate_stem: str,
    existing_stems: list[str],
    *,
    jaccard_threshold: float = DEFAULT_JACCARD_DUPLICATE_THRESHOLD,
) -> bool:
    norm_candidate = normalize_stem(candidate_stem)
    if not norm_candidate:
        return True
    for existing in existing_stems:
        if normalize_stem(existing) == norm_candidate:
            return True
        if stem_token_jaccard(candidate_stem, existing) >= jaccard_threshold:
            return True
    return False


def deduplicate_questions(
    questions: list[PracticeQuestion],
    *,
    existing_stems: list[str] | None = None,
    jaccard_threshold: float = DEFAULT_JACCARD_DUPLICATE_THRESHOLD,
) -> list[PracticeQuestion]:
    """Filter out exact and near-duplicate questions while preserving order."""
    seen_stems = list(existing_stems) if existing_stems else []
    unique: list[PracticeQuestion] = []

    for question in questions:
        if is_duplicate_question(
            question.question,
            seen_stems,
            jaccard_threshold=jaccard_threshold,
        ):
            continue
        seen_stems.append(question.question)
        unique.append(question)

    return unique
