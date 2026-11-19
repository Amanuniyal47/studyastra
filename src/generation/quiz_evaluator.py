"""Streamlit-independent quiz evaluation engine (Stage 6.3).

Design guarantees:
- MCQ and True/False grading are strictly deterministic.
- Every question is worth 1.0 point; awarded scores are strictly in {0.0, 0.5, 1.0}.
- Unanswered questions score 0.0 and are tracked separately (`status="unanswered"`).
- Deterministic short-answer grading normalizes text, computes content-token overlap,
  and applies a negation polarity safeguard.
- LLM short-answer grading sends ONE batched JSON request via `BaseLLMProvider`
  grounded in the question, expected answer, explanation, and verified source chunks.
- Never silently falls back to deterministic grading when LLM grading fails; preserves
  caller-supplied `user_answers` untouched.
- Strictly verifies source chunks belong to the selected document and derives
  `source_pages` and `filename` deterministically from verified metadata.
"""

from dataclasses import dataclass
import json
import logging
import re
from typing import Any

import src.config as config
from src.generation.llm_provider import BaseLLMProvider, get_llm_provider
from src.generation.question_validation import (
    CONCRETE_QUESTION_TYPES,
    PracticeQuestion,
    QuestionSetResult,
)
from src.generation.summarizer import format_page_range
from src.retrieval.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

VALID_EVALUATOR_MODES = ("llm", "deterministic")
ALLOWED_SCORES = (0.0, 0.5, 1.0)
_MCQ_LETTER_TO_INDEX = {"a": 0, "b": 1, "c": 2, "d": 3}
_OPTION_PREFIX_RE = re.compile(r"^\s*(?:[A-Da-d][\)\.\:\-]\s*)")
_WORD_RE = re.compile(r"[a-z0-9]+")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_CONTRACTIONS: dict[str, str] = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "can't": "cannot",
    "cannot": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "never": "never",
}

_NEGATION_TOKENS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "neither",
        "nor",
        "without",
        "cannot",
        "lack",
        "lacks",
        "lacking",
        "absent",
        "unable",
        "false",
    }
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "which",
        "with",
        "during",
        "using",
        "via",
        "also",
        "can",
        "do",
        "does",
        "did",
    }
)

QUIZ_GRADER_SYSTEM_PROMPT = (
    "You are an objective academic quiz grader. "
    "Grade each student's short answer strictly against the provided expected answer, "
    "explanation, and verified PDF source excerpts. "
    "Award ONLY one of three discrete scores per question: "
    "1.0 (correct: accurately captures the core concept/facts, paraphrases accepted), "
    "0.5 (partially correct: captures part of the expected answer but misses a key detail or is incomplete), "
    "0.0 (incorrect: factually wrong, contradicts the source, reverses polarity, or off-topic). "
    "Return ONLY valid JSON matching the exact schema—no extra prose."
)


class QuizEvaluationError(RuntimeError):
    """Raised when quiz inputs, source verification, or LLM grading output are invalid."""


@dataclass
class QuestionEvaluation:
    question_id: str
    question_type: str
    difficulty: str
    question: str
    user_answer: str | None
    correct_answer: str
    correct_statement: str | None
    explanation: str
    score: float
    max_points: float
    status: str
    feedback: str
    evaluated_by: str
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
            "user_answer": self.user_answer,
            "correct_answer": self.correct_answer,
            "correct_statement": self.correct_statement,
            "explanation": self.explanation,
            "score": self.score,
            "max_points": self.max_points,
            "status": self.status,
            "feedback": self.feedback,
            "evaluated_by": self.evaluated_by,
            "source_chunk_ids": list(self.source_chunk_ids),
            "source_pages": list(self.source_pages),
            "page_range_label": self.page_range_label,
            "filename": self.filename,
            "document_id": self.document_id,
        }


@dataclass
class TypeScoreBreakdown:
    question_type: str
    earned_points: float
    total_points: float
    percentage: float
    correct_count: int
    partial_count: int
    incorrect_count: int
    unanswered_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "earned_points": self.earned_points,
            "total_points": self.total_points,
            "percentage": self.percentage,
            "correct_count": self.correct_count,
            "partial_count": self.partial_count,
            "incorrect_count": self.incorrect_count,
            "unanswered_count": self.unanswered_count,
        }


@dataclass
class QuizEvaluationResult:
    document_id: str
    filename: str
    evaluations: list[QuestionEvaluation]
    earned_points: float
    total_points: float
    percentage: float
    correct_count: int
    partial_count: int
    incorrect_count: int
    unanswered_count: int
    type_breakdown: dict[str, TypeScoreBreakdown]
    short_answer_evaluator: str
    provider: str | None
    model: str | None
    missed_questions: list[QuestionEvaluation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "evaluations": [ev.to_dict() for ev in self.evaluations],
            "earned_points": self.earned_points,
            "total_points": self.total_points,
            "percentage": self.percentage,
            "correct_count": self.correct_count,
            "partial_count": self.partial_count,
            "incorrect_count": self.incorrect_count,
            "unanswered_count": self.unanswered_count,
            "type_breakdown": {
                qtype: br.to_dict() for qtype, br in self.type_breakdown.items()
            },
            "short_answer_evaluator": self.short_answer_evaluator,
            "provider": self.provider,
            "model": self.model,
            "missed_questions": [ev.to_dict() for ev in self.missed_questions],
        }


def validate_evaluator_mode(mode: str | None) -> str:
    raw = config.QUIZ_SHORT_ANSWER_EVALUATOR if mode is None else mode
    normalized = (raw or "").strip().lower()
    if normalized not in VALID_EVALUATOR_MODES:
        valid = ", ".join(VALID_EVALUATOR_MODES)
        raise ValueError(
            f"Unsupported short-answer evaluator '{raw}'. Choose one of: {valid}."
        )
    return normalized


def _normalize_question_dict(question: PracticeQuestion | dict[str, Any]) -> dict[str, Any]:
    if isinstance(question, PracticeQuestion):
        return question.to_dict()
    if isinstance(question, dict):
        return dict(question)
    raise QuizEvaluationError("Question must be a PracticeQuestion or dictionary.")


def _normalize_question_set_dict(question_set: QuestionSetResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(question_set, QuestionSetResult):
        return question_set.to_dict()
    if isinstance(question_set, dict):
        return dict(question_set)
    raise QuizEvaluationError("question_set must be a QuestionSetResult or dictionary.")


def _clean_user_answer(answer: Any) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, bool):
        return "True" if answer else "False"
    if not isinstance(answer, str):
        return None
    cleaned = " ".join(answer.split()).strip()
    return cleaned if cleaned else None


def _normalize_for_match(text: str) -> str:
    lowered = text.lower().strip()
    for contraction, expanded in _CONTRACTIONS.items():
        lowered = re.sub(rf"\b{re.escape(contraction)}\b", expanded, lowered)
    return " ".join(_WORD_RE.findall(lowered))


def _extract_tokens_and_negation(text: str) -> tuple[list[str], set[str], bool]:
    """Return (all_tokens, content_tokens, has_negation) from normalized text."""
    lowered = text.lower().strip()
    for contraction, expanded in _CONTRACTIONS.items():
        lowered = re.sub(rf"\b{re.escape(contraction)}\b", expanded, lowered)
    tokens = _WORD_RE.findall(lowered)
    has_negation = any(tok in _NEGATION_TOKENS for tok in tokens)
    content = {
        tok for tok in tokens if tok not in _STOPWORDS and tok not in _NEGATION_TOKENS
    }
    return tokens, content, has_negation


def _extract_source_chunks_map(
    *,
    expected_document_id: str,
    expected_filename: str,
    raw_source_chunks: list[dict[str, Any]] | None,
    store: FAISSVectorStore | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a strictly verified `chunk_id -> chunk_record` map for one document."""
    if store is not None:
        doc = store.get_document(expected_document_id)
        if doc is None:
            raise QuizEvaluationError(
                f"Document '{expected_document_id}' is no longer indexed in the vector store."
            )
        store_chunks = store.get_document_chunks(expected_document_id)
        verified: dict[str, dict[str, Any]] = {}
        for chunk in store_chunks:
            if chunk["document_id"] != expected_document_id:
                raise QuizEvaluationError(
                    f"Cross-document chunk '{chunk['chunk_id']}' rejected."
                )
            verified[chunk["chunk_id"]] = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "filename": doc["filename"],
                "page": int(chunk["page"]),
                "page_range": str(chunk.get("page_range", chunk["page"])),
                "chunk_index": int(chunk["chunk_index"]),
                "text": chunk["text"],
            }
        return verified

    if not isinstance(raw_source_chunks, list) or not raw_source_chunks:
        raise QuizEvaluationError(
            "Verified source_chunks are required to ground quiz evaluation."
        )

    verified = {}
    for entry in raw_source_chunks:
        if not isinstance(entry, dict):
            raise QuizEvaluationError("Invalid source_chunk entry.")
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else entry
        cid = meta.get("chunk_id")
        doc_id = meta.get("document_id")
        page = meta.get("page")
        text = entry.get("text")
        if not isinstance(cid, str) or not cid.strip():
            raise QuizEvaluationError("Source chunk is missing valid 'chunk_id'.")
        if doc_id != expected_document_id:
            raise QuizEvaluationError(
                f"Cross-document source chunk '{cid}' (doc='{doc_id}') does not belong to "
                f"selected document '{expected_document_id}'."
            )
        if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
            raise QuizEvaluationError(f"Source chunk '{cid}' has invalid page number.")
        if not isinstance(text, str) or not text.strip():
            raise QuizEvaluationError(f"Source chunk '{cid}' has empty text.")

        verified[cid.strip()] = {
            "chunk_id": cid.strip(),
            "document_id": expected_document_id,
            "filename": expected_filename,
            "page": int(page),
            "page_range": str(meta.get("page_range", page)),
            "chunk_index": int(meta.get("chunk_index", 1)),
            "text": text,
        }

    return verified


def _verify_question_grounding(
    question: dict[str, Any],
    *,
    expected_document_id: str,
    expected_filename: str,
    verified_chunks_map: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[str], list[int], str, list[dict[str, Any]]]:
    """Verify question belongs to expected document and resolve verified source chunks/pages."""
    q_doc_id = question.get("document_id")
    if q_doc_id != expected_document_id:
        raise QuizEvaluationError(
            f"Question '{question.get('question_id')}' belongs to document '{q_doc_id}', "
            f"not selected document '{expected_document_id}'."
        )

    raw_cids = question.get("source_chunk_ids")
    if not isinstance(raw_cids, list) or not raw_cids:
        raise QuizEvaluationError(
            f"Question '{question.get('question_id')}' is missing verified 'source_chunk_ids'."
        )

    cleaned_cids: list[str] = []
    seen: set[str] = set()
    for cid in raw_cids:
        if not isinstance(cid, str) or not cid.strip():
            raise QuizEvaluationError(
                f"Question '{question.get('question_id')}' contains invalid chunk ID."
            )
        c = cid.strip()
        if c not in seen:
            seen.add(c)
            cleaned_cids.append(c)

    if verified_chunks_map is not None:
        supporting_chunks: list[dict[str, Any]] = []
        derived_pages: set[int] = set()
        for cid in cleaned_cids:
            chunk = verified_chunks_map.get(cid)
            if chunk is None:
                raise QuizEvaluationError(
                    f"Question '{question.get('question_id')}' references unverified or "
                    f"cross-document chunk ID '{cid}'."
                )
            if chunk["document_id"] != expected_document_id:
                raise QuizEvaluationError(
                    f"Question '{question.get('question_id')}' references cross-document chunk '{cid}'."
                )
            supporting_chunks.append(chunk)
            derived_pages.add(int(chunk["page"]))

        pages = sorted(derived_pages)
        return cleaned_cids, pages, format_page_range(pages), supporting_chunks

    # Direct single-question helper call without external chunk map: require valid positive pages
    raw_pages = question.get("source_pages")
    if not isinstance(raw_pages, list) or not raw_pages or any(
        not isinstance(p, int) or isinstance(p, bool) or p <= 0 for p in raw_pages
    ):
        raise QuizEvaluationError(
            f"Question '{question.get('question_id')}' is missing valid source_pages."
        )
    pages = sorted({int(p) for p in raw_pages})
    return cleaned_cids, pages, format_page_range(pages), []


def _status_for_score(score: float, *, unanswered: bool = False) -> str:
    if unanswered:
        return "unanswered"
    if score == 1.0:
        return "correct"
    if score == 0.5:
        return "partial"
    return "incorrect"


def grade_mcq(
    question: PracticeQuestion | dict[str, Any],
    user_answer: str | None,
    *,
    verified_chunks_map: dict[str, dict[str, Any]] | None = None,
    expected_document_id: str | None = None,
    expected_filename: str | None = None,
) -> QuestionEvaluation:
    """Deterministically grade one MCQ (1.0 correct, 0.0 incorrect/unanswered)."""
    q = _normalize_question_dict(question)
    if q.get("question_type") != "mcq":
        raise ValueError(f"grade_mcq expects question_type='mcq', got '{q.get('question_type')}'.")

    doc_id = expected_document_id or q["document_id"]
    filename = expected_filename or q["filename"]
    cids, pages, page_label, _ = _verify_question_grounding(
        q,
        expected_document_id=doc_id,
        expected_filename=filename,
        verified_chunks_map=verified_chunks_map,
    )

    cleaned_user = _clean_user_answer(user_answer)
    correct_answer = str(q["correct_answer"]).strip()
    options = [str(opt).strip() for opt in (q.get("options") or [])]

    if cleaned_user is None:
        return QuestionEvaluation(
            question_id=str(q["question_id"]),
            question_type="mcq",
            difficulty=str(q["difficulty"]),
            question=str(q["question"]),
            user_answer=None,
            correct_answer=correct_answer,
            correct_statement=None,
            explanation=str(q["explanation"]),
            score=0.0,
            max_points=1.0,
            status="unanswered",
            feedback="No answer was selected.",
            evaluated_by="deterministic",
            source_chunk_ids=cids,
            source_pages=pages,
            page_range_label=page_label,
            filename=filename,
            document_id=doc_id,
        )

    # Resolve optional A/B/C/D selection against options
    resolved_user = _OPTION_PREFIX_RE.sub("", cleaned_user).strip()
    letter_match = re.fullmatch(r"(?:option\s+)?([a-d])[\)\.\:]?", cleaned_user.lower())
    if letter_match and len(options) == 4:
        resolved_user = options[_MCQ_LETTER_TO_INDEX[letter_match.group(1)]]

    is_correct = _normalize_for_match(resolved_user) == _normalize_for_match(correct_answer)
    score = 1.0 if is_correct else 0.0
    status = "correct" if is_correct else "incorrect"
    feedback = (
        "Correct!"
        if is_correct
        else f"Incorrect. The correct answer is '{correct_answer}'."
    )

    return QuestionEvaluation(
        question_id=str(q["question_id"]),
        question_type="mcq",
        difficulty=str(q["difficulty"]),
        question=str(q["question"]),
        user_answer=cleaned_user,
        correct_answer=correct_answer,
        correct_statement=None,
        explanation=str(q["explanation"]),
        score=score,
        max_points=1.0,
        status=status,
        feedback=feedback,
        evaluated_by="deterministic",
        source_chunk_ids=cids,
        source_pages=pages,
        page_range_label=page_label,
        filename=filename,
        document_id=doc_id,
    )


def grade_true_false(
    question: PracticeQuestion | dict[str, Any],
    user_answer: str | bool | None,
    *,
    verified_chunks_map: dict[str, dict[str, Any]] | None = None,
    expected_document_id: str | None = None,
    expected_filename: str | None = None,
) -> QuestionEvaluation:
    """Deterministically grade one True/False question (1.0 correct, 0.0 incorrect/unanswered)."""
    q = _normalize_question_dict(question)
    if q.get("question_type") != "true_false":
        raise ValueError(
            f"grade_true_false expects question_type='true_false', got '{q.get('question_type')}'."
        )

    doc_id = expected_document_id or q["document_id"]
    filename = expected_filename or q["filename"]
    cids, pages, page_label, _ = _verify_question_grounding(
        q,
        expected_document_id=doc_id,
        expected_filename=filename,
        verified_chunks_map=verified_chunks_map,
    )

    cleaned_user = _clean_user_answer(user_answer)
    correct_answer = str(q["correct_answer"]).strip().capitalize()
    correct_statement = q.get("correct_statement")

    if cleaned_user is None:
        return QuestionEvaluation(
            question_id=str(q["question_id"]),
            question_type="true_false",
            difficulty=str(q["difficulty"]),
            question=str(q["question"]),
            user_answer=None,
            correct_answer=correct_answer,
            correct_statement=correct_statement,
            explanation=str(q["explanation"]),
            score=0.0,
            max_points=1.0,
            status="unanswered",
            feedback="No answer was selected.",
            evaluated_by="deterministic",
            source_chunk_ids=cids,
            source_pages=pages,
            page_range_label=page_label,
            filename=filename,
            document_id=doc_id,
        )

    normalized_user = cleaned_user.strip().lower().rstrip(".")
    is_correct = normalized_user == correct_answer.lower()
    score = 1.0 if is_correct else 0.0
    status = "correct" if is_correct else "incorrect"
    feedback = (
        "Correct!"
        if is_correct
        else f"Incorrect. The statement is {correct_answer}."
    )

    return QuestionEvaluation(
        question_id=str(q["question_id"]),
        question_type="true_false",
        difficulty=str(q["difficulty"]),
        question=str(q["question"]),
        user_answer=cleaned_user.capitalize() if normalized_user in ("true", "false") else cleaned_user,
        correct_answer=correct_answer,
        correct_statement=correct_statement,
        explanation=str(q["explanation"]),
        score=score,
        max_points=1.0,
        status=status,
        feedback=feedback,
        evaluated_by="deterministic",
        source_chunk_ids=cids,
        source_pages=pages,
        page_range_label=page_label,
        filename=filename,
        document_id=doc_id,
    )


def grade_short_answer_deterministic(
    question: PracticeQuestion | dict[str, Any],
    user_answer: str | None,
    *,
    correct_threshold: float | None = None,
    partial_threshold: float | None = None,
    verified_chunks_map: dict[str, dict[str, Any]] | None = None,
    expected_document_id: str | None = None,
    expected_filename: str | None = None,
) -> QuestionEvaluation:
    """Deterministically grade a Short Answer using content-token overlap & negation safeguard."""
    q = _normalize_question_dict(question)
    if q.get("question_type") != "short_answer":
        raise ValueError(
            f"grade_short_answer_deterministic expects question_type='short_answer', "
            f"got '{q.get('question_type')}'."
        )

    hi = (
        config.QUIZ_SHORT_ANSWER_CORRECT_THRESHOLD
        if correct_threshold is None
        else float(correct_threshold)
    )
    lo = (
        config.QUIZ_SHORT_ANSWER_PARTIAL_THRESHOLD
        if partial_threshold is None
        else float(partial_threshold)
    )
    if not (0.0 < lo < hi <= 1.0):
        raise ValueError(
            f"Thresholds must satisfy 0 < partial ({lo}) < correct ({hi}) <= 1.0."
        )

    doc_id = expected_document_id or q["document_id"]
    filename = expected_filename or q["filename"]
    cids, pages, page_label, _ = _verify_question_grounding(
        q,
        expected_document_id=doc_id,
        expected_filename=filename,
        verified_chunks_map=verified_chunks_map,
    )

    cleaned_user = _clean_user_answer(user_answer)
    expected_answer = str(q["correct_answer"]).strip()

    if cleaned_user is None:
        return QuestionEvaluation(
            question_id=str(q["question_id"]),
            question_type="short_answer",
            difficulty=str(q["difficulty"]),
            question=str(q["question"]),
            user_answer=None,
            correct_answer=expected_answer,
            correct_statement=None,
            explanation=str(q["explanation"]),
            score=0.0,
            max_points=1.0,
            status="unanswered",
            feedback="No answer was submitted.",
            evaluated_by="deterministic",
            source_chunk_ids=cids,
            source_pages=pages,
            page_range_label=page_label,
            filename=filename,
            document_id=doc_id,
        )

    user_all, user_content, user_neg = _extract_tokens_and_negation(cleaned_user)
    exp_all, exp_content, exp_neg = _extract_tokens_and_negation(expected_answer)

    # Negation polarity safeguard: conflicting negation prevents false credit
    if user_neg != exp_neg:
        return QuestionEvaluation(
            question_id=str(q["question_id"]),
            question_type="short_answer",
            difficulty=str(q["difficulty"]),
            question=str(q["question"]),
            user_answer=cleaned_user,
            correct_answer=expected_answer,
            correct_statement=None,
            explanation=str(q["explanation"]),
            score=0.0,
            max_points=1.0,
            status="incorrect",
            feedback=(
                "Incorrect. Your response reverses the negation/polarity of the expected answer."
            ),
            evaluated_by="deterministic",
            source_chunk_ids=cids,
            source_pages=pages,
            page_range_label=page_label,
            filename=filename,
            document_id=doc_id,
        )

    # Exact normalized match
    if _normalize_for_match(cleaned_user) == _normalize_for_match(expected_answer):
        overlap = 1.0
    else:
        u_set = user_content if user_content else set(user_all)
        e_set = exp_content if exp_content else set(exp_all)
        if not u_set or not e_set:
            overlap = 0.0
        elif len(e_set) <= 3 and e_set.issubset(u_set):
            # Student included all concise key terms (e.g., full sentence containing "ATP")
            overlap = 1.0
        else:
            common = len(u_set & e_set)
            precision = common / len(u_set)
            recall = common / len(e_set)
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            # Blend F1 and recall slightly toward recall so complete explanations aren't overly penalized
            overlap = max(f1, 0.5 * f1 + 0.5 * recall) if precision >= 0.35 else f1

    if overlap >= hi:
        score = 1.0
        status = "correct"
        feedback = "Correct! Your answer matches the key concepts in the expected answer."
    elif overlap >= lo:
        score = 0.5
        status = "partial"
        feedback = (
            "Partially correct. You captured some key concepts, but missed important details "
            "from the expected answer."
        )
    else:
        score = 0.0
        status = "incorrect"
        feedback = "Incorrect. Review the expected answer and source pages below."

    return QuestionEvaluation(
        question_id=str(q["question_id"]),
        question_type="short_answer",
        difficulty=str(q["difficulty"]),
        question=str(q["question"]),
        user_answer=cleaned_user,
        correct_answer=expected_answer,
        correct_statement=None,
        explanation=str(q["explanation"]),
        score=score,
        max_points=1.0,
        status=status,
        feedback=feedback,
        evaluated_by="deterministic",
        source_chunk_ids=cids,
        source_pages=pages,
        page_range_label=page_label,
        filename=filename,
        document_id=doc_id,
    )


def _extract_grader_json(raw_text: str) -> dict[str, Any]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise QuizEvaluationError("LLM grader returned an empty response.")

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

    parsed: Any = None
    last_err: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_err = exc

    if parsed is None:
        raise QuizEvaluationError(
            f"Malformed LLM grader output: could not parse JSON ({last_err})."
        ) from last_err

    if not isinstance(parsed, dict) or not isinstance(parsed.get("evaluations"), list):
        raise QuizEvaluationError(
            "Malformed LLM grader JSON: expected top-level object with 'evaluations' list."
        )

    return parsed


def _parse_discrete_score(raw_score: Any, question_id: str) -> float:
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise QuizEvaluationError(
            f"Invalid score for '{question_id}': expected numeric 0.0, 0.5, or 1.0, got {raw_score!r}."
        )
    value = float(raw_score)
    if value not in ALLOWED_SCORES:
        raise QuizEvaluationError(
            f"Invalid score {value} for '{question_id}': must be strictly one of 0.0, 0.5, or 1.0."
        )
    return value


def _build_batched_grader_prompt(
    *,
    filename: str,
    grading_items: list[dict[str, Any]],
) -> str:
    blocks: list[str] = []
    for item in grading_items:
        chunk_lines = "\n".join(
            f"  [Chunk ID: {ch['chunk_id']} | Page {ch['page_range']}]: {ch['text'].strip()}"
            for ch in item["supporting_chunks"]
        )
        blocks.append(
            f"Question ID: {item['question_id']}\n"
            f"Question: {item['question']}\n"
            f"Expected Answer: {item['correct_answer']}\n"
            f"Explanation: {item['explanation']}\n"
            f"Verified Source Excerpts:\n{chunk_lines}\n"
            f"Student Answer: {item['user_answer']}"
        )

    joined = "\n\n---\n\n".join(blocks)
    return (
        f"Document: {filename}\n"
        f"Grade ALL {len(grading_items)} student short-answer response(s) below in ONE JSON object.\n\n"
        "Required JSON Output Format:\n"
        "{\n"
        '  "evaluations": [\n'
        "    {\n"
        '      "question_id": "<exact Question ID from below>",\n'
        '      "score": <0.0 | 0.5 | 1.0>,\n'
        '      "feedback": "<1-2 concise sentences explaining what was right or missing>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Strict Rules:\n"
        "- Include EVERY Question ID listed below exactly once.\n"
        "- Do not invent Question IDs or omit any Question ID.\n"
        "- Score MUST be strictly 0.0, 0.5, or 1.0.\n"
        "- Base grading strictly on the expected answer, explanation, and verified source excerpts.\n\n"
        f"Items to Grade:\n{joined}"
    )


def grade_short_answers_llm(
    questions: list[PracticeQuestion | dict[str, Any]],
    user_answers: dict[str, str | None],
    *,
    verified_chunks_map: dict[str, dict[str, Any]],
    document_id: str,
    filename: str,
    provider: BaseLLMProvider | None = None,
) -> dict[str, QuestionEvaluation]:
    """Grade all short-answer questions in ONE batched LLM call with strict validation.

    Unanswered short answers are scored 0.0 deterministically without calling the LLM.
    Provider errors and malformed grader responses propagate cleanly (no silent fallback).
    """
    evaluations: dict[str, QuestionEvaluation] = {}
    answered_items: list[dict[str, Any]] = []

    for raw_q in questions:
        q = _normalize_question_dict(raw_q)
        if q.get("question_type") != "short_answer":
            raise ValueError(
                f"grade_short_answers_llm only accepts short_answer questions, "
                f"got '{q.get('question_type')}'."
            )

        qid = str(q["question_id"])
        cids, pages, page_label, supporting_chunks = _verify_question_grounding(
            q,
            expected_document_id=document_id,
            expected_filename=filename,
            verified_chunks_map=verified_chunks_map,
        )
        cleaned_user = _clean_user_answer(user_answers.get(qid))

        if cleaned_user is None:
            evaluations[qid] = QuestionEvaluation(
                question_id=qid,
                question_type="short_answer",
                difficulty=str(q["difficulty"]),
                question=str(q["question"]),
                user_answer=None,
                correct_answer=str(q["correct_answer"]),
                correct_statement=None,
                explanation=str(q["explanation"]),
                score=0.0,
                max_points=1.0,
                status="unanswered",
                feedback="No answer was submitted.",
                evaluated_by="deterministic",
                source_chunk_ids=cids,
                source_pages=pages,
                page_range_label=page_label,
                filename=filename,
                document_id=document_id,
            )
            continue

        answered_items.append(
            {
                "question_id": qid,
                "difficulty": str(q["difficulty"]),
                "question": str(q["question"]),
                "correct_answer": str(q["correct_answer"]),
                "explanation": str(q["explanation"]),
                "user_answer": cleaned_user,
                "source_chunk_ids": cids,
                "source_pages": pages,
                "page_range_label": page_label,
                "supporting_chunks": supporting_chunks,
            }
        )

    if not answered_items:
        return evaluations

    active_provider = provider if provider is not None else get_llm_provider()
    prompt = _build_batched_grader_prompt(filename=filename, grading_items=answered_items)
    max_tokens = min(1400, max(250, len(answered_items) * 140 + 100))

    # LLMProviderError / LLMConfigurationError propagate cleanly — NEVER silent fallback!
    raw_output = active_provider.generate(
        QUIZ_GRADER_SYSTEM_PROMPT,
        prompt,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    payload = _extract_grader_json(raw_output)
    raw_evals = payload["evaluations"]

    expected_by_id = {item["question_id"]: item for item in answered_items}
    seen_ids: set[str] = set()

    for entry in raw_evals:
        if not isinstance(entry, dict):
            raise QuizEvaluationError("Every entry in grader 'evaluations' must be a JSON object.")

        raw_qid = entry.get("question_id")
        if not isinstance(raw_qid, str) or not raw_qid.strip():
            raise QuizEvaluationError("Grader evaluation entry is missing 'question_id'.")
        qid = raw_qid.strip()

        if qid not in expected_by_id:
            raise QuizEvaluationError(
                f"LLM grader returned unknown or unexpected question_id '{qid}'."
            )
        if qid in seen_ids:
            raise QuizEvaluationError(
                f"LLM grader returned duplicate evaluation for question_id '{qid}'."
            )
        seen_ids.add(qid)

        score = _parse_discrete_score(entry.get("score"), qid)
        feedback = entry.get("feedback")
        if not isinstance(feedback, str) or not feedback.strip():
            raise QuizEvaluationError(
                f"LLM grader returned empty feedback for question_id '{qid}'."
            )

        item = expected_by_id[qid]
        evaluations[qid] = QuestionEvaluation(
            question_id=qid,
            question_type="short_answer",
            difficulty=item["difficulty"],
            question=item["question"],
            user_answer=item["user_answer"],
            correct_answer=item["correct_answer"],
            correct_statement=None,
            explanation=item["explanation"],
            score=score,
            max_points=1.0,
            status=_status_for_score(score),
            feedback=" ".join(feedback.split()).strip(),
            evaluated_by="llm",
            source_chunk_ids=item["source_chunk_ids"],
            source_pages=item["source_pages"],
            page_range_label=item["page_range_label"],
            filename=filename,
            document_id=document_id,
        )

    missing_ids = sorted(set(expected_by_id.keys()) - seen_ids)
    if missing_ids:
        raise QuizEvaluationError(
            f"LLM grader omitted evaluation(s) for question_id(s): {', '.join(missing_ids)}."
        )

    return evaluations


def evaluate_quiz(
    question_set: QuestionSetResult | dict[str, Any],
    user_answers: dict[str, Any],
    *,
    evaluator_mode: str | None = None,
    provider: BaseLLMProvider | None = None,
    store: FAISSVectorStore | None = None,
    correct_threshold: float | None = None,
    partial_threshold: float | None = None,
) -> QuizEvaluationResult:
    """Evaluate all submitted quiz answers for a generated QuestionSetResult.

    Does not mutate `user_answers`, so submitted answers remain preserved if
    evaluation fails.
    """
    if not isinstance(user_answers, dict):
        raise ValueError("user_answers must be a dictionary keyed by question_id.")

    mode = validate_evaluator_mode(evaluator_mode)
    qset = _normalize_question_set_dict(question_set)

    document_id = qset.get("document_id")
    filename = qset.get("filename")
    if not isinstance(document_id, str) or not document_id.strip():
        raise QuizEvaluationError("question_set is missing valid 'document_id'.")
    if not isinstance(filename, str) or not filename.strip():
        raise QuizEvaluationError("question_set is missing valid 'filename'.")

    raw_questions = qset.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise QuizEvaluationError("question_set contains no questions to evaluate.")

    verified_chunks_map = _extract_source_chunks_map(
        expected_document_id=document_id,
        expected_filename=filename,
        raw_source_chunks=qset.get("source_chunks"),
        store=store,
    )

    normalized_questions = [_normalize_question_dict(q) for q in raw_questions]

    # Pre-validate ALL questions & source chunks BEFORE making any LLM API call
    seen_qids: set[str] = set()
    for q in normalized_questions:
        qid = q.get("question_id")
        if not isinstance(qid, str) or not qid.strip():
            raise QuizEvaluationError("Every question must have a non-empty 'question_id'.")
        if qid in seen_qids:
            raise QuizEvaluationError(f"Duplicate question_id '{qid}' in question_set.")
        seen_qids.add(qid)
        _verify_question_grounding(
            q,
            expected_document_id=document_id,
            expected_filename=filename,
            verified_chunks_map=verified_chunks_map,
        )

    evals_by_id: dict[str, QuestionEvaluation] = {}
    short_answer_questions: list[dict[str, Any]] = []

    for q in normalized_questions:
        qid = str(q["question_id"])
        qtype = q.get("question_type")
        if qtype == "mcq":
            evals_by_id[qid] = grade_mcq(
                q,
                user_answers.get(qid),
                verified_chunks_map=verified_chunks_map,
                expected_document_id=document_id,
                expected_filename=filename,
            )
        elif qtype == "true_false":
            evals_by_id[qid] = grade_true_false(
                q,
                user_answers.get(qid),
                verified_chunks_map=verified_chunks_map,
                expected_document_id=document_id,
                expected_filename=filename,
            )
        elif qtype == "short_answer":
            short_answer_questions.append(q)
        else:
            raise QuizEvaluationError(f"Unsupported question_type '{qtype}' in question_set.")

    used_provider_name: str | None = None
    used_model_name: str | None = None

    if short_answer_questions:
        if mode == "deterministic":
            for q in short_answer_questions:
                qid = str(q["question_id"])
                evals_by_id[qid] = grade_short_answer_deterministic(
                    q,
                    user_answers.get(qid),
                    correct_threshold=correct_threshold,
                    partial_threshold=partial_threshold,
                    verified_chunks_map=verified_chunks_map,
                    expected_document_id=document_id,
                    expected_filename=filename,
                )
        else:
            has_any_answered_short = any(
                _clean_user_answer(user_answers.get(str(q["question_id"]))) is not None
                for q in short_answer_questions
            )
            active_provider = (
                (provider if provider is not None else get_llm_provider())
                if has_any_answered_short
                else provider
            )
            short_evals = grade_short_answers_llm(
                short_answer_questions,
                user_answers,
                verified_chunks_map=verified_chunks_map,
                document_id=document_id,
                filename=filename,
                provider=active_provider,
            )
            evals_by_id.update(short_evals)
            if has_any_answered_short and active_provider is not None:
                used_provider_name = active_provider.provider_name
                used_model_name = active_provider.model_name

    # Reconstruct evaluations in original question order
    ordered_evals = [evals_by_id[str(q["question_id"])] for q in normalized_questions]

    earned_points = round(sum(ev.score for ev in ordered_evals), 2)
    total_points = float(len(ordered_evals))
    percentage = round((earned_points / total_points) * 100.0, 1) if total_points > 0 else 0.0

    correct_count = sum(1 for ev in ordered_evals if ev.status == "correct")
    partial_count = sum(1 for ev in ordered_evals if ev.status == "partial")
    incorrect_count = sum(1 for ev in ordered_evals if ev.status == "incorrect")
    unanswered_count = sum(1 for ev in ordered_evals if ev.status == "unanswered")

    # Deterministic per-type breakdown in canonical order
    type_breakdown: dict[str, TypeScoreBreakdown] = {}
    for qtype in CONCRETE_QUESTION_TYPES:
        subset = [ev for ev in ordered_evals if ev.question_type == qtype]
        if not subset:
            continue
        t_earned = round(sum(ev.score for ev in subset), 2)
        t_total = float(len(subset))
        type_breakdown[qtype] = TypeScoreBreakdown(
            question_type=qtype,
            earned_points=t_earned,
            total_points=t_total,
            percentage=round((t_earned / t_total) * 100.0, 1),
            correct_count=sum(1 for ev in subset if ev.status == "correct"),
            partial_count=sum(1 for ev in subset if ev.status == "partial"),
            incorrect_count=sum(1 for ev in subset if ev.status == "incorrect"),
            unanswered_count=sum(1 for ev in subset if ev.status == "unanswered"),
        )

    missed_questions = [ev for ev in ordered_evals if ev.score < 1.0]

    return QuizEvaluationResult(
        document_id=document_id,
        filename=filename,
        evaluations=ordered_evals,
        earned_points=earned_points,
        total_points=total_points,
        percentage=percentage,
        correct_count=correct_count,
        partial_count=partial_count,
        incorrect_count=incorrect_count,
        unanswered_count=unanswered_count,
        type_breakdown=type_breakdown,
        short_answer_evaluator=mode,
        provider=used_provider_name,
        model=used_model_name,
        missed_questions=missed_questions,
    )
