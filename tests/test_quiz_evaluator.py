import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pymupdf

import src.config as config
from src.generation.llm_provider import BaseLLMProvider, LLMProviderError
from src.generation.question_validation import PracticeQuestion, QuestionSetResult
from src.generation.quiz_evaluator import (
    QuizEvaluationError,
    evaluate_quiz,
    grade_mcq,
    grade_short_answer_deterministic,
    grade_short_answers_llm,
    grade_true_false,
)
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import FAISSVectorStore


class _MockGraderProvider(BaseLLMProvider):
    provider_name = "mock_groq"

    def __init__(self, response_text: str | None = None, *, fail: bool = False):
        super().__init__(model_name="mock-grader-model")
        self.response_text = response_text
        self.fail = fail
        self.calls: list[dict] = []

    def is_configured(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.fail:
            raise LLMProviderError("Simulated Groq grading timeout.")
        return self.response_text or ""


def _sample_question_set() -> QuestionSetResult:
    q1 = PracticeQuestion(
        question_id="q1",
        question_type="mcq",
        difficulty="easy",
        question="Where does photosynthesis occur in plant cells?",
        options=["Chloroplasts", "Ribosomes", "Lysosomes", "Nucleus"],
        correct_answer="Chloroplasts",
        correct_statement=None,
        explanation="Photosynthesis occurs in chloroplasts.",
        source_chunk_ids=["bio_p1_c1"],
        source_pages=[1],
        page_range_label="Page 1",
        filename="biology.pdf",
        document_id="doc_bio",
    )
    q2 = PracticeQuestion(
        question_id="q2",
        question_type="true_false",
        difficulty="medium",
        question="Mitochondria convert sunlight directly into glucose.",
        options=None,
        correct_answer="False",
        correct_statement="Chloroplasts convert sunlight into glucose, while mitochondria generate ATP.",
        explanation="Chloroplasts handle photosynthesis; mitochondria generate ATP.",
        source_chunk_ids=["bio_p1_c1", "bio_p3_c2"],
        source_pages=[1, 3],
        page_range_label="Pages 1, 3",
        filename="biology.pdf",
        document_id="doc_bio",
    )
    q3 = PracticeQuestion(
        question_id="q3",
        question_type="short_answer",
        difficulty="medium",
        question="What molecule do mitochondria generate during cellular respiration?",
        options=None,
        correct_answer="ATP using oxygen during cellular respiration",
        correct_statement=None,
        explanation="Mitochondria produce ATP via oxidative phosphorylation using oxygen.",
        source_chunk_ids=["bio_p3_c2"],
        source_pages=[3],
        page_range_label="Page 3",
        filename="biology.pdf",
        document_id="doc_bio",
    )
    q4 = PracticeQuestion(
        question_id="q4",
        question_type="short_answer",
        difficulty="hard",
        question="What two outputs are produced by photosynthesis in chloroplasts?",
        options=None,
        correct_answer="Glucose and oxygen",
        correct_statement=None,
        explanation="Photosynthesis converts sunlight into glucose and releases oxygen.",
        source_chunk_ids=["bio_p1_c1"],
        source_pages=[1],
        page_range_label="Page 1",
        filename="biology.pdf",
        document_id="doc_bio",
    )

    source_chunks = [
        {
            "text": "Photosynthesis occurs in chloroplasts and converts sunlight into glucose and oxygen.",
            "metadata": {
                "chunk_id": "bio_p1_c1",
                "document_id": "doc_bio",
                "filename": "biology.pdf",
                "page": 1,
                "page_range": "1",
                "chunk_index": 1,
                "token_count": 14,
            },
        },
        {
            "text": "Mitochondria generate ATP via cellular respiration using oxygen.",
            "metadata": {
                "chunk_id": "bio_p3_c2",
                "document_id": "doc_bio",
                "filename": "biology.pdf",
                "page": 3,
                "page_range": "3",
                "chunk_index": 2,
                "token_count": 10,
            },
        },
    ]

    return QuestionSetResult(
        questions=[q1, q2, q3, q4],
        document_id="doc_bio",
        filename="biology.pdf",
        scope="full",
        question_type="mixed",
        difficulty="medium",
        requested_count=4,
        generated_count=4,
        type_distribution={"mcq": 1, "short_answer": 2, "true_false": 1},
        pages_covered=[1, 3],
        page_range_label="Pages 1, 3",
        chunks_used=2,
        batches_processed=1,
        provider="groq",
        model="openai/gpt-oss-120b",
        source_chunks=source_chunks,
    )


class TestStage63aQuizEvaluator(unittest.TestCase):
    def test_config_defaults(self):
        self.assertEqual(config.QUIZ_SHORT_ANSWER_EVALUATOR, "llm")
        self.assertEqual(config.QUIZ_SHORT_ANSWER_CORRECT_THRESHOLD, 0.75)
        self.assertEqual(config.QUIZ_SHORT_ANSWER_PARTIAL_THRESHOLD, 0.40)

    def test_mcq_and_true_false_deterministic_grading(self):
        qset = _sample_question_set()
        mcq = qset.questions[0]
        tf = qset.questions[1]

        ev_mcq_ok = grade_mcq(mcq, "Chloroplasts")
        self.assertEqual(ev_mcq_ok.score, 1.0)
        self.assertEqual(ev_mcq_ok.status, "correct")
        self.assertEqual(ev_mcq_ok.evaluated_by, "deterministic")

        ev_mcq_letter = grade_mcq(mcq, "A")
        self.assertEqual(ev_mcq_letter.score, 1.0)

        ev_mcq_wrong = grade_mcq(mcq, "Ribosomes")
        self.assertEqual(ev_mcq_wrong.score, 0.0)
        self.assertEqual(ev_mcq_wrong.status, "incorrect")

        ev_mcq_blank = grade_mcq(mcq, "   ")
        self.assertEqual(ev_mcq_blank.score, 0.0)
        self.assertEqual(ev_mcq_blank.status, "unanswered")

        ev_tf_ok = grade_true_false(tf, "False")
        self.assertEqual(ev_tf_ok.score, 1.0)
        self.assertEqual(ev_tf_ok.status, "correct")
        self.assertIn("Chloroplasts", ev_tf_ok.correct_statement)

        ev_tf_wrong = grade_true_false(tf, "True")
        self.assertEqual(ev_tf_wrong.score, 0.0)
        self.assertEqual(ev_tf_wrong.status, "incorrect")

        ev_tf_blank = grade_true_false(tf, None)
        self.assertEqual(ev_tf_blank.score, 0.0)
        self.assertEqual(ev_tf_blank.status, "unanswered")

    def test_deterministic_short_answer_grading_partial_and_negation_safeguard(self):
        qset = _sample_question_set()
        sa_q3 = qset.questions[2]  # Expected: "ATP using oxygen during cellular respiration"
        sa_q4 = qset.questions[3]  # Expected: "Glucose and oxygen"

        # Full conceptual match
        ev_full = grade_short_answer_deterministic(
            sa_q4, "Photosynthesis produces glucose and oxygen in chloroplasts."
        )
        self.assertEqual(ev_full.score, 1.0)
        self.assertEqual(ev_full.status, "correct")

        # Partial match (captures glucose, misses oxygen)
        ev_partial = grade_short_answer_deterministic(sa_q4, "Glucose")
        self.assertEqual(ev_partial.score, 0.5)
        self.assertEqual(ev_partial.status, "partial")

        # Negation polarity reversal MUST score 0.0 even with identical content words!
        ev_negated = grade_short_answer_deterministic(
            sa_q4, "Photosynthesis does NOT produce glucose and oxygen."
        )
        self.assertEqual(ev_negated.score, 0.0)
        self.assertEqual(ev_negated.status, "incorrect")
        self.assertIn("negation", ev_negated.feedback.lower())

        # Unanswered
        ev_blank = grade_short_answer_deterministic(sa_q3, "")
        self.assertEqual(ev_blank.score, 0.0)
        self.assertEqual(ev_blank.status, "unanswered")

    def test_mixed_quiz_batched_llm_grading_partial_unanswered_and_breakdown(self):
        qset = _sample_question_set()
        answers = {
            "q1": "Chloroplasts",                     # MCQ correct -> 1.0
            "q2": "True",                             # T/F incorrect -> 0.0
            "q3": "They produce ATP.",                # Short partial -> 0.5 (via LLM)
            "q4": "   ",                              # Short unanswered -> 0.0 (deterministic)
        }
        answers_snapshot = copy.deepcopy(answers)

        # Single batched LLM call should receive ONLY answered short answer q3 (not blank q4!)
        grader_json = json.dumps(
            {
                "evaluations": [
                    {
                        "question_id": "q3",
                        "score": 0.5,
                        "feedback": "Correctly identified ATP, but omitted the role of oxygen and cellular respiration.",
                    }
                ]
            }
        )
        provider = _MockGraderProvider(grader_json)

        result = evaluate_quiz(qset, answers, evaluator_mode="llm", provider=provider)

        # Exactly 1 batched LLM call
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("q3", provider.calls[0]["user_prompt"])
        self.assertNotIn("Question ID: q4", provider.calls[0]["user_prompt"])
        self.assertIn("bio_p3_c2", provider.calls[0]["user_prompt"])

        # Totals: 1.0 + 0.0 + 0.5 + 0.0 = 1.5 / 4.0 = 37.5%
        self.assertEqual(result.earned_points, 1.5)
        self.assertEqual(result.total_points, 4.0)
        self.assertEqual(result.percentage, 37.5)
        self.assertEqual(result.correct_count, 1)
        self.assertEqual(result.partial_count, 1)
        self.assertEqual(result.incorrect_count, 1)
        self.assertEqual(result.unanswered_count, 1)
        self.assertEqual(result.provider, "mock_groq")
        self.assertEqual(result.model, "mock-grader-model")

        # Per-type breakdown
        self.assertEqual(result.type_breakdown["mcq"].earned_points, 1.0)
        self.assertEqual(result.type_breakdown["mcq"].percentage, 100.0)
        self.assertEqual(result.type_breakdown["true_false"].earned_points, 0.0)
        self.assertEqual(result.type_breakdown["true_false"].percentage, 0.0)
        self.assertEqual(result.type_breakdown["short_answer"].earned_points, 0.5)
        self.assertEqual(result.type_breakdown["short_answer"].total_points, 2.0)
        self.assertEqual(result.type_breakdown["short_answer"].percentage, 25.0)
        self.assertEqual(result.type_breakdown["short_answer"].partial_count, 1)
        self.assertEqual(result.type_breakdown["short_answer"].unanswered_count, 1)

        # Missed questions include q2, q3, q4 with preserved verified source pages & filename
        self.assertEqual([m.question_id for m in result.missed_questions], ["q2", "q3", "q4"])
        self.assertEqual(result.missed_questions[0].source_pages, [1, 3])
        self.assertEqual(result.missed_questions[0].filename, "biology.pdf")

        # Original user_answers dict preserved untouched
        self.assertEqual(answers, answers_snapshot)

    def test_batched_llm_grades_multiple_short_answers_in_one_call(self):
        qset = _sample_question_set()
        answers = {
            "q1": "Chloroplasts",
            "q2": "False",
            "q3": "Mitochondria produce ATP using oxygen.",
            "q4": "Glucose and oxygen",
        }
        grader_json = json.dumps(
            {
                "evaluations": [
                    {"question_id": "q3", "score": 1.0, "feedback": "Accurate."},
                    {"question_id": "q4", "score": 1.0, "feedback": "Complete and accurate."},
                ]
            }
        )
        provider = _MockGraderProvider(grader_json)
        result = evaluate_quiz(qset, answers, evaluator_mode="llm", provider=provider)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.earned_points, 4.0)
        self.assertEqual(result.percentage, 100.0)
        self.assertEqual(result.missed_questions, [])

    def test_llm_grader_validation_rejects_invalid_duplicate_missing_unknown_and_malformed(self):
        qset = _sample_question_set()
        answers = {"q1": "Chloroplasts", "q2": "False", "q3": "ATP", "q4": "Glucose"}

        bad_payloads = [
            "not valid json",
            '{"wrong_key": []}',
            # Missing q4
            json.dumps({"evaluations": [{"question_id": "q3", "score": 1.0, "feedback": "ok"}]}),
            # Unknown question_id q99
            json.dumps(
                {
                    "evaluations": [
                        {"question_id": "q3", "score": 1.0, "feedback": "ok"},
                        {"question_id": "q99", "score": 1.0, "feedback": "ok"},
                    ]
                }
            ),
            # Duplicate q3
            json.dumps(
                {
                    "evaluations": [
                        {"question_id": "q3", "score": 1.0, "feedback": "ok"},
                        {"question_id": "q3", "score": 0.5, "feedback": "dup"},
                    ]
                }
            ),
            # Non-discrete invalid score 0.7
            json.dumps(
                {
                    "evaluations": [
                        {"question_id": "q3", "score": 0.7, "feedback": "ok"},
                        {"question_id": "q4", "score": 1.0, "feedback": "ok"},
                    ]
                }
            ),
            # Boolean score True must be rejected
            json.dumps(
                {
                    "evaluations": [
                        {"question_id": "q3", "score": True, "feedback": "ok"},
                        {"question_id": "q4", "score": 1.0, "feedback": "ok"},
                    ]
                }
            ),
            # Empty feedback
            json.dumps(
                {
                    "evaluations": [
                        {"question_id": "q3", "score": 1.0, "feedback": "   "},
                        {"question_id": "q4", "score": 1.0, "feedback": "ok"},
                    ]
                }
            ),
        ]

        for payload in bad_payloads:
            provider = _MockGraderProvider(payload)
            with self.assertRaises(QuizEvaluationError):
                evaluate_quiz(qset, answers, evaluator_mode="llm", provider=provider)

    def test_llm_provider_failure_propagates_without_silent_fallback_and_preserves_answers(self):
        qset = _sample_question_set()
        answers = {"q1": "Chloroplasts", "q2": "False", "q3": "ATP", "q4": "Glucose"}
        answers_before = copy.deepcopy(answers)

        failing_provider = _MockGraderProvider(fail=True)
        with self.assertRaises(LLMProviderError):
            evaluate_quiz(qset, answers, evaluator_mode="llm", provider=failing_provider)

        # No silent fallback happened and answers were untouched
        self.assertEqual(len(failing_provider.calls), 1)
        self.assertEqual(answers, answers_before)

    def test_cross_document_and_unverified_source_chunks_rejected(self):
        qset_dict = _sample_question_set().to_dict()
        answers = {"q1": "Chloroplasts", "q2": "False", "q3": "ATP", "q4": "Glucose"}

        # 1. Tamper source_chunk document_id to another document
        tampered_doc = copy.deepcopy(qset_dict)
        tampered_doc["source_chunks"][0]["metadata"]["document_id"] = "other_doc"
        with self.assertRaises(QuizEvaluationError):
            evaluate_quiz(tampered_doc, answers, evaluator_mode="deterministic")

        # 2. Question cites unknown chunk ID not present in source_chunks
        tampered_cid = copy.deepcopy(qset_dict)
        tampered_cid["questions"][0]["source_chunk_ids"] = ["forged_chunk_999"]
        with self.assertRaises(QuizEvaluationError):
            evaluate_quiz(tampered_cid, answers, evaluator_mode="deterministic")

        # 3. Question belongs to another document_id
        tampered_q_doc = copy.deepcopy(qset_dict)
        tampered_q_doc["questions"][0]["document_id"] = "other_doc"
        with self.assertRaises(QuizEvaluationError):
            evaluate_quiz(tampered_q_doc, answers, evaluator_mode="deterministic")

        # 4. Hallucinated page number in question dict is overridden by verified source chunk page!
        tampered_page = copy.deepcopy(qset_dict)
        tampered_page["questions"][0]["source_pages"] = [999]
        res = evaluate_quiz(tampered_page, answers, evaluator_mode="deterministic")
        self.assertEqual(res.evaluations[0].source_pages, [1])
        self.assertEqual(res.evaluations[0].page_range_label, "Page 1")

    def test_authoritative_vector_store_verification(self):
        class _DummyEmbedder:
            def get_embedding_dimension(self):
                return 4

            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 4), dtype=np.float32)

        class _Upload:
            def __init__(self, data: bytes, name: str):
                self._data = data
                self.name = name

            def getvalue(self) -> bytes:
                return self._data

        doc = pymupdf.open()
        p = doc.new_page()
        p.insert_textbox(pymupdf.Rect(50, 50, 540, 760), "Chloroplasts perform photosynthesis.", fontsize=10)
        pdf_bytes = doc.tobytes()
        doc.close()

        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vs"
            with patch("src.retrieval.embeddings.load_embedding_model", return_value=_DummyEmbedder()):
                svc = EmbeddingService(model_name="all-MiniLM-L6-v2")
                store = FAISSVectorStore(store_dir=store_dir, embedding_service=svc)
                added = store.add_pdf(_Upload(pdf_bytes, "real.pdf"), embedding_service=svc)
                doc_id = added["document_id"]
                real_chunk = store.get_document_chunks(doc_id)[0]

                qset_dict = {
                    "document_id": doc_id,
                    "filename": "real.pdf",
                    "questions": [
                        {
                            "question_id": "q1",
                            "question_type": "mcq",
                            "difficulty": "easy",
                            "question": "Which organelle performs photosynthesis?",
                            "options": ["Chloroplasts", "Nucleus", "Ribosome", "Vacuole"],
                            "correct_answer": "Chloroplasts",
                            "correct_statement": None,
                            "explanation": "Chloroplasts perform photosynthesis.",
                            "source_chunk_ids": [real_chunk["chunk_id"]],
                            "source_pages": [999],  # Must be overwritten with verified page 1 from store!
                            "page_range_label": "Page 999",
                            "filename": "real.pdf",
                            "document_id": doc_id,
                        }
                    ],
                    "source_chunks": [],
                }
                res = evaluate_quiz(
                    qset_dict,
                    {"q1": "Chloroplasts"},
                    evaluator_mode="deterministic",
                    store=store,
                )
                self.assertEqual(res.evaluations[0].score, 1.0)
                self.assertEqual(res.evaluations[0].source_pages, [1])
                self.assertEqual(res.evaluations[0].page_range_label, "Page 1")


if __name__ == "__main__":
    unittest.main()
