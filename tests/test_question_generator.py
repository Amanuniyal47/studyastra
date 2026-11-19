import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pymupdf

import src.config as config
from src.generation.llm_provider import BaseLLMProvider, LLMProviderError
from src.generation.question_generator import (
    generate_document_questions,
    generate_page_range_questions,
    generate_practice_questions,
    generate_topic_questions,
)
from src.generation.question_validation import QuestionGenerationError
from src.ingestion.chunker import estimate_tokens
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import FAISSVectorStore


class _NamedUpload:
    def __init__(self, data: bytes, name: str):
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data


def _make_pdf(pages: list[str], name: str) -> _NamedUpload:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(45, 45, 550, 790), text, fontsize=9)
    raw = doc.tobytes()
    doc.close()
    return _NamedUpload(raw, name=name)


class _TopicEmbedder:
    TOPICS = (
        ("photosynthesis", "chloroplast", "glucose", "sunlight", "plant"),
        ("mitochondria", "respiration", "atp", "cellular", "oxygen"),
        ("faiss", "vector", "embedding", "index", "similarity"),
        ("history", "treaty", "empire", "revolution", "monarchy"),
    )

    def get_embedding_dimension(self) -> int:
        return 5

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ):
        rows = []
        for text in texts:
            lower = text.lower()
            vec = np.zeros(5, dtype=np.float32)
            for idx, keywords in enumerate(self.TOPICS):
                vec[idx] = float(sum(1 for kw in keywords if kw in lower))
            vec[4] = 0.05
            rows.append(vec)
        return np.array(rows, dtype=np.float32)


def _extract_chunk_ids_from_prompt(user_prompt: str) -> list[str]:
    return re.findall(r"\[Chunk ID: ([^|\]]+) \| Source:", user_prompt)


def _extract_requested_batch_plan(user_prompt: str) -> list[str]:
    mixed_match = re.search(r"distributed as: ([^.\n]+)\.", user_prompt)
    if mixed_match:
        plan: list[str] = []
        for part in mixed_match.group(1).split(","):
            m = re.match(r"\s*(\d+)\s+(mcq|short_answer|true_false)\s*", part)
            if m:
                plan.extend([m.group(2)] * int(m.group(1)))
        return plan

    single_match = re.search(r"Generate exactly (\d+) '(mcq|short_answer|true_false)'", user_prompt)
    if single_match:
        return [single_match.group(2)] * int(single_match.group(1))
    return ["mcq"]


def _extract_difficulty_from_prompt(user_prompt: str) -> str:
    m = re.search(r"Difficulty:\s*(Easy|Medium|Hard):", user_prompt)
    return m.group(1).lower() if m else "medium"


class _DynamicMockQuestionProvider(BaseLLMProvider):
    provider_name = "mock_groq"

    def __init__(
        self,
        model_name: str = "mock-question-model",
        *,
        custom_handler=None,
        fail_on_call: int | None = None,
    ):
        super().__init__(model_name=model_name)
        self.calls: list[dict] = []
        self.custom_handler = custom_handler
        self.fail_on_call = fail_on_call
        self._counter = 0

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
        call_num = len(self.calls) + 1
        total_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        self.calls.append(
            {
                "call_num": call_num,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "total_prompt_tokens": total_tokens,
            }
        )

        if self.fail_on_call == call_num:
            raise LLMProviderError(f"Simulated Groq API error on call {call_num}")

        if self.custom_handler is not None:
            return self.custom_handler(call_num, system_prompt, user_prompt, max_tokens)

        chunk_ids = [cid.strip() for cid in _extract_chunk_ids_from_prompt(user_prompt)]
        plan = _extract_requested_batch_plan(user_prompt)
        difficulty = _extract_difficulty_from_prompt(user_prompt)

        distinct_stems = [
            "chloroplast sunlight conversion pathway",
            "mitochondrial oxygen phosphorylation mechanism",
            "cytoplasmic glucose breakdown stage",
            "daughter cell nuclear division fidelity",
            "dense embedding similarity indexing",
            "monotonic vector identifier persistence",
            "krebs cycle enzymatic intermediate step",
            "cellular energy homeostasis regulation",
            "transactional manifest atomic switching",
            "chromosomal replication checkpoint control",
            "metabolic electron transport gradient",
            "biochemical substrate catalysis kinetics",
            "thylakoid membrane pigment absorption",
            "oxidative adenosine triphosphate yield",
        ]

        items = []
        for idx, qtype in enumerate(plan):
            self._counter += 1
            cid = chunk_ids[idx % len(chunk_ids)]
            uid = self._counter
            phrase = distinct_stems[(uid - 1) % len(distinct_stems)]
            if qtype == "mcq":
                items.append(
                    {
                        "question_type": "mcq",
                        "difficulty": difficulty,
                        "question": f"Regarding {phrase} (#{uid}), what outcome occurs in {cid}?",
                        "options": [
                            f"Primary mechanism {uid}",
                            f"Unrelated pathway {uid}",
                            f"Secondary distractor {uid}",
                            f"Incorrect organelle {uid}",
                        ],
                        "correct_answer": f"Primary mechanism {uid}",
                        "explanation": f"Directly supported by source chunk {cid}.",
                        "source_chunk_ids": [cid],
                        "source_pages": [999],  # Hallucinated page must be ignored!
                    }
                )
            elif qtype == "short_answer":
                items.append(
                    {
                        "question_type": "short_answer",
                        "difficulty": difficulty,
                        "question": f"Describe how {phrase} (#{uid}) operates according to {cid}.",
                        "expected_answer": f"Grounded expected answer {uid}.",
                        "explanation": f"Excerpt {cid} defines {phrase}.",
                        "source_chunk_ids": [cid],
                    }
                )
            else:
                items.append(
                    {
                        "question_type": "true_false",
                        "difficulty": difficulty,
                        "question": f"During {phrase} (#{uid}), {cid} reports an inverted effect.",
                        "correct_answer": "False",
                        "correct_statement": f"During {phrase} (#{uid}), {cid} reports the accurate forward effect.",
                        "explanation": f"Excerpt {cid} clarifies {phrase}.",
                        "source_chunk_ids": [cid],
                    }
                )

        return json.dumps({"questions": items})


class TestStage62bQuestionGenerator(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store_dir = Path(self._tmpdir.name) / "vector_store"
        self.embedder = _TopicEmbedder()
        self._patcher = patch(
            "src.retrieval.embeddings.load_embedding_model",
            return_value=self.embedder,
        )
        self._patcher.start()
        self.service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        self.store = FAISSVectorStore(store_dir=self.store_dir, embedding_service=self.service)

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def _index_two_pdfs(self) -> tuple[str, str]:
        pdf_bio = _make_pdf(
            [
                "Page 1: Photosynthesis occurs inside chloroplasts converting sunlight into glucose.",
                "Page 2: Mitochondria generate ATP via cellular respiration using oxygen.",
                "Page 3: Glycolysis breaks down glucose in the cytoplasm prior to the Krebs cycle.",
                "Page 4: Mitosis produces two genetically identical daughter cells.",
            ],
            name="biology.pdf",
        )
        pdf_cs = _make_pdf(
            [
                "Page 1: FAISS performs fast dense vector similarity search over embeddings.",
                "Page 2: IndexIDMap2 maps stable int64 IDs to vectors for safe deletion.",
            ],
            name="faiss_guide.pdf",
        )
        r_bio = self.store.add_pdf(pdf_bio, embedding_service=self.service)
        r_cs = self.store.add_pdf(pdf_cs, embedding_service=self.service)
        return r_bio["document_id"], r_cs["document_id"]

    def test_all_question_types_and_difficulties_with_verified_chunk_pages_and_no_faiss_mutation(self):
        bio_id, _ = self._index_two_pdfs()
        vectors_before = self.store.total_vectors
        docs_before = self.store.total_documents

        for qtype in ("mcq", "short_answer", "true_false"):
            for diff in ("easy", "medium", "hard"):
                provider = _DynamicMockQuestionProvider()
                result = generate_document_questions(
                    self.store,
                    bio_id,
                    question_type=qtype,
                    difficulty=diff,
                    count=3,
                    provider=provider,
                )
                self.assertEqual(result.generated_count, 3)
                self.assertEqual(result.requested_count, 3)
                self.assertEqual(result.filename, "biology.pdf")
                self.assertEqual(result.document_id, bio_id)
                self.assertEqual(result.difficulty, diff)
                self.assertEqual(result.type_distribution[qtype], 3)
                self.assertIsNone(result.notice)

                for q in result.questions:
                    self.assertEqual(q.question_type, qtype)
                    self.assertEqual(q.difficulty, diff)
                    self.assertTrue(all(1 <= p <= 4 for p in q.source_pages))
                    self.assertNotIn(999, q.source_pages)
                    self.assertEqual(q.filename, "biology.pdf")

                # Never mixes content from faiss_guide.pdf
                self.assertNotIn("faiss_guide.pdf", provider.calls[0]["user_prompt"])

        # Generated questions never mutate FAISS
        self.assertEqual(self.store.total_vectors, vectors_before)
        self.assertEqual(self.store.total_documents, docs_before)

    def test_mixed_mode_balanced_distribution_10_questions(self):
        bio_id, _ = self._index_two_pdfs()
        provider = _DynamicMockQuestionProvider()

        result = generate_document_questions(
            self.store,
            bio_id,
            question_type="mixed",
            difficulty="medium",
            count=10,
            provider=provider,
        )
        self.assertEqual(result.generated_count, 10)
        self.assertEqual(result.question_type, "mixed")
        self.assertEqual(
            result.type_distribution,
            {"mcq": 4, "short_answer": 3, "true_false": 3},
        )
        self.assertEqual([q.question_id for q in result.questions], [f"q{i}" for i in range(1, 11)])

    def test_page_range_and_topic_scoping_strictly_isolated(self):
        bio_id, cs_id = self._index_two_pdfs()
        provider_pr = _DynamicMockQuestionProvider()

        res_pr = generate_page_range_questions(
            self.store,
            bio_id,
            start_page=2,
            end_page=3,
            question_type="short_answer",
            difficulty="medium",
            count=2,
            provider=provider_pr,
        )
        self.assertEqual(res_pr.scope, "page_range")
        self.assertEqual(res_pr.generated_count, 2)
        self.assertTrue(set(res_pr.pages_covered).issubset({2, 3}))
        self.assertNotIn("Page 1:", provider_pr.calls[0]["user_prompt"])
        self.assertNotIn("Page 4:", provider_pr.calls[0]["user_prompt"])

        with self.assertRaises(ValueError):
            generate_page_range_questions(
                self.store, bio_id, start_page=3, end_page=1, provider=provider_pr
            )
        with self.assertRaises(ValueError):
            generate_page_range_questions(
                self.store, bio_id, start_page=10, end_page=12, provider=provider_pr
            )

        # Topic scope against biology.pdf for a FAISS query must NOT pull chunks from faiss_guide.pdf
        provider_topic = _DynamicMockQuestionProvider()
        res_topic_bio = generate_topic_questions(
            self.store,
            bio_id,
            "FAISS vector embedding similarity",
            question_type="mcq",
            difficulty="easy",
            count=2,
            top_k=2,
            embedding_service=self.service,
            provider=provider_topic,
        )
        self.assertEqual(res_topic_bio.filename, "biology.pdf")
        self.assertNotIn("faiss_guide.pdf", provider_topic.calls[0]["user_prompt"])
        self.assertIsNotNone(res_topic_bio.notice)  # Low semantic similarity warning

        # Topic scope against faiss_guide.pdf succeeds cleanly with high relevance
        provider_cs = _DynamicMockQuestionProvider()
        res_topic_cs = generate_topic_questions(
            self.store,
            cs_id,
            "FAISS vector similarity",
            question_type="mcq",
            difficulty="easy",
            count=2,
            top_k=2,
            embedding_service=self.service,
            provider=provider_cs,
        )
        self.assertEqual(res_topic_cs.filename, "faiss_guide.pdf")
        self.assertIsNone(res_topic_cs.notice)

    def test_large_pdf_token_budget_enforced_evenly_spaced_and_bounded_previous_stems(self):
        pages = [
            f"Page {p}: "
            + " ".join(
                f"Fact {p}.{i}: biological mechanism {p}_{i} regulates energy transfer in cells."
                for i in range(1, 28)
            )
            for p in range(1, 13)
        ]
        added = self.store.add_pdf(_make_pdf(pages, name="large_bio.pdf"), embedding_service=self.service)
        doc_id = added["document_id"]

        budget = 950
        provider = _DynamicMockQuestionProvider()
        progress_events: list[tuple[int, int, str]] = []

        result = generate_document_questions(
            self.store,
            doc_id,
            question_type="mixed",
            difficulty="hard",
            count=6,
            provider=provider,
            max_batch_tokens=budget,
            progress_callback=lambda step, total, msg: progress_events.append((step, total, msg)),
        )

        self.assertEqual(result.generated_count, 6)
        self.assertEqual(result.type_distribution, {"mcq": 2, "short_answer": 2, "true_false": 2})
        self.assertGreater(result.batches_processed, 1)
        # Evenly spaced batches should span from early page 1 through late page 12
        self.assertIn(1, result.pages_covered)
        self.assertIn(12, result.pages_covered)

        # CRITICAL: Every single LLM call MUST strictly satisfy total_prompt_tokens <= budget
        for call in provider.calls:
            self.assertLessEqual(
                call["total_prompt_tokens"],
                budget,
                f"Call {call['call_num']} exceeded token budget: "
                f"{call['total_prompt_tokens']} > {budget}",
            )
            self.assertLessEqual(call["max_tokens"], config.QUESTION_MAX_OUTPUT_TOKENS)

        # Progress events reported monotonically
        self.assertGreaterEqual(len(progress_events), result.batches_processed)
        self.assertTrue(all(step <= total for step, total, _ in progress_events))

    def test_duplicate_filtering_bounded_topup_and_shortfall_notice(self):
        bio_id, _ = self._index_two_pdfs()
        bio_chunks = self.store.get_document_chunks(bio_id)
        cid1 = bio_chunks[0]["chunk_id"]
        cid2 = bio_chunks[1]["chunk_id"]

        # Call 1 returns 2 valid + 1 duplicate; Top-up call 2 returns only 1 new valid + 1 duplicate
        # -> Total 3 valid out of 5 requested -> returns 3 with clear shortfall notice (never > 5, never weak!)
        def _shortfall_handler(call_num, sys_prompt, user_prompt, max_tokens):
            if call_num == 1:
                return json.dumps(
                    {
                        "questions": [
                            {
                                "question_type": "short_answer",
                                "difficulty": "easy",
                                "question": "What organelle performs photosynthesis in plants?",
                                "expected_answer": "Chloroplasts",
                                "explanation": "Chloroplasts convert sunlight into glucose.",
                                "source_chunk_ids": [cid1],
                            },
                            {
                                # Exact duplicate of item 1
                                "question_type": "short_answer",
                                "difficulty": "easy",
                                "question": "What organelle performs photosynthesis in plants?",
                                "expected_answer": "Chloroplasts",
                                "explanation": "Duplicate question.",
                                "source_chunk_ids": [cid1],
                            },
                            {
                                "question_type": "short_answer",
                                "difficulty": "easy",
                                "question": "What energy molecule do mitochondria generate?",
                                "expected_answer": "ATP",
                                "explanation": "Mitochondria produce ATP using oxygen.",
                                "source_chunk_ids": [cid2],
                            },
                        ]
                    }
                )
            return json.dumps(
                {
                    "questions": [
                        {
                            "question_type": "short_answer",
                            "difficulty": "easy",
                            "question": "Which input does photosynthesis convert into glucose?",
                            "expected_answer": "Sunlight",
                            "explanation": "Sunlight is converted into glucose in chloroplasts.",
                            "source_chunk_ids": [cid1],
                        },
                        {
                            # Duplicate of earlier stem -> dropped
                            "question_type": "short_answer",
                            "difficulty": "easy",
                            "question": "What energy molecule do mitochondria generate?",
                            "expected_answer": "ATP",
                            "explanation": "Duplicate.",
                            "source_chunk_ids": [cid2],
                        },
                    ]
                }
            )

        provider = _DynamicMockQuestionProvider(custom_handler=_shortfall_handler)
        result = generate_document_questions(
            self.store,
            bio_id,
            question_type="short_answer",
            difficulty="easy",
            count=5,
            provider=provider,
            max_retries=0,
        )

        self.assertEqual(result.requested_count, 5)
        self.assertEqual(result.generated_count, 3)
        self.assertEqual(len(result.questions), 3)
        self.assertEqual(result.batches_processed, 2)  # 1 main + 1 bounded top-up
        self.assertIsNotNone(result.notice)
        self.assertIn("Requested 5", result.notice)
        self.assertIn("yielded 3", result.notice)

    def test_malformed_json_retry_succeeds_and_persistent_malformed_raises(self):
        bio_id, _ = self._index_two_pdfs()
        cid = self.store.get_document_chunks(bio_id)[0]["chunk_id"]

        # Attempt 1 malformed, attempt 2 valid JSON -> succeeds within retry limit
        def _retry_then_ok(call_num, sys_prompt, user_prompt, max_tokens):
            if call_num == 1:
                return "Here are some questions (not JSON!)"
            self.assertIn("IMPORTANT RETRY NOTICE", user_prompt)
            return json.dumps(
                {
                    "questions": [
                        {
                            "question_type": "short_answer",
                            "difficulty": "easy",
                            "question": "What product is created by photosynthesis in chloroplasts?",
                            "expected_answer": "Glucose",
                            "explanation": "Photosynthesis converts sunlight into glucose.",
                            "source_chunk_ids": [cid],
                        }
                    ]
                }
            )

        provider_ok = _DynamicMockQuestionProvider(custom_handler=_retry_then_ok)
        res = generate_document_questions(
            self.store,
            bio_id,
            question_type="short_answer",
            difficulty="easy",
            count=1,
            provider=provider_ok,
            max_retries=1,
        )
        self.assertEqual(res.generated_count, 1)
        self.assertEqual(len(provider_ok.calls), 2)

        # Persistent malformed JSON raises QuestionGenerationError after max_retries
        provider_bad = _DynamicMockQuestionProvider(
            custom_handler=lambda *args: "completely broken output"
        )
        with self.assertRaises(QuestionGenerationError):
            generate_document_questions(
                self.store,
                bio_id,
                question_type="short_answer",
                difficulty="easy",
                count=1,
                provider=provider_bad,
                max_retries=1,
            )
        self.assertEqual(len(provider_bad.calls), 2)

    def test_llm_api_failure_mid_batch_propagates_cleanly_without_partial_result(self):
        pages = [
            f"Page {p}: " + " ".join(f"Biology concept {p}.{i} detail." for i in range(35))
            for p in range(1, 7)
        ]
        added = self.store.add_pdf(_make_pdf(pages, name="multi_batch.pdf"), embedding_service=self.service)
        doc_id = added["document_id"]
        vectors_before = self.store.total_vectors

        failing_provider = _DynamicMockQuestionProvider(fail_on_call=2)
        with self.assertRaises(LLMProviderError):
            generate_document_questions(
                self.store,
                doc_id,
                question_type="mcq",
                difficulty="medium",
                count=4,
                provider=failing_provider,
                max_batch_tokens=650,
            )

        self.assertEqual(len(failing_provider.calls), 2)
        self.assertEqual(self.store.total_vectors, vectors_before)

    def test_invalid_inputs_and_tiny_budget_rejected_before_calling_llm(self):
        bio_id, _ = self._index_two_pdfs()
        provider = _DynamicMockQuestionProvider()

        with self.assertRaises(ValueError):
            generate_document_questions(self.store, bio_id, count=0, provider=provider)
        with self.assertRaises(ValueError):
            generate_document_questions(
                self.store, bio_id, count=config.QUESTION_MAX_COUNT + 1, provider=provider
            )
        with self.assertRaises(ValueError):
            generate_document_questions(
                self.store, bio_id, count=2, provider=provider, max_batch_tokens=50
            )
        with self.assertRaises(KeyError):
            generate_document_questions(self.store, "missing_doc_id", count=2, provider=provider)
        with self.assertRaises(ValueError):
            generate_practice_questions(self.store, bio_id, scope="invalid", provider=provider)

        self.assertEqual(len(provider.calls), 0)


if __name__ == "__main__":
    unittest.main()
