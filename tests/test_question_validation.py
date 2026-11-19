import json
import unittest

import src.config as config
from src.generation.question_prompts import (
    QUESTION_SYSTEM_PROMPT,
    build_question_generation_prompt,
    estimate_output_token_allowance,
    format_previous_stems_block,
    format_question_chunk_entry,
)
from src.generation.question_validation import (
    QuestionGenerationError,
    QuestionValidationError,
    allocate_mixed_counts,
    compute_type_distribution,
    deduplicate_questions,
    extract_json_payload,
    is_duplicate_question,
    parse_and_validate_questions,
    validate_difficulty,
    validate_question_item,
    validate_question_type,
)


def _sample_allowed_chunks() -> dict[str, dict]:
    return {
        "bio123456789_p1_c1": {
            "chunk_id": "bio123456789_p1_c1",
            "document_id": "bio_doc_id",
            "filename": "biology.pdf",
            "page": 1,
            "page_range": "1",
            "chunk_index": 1,
            "text": "Photosynthesis occurs in chloroplasts and converts sunlight into glucose.",
        },
        "bio123456789_p3_c2": {
            "chunk_id": "bio123456789_p3_c2",
            "document_id": "bio_doc_id",
            "filename": "biology.pdf",
            "page": 3,
            "page_range": "3",
            "chunk_index": 2,
            "text": "Mitochondria generate ATP via cellular respiration using oxygen.",
        },
        "other9999999_p2_c1": {
            "chunk_id": "other9999999_p2_c1",
            "document_id": "other_doc_id",
            "filename": "history.pdf",
            "page": 2,
            "page_range": "2",
            "chunk_index": 1,
            "text": "Cross-document chunk that must always be rejected for bio_doc_id.",
        },
    }


class TestStage62aConfigAndAllocation(unittest.TestCase):
    def test_stage62_config_defaults(self):
        self.assertGreater(config.QUESTION_MAX_BATCH_TOKENS, 500)
        self.assertGreater(config.QUESTION_DEFAULT_COUNT, 0)
        self.assertGreaterEqual(config.QUESTION_MAX_COUNT, config.QUESTION_DEFAULT_COUNT)
        self.assertGreater(config.QUESTION_TOPIC_TOP_K, 0)
        self.assertGreaterEqual(config.QUESTION_MAX_RETRIES, 0)
        self.assertGreater(config.QUESTION_MAX_OUTPUT_TOKENS, 200)
        self.assertGreater(config.QUESTION_MAX_PREVIOUS_STEMS, 0)

    def test_allocate_mixed_counts_balanced(self):
        self.assertEqual(allocate_mixed_counts(10), {"mcq": 4, "short_answer": 3, "true_false": 3})
        self.assertEqual(allocate_mixed_counts(9), {"mcq": 3, "short_answer": 3, "true_false": 3})
        self.assertEqual(allocate_mixed_counts(1), {"mcq": 1, "short_answer": 0, "true_false": 0})
        self.assertEqual(allocate_mixed_counts(2), {"mcq": 1, "short_answer": 1, "true_false": 0})
        with self.assertRaises(ValueError):
            allocate_mixed_counts(0)

    def test_validate_type_and_difficulty_aliases(self):
        self.assertEqual(validate_question_type("MCQ"), "mcq")
        self.assertEqual(validate_question_type("Short Answer"), "short_answer")
        self.assertEqual(validate_question_type("True / False"), "true_false")
        self.assertEqual(validate_question_type("Mixed"), "mixed")
        self.assertEqual(validate_difficulty("Medium"), "medium")
        with self.assertRaises(ValueError):
            validate_question_type("essay")
        with self.assertRaises(ValueError):
            validate_difficulty("impossible")


class TestStage62aJsonExtractionAndValidation(unittest.TestCase):
    def setUp(self):
        self.allowed = _sample_allowed_chunks()

    def test_extract_json_handles_plain_fenced_and_preamble(self):
        plain = json.dumps({"questions": [{"question": "Q1"}]})
        self.assertEqual(len(extract_json_payload(plain)["questions"]), 1)

        fenced = f"Here is the JSON:\n```json\n{plain}\n```"
        self.assertEqual(len(extract_json_payload(fenced)["questions"]), 1)

        with self.assertRaises(QuestionGenerationError):
            extract_json_payload("not json at all")
        with self.assertRaises(QuestionGenerationError):
            extract_json_payload('{"wrong_key": []}')

    def test_mcq_validation_and_chunk_derived_pages_ignore_llm_page_hallucination(self):
        item = {
            "question_type": "mcq",
            "difficulty": "easy",
            "question": "Where does photosynthesis occur in plant cells?",
            "options": [
                "A) Chloroplasts",
                "B) Ribosomes",
                "C) Lysosomes",
                "D) Golgi apparatus",
            ],
            "correct_answer": "A",
            "explanation": "Photosynthesis occurs in chloroplasts and converts sunlight into glucose.",
            "source_chunk_ids": ["bio123456789_p1_c1"],
            # Deliberately bogus LLM-generated page/filename fields that MUST be ignored
            "source_pages": [999],
            "filename": "fake.pdf",
        }
        q = validate_question_item(
            item,
            expected_type="mcq",
            expected_difficulty="easy",
            allowed_chunks_by_id=self.allowed,
            document_id="bio_doc_id",
            filename="biology.pdf",
        )
        self.assertEqual(q.question_type, "mcq")
        self.assertEqual(q.options, ["Chloroplasts", "Ribosomes", "Lysosomes", "Golgi apparatus"])
        self.assertEqual(q.correct_answer, "Chloroplasts")
        self.assertEqual(q.source_chunk_ids, ["bio123456789_p1_c1"])
        # Derived strictly from verified chunk (page 1), NOT hallucinated page 999
        self.assertEqual(q.source_pages, [1])
        self.assertEqual(q.page_range_label, "Page 1")
        self.assertEqual(q.filename, "biology.pdf")
        self.assertEqual(q.document_id, "bio_doc_id")

    def test_mcq_rejects_bad_options_duplicates_or_unmapped_answer(self):
        base = {
            "question": "Which organelle performs photosynthesis?",
            "options": ["Chloroplasts", "Ribosomes", "Lysosomes", "Nucleus"],
            "correct_answer": "Chloroplasts",
            "explanation": "Chloroplasts perform photosynthesis.",
            "source_chunk_ids": ["bio123456789_p1_c1"],
        }

        # Only 3 options
        bad_count = {**base, "options": ["Chloroplasts", "Ribosomes", "Nucleus"]}
        with self.assertRaises(QuestionValidationError):
            validate_question_item(
                bad_count,
                expected_type="mcq",
                expected_difficulty="easy",
                allowed_chunks_by_id=self.allowed,
                document_id="bio_doc_id",
                filename="biology.pdf",
            )

        # Duplicate options
        dup_opts = {**base, "options": ["Chloroplasts", "chloroplasts", "Ribosomes", "Nucleus"]}
        with self.assertRaises(QuestionValidationError):
            validate_question_item(
                dup_opts,
                expected_type="mcq",
                expected_difficulty="easy",
                allowed_chunks_by_id=self.allowed,
                document_id="bio_doc_id",
                filename="biology.pdf",
            )

        # Correct answer not matching any option
        bad_ans = {**base, "correct_answer": "Mitochondria"}
        with self.assertRaises(QuestionValidationError):
            validate_question_item(
                bad_ans,
                expected_type="mcq",
                expected_difficulty="easy",
                allowed_chunks_by_id=self.allowed,
                document_id="bio_doc_id",
                filename="biology.pdf",
            )

    def test_short_answer_and_true_false_validation(self):
        short_item = {
            "question_type": "short_answer",
            "difficulty": "medium",
            "question": "What molecule do mitochondria produce during cellular respiration?",
            "expected_answer": "ATP",
            "explanation": "Mitochondria generate ATP via cellular respiration using oxygen.",
            "source_chunk_ids": ["bio123456789_p3_c2"],
        }
        q_short = validate_question_item(
            short_item,
            expected_type="short_answer",
            expected_difficulty="medium",
            allowed_chunks_by_id=self.allowed,
            document_id="bio_doc_id",
            filename="biology.pdf",
        )
        self.assertEqual(q_short.correct_answer, "ATP")
        self.assertEqual(q_short.source_pages, [3])

        tf_false = {
            "question_type": "true_false",
            "difficulty": "medium",
            "question": "Mitochondria convert sunlight directly into glucose.",
            "correct_answer": "False",
            "correct_statement": "Chloroplasts convert sunlight into glucose, while mitochondria generate ATP.",
            "explanation": "Photosynthesis takes place in chloroplasts, whereas mitochondria produce ATP.",
            "source_chunk_ids": ["bio123456789_p1_c1", "bio123456789_p3_c2"],
        }
        q_tf = validate_question_item(
            tf_false,
            expected_type="true_false",
            expected_difficulty="medium",
            allowed_chunks_by_id=self.allowed,
            document_id="bio_doc_id",
            filename="biology.pdf",
        )
        self.assertEqual(q_tf.correct_answer, "False")
        self.assertIn("Chloroplasts", q_tf.correct_statement)
        self.assertEqual(q_tf.source_pages, [1, 3])
        self.assertEqual(q_tf.page_range_label, "Pages 1, 3")

        # False T/F without corrected statement must fail
        tf_missing_correction = {**tf_false, "correct_statement": ""}
        with self.assertRaises(QuestionValidationError):
            validate_question_item(
                tf_missing_correction,
                expected_type="true_false",
                expected_difficulty="medium",
                allowed_chunks_by_id=self.allowed,
                document_id="bio_doc_id",
                filename="biology.pdf",
            )

        # Invalid T/F boolean answer must fail
        tf_bad_label = {**tf_false, "correct_answer": "Maybe"}
        with self.assertRaises(QuestionValidationError):
            validate_question_item(
                tf_bad_label,
                expected_type="true_false",
                expected_difficulty="medium",
                allowed_chunks_by_id=self.allowed,
                document_id="bio_doc_id",
                filename="biology.pdf",
            )

    def test_unknown_missing_and_cross_document_chunk_ids_rejected(self):
        base = {
            "question": "What do chloroplasts convert sunlight into?",
            "expected_answer": "Glucose",
            "explanation": "Photosynthesis produces glucose.",
        }
        for bad_cids in ([], ["unknown_chunk_999"], ["other9999999_p2_c1"]):
            with self.assertRaises(QuestionValidationError):
                validate_question_item(
                    {**base, "source_chunk_ids": bad_cids},
                    expected_type="short_answer",
                    expected_difficulty="easy",
                    allowed_chunks_by_id=self.allowed,
                    document_id="bio_doc_id",
                    filename="biology.pdf",
                )

    def test_parse_and_validate_mixed_caps_max_questions_and_deduplicates(self):
        payload = {
            "questions": [
                {
                    "question_type": "mcq",
                    "difficulty": "hard",
                    "question": "Which organelle converts sunlight into glucose?",
                    "options": ["Chloroplasts", "Mitochondria", "Nucleus", "Vacuole"],
                    "correct_answer": "Chloroplasts",
                    "explanation": "Chloroplasts carry out photosynthesis.",
                    "source_chunk_ids": ["bio123456789_p1_c1"],
                },
                {
                    # Invalid item (unknown chunk) should be rejected without crashing valid ones
                    "question_type": "short_answer",
                    "difficulty": "hard",
                    "question": "Invalid question with unknown chunk reference.",
                    "expected_answer": "Nothing",
                    "explanation": "Should be dropped.",
                    "source_chunk_ids": ["fake_chunk_id"],
                },
                {
                    "question_type": "short_answer",
                    "difficulty": "hard",
                    "question": "Which organelle converts sunlight into glucose?",  # Duplicate stem
                    "expected_answer": "Chloroplasts",
                    "explanation": "Duplicate stem test.",
                    "source_chunk_ids": ["bio123456789_p1_c1"],
                },
                {
                    "question_type": "true_false",
                    "difficulty": "hard",
                    "question": "Mitochondria generate ATP via cellular respiration using oxygen.",
                    "correct_answer": "True",
                    "correct_statement": "Mitochondria generate ATP via cellular respiration using oxygen.",
                    "explanation": "Explicitly stated on page 3.",
                    "source_chunk_ids": ["bio123456789_p3_c2"],
                },
            ]
        }

        valid, rejections = parse_and_validate_questions(
            json.dumps(payload),
            expected_type="mixed",
            expected_difficulty="hard",
            allowed_chunks_by_id=self.allowed,
            document_id="bio_doc_id",
            filename="biology.pdf",
        )
        self.assertEqual(len(valid), 3)
        self.assertEqual(len(rejections), 1)

        deduped = deduplicate_questions(valid)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(
            compute_type_distribution(deduped),
            {"mcq": 1, "short_answer": 0, "true_false": 1},
        )
        self.assertTrue(
            is_duplicate_question(
                "Which organelle converts sunlight into glucose",
                ["Which organelle converts sunlight into glucose?"],
            )
        )


class TestStage62aPrompts(unittest.TestCase):
    def test_prompt_builders_include_chunk_ids_and_bound_previous_stems(self):
        chunk = _sample_allowed_chunks()["bio123456789_p1_c1"]
        entry = format_question_chunk_entry(chunk)
        self.assertIn("Chunk ID: bio123456789_p1_c1", entry)
        self.assertIn("Page 1", entry)

        many_stems = [f"Previous question number {i} about biology concepts?" for i in range(50)]
        bounded_block = format_previous_stems_block(many_stems, max_stems=5)
        self.assertEqual(bounded_block.count("\n- ("), 5)
        self.assertIn("question number 49", bounded_block)
        self.assertNotIn("question number 0", bounded_block)

        prompt = build_question_generation_prompt(
            filename="biology.pdf",
            scope_description="Full document",
            question_type="mixed",
            difficulty="medium",
            question_count=4,
            formatted_chunks=[entry],
            type_counts={"mcq": 2, "short_answer": 1, "true_false": 1},
            previous_stems=many_stems,
            max_previous_stems=3,
            retry_reminder=True,
        )
        self.assertIn("2 mcq, 1 short_answer, 1 true_false", prompt)
        self.assertIn("IMPORTANT RETRY NOTICE", prompt)
        self.assertIn("source_chunk_ids", prompt)
        self.assertIn("source_chunk_ids", QUESTION_SYSTEM_PROMPT)

        allowance = estimate_output_token_allowance(5, "mcq")
        self.assertGreaterEqual(allowance, 250)
        self.assertLessEqual(allowance, config.QUESTION_MAX_OUTPUT_TOKENS)


if __name__ == "__main__":
    unittest.main()
