"""Unit tests for the grounded Flashcard Generator powered by Qwen2.5 7B."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pymupdf

import src.config as config
from src.generation.flashcard_generator import (
    Flashcard,
    FlashcardGenerationError,
    FlashcardSetResult,
    export_flashcards_to_anki_csv,
    export_flashcards_to_markdown,
    extract_flashcard_json_payload,
    generate_flashcards,
    parse_and_validate_flashcards,
)
from src.generation.llm_provider import BaseLLMProvider, LLMProviderError
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


class _MockFlashcardProvider(BaseLLMProvider):
    provider_name = "mock_ollama"

    def __init__(self, responses: list[str] | None = None):
        super().__init__(model_name="qwen2.5:7b")
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def is_configured(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
        })
        if self.responses:
            return self.responses.pop(0)
        return json.dumps({
            "flashcards": [
                {
                    "front": "What is ATP?",
                    "back": "Adenosine triphosphate is the primary energy currency of the cell.",
                    "source_chunk_ids": [],
                }
            ]
        })


class TestFlashcardGenerator(unittest.TestCase):
    def test_extract_flashcard_json_payload_valid(self):
        raw = json.dumps({"flashcards": [{"front": "Q1", "back": "A1"}]})
        parsed = extract_flashcard_json_payload(raw)
        self.assertIn("flashcards", parsed)
        self.assertEqual(len(parsed["flashcards"]), 1)

    def test_extract_flashcard_json_payload_fenced(self):
        raw = "```json\n" + json.dumps({"flashcards": [{"front": "Q2", "back": "A2"}]}) + "\n```"
        parsed = extract_flashcard_json_payload(raw)
        self.assertIn("flashcards", parsed)

    def test_extract_flashcard_json_payload_empty_raises(self):
        with self.assertRaises(FlashcardGenerationError):
            extract_flashcard_json_payload("")

    def test_parse_and_validate_flashcards(self):
        chunks = {
            "chunk_1": {"chunk_id": "chunk_1", "page": 2, "filename": "doc.pdf"},
            "chunk_2": {"chunk_id": "chunk_2", "page": 3, "filename": "doc.pdf"},
        }
        payload = json.dumps({
            "flashcards": [
                {
                    "front": "What is photosynthesis?",
                    "back": "Process by which plants use sunlight to synthesize nutrients.",
                    "source_chunk_ids": ["chunk_1"],
                },
                {
                    "front": "What is photosynthesis?",  # Duplicate front
                    "back": "Same concept duplicate.",
                    "source_chunk_ids": ["chunk_1"],
                },
                {
                    "front": "What is chlorophyll?",
                    "back": "Green pigment absorbing light energy.",
                    "source_chunk_ids": ["chunk_2"],
                },
                {
                    "front": "What is ATP?",
                    "back": "Unbacked hallucination citing fake chunk.",
                    "source_chunk_ids": ["invalid_chunk"],  # Must be rejected
                },
            ]
        })

        cards, rejections = parse_and_validate_flashcards(
            payload,
            allowed_chunks_by_id=chunks,
            document_id="doc_123",
            filename="doc.pdf",
            max_cards=5,
        )

        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0].front, "What is photosynthesis?")
        self.assertEqual(cards[0].source_pages, [2])
        self.assertEqual(cards[1].front, "What is chlorophyll?")
        self.assertEqual(cards[1].source_pages, [3])
        self.assertTrue(any("invalid_chunk" in r or "no valid chunk" in r for r in rejections))

    def test_generate_flashcards_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            store = FAISSVectorStore(store_dir=store_dir)

            sample_chunks = [
                {
                    "chunk_id": "c1",
                    "chunk_index": 1,
                    "text": "Mitochondria generate most of the chemical energy needed by the cell.",
                    "page": 1,
                    "page_range": "1",
                    "filename": "cell_bio.pdf",
                    "document_id": "doc_bio",
                    "token_count": 12,
                    "chunk_type": "text",
                },
                {
                    "chunk_id": "c2",
                    "chunk_index": 2,
                    "text": "ATP is the biochemical energy currency produced in mitochondria.",
                    "page": 2,
                    "page_range": "2",
                    "filename": "cell_bio.pdf",
                    "document_id": "doc_bio",
                    "token_count": 11,
                    "chunk_type": "text",
                },
            ]
            embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            store.add_document_chunks(
                document_id="doc_bio",
                filename="cell_bio.pdf",
                page_count=2,
                chunks=sample_chunks,
                embeddings=embeddings,
            )

            mock_response = json.dumps({
                "flashcards": [
                    {
                        "front": "What role do mitochondria play?",
                        "back": "They generate most of the cell's chemical energy.",
                        "source_chunk_ids": ["c1"],
                    },
                    {
                        "front": "What is ATP?",
                        "back": "Biochemical energy currency produced in mitochondria.",
                        "source_chunk_ids": ["c2"],
                    },
                ]
            })
            provider = _MockFlashcardProvider([mock_response])

            result = generate_flashcards(
                store,
                "doc_bio",
                scope="full",
                count=2,
                provider=provider,
            )

            self.assertIsInstance(result, FlashcardSetResult)
            self.assertEqual(len(result.cards), 2)
            self.assertEqual(result.document_id, "doc_bio")
            self.assertEqual(result.filename, "cell_bio.pdf")
            self.assertEqual(result.cards[0].source_pages, [1])
            self.assertEqual(result.cards[1].source_pages, [2])
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(provider.calls[0]["kwargs"].get("format"), "json")

            res_dict = result.to_dict()
            self.assertEqual(len(res_dict["cards"]), 2)
            self.assertEqual(res_dict["cards"][0]["front"], "What role do mitochondria play?")


    def test_extract_flashcard_json_payload_root_list(self):
        raw = json.dumps([
            {"front": "What is DNA?", "back": "Deoxyribonucleic acid carries genetic code."},
            {"question": "What is RNA?", "answer": "Ribonucleic acid translates genetic code."},
        ])
        parsed = extract_flashcard_json_payload(raw)
        self.assertIn("flashcards", parsed)
        self.assertEqual(len(parsed["flashcards"]), 2)

    def test_extract_flashcard_json_payload_cards_synonym_key(self):
        raw = json.dumps({"cards": [{"front": "Q1", "back": "A1"}]})
        parsed = extract_flashcard_json_payload(raw)
        self.assertIn("flashcards", parsed)
        self.assertEqual(len(parsed["flashcards"]), 1)

    def test_parse_and_validate_concise_answers_and_synonyms(self):
        chunks = {
            "chunk_1": {"chunk_id": "chunk_1", "page": 1, "filename": "doc.pdf"},
        }
        payload = json.dumps([
            {
                "question": "What is the primary cellular energy molecule?",
                "answer": "ATP",  # 3 chars, concise valid answer
                "source_chunk_ids": "chunk_1",  # string format
            },
            {
                "term": "Deoxyribonucleic acid",
                "definition": "DNA",  # 3 chars, concise definition
                "source_chunk_ids": ["chunk_1"],
            },
            {
                "prompt": "Base pairs in RNA",
                "response": "4",  # 1 char, concise numeric answer
                # Missing source_chunk_ids -> should fallback to batch chunks
            },
        ])

        cards, rejections = parse_and_validate_flashcards(
            payload,
            allowed_chunks_by_id=chunks,
            document_id="doc_123",
            filename="doc.pdf",
            max_cards=5,
        )

        self.assertEqual(len(cards), 3)
        self.assertEqual(cards[0].front, "What is the primary cellular energy molecule?")
        self.assertEqual(cards[0].back, "ATP")
        self.assertEqual(cards[0].source_pages, [1])

        self.assertEqual(cards[1].front, "Deoxyribonucleic acid")
        self.assertEqual(cards[1].back, "DNA")

        self.assertEqual(cards[2].front, "Base pairs in RNA")
        self.assertEqual(cards[2].back, "4")
        self.assertEqual(cards[2].source_pages, [1])

    def test_multi_batch_chunk_distribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            store = FAISSVectorStore(store_dir=store_dir)
            sample_chunks = [
                {
                    "chunk_id": f"c{i}",
                    "chunk_index": i,
                    "text": f"Chapter {i}: Detailed biology section describing key cellular processes #{i}.",
                    "page": i,
                    "page_range": str(i),
                    "filename": "cell_bio.pdf",
                    "document_id": "doc_bio",
                    "token_count": 25,
                    "chunk_type": "text",
                }
                for i in range(1, 5)
            ]
            embeddings = np.eye(4, dtype=np.float32)
            store.add_document_chunks(
                document_id="doc_bio",
                filename="cell_bio.pdf",
                page_count=4,
                chunks=sample_chunks,
                embeddings=embeddings,
            )

            # Return 1 card per batch
            mock_responses = [
                json.dumps({"flashcards": [{"front": f"Topic from Batch {i}", "back": f"Answer {i}", "source_chunk_ids": [f"c{i}"]}]})
                for i in range(1, 5)
            ]
            provider = _MockFlashcardProvider(mock_responses)

            result = generate_flashcards(
                store,
                "doc_bio",
                scope="full",
                count=3,
                provider=provider,
                max_batch_tokens=300,  # Small budget forcing multiple batches
            )

            self.assertGreaterEqual(result.batches_processed, 2)
            self.assertEqual(len(result.cards), 3)
            self.assertGreater(result.chunks_used, 0)

    def test_bounded_retry_on_malformed_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = Path(tmpdir) / "vector_store"
            store = FAISSVectorStore(store_dir=store_dir)
            sample_chunks = [
                {
                    "chunk_id": "c1",
                    "chunk_index": 1,
                    "text": "Cellular respiration produces ATP.",
                    "page": 1,
                    "page_range": "1",
                    "filename": "cell.pdf",
                    "document_id": "doc_cell",
                    "token_count": 10,
                    "chunk_type": "text",
                }
            ]
            embeddings = np.array([[1.0]], dtype=np.float32)
            store.add_document_chunks(
                document_id="doc_cell",
                filename="cell.pdf",
                page_count=1,
                chunks=sample_chunks,
                embeddings=embeddings,
            )

            # First response is malformed, retry response is valid JSON
            mock_responses = [
                "Malformed plain text with no json",
                json.dumps({"flashcards": [{"front": "What does respiration produce?", "back": "ATP", "source_chunk_ids": ["c1"]}]}),
            ]
            provider = _MockFlashcardProvider(mock_responses)

            result = generate_flashcards(
                store,
                "doc_cell",
                scope="full",
                count=1,
                provider=provider,
            )

            self.assertEqual(len(result.cards), 1)
            self.assertEqual(result.cards[0].back, "ATP")
            self.assertEqual(len(provider.calls), 2)  # Initial call + retry call

    def test_export_flashcards_to_markdown(self):
        card = Flashcard(
            card_id="fc_1",
            front="What is ATP?",
            back="Adenosine triphosphate",
            source_chunk_ids=["c1"],
            source_pages=[1],
            page_range_label="Page 1",
            filename="bio.pdf",
            document_id="doc_1",
        )
        md = export_flashcards_to_markdown([card], filename="bio.pdf", page_range_label="Page 1")
        self.assertIn("# Flashcard Deck: bio.pdf", md)
        self.assertIn("### Card 1", md)
        self.assertIn("What is ATP?", md)
        self.assertIn("Adenosine triphosphate", md)
        self.assertIn("Page 1", md)

    def test_export_flashcards_to_anki_csv(self):
        card = Flashcard(
            card_id="fc_1",
            front="What is Mitochondria?",
            back="Powerhouse of the cell, generates ATP.",
            source_chunk_ids=["c1"],
            source_pages=[2],
            page_range_label="Page 2",
            filename="bio.pdf",
            document_id="doc_1",
        )
        csv_out = export_flashcards_to_anki_csv([card])
        lines = csv_out.strip().split("\n")
        self.assertEqual(len(lines), 2)  # Header + 1 row
        self.assertIn("Front", lines[0])
        self.assertIn("Back", lines[0])
        self.assertIn('"What is Mitochondria?"', lines[1])
        self.assertIn('"Powerhouse of the cell, generates ATP."', lines[1])


if __name__ == "__main__":
    unittest.main()
