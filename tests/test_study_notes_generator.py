"""Unit tests for the grounded Academic Study Notes Generator powered by Qwen2.5 7B."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import src.config as config
from src.generation.llm_provider import BaseLLMProvider, LLMProviderError
from src.generation.study_notes_generator import (
    StudyNotesResult,
    _reduce_notes_hierarchically,
    build_notes_prompt,
    export_notes_to_markdown,
    generate_study_notes,
    validate_and_filter_page_citations,
    validate_notes_structure,
)
from src.retrieval.vector_store import FAISSVectorStore


class _MockNotesProvider(BaseLLMProvider):
    provider_name = "mock_ollama"

    def __init__(self, response_text: str = "# Study Notes\n## 1. Overview\nHigh yield notes.", responses: list[str] | None = None):
        super().__init__(model_name="qwen2.5:7b")
        self.response_text = response_text
        self.responses = list(responses) if responses is not None else None
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
        return self.response_text


class TestStudyNotesGenerator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store_dir = Path(self.tmpdir.name) / "vector_store"
        self.store = FAISSVectorStore(store_dir=self.store_dir)

        self.sample_chunks = [
            {
                "chunk_id": "c1",
                "chunk_index": 1,
                "text": "Cellular respiration occurs in the mitochondria. Glycolysis breaks down glucose.",
                "page": 1,
                "page_range": "1",
                "filename": "biology.pdf",
                "document_id": "doc_bio",
                "token_count": 12,
                "chunk_type": "text",
            },
            {
                "chunk_id": "c2",
                "chunk_index": 2,
                "text": "The Krebs cycle produces NADH and FADH2 which power the electron transport chain.",
                "page": 2,
                "page_range": "2",
                "filename": "biology.pdf",
                "document_id": "doc_bio",
                "token_count": 15,
                "chunk_type": "text",
            },
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.store.add_document_chunks(
            document_id="doc_bio",
            filename="biology.pdf",
            page_count=2,
            chunks=self.sample_chunks,
            embeddings=embeddings,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_build_notes_prompt(self):
        prompt = build_notes_prompt(
            filename="test.pdf",
            scope_description="Pages 1-2",
            formatted_chunks=["[Page 1]: Hello", "[Page 2]: World"],
        )
        self.assertIn("Document: test.pdf", prompt)
        self.assertIn("Pages 1-2", prompt)
        self.assertIn("[Page 1]: Hello", prompt)

    def test_generate_study_notes_full_document(self):
        notes_body = (
            "# Cellular Respiration\n"
            "## 1. Executive Overview\nRespiration generates ATP.\n"
            "## 2. Key Concepts\nGlycolysis, Krebs cycle.\n"
        )
        provider = _MockNotesProvider(response_text=notes_body)

        result = generate_study_notes(
            self.store,
            "doc_bio",
            scope="full",
            provider=provider,
        )

        self.assertIsInstance(result, StudyNotesResult)
        self.assertEqual(result.document_id, "doc_bio")
        self.assertEqual(result.filename, "biology.pdf")
        self.assertEqual(result.chunks_used, 2)
        self.assertEqual(result.pages_covered, [1, 2])
        self.assertEqual(result.notes_text, notes_body.strip())
        self.assertEqual(len(provider.calls), 1)

        result_dict = result.to_dict()
        self.assertEqual(result_dict["document_id"], "doc_bio")
        self.assertEqual(len(result_dict["source_chunks"]), 2)

    def test_generate_study_notes_page_range(self):
        provider = _MockNotesProvider()
        result = generate_study_notes(
            self.store,
            "doc_bio",
            scope="page_range",
            start_page=1,
            end_page=1,
            provider=provider,
        )
        self.assertEqual(result.chunks_used, 1)
        self.assertEqual(result.pages_covered, [1])

    def test_generate_study_notes_invalid_doc_raises(self):
        provider = _MockNotesProvider()
        with self.assertRaises(KeyError):
            generate_study_notes(self.store, "nonexistent_doc", provider=provider)

    def test_generate_study_notes_invalid_page_range_raises(self):
        provider = _MockNotesProvider()
        with self.assertRaises(ValueError):
            generate_study_notes(
                self.store,
                "doc_bio",
                scope="page_range",
                start_page=5,
                end_page=2,
                provider=provider,
            )

    def test_multi_batch_map_reduce_synthesis(self):
        # Seed 4 chunks with sufficient length to force multiple batches
        extra_chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "chunk_index": i,
                "text": f"Chapter {i}: Detailed discussion on biological mechanism #{i} with key processes in cellular biology. " * 3,
                "page": i,
                "page_range": str(i),
                "filename": "multi.pdf",
                "document_id": "doc_multi",
                "token_count": 50,
                "chunk_type": "text",
            }
            for i in range(1, 5)
        ]
        embeddings = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        self.store.add_document_chunks(
            document_id="doc_multi",
            filename="multi.pdf",
            page_count=4,
            chunks=extra_chunks,
            embeddings=embeddings,
        )

        # 4 map responses + 1 reduce response
        map_responses = [
            f"# Section {i} Notes\n- Process {i} occurs at [Page {i}]."
            for i in range(1, 5)
        ]
        reduce_response = (
            "# Master Academic Study Notes: multi.pdf\n"
            "## 1. Executive Overview\nCovers 4 biological mechanisms.\n"
            "## 2. Chapters & Topics Deep Dive\nChapters 1-4 cover processes.\n"
            "## 3. Key Concepts & Definitions Dictionary\nMechanisms 1 to 4.\n"
            "## 4. Core Principles & Mechanisms\nSequential cellular reactions.\n"
            "## 5. Important Facts & Formulas\nFour primary phases.\n"
            "## 6. High-Yield Summary & Review Takeaways\nAll processes verified at [Page 1], [Page 2], [Page 3], [Page 4]."
        )
        provider = _MockNotesProvider(
            response_text=reduce_response,
            responses=map_responses + [reduce_response],
        )

        result = generate_study_notes(
            self.store,
            "doc_multi",
            scope="full",
            provider=provider,
            max_batch_tokens=220,  # Small budget forcing multiple batches
        )

        self.assertIsInstance(result, StudyNotesResult)
        self.assertGreaterEqual(result.batches_processed, 2)
        self.assertGreaterEqual(result.reduction_levels, 1)
        self.assertEqual(result.pages_covered, [1, 2, 3, 4])
        self.assertEqual(result.chunks_used, 4)
        self.assertIn("Master Academic Study Notes", result.notes_text)
        self.assertIn("Executive Overview", result.notes_text)

    def test_validate_and_filter_page_citations(self):
        raw_text = (
            "Photosynthesis produces glucose [Page 1]. "
            "Chlorophyll is a pigment (Page 2). "
            "Hallucinated alien concept [Page 99] not in book (Page 150)."
        )
        filtered = validate_and_filter_page_citations(raw_text, allowed_pages=[1, 2])
        self.assertIn("[Page 1]", filtered)
        self.assertIn("(Page 2)", filtered)
        self.assertNotIn("[Page 99]", filtered)
        self.assertNotIn("(Page 150)", filtered)

    def test_validate_notes_structure(self):
        # Empty text
        valid, issues = validate_notes_structure("")
        self.assertFalse(valid)
        self.assertTrue(any("empty" in i for i in issues))

        # Too short text
        valid, issues = validate_notes_structure("# Short")
        self.assertFalse(valid)
        self.assertTrue(any("too brief" in i for i in issues))

        # Text without headings
        valid, issues = validate_notes_structure(
            "This is a long piece of text that describes biology concepts without any markdown headings at all." * 2
        )
        self.assertFalse(valid)
        self.assertTrue(any("heading" in i for i in issues))

        # Valid structured notes
        good_text = (
            "# Cellular Respiration\n"
            "## 1. Overview\nComprehensive overview of respiration.\n"
            "## 2. Key Concepts\nATP, NADH, and electron transport chain.\n"
            "## 3. Review Summary\nHigh-yield facts."
        )
        valid, issues = validate_notes_structure(good_text)
        self.assertTrue(valid)
        self.assertEqual(len(issues), 0)

    def test_bounded_retry_on_empty_batch(self):
        # Empty map response, then valid retry response
        provider = _MockNotesProvider(
            responses=[
                "",  # attempt 1 returns empty
                "# Single Batch Study Notes\n## 1. Overview\nOverview after retry.\n## 2. Concepts\nDefinitions.",
            ]
        )

        result = generate_study_notes(
            self.store,
            "doc_bio",
            scope="full",
            provider=provider,
        )

        self.assertEqual(result.document_id, "doc_bio")
        self.assertIn("Single Batch Study Notes", result.notes_text)
        self.assertEqual(len(provider.calls), 2)  # Initial call + retry call

    def test_export_notes_to_markdown(self):
        result = StudyNotesResult(
            title="Academic Study Notes: biology.pdf (Full Document)",
            notes_text="## 1. Overview\nRespiration generates ATP.",
            document_id="doc_bio",
            filename="biology.pdf",
            scope="full",
            pages_covered=[1, 2],
            page_range_label="Pages 1-2",
            chunks_used=2,
            provider="mock_ollama",
            model="qwen2.5:7b",
            source_chunks=[],
            batches_processed=1,
            reduction_levels=0,
        )

        md = export_notes_to_markdown(result)
        self.assertIn("# Academic Study Notes: biology.pdf (Full Document)", md)
        self.assertIn("**Document:** biology.pdf", md)
        self.assertIn("**Pages Covered:** Pages 1-2", md)
        self.assertIn("Respiration generates ATP.", md)

    def test_recursive_reduction_hierarchical(self):
        # Provide 6 section notes with substantial text
        section_notes = [
            f"# Section {i}\n" + ("Detailed analysis of biological process, enzymatic catalysis, and electron transport. " * 10)
            for i in range(1, 7)
        ]
        provider = _MockNotesProvider(
            response_text="# Reduced Level Output\n## Overview\nCombined notes."
        )

        final_text, levels = _reduce_notes_hierarchically(
            section_notes,
            filename="test.pdf",
            scope_description="Full Document",
            provider=provider,
            max_batch_tokens=320,  # Forces grouping into small subsets across levels
        )

        self.assertIn("Reduced Level Output", final_text)
        self.assertGreaterEqual(levels, 1)
        self.assertGreater(len(provider.calls), 1)

    def test_12_large_document_multi_batch_coverage(self):
        """Verify generate_study_notes with multiple batches correctly populates coverage fields."""
        chunks = []
        for i in range(1, 13):
            p = ((i - 1) % 4) + 1
            chunks.append({
                "chunk_id": f"doc_large_c{i}",
                "vector_id": i,
                "document_id": "doc_large",
                "filename": "neuroscience.pdf",
                "page": p,
                "page_range": str(p),
                "chunk_index": i,
                "text": f"Neuroscience principle chunk {i} on page {p}. Synaptic plasticity governs memory consolidation. " * 8,
                "token_count": 90,
            })

        mock_store = MagicMock()
        mock_store.get_document_chunks.return_value = chunks
        mock_store.get_document.return_value = {"filename": "neuroscience.pdf"}

        provider = _MockNotesProvider(
            response_text="# Master Study Notes\n## 1. Overview\nComprehensive breakdown [Page 1].\n## 2. Details\nSynaptic transmission mechanisms [Page 2]."
        )

        result = generate_study_notes(
            mock_store,
            "doc_large",
            scope="full",
            provider=provider,
            max_batch_tokens=300,  # forces multiple batches
        )

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.total_eligible_chunks, 12)
        self.assertEqual(result.processed_chunks, 12)
        self.assertEqual(result.failed_chunks, 0)
        self.assertGreater(result.batches_processed, 1)
        self.assertEqual(result.pages_covered, [1, 2, 3, 4])

    def test_13_failed_map_batch_records_partial_coverage_and_failed_indices(self):
        """Verify that when a map batch fails all retries, status is partial with failed accounting."""
        chunks = []
        for i in range(1, 9):
            chunks.append({
                "chunk_id": f"chunk_{i}",
                "vector_id": i,
                "document_id": "doc_fail",
                "filename": "neuro.pdf",
                "page": (i % 2) + 1,
                "chunk_index": i,
                "text": f"Neuro chunk {i} discussing synaptic plasticity and long-term potentiation mechanisms. " * 8,
                "token_count": 80,
            })

        mock_store = MagicMock()
        mock_store.get_document_chunks.return_value = chunks
        mock_store.get_document.return_value = {"filename": "neuro.pdf"}

        # Custom provider that fails on batch 2 (both attempt 1 and retry)
        call_count = 0
        def _failing_generate(sys_p, usr_p, **kwargs):
            nonlocal call_count
            call_count += 1
            if "Section Batch: 2" in usr_p:
                raise LLMProviderError("Transient rate limit on batch 2")
            return "# Section Notes\n## Overview\nValid notes content."

        provider = MagicMock()
        provider.provider_name = "ollama"
        provider.model_name = "qwen2.5:7b"
        provider.generate.side_effect = _failing_generate

        result = generate_study_notes(
            mock_store,
            "doc_fail",
            scope="full",
            provider=provider,
            max_batch_tokens=300,
        )

        self.assertEqual(result.status, "partial")
        self.assertGreater(result.failed_chunks, 0)
        self.assertIn(2, result.failed_batch_indices)
        self.assertEqual(result.processed_chunks + result.failed_chunks, 8)
        self.assertIn("Section Notes", result.notes_text)

    def test_14_max_reduction_levels_exceeded_records_partial(self):
        """Verify that when reduction reaches max_levels limit, status is marked partial."""
        chunks = []
        for i in range(1, 15):
            chunks.append({
                "chunk_id": f"chunk_{i}",
                "vector_id": i,
                "document_id": "doc_limit",
                "filename": "physics.pdf",
                "page": (i % 3) + 1,
                "chunk_index": i,
                "text": f"Physics quantum mechanics chunk {i} on wavefunctions and operator algebra. " * 8,
                "token_count": 80,
            })

        mock_store = MagicMock()
        mock_store.get_document_chunks.return_value = chunks
        mock_store.get_document.return_value = {"filename": "physics.pdf"}

        provider = _MockNotesProvider(
            response_text="# Physics Notes\n## Overview\nElectromagnetism and quantum mechanics synthesis."
        )

        # Set MAX_REDUCTION_LEVELS to 1 to force level exhaustion during multi-batch reduction
        with unittest.mock.patch("src.config.MAX_REDUCTION_LEVELS", 1):
            result = generate_study_notes(
                mock_store,
                "doc_limit",
                scope="full",
                provider=provider,
                max_batch_tokens=250,
            )

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.total_eligible_chunks, 14)


if __name__ == "__main__":
    unittest.main()
