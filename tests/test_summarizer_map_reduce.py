"""Targeted tests for Scalable Map-Reduce Summarization (Stage 11.3 Work A)."""

import unittest
from unittest.mock import MagicMock
from typing import Any

from src.generation.llm_provider import BaseLLMProvider, LLMProviderError
from src.generation.summarizer import (
    _summarize_chunks_hierarchical,
    SummaryResult,
    SummarizationError,
)


class _MockHierarchicalProvider(BaseLLMProvider):
    def __init__(self, prefix: str = "Summary"):
        super().__init__(model_name="qwen2.5:7b")
        self.calls: list[dict[str, Any]] = []
        self.prefix = prefix

    def is_configured(self) -> bool:
        return True

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        self.calls.append({"system": system_prompt, "user": user_prompt, "kwargs": kwargs})
        # Extract page range from prompt if possible
        return f"{self.prefix} for call {len(self.calls)}: Grounded academic synthesis of provided concepts."


class TestSummarizerMapReduceScalability(unittest.TestCase):
    """Test multi-batch, multi-level map-reduce summarization on large chunk collections."""

    def _create_chunks(self, count: int, pages: int = 50) -> list[dict[str, Any]]:
        chunks = []
        for i in range(1, count + 1):
            p = ((i - 1) % pages) + 1
            chunks.append({
                "chunk_id": f"doc_large_c{i}",
                "vector_id": i,
                "document_id": "doc_large",
                "filename": "textbook.pdf",
                "page": p,
                "page_range": str(p),
                "chunk_index": i,
                "text": f"Chapter {((p-1)//10)+1} content for chunk {i}: Essential academic explanations and formulas for page {p}. " * 6,
                "token_count": 80,
            })
        return chunks

    def test_01_multi_batch_reduction_converges_and_tracks_coverage(self):
        """Verify that a large chunk set produces multiple map batches and hierarchical reduction."""
        chunks = self._create_chunks(60, pages=20)
        provider = _MockHierarchicalProvider()

        # Set max_batch_tokens low enough to force multiple map batches and reduction levels
        result = _summarize_chunks_hierarchical(
            chunks=chunks,
            document_id="doc_large",
            filename="textbook.pdf",
            mode="full",
            scope_description="Entire Document",
            length="medium",
            provider=provider,
            max_batch_tokens=800,
        )

        self.assertIsInstance(result, SummaryResult)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.total_eligible_chunks, 60)
        self.assertEqual(result.processed_chunks, 60)
        self.assertEqual(result.failed_chunks, 0)
        self.assertTrue(result.hierarchical)
        self.assertGreater(result.batches_processed, 1)
        self.assertGreater(result.reduction_levels, 0)
        self.assertEqual(len(result.pages_covered), 20)
        self.assertIn("Pages 1-20", result.page_range_label)
        self.assertTrue(len(provider.calls) >= 2)

    def test_02_failure_propagates_without_silent_corruption(self):
        """Verify that provider failure raises cleanly and does not return incomplete result."""
        chunks = self._create_chunks(15, pages=5)
        failing_provider = MagicMock()
        failing_provider.provider_name = "ollama"
        failing_provider.model_name = "qwen2.5:7b"
        failing_provider.generate.side_effect = LLMProviderError("Ollama daemon out of memory")

        with self.assertRaises(LLMProviderError):
            _summarize_chunks_hierarchical(
                chunks=chunks,
                document_id="doc_large",
                filename="textbook.pdf",
                mode="full",
                scope_description="Entire Document",
                length="medium",
                provider=failing_provider,
                max_batch_tokens=600,
            )

    def test_03_progress_callback_reports_steps(self):
        """Verify progress callback is invoked during map and reduce stages."""
        chunks = self._create_chunks(20, pages=5)
        provider = _MockHierarchicalProvider()
        progress_steps = []

        def _cb(step, total, msg):
            progress_steps.append((step, total, msg))

        result = _summarize_chunks_hierarchical(
            chunks=chunks,
            document_id="doc_large",
            filename="textbook.pdf",
            mode="full",
            scope_description="Entire Document",
            length="short",
            provider=provider,
            max_batch_tokens=600,
            progress_callback=_cb,
        )

        self.assertGreater(len(progress_steps), 1)
        self.assertEqual(result.status, "complete")
        self.assertIn("Summarizing batch", progress_steps[0][2])

    def test_04_max_reduction_levels_exhaustion_marks_partial(self):
        """Verify exceeding MAX_REDUCTION_LEVELS produces status='partial' instead of 'complete'."""
        chunks = self._create_chunks(40, pages=10)
        provider = _MockHierarchicalProvider()

        # Mock config.MAX_REDUCTION_LEVELS to 1 so reduction halts after level 1
        with unittest.mock.patch("src.config.MAX_REDUCTION_LEVELS", 1):
            result = _summarize_chunks_hierarchical(
                chunks=chunks,
                document_id="doc_large",
                filename="textbook.pdf",
                mode="full",
                scope_description="Entire Document",
                length="medium",
                provider=provider,
                max_batch_tokens=500,
            )

        self.assertEqual(result.status, "partial")
        self.assertIn("Reached maximum reduction level limit", result.match_notice)
        self.assertEqual(result.total_eligible_chunks, 40)
        self.assertEqual(result.processed_chunks, 40)

    def test_05_oversized_intermediate_summary_forces_compression_and_convergence(self):
        """Verify oversized intermediate summaries trigger compression to guarantee convergence."""
        chunks = self._create_chunks(16, pages=4)
        # Create a provider that produces long intermediate summaries
        verbose_provider = _MockHierarchicalProvider(prefix="Detailed summary " * 20)

        result = _summarize_chunks_hierarchical(
            chunks=chunks,
            document_id="doc_large",
            filename="textbook.pdf",
            mode="full",
            scope_description="Entire Document",
            length="medium",
            provider=verbose_provider,
            max_batch_tokens=700,
        )

        self.assertIn(result.status, ("complete", "partial"))
        self.assertTrue(result.hierarchical)
        self.assertGreater(result.batches_processed, 1)

    def test_06_odd_number_of_intermediate_inputs_handled_safely(self):
        """Verify odd numbers of input chunks/batches converge without dropping the remainder."""
        chunks = self._create_chunks(15, pages=5)
        provider = _MockHierarchicalProvider()

        result = _summarize_chunks_hierarchical(
            chunks=chunks,
            document_id="doc_large",
            filename="textbook.pdf",
            mode="full",
            scope_description="Entire Document",
            length="medium",
            provider=provider,
            max_batch_tokens=600,
        )

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.total_eligible_chunks, 15)
        self.assertEqual(result.processed_chunks, 15)
        self.assertEqual(result.failed_chunks, 0)


if __name__ == "__main__":
    unittest.main()
