from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pymupdf

from src.generation.llm_provider import BaseLLMProvider, LLMProviderError
from src.generation.summarizer import (
    SummarizationError,
    format_page_range,
    summarize_document,
    summarize_page_range,
    summarize_section,
    summarize_topic,
)
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


class _RecordingMockProvider(BaseLLMProvider):
    provider_name = "mock_groq"

    def __init__(
        self,
        model_name: str = "mock-model-70b",
        *,
        fail_on_call: int | None = None,
        response_builder=None,
    ):
        super().__init__(model_name=model_name)
        self.calls: list[dict] = []
        self.fail_on_call = fail_on_call
        self.response_builder = response_builder

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
            raise LLMProviderError(f"Simulated API failure on call {call_num}")

        if self.response_builder is not None:
            return self.response_builder(call_num, system_prompt, user_prompt, max_tokens)

        # Preserve any page markers seen in the prompt so downstream tests can verify propagation
        import re

        pages_seen = sorted({int(p) for p in re.findall(r"Page (\d+)", user_prompt)})
        if not pages_seen:
            pages_seen = sorted({int(p) for p in re.findall(r"\[p\. (\d+)\]", user_prompt)})
        cite_str = ", ".join(f"[p. {p}]" for p in pages_seen) if pages_seen else "[no-pages]"
        return f"Condensed facts and definitions {cite_str}."


class TestStage61bDocumentChunkHelpersAndSummarizer(unittest.TestCase):
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

    def _index_sample_pdfs(self) -> tuple[str, str]:
        pdf_bio = _make_pdf(
            [
                "Chapter 1: Photosynthesis\nPhotosynthesis in plant chloroplasts converts sunlight into glucose.",
                "Chapter 2: Cellular Respiration\nMitochondria produce ATP during cellular respiration using oxygen.",
                "Respiration continues with glycolysis, the Krebs cycle, and oxidative phosphorylation.",
                "Chapter 3: Cell Division\nMitosis divides a parent cell into two identical daughter cells.",
            ],
            name="biology.pdf",
        )
        pdf_cs = _make_pdf(
            [
                "FAISS Indexing Guide\nFAISS indexes dense vector embeddings for fast similarity search.",
                "Vector quantization reduces memory footprint for million-scale embedding indexes.",
            ],
            name="faiss_guide.pdf",
        )
        r_bio = self.store.add_pdf(pdf_bio, embedding_service=self.service)
        r_cs = self.store.add_pdf(pdf_cs, embedding_service=self.service)
        return r_bio["document_id"], r_cs["document_id"]

    def test_document_chunk_helpers_isolated_ordered_and_immutable(self):
        bio_id, cs_id = self._index_sample_pdfs()

        bio_chunks = self.store.get_document_chunks(bio_id)
        self.assertEqual(len(bio_chunks), 4)
        self.assertTrue(all(c["document_id"] == bio_id for c in bio_chunks))
        self.assertEqual([c["page"] for c in bio_chunks], [1, 2, 3, 4])
        self.assertEqual(self.store.get_document_page_numbers(bio_id), [1, 2, 3, 4])
        self.assertEqual(self.store.get_document_page_numbers(cs_id), [1, 2])

        # Mutating returned deep copy must NOT mutate internal store metadata
        bio_chunks[0]["text"] = "TAMPERED"
        fresh_chunks = self.store.get_document_chunks(bio_id)
        self.assertNotEqual(fresh_chunks[0]["text"], "TAMPERED")

        # Page range filtering
        mid = self.store.get_document_chunks(bio_id, start_page=2, end_page=3)
        self.assertEqual([c["page"] for c in mid], [2, 3])

        with self.assertRaises(ValueError):
            self.store.get_document_chunks(bio_id, start_page=4, end_page=2)
        with self.assertRaises(KeyError):
            self.store.get_document_chunks("nonexistent_doc")

    def test_format_page_range_helper(self):
        self.assertEqual(format_page_range([3]), "Page 3")
        self.assertEqual(format_page_range([1, 2, 3, 4]), "Pages 1-4")
        self.assertEqual(format_page_range([1, 2, 5, 7, 8]), "Pages 1-2, 5, 7-8")

    def test_full_summary_lengths_and_never_indexes_summaries(self):
        bio_id, _ = self._index_sample_pdfs()
        vectors_before = self.store.total_vectors
        docs_before = self.store.total_documents

        max_tokens_by_length = {}
        for length in ("short", "medium", "detailed"):
            provider = _RecordingMockProvider()
            result = summarize_document(
                self.store,
                bio_id,
                length=length,
                provider=provider,
            )
            self.assertEqual(result.filename, "biology.pdf")
            self.assertEqual(result.document_id, bio_id)
            self.assertEqual(result.mode, "full")
            self.assertEqual(result.length, length)
            self.assertEqual(result.pages_covered, [1, 2, 3, 4])
            self.assertEqual(result.page_range_label, "Pages 1-4")
            self.assertEqual(result.chunks_used, 4)
            self.assertEqual(result.provider, "mock_groq")
            self.assertEqual(result.model, "mock-model-70b")
            self.assertFalse(result.hierarchical)
            max_tokens_by_length[length] = provider.calls[0]["max_tokens"]

            # Never mixes content from faiss_guide.pdf
            self.assertNotIn("faiss_guide.pdf", provider.calls[0]["user_prompt"])
            self.assertNotIn("Vector quantization", provider.calls[0]["user_prompt"])

        self.assertLess(max_tokens_by_length["short"], max_tokens_by_length["medium"])
        self.assertLess(max_tokens_by_length["medium"], max_tokens_by_length["detailed"])

        # Summaries are never indexed into FAISS
        self.assertEqual(self.store.total_vectors, vectors_before)
        self.assertEqual(self.store.total_documents, docs_before)

    def test_page_range_summary_and_validation(self):
        bio_id, _ = self._index_sample_pdfs()
        provider = _RecordingMockProvider()

        res = summarize_page_range(
            self.store,
            bio_id,
            start_page=2,
            end_page=3,
            length="short",
            provider=provider,
        )
        self.assertEqual(res.mode, "page_range")
        self.assertEqual(res.pages_covered, [2, 3])
        self.assertEqual(res.page_range_label, "Pages 2-3")
        self.assertEqual(res.chunks_used, 2)
        prompt = provider.calls[0]["user_prompt"]
        self.assertIn("Cellular Respiration", prompt)
        self.assertNotIn("Photosynthesis", prompt)
        self.assertNotIn("Mitosis", prompt)

        with self.assertRaises(ValueError):
            summarize_page_range(self.store, bio_id, start_page=3, end_page=1, provider=provider)
        with self.assertRaises(ValueError):
            summarize_page_range(self.store, bio_id, start_page=10, end_page=12, provider=provider)

    def test_section_summary_exact_heading_vs_semantic_fallback_vs_unreliable(self):
        bio_id, _ = self._index_sample_pdfs()

        # 1. Exact heading match stops cleanly before Chapter 3
        provider_exact = _RecordingMockProvider()
        res_exact = summarize_section(
            self.store,
            bio_id,
            "Chapter 2: Cellular Respiration",
            length="medium",
            provider=provider_exact,
            embedding_service=self.service,
        )
        self.assertEqual(res_exact.match_strategy, "exact_heading")
        self.assertIsNone(res_exact.match_notice)
        self.assertEqual(res_exact.pages_covered, [2, 3])
        self.assertIn("Cellular Respiration", provider_exact.calls[0]["user_prompt"])
        self.assertNotIn("Cell Division", provider_exact.calls[0]["user_prompt"])

        # 2. Semantic fallback clearly disclosed when heading text isn't verbatim
        provider_fallback = _RecordingMockProvider()
        res_fallback = summarize_section(
            self.store,
            bio_id,
            "sunlight glucose chloroplast mechanism",
            length="short",
            top_k=1,
            provider=provider_fallback,
            embedding_service=self.service,
        )
        self.assertEqual(res_fallback.match_strategy, "semantic_fallback")
        self.assertIsNotNone(res_fallback.match_notice)
        self.assertIn("semantic fallback", res_fallback.match_notice.lower())
        self.assertEqual(res_fallback.pages_covered, [1])

        # 3. Unreliable match raises clear ValueError instead of misleading summary
        with self.assertRaises(ValueError) as ctx:
            summarize_section(
                self.store,
                bio_id,
                "ancient empire treaty monarchy",
                min_similarity=0.5,
                provider=provider_fallback,
                embedding_service=self.service,
            )
        self.assertIn("could not be reliably identified", str(ctx.exception).lower())

    def test_topic_summary_strictly_scoped_to_single_pdf(self):
        bio_id, cs_id = self._index_sample_pdfs()
        provider = _RecordingMockProvider()

        # Even when querying FAISS topic against biology.pdf, cs_notes.pdf MUST NOT leak in
        res_bio = summarize_topic(
            self.store,
            bio_id,
            "FAISS vector similarity",
            top_k=2,
            length="short",
            embedding_service=self.service,
            provider=provider,
        )
        self.assertEqual(res_bio.document_id, bio_id)
        self.assertEqual(res_bio.filename, "biology.pdf")
        self.assertTrue(all(c["metadata"]["document_id"] == bio_id for c in res_bio.source_chunks))
        self.assertNotIn("faiss_guide.pdf", provider.calls[0]["user_prompt"])
        # Low relevance notice surfaced since biology.pdf isn't about FAISS
        self.assertIsNotNone(res_bio.match_notice)

        # Querying FAISS topic against faiss_guide.pdf returns high-relevance CS chunks
        provider_cs = _RecordingMockProvider()
        res_cs = summarize_topic(
            self.store,
            cs_id,
            "FAISS vector similarity",
            top_k=2,
            length="medium",
            embedding_service=self.service,
            provider=provider_cs,
        )
        self.assertEqual(res_cs.filename, "faiss_guide.pdf")
        self.assertEqual(res_cs.pages_covered, [1, 2])
        self.assertIsNone(res_cs.match_notice)

    def test_hierarchical_multi_level_budget_enforced_with_prompt_overhead_and_page_preservation(self):
        # Build a 12-page PDF with distinct page facts so multiple map batches & >=2 reduce levels run
        pages = []
        for p in range(1, 13):
            body = " ".join(
                f"Page {p} fact {i}: biological concept {p}_{i} defines energy transfer example {i}."
                for i in range(1, 26)
            )
            pages.append(f"Section {p}: Topic {p}\n{body}")

        large_pdf = _make_pdf(pages, name="large_biology.pdf")
        res_add = self.store.add_pdf(large_pdf, embedding_service=self.service)
        doc_id = res_add["document_id"]

        # Choose a tight token budget (420 tokens) so:
        # - Prompt overhead (~180-220 tokens) MUST be accounted for
        # - Multiple map batches are forced
        # - Intermediate summaries require >= 2 reduction levels
        budget = 420

        def _multi_level_response(call_num, sys_prompt, usr_prompt, max_tokens):
            import re

            pages_found = sorted({int(x) for x in re.findall(r"Page (\d+)|Pages (\d+)", usr_prompt) for x in x if x})
            p_tag = ", ".join(f"[p. {p}]" for p in pages_found[:4]) or "[p. 1]"
            # Map responses (~45 tokens each) force intermediate level-1 reduce grouping under 420 budget
            if "Batch:" in usr_prompt:
                return (
                    f"Key definition and concrete example preserved from {p_tag}: "
                    "energy transfer mechanism maintains cellular homeostasis and metabolic output "
                    "across sequential biochemical stages with verifiable study milestones."
                )
            # Reduce responses compact cleanly while keeping page citations
            return f"Consolidated study findings covering {p_tag}."

        provider = _RecordingMockProvider(response_builder=_multi_level_response)
        progress_events: list[tuple[int, int, str]] = []

        result = summarize_document(
            self.store,
            doc_id,
            length="medium",
            provider=provider,
            max_batch_tokens=budget,
            progress_callback=lambda step, total, msg: progress_events.append((step, total, msg)),
        )

        self.assertTrue(result.hierarchical)
        self.assertGreater(result.batches_processed, 4)
        self.assertGreaterEqual(result.reduction_levels, 2)
        self.assertEqual(result.pages_covered, list(range(1, 13)))
        self.assertEqual(result.page_range_label, "Pages 1-12")
        self.assertEqual(result.filename, "large_biology.pdf")

        # CRITICAL: Verify EVERY SINGLE LLM call stayed strictly within budget INCLUDING prompt overhead
        for call in provider.calls:
            self.assertLessEqual(
                call["total_prompt_tokens"],
                budget,
                f"Call {call['call_num']} exceeded token budget: "
                f"{call['total_prompt_tokens']} > {budget}",
            )

        # Verify final reduce prompt still carries earliest and latest page ranges through multi-level reduction
        final_prompt = provider.calls[-1]["user_prompt"]
        self.assertIn("Pages 1-12", final_prompt)
        self.assertIn("p. 1", final_prompt)
        self.assertIn("12", final_prompt)

        # Verify progress callback reported monotonically
        self.assertGreaterEqual(len(progress_events), result.batches_processed)
        steps = [e[0] for e in progress_events]
        self.assertEqual(steps, list(range(1, len(progress_events) + 1)))
        self.assertTrue(all(step <= total for step, total, _ in progress_events))

    def test_budget_too_small_for_prompt_overhead_raises_value_error(self):
        bio_id, _ = self._index_sample_pdfs()
        provider = _RecordingMockProvider()
        with self.assertRaises(ValueError):
            summarize_document(
                self.store,
                bio_id,
                provider=provider,
                max_batch_tokens=50,
            )
        self.assertEqual(len(provider.calls), 0)

    def test_failed_api_call_mid_batch_raises_and_never_returns_partial_summary(self):
        pages = [
            f"Page {p}: " + " ".join(f"Detail {p}.{i} about biology concepts." for i in range(40))
            for p in range(1, 7)
        ]
        res_add = self.store.add_pdf(_make_pdf(pages, name="fail_test.pdf"), embedding_service=self.service)
        doc_id = res_add["document_id"]
        vectors_before = self.store.total_vectors

        failing_provider = _RecordingMockProvider(fail_on_call=2)
        with self.assertRaises(LLMProviderError):
            summarize_document(
                self.store,
                doc_id,
                provider=failing_provider,
                max_batch_tokens=420,
            )

        self.assertEqual(len(failing_provider.calls), 2)
        self.assertEqual(self.store.total_vectors, vectors_before)


if __name__ == "__main__":
    unittest.main()
