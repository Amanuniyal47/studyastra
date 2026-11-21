"""Stage 11.3 Large-PDF Verification and Scalability Benchmark Harness.

Measures:
- Extraction time (layout-aware).
- Chunking time & chunk counts.
- Batched embedding time & peak RAM across 10, 50, 100, 250, 500 pages.
- Map-reduce batch counts and reduction levels for Summarization and Study Notes.
- Progress callback behavior across all stages.
"""

import io
import time
import tracemalloc
from typing import Any, Dict, List
import numpy as np

from src.ingestion.pdf_loader import extract_pdf_text
from src.ingestion.chunker import chunk_text
from src.retrieval.embeddings import EmbeddingService
from src.generation.summarizer import _summarize_chunks_hierarchical
from src.generation.study_notes_generator import generate_study_notes
from src.generation.llm_provider import BaseLLMProvider
from tests.benchmarks.synthetic_data import generate_synthetic_pdf
from tests.benchmarks.profiler import get_current_process_memory


class _FastMockProvider(BaseLLMProvider):
    def __init__(self):
        super().__init__(model_name="qwen2.5:7b")
        self.call_count = 0

    def is_configured(self) -> bool:
        return True

    def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        self.call_count += 1
        return (
            "# Master Study Summary / Notes\n"
            "## 1. Executive Overview\n"
            "Comprehensive synthesis of all biological, computational, and physical systems [Page 1].\n"
            "## 2. Key Concepts & Mechanisms\n"
            "Detailed explanations of photosynthetic reactions, respiration, and operating system paging [Page 2]."
        )


def run_page_scale_benchmark(page_counts: List[int]) -> List[Dict[str, Any]]:
    results = []
    print("=" * 80)
    print("STAGE 11.3 LARGE-PDF SCALE VERIFICATION (10, 50, 100, 250, 500 PAGES)")
    print("=" * 80)

    for pages in page_counts:
        print(f"\n--- Testing Document: {pages} Pages ---")
        t_start = time.perf_counter()

        # 1. Synthetic PDF Generation
        pdf_bytes = generate_synthetic_pdf(pages)
        pdf_size_mb = len(pdf_bytes) / (1024 * 1024)

        # 2. Text Extraction (Layout-Aware)
        extract_callbacks = []
        def _on_extract(p, total, msg):
            extract_callbacks.append(p)

        tracemalloc.start()
        t0 = time.perf_counter()
        pages_text = extract_pdf_text(io.BytesIO(pdf_bytes), progress_callback=_on_extract)
        t_extract = time.perf_counter() - t0
        _, peak_extract_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 3. Chunking
        t0 = time.perf_counter()
        chunks = chunk_text(pages_text, document_id=f"doc_{pages}p", filename=f"doc_{pages}p.pdf")
        t_chunk = time.perf_counter() - t0

        # 4. Batched Embedding
        service = EmbeddingService(batch_size=32)
        total_chunks = len(chunks)
        embed_batches_count = (total_chunks + 31) // 32
        embed_callbacks = []
        def _on_embed(c, total, msg):
            embed_callbacks.append(c)

        # To keep 500-page benchmark swift on CPU without 15 minutes of embedding encoding,
        # we mock _encode_raw for vectors while exercising full batching & callback infrastructure
        def _mock_encode(batch_texts, batch_size):
            return np.ones((len(batch_texts), 384), dtype=np.float32)

        tracemalloc.start()
        t0 = time.perf_counter()
        with (
            unittest_mock_dim(service, 384),
            unittest_mock_encode(service, _mock_encode),
        ):
            embeddings = service.embed_texts(
                [c["text"] for c in chunks],
                batch_size=32,
                normalize=True,
                progress_callback=_on_embed,
            )
        t_embed = time.perf_counter() - t0
        _, peak_embed_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 5. Scalable Map-Reduce Summarization
        mock_provider_sum = _FastMockProvider()
        sum_callbacks = []
        def _on_sum(s, total, msg):
            sum_callbacks.append(s)

        t0 = time.perf_counter()
        sum_result = _summarize_chunks_hierarchical(
            chunks=chunks,
            document_id=f"doc_{pages}p",
            filename=f"doc_{pages}p.pdf",
            mode="full",
            scope_description="Full Document",
            length="medium",
            provider=mock_provider_sum,
            max_batch_tokens=2400,
            progress_callback=_on_sum,
        )
        t_sum = time.perf_counter() - t0

        # 6. Scalable Hierarchical Study Notes
        mock_provider_notes = _FastMockProvider()
        notes_callbacks = []
        def _on_notes(s, total, msg):
            notes_callbacks.append(s)

        mock_store = FastMockStore(chunks, f"doc_{pages}p.pdf")
        t0 = time.perf_counter()
        notes_result = generate_study_notes(
            mock_store,
            f"doc_{pages}p",
            scope="full",
            provider=mock_provider_notes,
            max_batch_tokens=2400,
            progress_callback=_on_notes,
        )
        t_notes = time.perf_counter() - t0

        total_elapsed = time.perf_counter() - t_start

        rss_mb, peak_rss_mb = get_current_process_memory()

        record = {
            "pages": pages,
            "pdf_size_mb": pdf_size_mb,
            "extracted_pages": len(pages_text),
            "total_chunks": total_chunks,
            "t_extract_sec": t_extract,
            "peak_extract_heap_mb": peak_extract_ram / (1024 * 1024),
            "t_chunk_sec": t_chunk,
            "embed_batches": embed_batches_count,
            "t_embed_sec": t_embed,
            "peak_embed_heap_mb": peak_embed_ram / (1024 * 1024),
            "sum_batches": sum_result.batches_processed,
            "sum_levels": sum_result.reduction_levels,
            "sum_status": sum_result.status,
            "notes_batches": notes_result.batches_processed,
            "notes_levels": notes_result.reduction_levels,
            "notes_status": notes_result.status,
            "process_rss_mb": rss_mb,
            "process_peak_rss_mb": peak_rss_mb,
            "total_elapsed_sec": total_elapsed,
        }
        results.append(record)

        print(f"Pages: {pages:3d} | Size: {pdf_size_mb:.2f} MB | Chunks: {total_chunks:5d} | "
              f"Extract: {t_extract:.2f}s | Chunk: {t_chunk:.3f}s | Embed ({embed_batches_count} batches): {t_embed:.2f}s")
        print(f"Summary: {sum_result.batches_processed} batches, {sum_result.reduction_levels} reduction levels, Status: {sum_result.status}")
        print(f"Notes:   {notes_result.batches_processed} batches, {notes_result.reduction_levels} reduction levels, Status: {notes_result.status}")
        print(f"Incremental Heap (tracemalloc): Extract={peak_extract_ram / (1024*1024):.2f} MB, Embed={peak_embed_ram / (1024*1024):.2f} MB")
        print(f"Process Working Set (RSS): Current={rss_mb:.2f} MB, Peak={peak_rss_mb:.2f} MB")

    return results


class FastMockStore:
    def __init__(self, chunks, filename):
        self._chunks = chunks
        self._filename = filename

    def is_document_indexed(self, doc_id):
        return True

    def get_document(self, doc_id):
        return {"filename": self._filename}

    def get_document_chunks(self, doc_id, **kwargs):
        return self._chunks


from unittest.mock import patch

def unittest_mock_dim(service, dim):
    return patch.object(service, "get_dimension", return_value=dim)

def unittest_mock_encode(service, func):
    return patch.object(service, "_encode_raw", side_effect=func)


if __name__ == "__main__":
    page_scales = [10, 50, 100, 250, 500]
    run_page_scale_benchmark(page_scales)
