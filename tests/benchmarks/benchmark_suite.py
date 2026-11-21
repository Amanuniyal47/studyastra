"""Stage 11.1 Comprehensive Benchmarking Suite.

Measures:
  A. PDF ingestion & extraction scalability (10, 50, 100 pages).
  B. Text chunking & token throughput.
  C. Embedding generation (cold vs warm, batch sizes 8, 16, 32, CPU vs CUDA).
  D. FAISS vector store (init, indexing, commit, retrieval latency, collection sizes, disk usage).
  E. Ollama performance (health checks, TTFT, generation throughput, timeout handling).
  F. Conversational RAG pipeline (Turn 1, Turn 2 with query rewrite, prompt budgeting, latency).
  G. Application startup latency.
"""

import io
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

import src.config as config
from src.ingestion.pdf_loader import extract_pdf_text
from src.ingestion.chunker import chunk_text, estimate_tokens
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import FAISSVectorStore
from src.generation.llm_provider import OllamaLLMProvider, BaseLLMProvider
from src.chat.conversation_memory import ConversationManager
from src.chat.chat_orchestrator import ChatOrchestrator
from tests.benchmarks.profiler import (
    HardwareProfiler,
    MemoryTracker,
    BenchmarkMetric,
    get_current_process_memory,
)
from tests.benchmarks.synthetic_data import (
    generate_synthetic_pdf,
    BENCHMARK_QUERIES,
)


class BenchmarkSuite:
    """Orchestrates comprehensive, reproducible performance benchmarks across all subsystems."""

    def __init__(
        self,
        repetitions: int = 3,
        live_llm: bool = False,
        quick: bool = False,
        temp_dir: Optional[Path] = None,
    ):
        self.repetitions = max(1, repetitions)
        self.live_llm = live_llm
        self.quick = quick
        self.temp_dir = temp_dir
        self.metrics: List[BenchmarkMetric] = []
        self.system_info = HardwareProfiler.get_system_info()

    # ----------------------------------------------------------------------
    # A. PDF Ingestion & Extraction Scalability
    # ----------------------------------------------------------------------
    def run_pdf_ingestion_benchmarks(self) -> List[BenchmarkMetric]:
        """Benchmark PDF text extraction across 10, 50, and 100-page synthetic documents."""
        scales = [10, 50] if self.quick else [10, 50, 100]
        results = []

        for pages in scales:
            pdf_bytes = generate_synthetic_pdf(pages)
            pdf_size_kb = len(pdf_bytes) / 1024

            latencies = []
            heap_peaks = []
            rss_deltas = []

            # Multi-repetition measurement
            for rep in range(self.repetitions):
                stream = io.BytesIO(pdf_bytes)
                with MemoryTracker(track_heap=True) as mem:
                    t0 = time.perf_counter()
                    extracted = extract_pdf_text(stream)
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                latencies.append(elapsed_ms)
                heap_peaks.append(mem.delta_heap_peak_mb)
                rss_deltas.append(mem.delta_rss_mb)

            cold_latency = latencies[0]
            warm_latency = sum(latencies[1:]) / len(latencies[1:]) if len(latencies) > 1 else latencies[0]
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            pages_per_sec = (pages / (avg_latency / 1000.0)) if avg_latency > 0 else 0.0

            metric = BenchmarkMetric(
                name=f"pdf_extraction_{pages}p",
                category="pdf_ingestion",
                status="measured",
                repetition_count=self.repetitions,
                scale=f"{pages} pages ({pdf_size_kb:.1f} KB)",
                cold_latency_ms=round(cold_latency, 2),
                warm_latency_ms=round(warm_latency, 2),
                avg_latency_ms=round(avg_latency, 2),
                min_latency_ms=round(min_latency, 2),
                max_latency_ms=round(max_latency, 2),
                throughput_value=round(pages_per_sec, 1),
                throughput_unit="pages/sec",
                heap_peak_mb=round(max(heap_peaks), 3),
                process_rss_mb=round(max(rss_deltas), 3),
                notes=f"PyMuPDF in-memory text extraction. Pages extracted: {len(extracted)}.",
                limitations="Pure text extraction without OCR; image/scanned documents are not evaluated here."
            )
            results.append(metric)
            self.metrics.append(metric)

        return results

    # ----------------------------------------------------------------------
    # B. Text Chunking Scalability
    # ----------------------------------------------------------------------
    def run_chunking_benchmarks(self) -> List[BenchmarkMetric]:
        """Benchmark sentence- and paragraph-aware token chunking across document scales."""
        scales = [10, 50] if self.quick else [10, 50, 100]
        results = []

        for pages in scales:
            pdf_bytes = generate_synthetic_pdf(pages)
            extracted = extract_pdf_text(io.BytesIO(pdf_bytes))

            latencies = []
            heap_peaks = []
            chunk_counts = []
            token_counts = []

            for rep in range(self.repetitions):
                with MemoryTracker(track_heap=True) as mem:
                    t0 = time.perf_counter()
                    chunks = chunk_text(
                        extracted,
                        document_id=f"doc_{pages}p",
                        filename=f"doc_{pages}p.pdf",
                        chunk_size=config.CHUNK_SIZE_TOKENS,
                        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                latencies.append(elapsed_ms)
                heap_peaks.append(mem.delta_heap_peak_mb)
                chunk_counts.append(len(chunks))
                token_counts.append(sum(c.get("token_count", 0) for c in chunks))

            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            total_tokens = token_counts[0]
            tokens_per_sec = (total_tokens / (avg_latency / 1000.0)) if avg_latency > 0 else 0.0

            metric = BenchmarkMetric(
                name=f"text_chunking_{pages}p",
                category="chunking",
                status="measured",
                repetition_count=self.repetitions,
                scale=f"{pages} pages ({chunk_counts[0]} chunks, {total_tokens} tokens)",
                cold_latency_ms=round(latencies[0], 2),
                warm_latency_ms=round(sum(latencies[1:]) / len(latencies[1:]), 2) if len(latencies) > 1 else round(latencies[0], 2),
                avg_latency_ms=round(avg_latency, 2),
                min_latency_ms=round(min_latency, 2),
                max_latency_ms=round(max_latency, 2),
                throughput_value=round(tokens_per_sec, 0),
                throughput_unit="tokens/sec",
                heap_peak_mb=round(max(heap_peaks), 3),
                notes=f"Configured chunk_size={config.CHUNK_SIZE_TOKENS}, overlap={config.CHUNK_OVERLAP_TOKENS}.",
                limitations="Tokens estimated using regex heuristic (approx. 4 chars per token)."
            )
            results.append(metric)
            self.metrics.append(metric)

        return results

    # ----------------------------------------------------------------------
    # C. Embedding Generation
    # ----------------------------------------------------------------------
    def run_embedding_benchmarks(
        self, embed_service: Optional[EmbeddingService] = None
    ) -> List[BenchmarkMetric]:
        """Benchmark embedding service cold load, warm reuse, batch sizes, and CPU vs CUDA."""
        results = []

        # 1. Cold vs Warm Model Initialization
        if embed_service is None:
            with MemoryTracker(track_heap=True) as mem:
                t0 = time.perf_counter()
                embed_service = EmbeddingService(model_name=config.EMBEDDING_MODEL)
                cold_init_ms = (time.perf_counter() - t0) * 1000

            metric_cold_init = BenchmarkMetric(
                name="embedding_model_cold_init",
                category="embedding",
                status="measured",
                repetition_count=1,
                scale="all-MiniLM-L6-v2 (384-dim)",
                cold_latency_ms=round(cold_init_ms, 2),
                avg_latency_ms=round(cold_init_ms, 2),
                heap_peak_mb=round(mem.delta_heap_peak_mb, 2),
                process_rss_mb=round(mem.delta_rss_mb, 2),
                notes="Initial disk-to-RAM PyTorch model weight loading.",
                limitations="Measured on first instantiation in process."
            )
            results.append(metric_cold_init)
            self.metrics.append(metric_cold_init)

        # Warm reuse
        t0 = time.perf_counter()
        _ = embed_service.get_dimension()
        warm_init_ms = (time.perf_counter() - t0) * 1000

        metric_warm = BenchmarkMetric(
            name="embedding_model_warm_reuse",
            category="embedding",
            status="measured",
            repetition_count=1,
            scale="Cached instance",
            warm_latency_ms=round(warm_init_ms, 3),
            avg_latency_ms=round(warm_init_ms, 3),
            notes="Subsequent access to cached EmbeddingService singleton.",
            limitations="In-process reference lookup."
        )
        results.append(metric_warm)
        self.metrics.append(metric_warm)

        # 2. Embedding Generation across Batch Sizes (8, 16, 32)
        pdf_bytes = generate_synthetic_pdf(10)
        chunks = chunk_text(
            extract_pdf_text(io.BytesIO(pdf_bytes)),
            document_id="doc_bench_embed",
            filename="bench.pdf",
            chunk_size=config.CHUNK_SIZE_TOKENS,
            chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        )
        texts = [c["text"] for c in chunks]
        # Duplicate to 24 chunks to benchmark batch variations reliably
        sample_texts = (texts * 4)[:24]

        batch_sizes = [8, 16, 32] if not self.quick else [16, 32]
        for b_size in batch_sizes:
            latencies = []
            heap_peaks = []
            for rep in range(self.repetitions):
                with MemoryTracker(track_heap=True) as mem:
                    t0 = time.perf_counter()
                    _ = embed_service.embed_texts(sample_texts, batch_size=b_size)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed_ms)
                heap_peaks.append(mem.delta_heap_peak_mb)

            avg_latency = sum(latencies) / len(latencies)
            chunks_per_sec = (len(sample_texts) / (avg_latency / 1000.0)) if avg_latency > 0 else 0.0

            cuda_status = "CUDA" if self.system_info.get("cuda_available") else "CPU"
            metric_batch = BenchmarkMetric(
                name=f"embedding_generation_batch_{b_size}",
                category="embedding",
                status="measured",
                repetition_count=self.repetitions,
                scale=f"{len(sample_texts)} chunks (batch_size={b_size}, {cuda_status})",
                cold_latency_ms=round(latencies[0], 2),
                warm_latency_ms=round(sum(latencies[1:]) / len(latencies[1:]), 2) if len(latencies) > 1 else round(latencies[0], 2),
                avg_latency_ms=round(avg_latency, 2),
                min_latency_ms=round(min(latencies), 2),
                max_latency_ms=round(max(latencies), 2),
                throughput_value=round(chunks_per_sec, 1),
                throughput_unit="chunks/sec",
                heap_peak_mb=round(max(heap_peaks), 3),
                notes=f"Executed on {cuda_status} device.",
                limitations=f"CUDA device availability: {self.system_info.get('cuda_available')}."
            )
            results.append(metric_batch)
            self.metrics.append(metric_batch)

        return results

    # ----------------------------------------------------------------------
    # D. FAISS Vector Database
    # ----------------------------------------------------------------------
    def run_faiss_benchmarks(
        self, embed_service: Optional[EmbeddingService] = None
    ) -> List[BenchmarkMetric]:
        """Benchmark FAISS index initialization, document indexing, commit time, and retrieval."""
        results = []
        if embed_service is None:
            embed_service = EmbeddingService(model_name=config.EMBEDDING_MODEL)

        # 1. VectorStore Initialization
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_path = Path(tmp_str)
            t0 = time.perf_counter()
            _ = FAISSVectorStore(store_dir=tmp_path, embedding_service=embed_service)
            init_ms = (time.perf_counter() - t0) * 1000

        metric_init = BenchmarkMetric(
            name="faiss_store_initialization",
            category="vector_db",
            status="measured",
            repetition_count=1,
            scale="Empty store directory",
            cold_latency_ms=round(init_ms, 2),
            avg_latency_ms=round(init_ms, 2),
            notes="Empty FAISS IndexIDMap2(IndexFlatIP) initialization.",
            limitations="Filesystem creation on temporary directory."
        )
        results.append(metric_init)
        self.metrics.append(metric_init)

        # 2. Indexing & Commit Latency across Scales (10, 50, 100 pages)
        scales = [10, 50] if self.quick else [10, 50, 100]
        with tempfile.TemporaryDirectory() as store_str:
            store_dir = Path(store_str)
            store = FAISSVectorStore(store_dir=store_dir, embedding_service=embed_service)

            for pages in scales:
                pdf_bytes = generate_synthetic_pdf(pages)
                chunks = chunk_text(
                    extract_pdf_text(io.BytesIO(pdf_bytes)),
                    document_id=f"doc_{pages}p",
                    filename=f"doc_{pages}p.pdf",
                    chunk_size=config.CHUNK_SIZE_TOKENS,
                    chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
                )
                embeddings = embed_service.embed_texts([c["text"] for c in chunks])

                with MemoryTracker(track_heap=True) as mem:
                    t0 = time.perf_counter()
                    _ = store.add_document_chunks(
                        document_id=f"doc_{pages}p",
                        filename=f"doc_{pages}p.pdf",
                        page_count=pages,
                        chunks=chunks,
                        embeddings=embeddings,
                    )
                    commit_ms = (time.perf_counter() - t0) * 1000

                active_gen_dir = store.generations_dir / f"gen_{store.active_generation}"
                index_size_kb = (active_gen_dir / "index.faiss").stat().st_size / 1024
                meta_size_kb = (active_gen_dir / "metadata.json").stat().st_size / 1024

                metric_commit = BenchmarkMetric(
                    name=f"faiss_indexing_commit_{pages}p",
                    category="vector_db",
                    status="measured",
                    repetition_count=1,
                    scale=f"{pages} pages ({len(chunks)} chunks)",
                    cold_latency_ms=round(commit_ms, 2),
                    avg_latency_ms=round(commit_ms, 2),
                    heap_peak_mb=round(mem.delta_heap_peak_mb, 3),
                    process_rss_mb=round(mem.delta_rss_mb, 3),
                    notes=f"Snapshot disk size: {index_size_kb + meta_size_kb:.1f} KB (Index: {index_size_kb:.1f} KB).",
                    limitations="Transaction includes deepcopy of chunk metadata."
                )
                results.append(metric_commit)
                self.metrics.append(metric_commit)

            # 3. Vector Retrieval Latency (top_k=4 and top_k=10)
            for top_k in [4, 10]:
                latencies = []
                for query_item in BENCHMARK_QUERIES:
                    q_text = query_item["query"]
                    for rep in range(self.repetitions):
                        t0 = time.perf_counter()
                        q_vec = embed_service.embed_query(q_text)
                        hits = store.search(q_vec, document_id="doc_10p", top_k=top_k)
                        latencies.append((time.perf_counter() - t0) * 1000)

                avg_latency = sum(latencies) / len(latencies)
                throughput_qps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0

                metric_search = BenchmarkMetric(
                    name=f"faiss_search_top_k_{top_k}",
                    category="vector_db",
                    status="measured",
                    repetition_count=len(latencies),
                    scale=f"top_k={top_k} (over indexed collection)",
                    cold_latency_ms=round(latencies[0], 2),
                    warm_latency_ms=round(sum(latencies[1:]) / len(latencies[1:]), 2),
                    avg_latency_ms=round(avg_latency, 2),
                    min_latency_ms=round(min(latencies), 2),
                    max_latency_ms=round(max(latencies), 2),
                    throughput_value=round(throughput_qps, 1),
                    throughput_unit="queries/sec",
                    notes=f"Combined query embedding + FAISS IP cosine search. Hits: {len(hits)}.",
                    limitations="In-memory search on local FAISS Flat index."
                )
                results.append(metric_search)
                self.metrics.append(metric_search)

        return results

    # ----------------------------------------------------------------------
    # E. Ollama Performance & Health Checks
    # ----------------------------------------------------------------------
    def run_ollama_benchmarks(
        self, provider: Optional[BaseLLMProvider] = None
    ) -> List[BenchmarkMetric]:
        """Benchmark Ollama connection check, model check, TTFT, stream velocity, and timeout handling."""
        results = []
        if provider is None:
            provider = OllamaLLMProvider(model_name="qwen2.5:7b")

        # 1. Connection Health Check Latency
        conn_latencies = []
        conn_ok = False
        for _ in range(self.repetitions):
            t0 = time.perf_counter()
            ok, msg = provider.check_connection()
            conn_latencies.append((time.perf_counter() - t0) * 1000)
            conn_ok = ok

        status_str = "measured" if conn_ok else "unavailable"
        notes_str = "HTTP GET to /api/tags." if conn_ok else f"Ollama not reachable: {msg}"

        metric_conn = BenchmarkMetric(
            name="ollama_connection_check",
            category="ollama",
            status=status_str,
            repetition_count=self.repetitions,
            scale="HTTP /api/tags",
            cold_latency_ms=round(conn_latencies[0], 2),
            warm_latency_ms=round(sum(conn_latencies[1:]) / len(conn_latencies[1:]), 2) if len(conn_latencies) > 1 else round(conn_latencies[0], 2),
            avg_latency_ms=round(sum(conn_latencies) / len(conn_latencies), 2),
            min_latency_ms=round(min(conn_latencies), 2),
            max_latency_ms=round(max(conn_latencies), 2),
            notes=notes_str,
            limitations="Uncached HTTP round-trip on every call."
        )
        results.append(metric_conn)
        self.metrics.append(metric_conn)

        # 2. Model Availability Check Latency
        model_latencies = []
        model_ok = False
        for _ in range(self.repetitions):
            t0 = time.perf_counter()
            ok, msg = provider.check_model_available()
            model_latencies.append((time.perf_counter() - t0) * 1000)
            model_ok = ok

        metric_model = BenchmarkMetric(
            name="ollama_model_availability_check",
            category="ollama",
            status="measured" if model_ok else "unavailable",
            repetition_count=self.repetitions,
            scale=f"Model: {getattr(provider, 'model_name', 'qwen2.5:7b')}",
            cold_latency_ms=round(model_latencies[0], 2),
            warm_latency_ms=round(sum(model_latencies[1:]) / len(model_latencies[1:]), 2) if len(model_latencies) > 1 else round(model_latencies[0], 2),
            avg_latency_ms=round(sum(model_latencies) / len(model_latencies), 2),
            min_latency_ms=round(min(model_latencies), 2),
            max_latency_ms=round(max(model_latencies), 2),
            notes="Tag parsing from /api/tags response.",
            limitations="Blocks UI thread if un-cached."
        )
        results.append(metric_model)
        self.metrics.append(metric_model)

        # 3. Unavailable Server Timeout Handling (Deliberate timeout test)
        bad_provider = OllamaLLMProvider(base_url="http://localhost:11439", timeout=1.0)
        t0 = time.perf_counter()
        bad_ok, bad_msg = bad_provider.check_connection()
        timeout_ms = (time.perf_counter() - t0) * 1000

        metric_timeout = BenchmarkMetric(
            name="ollama_unavailable_server_timeout",
            category="ollama",
            status="measured",
            repetition_count=1,
            scale="Unreachable port (http://localhost:11439, timeout=1.0s)",
            cold_latency_ms=round(timeout_ms, 2),
            avg_latency_ms=round(timeout_ms, 2),
            notes=f"Graceful connection failure response: {bad_ok} ({bad_msg}).",
            limitations="Measures OS socket connection refusal or HTTP timeout ceiling."
        )
        results.append(metric_timeout)
        self.metrics.append(metric_timeout)

        # 4. Live Generation & Streaming (Optional / Live Hardware only)
        if conn_ok and model_ok and self.live_llm:
            sys_prompt = "You are a concise academic tutor."
            usr_prompt = "In 2 brief sentences, state the function of ATP synthase."

            # Cold generation & TTFT
            t0 = time.perf_counter()
            first_token_time = None
            tokens = []
            for tok in provider.generate_stream(sys_prompt, usr_prompt, temperature=0.1, max_tokens=100):
                if first_token_time is None:
                    first_token_time = time.perf_counter() - t0
                tokens.append(tok)
            total_duration_s = time.perf_counter() - t0
            tok_per_sec = len(tokens) / total_duration_s if total_duration_s > 0 else 0.0

            metric_stream = BenchmarkMetric(
                name="ollama_stream_generation",
                category="ollama",
                status="measured",
                repetition_count=1,
                scale="qwen2.5:7b stream (100 max tokens)",
                cold_latency_ms=round((first_token_time * 1000) if first_token_time else 0, 2),
                avg_latency_ms=round(total_duration_s * 1000, 2),
                throughput_value=round(tok_per_sec, 1),
                throughput_unit="tokens/sec",
                notes=f"Yielded {len(tokens)} chunks in {total_duration_s:.2f}s. TTFT: {(first_token_time * 1000):.1f}ms.",
                limitations="Dependent on GPU memory allocation and local Ollama daemon state."
            )
            results.append(metric_stream)
            self.metrics.append(metric_stream)
        else:
            metric_stream = BenchmarkMetric(
                name="ollama_stream_generation",
                category="ollama",
                status="unavailable",
                scale="qwen2.5:7b",
                notes="Live generation skipped (either live_llm is False or Ollama server unreachable).",
                limitations="Live generation requires local Ollama service running."
            )
            results.append(metric_stream)
            self.metrics.append(metric_stream)

        return results

    # ----------------------------------------------------------------------
    # F. Conversational RAG Latency
    # ----------------------------------------------------------------------
    def run_conversational_rag_benchmarks(
        self,
        embed_service: Optional[EmbeddingService] = None,
        provider: Optional[BaseLLMProvider] = None,
    ) -> List[BenchmarkMetric]:
        """Benchmark end-to-end multi-turn RAG (Turn 1 direct question, Turn 2 conversational follow-up)."""
        results = []
        if embed_service is None:
            embed_service = EmbeddingService(model_name=config.EMBEDDING_MODEL)
        if provider is None:
            provider = OllamaLLMProvider(model_name="qwen2.5:7b")

        ok_conn, _ = provider.check_connection()
        ok_model, _ = provider.check_model_available()

        with tempfile.TemporaryDirectory() as store_str:
            store_dir = Path(store_str)
            store = FAISSVectorStore(store_dir=store_dir, embedding_service=embed_service)

            # Ingest synthetic 10-page doc
            pdf_bytes = generate_synthetic_pdf(10)
            chunks = chunk_text(
                extract_pdf_text(io.BytesIO(pdf_bytes)),
                document_id="doc_chat_bench",
                filename="biology.pdf",
                chunk_size=config.CHUNK_SIZE_TOKENS,
                chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
            )
            embeddings = embed_service.embed_texts([c["text"] for c in chunks])
            store.add_document_chunks(
                document_id="doc_chat_bench",
                filename="biology.pdf",
                page_count=10,
                chunks=chunks,
                embeddings=embeddings,
            )

            mgr = ConversationManager()
            orchestrator = ChatOrchestrator(
                store=store,
                embedding_service=embed_service,
                provider=provider,
                conversation_manager=mgr,
            )

            if ok_conn and ok_model and self.live_llm:
                # Turn 1: Direct question
                t0 = time.perf_counter()
                res1 = orchestrator.orchestrate_turn(
                    query="What occurs in the thylakoid membranes during photosynthesis?",
                    document_id="doc_chat_bench",
                )
                turn1_ms = (time.perf_counter() - t0) * 1000

                metric_turn1 = BenchmarkMetric(
                    name="conversational_rag_turn1_direct",
                    category="conversational_rag",
                    status="measured" if res1.success else "unavailable",
                    repetition_count=1,
                    scale="Direct initial query (no rewrite needed)",
                    cold_latency_ms=round(turn1_ms, 2),
                    avg_latency_ms=round(turn1_ms, 2),
                    notes=f"Turn 1 success: {res1.success}. Cited {len(res1.citations)} sources.",
                    limitations="Includes retrieval, prompt assembly, and live Qwen2.5 7B generation."
                )
                results.append(metric_turn1)
                self.metrics.append(metric_turn1)

                # Turn 2: Follow-up question requiring query rewriting
                t0 = time.perf_counter()
                res2 = orchestrator.orchestrate_turn(
                    query="What pigments absorb light there?",
                    document_id="doc_chat_bench",
                    conversation_id=res1.conversation_id,
                )
                turn2_ms = (time.perf_counter() - t0) * 1000

                metric_turn2 = BenchmarkMetric(
                    name="conversational_rag_turn2_followup",
                    category="conversational_rag",
                    status="measured" if res2.success else "unavailable",
                    repetition_count=1,
                    scale="Follow-up pronoun query (requires query rewrite)",
                    cold_latency_ms=round(turn2_ms, 2),
                    avg_latency_ms=round(turn2_ms, 2),
                    notes=f"Turn 2 success: {res2.success}. Rewritten: {res2.is_rewritten} ('{res2.retrieval_query}').",
                    limitations="Includes query rewriter call + retrieval + prompt assembly + generation."
                )
                results.append(metric_turn2)
                self.metrics.append(metric_turn2)
            else:
                # Non-live / offline: Measure deterministic stages (retrieval + prompt assembly)
                from src.chat.conversational_retriever import retrieve_chat_context
                from src.chat.chat_prompts import build_grounded_chat_prompt

                t0 = time.perf_counter()
                retrieved = retrieve_chat_context(
                    query="What occurs in thylakoid membranes?",
                    document_id="doc_chat_bench",
                    store=store,
                    embedding_service=embed_service,
                )
                grounded_prompt = build_grounded_chat_prompt(
                    query="What occurs in thylakoid membranes?",
                    chunks=retrieved,
                )
                deterministic_ms = (time.perf_counter() - t0) * 1000

                metric_det = BenchmarkMetric(
                    name="conversational_rag_deterministic_pipeline",
                    category="conversational_rag",
                    status="measured",
                    repetition_count=1,
                    scale="Retrieval + Prompt Assembly (Offline)",
                    avg_latency_ms=round(deterministic_ms, 2),
                    notes="Deterministic retrieval + prompt context construction without LLM generation.",
                    limitations="Live generation skipped because live_llm is False or Ollama is offline."
                )
                results.append(metric_det)
                self.metrics.append(metric_det)

        return results

    # ----------------------------------------------------------------------
    # G. Application Startup & Import Latency
    # ----------------------------------------------------------------------
    def run_startup_benchmarks(self) -> List[BenchmarkMetric]:
        """Benchmark module import latency and cold initialization."""
        results = []

        modules_to_test = [
            ("src.config", "Configuration loading"),
            ("src.ingestion.pdf_loader", "PyMuPDF integration"),
            ("src.ingestion.chunker", "Chunker module"),
            ("src.retrieval.embeddings", "SentenceTransformers wrapper"),
            ("src.retrieval.vector_store", "FAISS vector store module"),
            ("src.chat.chat_orchestrator", "Conversational orchestrator module"),
        ]

        total_import_ms = 0.0
        for mod_name, desc in modules_to_test:
            t0 = time.perf_counter()
            __import__(mod_name)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_import_ms += elapsed_ms

        metric_import = BenchmarkMetric(
            name="application_module_imports",
            category="startup",
            status="measured",
            repetition_count=1,
            scale=f"{len(modules_to_test)} core modules",
            cold_latency_ms=round(total_import_ms, 2),
            avg_latency_ms=round(total_import_ms, 2),
            notes="Module import latency across core subsystems.",
            limitations="Cached by sys.modules on subsequent imports."
        )
        results.append(metric_import)
        self.metrics.append(metric_import)

        return results

    # ----------------------------------------------------------------------
    # Run Complete Suite
    # ----------------------------------------------------------------------
    def run_all(
        self,
        embed_service: Optional[EmbeddingService] = None,
        provider: Optional[BaseLLMProvider] = None,
    ) -> Dict[str, Any]:
        """Execute all benchmark categories and aggregate results."""
        self.metrics.clear()

        # Run startup
        self.run_startup_benchmarks()

        # Run PDF Ingestion
        self.run_pdf_ingestion_benchmarks()

        # Run Chunking
        self.run_chunking_benchmarks()

        # Run Embedding
        self.run_embedding_benchmarks(embed_service=embed_service)

        # Run FAISS
        self.run_faiss_benchmarks(embed_service=embed_service)

        # Run Ollama
        self.run_ollama_benchmarks(provider=provider)

        # Run Conversational RAG
        self.run_conversational_rag_benchmarks(
            embed_service=embed_service, provider=provider
        )

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "system_info": self.system_info,
            "config": {
                "repetitions": self.repetitions,
                "live_llm": self.live_llm,
                "quick": self.quick,
            },
            "metrics": [m.to_dict() for m in self.metrics],
        }
