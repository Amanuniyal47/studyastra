"""Stage 11.2 Verification Benchmark Script.

Measures:
1. Ollama health check latency (cold HTTP request vs warm TTL cache).
2. FAISS transaction staging latency and memory (deepcopy vs shallow copy + metadata clone) at 100, 1000, 5000 chunks.
3. Session state LRU pruning memory efficiency.
"""

import copy
import time
import tracemalloc
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from src.generation.llm_provider import OllamaLLMProvider
from src.retrieval.vector_store import _clone_chunk_metadata
from src.utils.session_lifecycle import prune_lru_cache


def benchmark_health_check():
    provider = OllamaLLMProvider()
    OllamaLLMProvider.invalidate_health_cache()

    # Simulate HTTP check with 50ms mock network latency (or real endpoint if mock)
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "qwen2.5:7b"}]}
        
        def mock_delay(*args, **kwargs):
            time.sleep(0.05)  # 50ms network round-trip simulation
            return mock_resp
            
        mock_get.side_effect = mock_delay

        # Cold check 1 (simulating first call)
        t0 = time.perf_counter()
        provider.check_connection(force=True)
        cold_conn_ms = (time.perf_counter() - t0) * 1000

        # Cold check 2 (simulating model check)
        t0 = time.perf_counter()
        provider.check_model_available("qwen2.5:7b", force=True)
        cold_model_ms = (time.perf_counter() - t0) * 1000

        # Warm checks (hits TTL cache)
        warm_times = []
        for _ in range(100):
            t0 = time.perf_counter()
            provider.check_connection()
            provider.check_model_available("qwen2.5:7b")
            warm_times.append((time.perf_counter() - t0) * 1000)

        warm_avg_ms = sum(warm_times) / len(warm_times)

    print(f"Health Check Cold Connection: {cold_conn_ms:.2f} ms")
    print(f"Health Check Cold Model: {cold_model_ms:.2f} ms")
    print(f"Health Check Warm (Cached) Total per rerun: {warm_avg_ms:.4f} ms")
    print(f"Speedup: {((cold_conn_ms + cold_model_ms) / max(0.0001, warm_avg_ms)):.1f}x")


def benchmark_faiss_staging():
    print("\n--- FAISS Staging Benchmark ---")
    chunk_counts = [100, 1000, 5000]

    for count in chunk_counts:
        # Create realistic chunk dataset
        chunks = [
            {
                "chunk_id": f"chunk_{i:05d}",
                "document_id": f"doc_{i // 50:03d}",
                "text": "This is synthetic educational text representing typical lecture notes and textbook content. " * 5,
                "metadata": {
                    "page_number": (i % 20) + 1,
                    "section_title": f"Section {i % 10}",
                    "token_count": 85,
                    "tags": ["ml", "study", "exam"],
                    "nested_props": {"difficulty": "medium", "importance": 0.8},
                },
            }
            for i in range(count)
        ]

        # 1. Old baseline: copy.deepcopy(chunks)
        tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(5):
            old_staged = copy.deepcopy(chunks)
        old_time_ms = ((time.perf_counter() - t0) / 5) * 1000
        old_current, old_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 2. Optimized: chunks.copy() + outer staging
        tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(5):
            new_staged = chunks.copy()
        new_time_ms = ((time.perf_counter() - t0) / 5) * 1000
        new_current, new_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 3. Read path: clone chunk metadata on retrieval (e.g. 5 retrieved chunks)
        retrieved_chunks = chunks[:5]
        tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(100):
            cloned = [_clone_chunk_metadata(c) for c in retrieved_chunks]
        read_time_ms = ((time.perf_counter() - t0) / 100) * 1000
        tracemalloc.stop()

        speedup = old_time_ms / max(0.00001, new_time_ms)
        ram_reduction = old_peak / max(1, new_peak)

        print(f"Chunks: {count:5d} | "
              f"Deepcopy: {old_time_ms:6.2f} ms ({old_peak / 1024:6.1f} KB) | "
              f"Optimized: {new_time_ms:6.3f} ms ({new_peak / 1024:6.1f} KB) | "
              f"Speedup: {speedup:6.1f}x | RAM reduction: {ram_reduction:5.1f}x | "
              f"Read clone (5 chunks): {read_time_ms:6.4f} ms")


def benchmark_session_lru():
    print("\n--- Session State LRU Pruning Benchmark ---")
    cache = {}
    for i in range(20):
        cache[f"doc_{i}"] = {"summary": "A" * 10000, "page_range": "1-10"}
        prune_lru_cache(cache, max_entries=5)

    print(f"Final cache size after 20 additions: {len(cache)} entries (Bounded to 5)")
    print(f"Retained document IDs: {list(cache.keys())}")


if __name__ == "__main__":
    benchmark_health_check()
    benchmark_faiss_staging()
    benchmark_session_lru()
