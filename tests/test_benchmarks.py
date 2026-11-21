"""Unit and integration tests for the Stage 11.1 benchmarking harness."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import numpy as np

from tests.benchmarks.profiler import (
    HardwareProfiler,
    MemoryTracker,
    ProcessMemorySnapshot,
    BenchmarkMetric,
    get_current_process_memory,
    get_gpu_memory,
)
from tests.benchmarks.synthetic_data import (
    generate_synthetic_pdf,
    generate_synthetic_pdf_stream,
    SAMPLE_ACADEMIC_PARAGRAPHS,
)
from tests.benchmarks.benchmark_suite import BenchmarkSuite
from tests.benchmarks.report_generator import ReportGenerator


class TestBenchmarkProfiler(unittest.TestCase):
    """Test profiler utilities, system metadata collection, and memory trackers."""

    def test_hardware_profiler_system_info(self):
        info = HardwareProfiler.get_system_info()
        self.assertIsInstance(info, dict)
        self.assertIn("os_name", info)
        self.assertIn("python_version", info)
        self.assertIn("cpu_count_logical", info)
        self.assertIn("total_ram_gb", info)
        self.assertIn("torch_version", info)
        self.assertIn("cuda_available", info)

    def test_current_process_memory(self):
        rss_mb, peak_rss_mb = get_current_process_memory()
        self.assertIsInstance(rss_mb, float)
        self.assertIsInstance(peak_rss_mb, float)
        self.assertGreaterEqual(rss_mb, 0.0)
        self.assertGreaterEqual(peak_rss_mb, 0.0)

    def test_gpu_memory_detection(self):
        vram, peak_vram = get_gpu_memory()
        # On CPU systems, vram should be None; on CUDA it should be float
        if vram is not None:
            self.assertIsInstance(vram, float)
            self.assertGreaterEqual(vram, 0.0)

    def test_memory_tracker_context_manager(self):
        with MemoryTracker(track_heap=True) as mem:
            _ = [i for i in range(100000)]
        self.assertGreaterEqual(mem.delta_heap_peak_mb, 0.0)
        self.assertGreaterEqual(mem.delta_rss_mb, 0.0)
        self.assertIsInstance(mem.start_snapshot, ProcessMemorySnapshot)
        self.assertIsInstance(mem.end_snapshot, ProcessMemorySnapshot)

    def test_check_ollama_status_unreachable_port(self):
        status = HardwareProfiler.check_ollama_status("http://localhost:11439")
        self.assertIsInstance(status, dict)
        self.assertFalse(status["is_reachable"])
        self.assertEqual(status["models"], [])

    def test_benchmark_metric_to_dict(self):
        metric = BenchmarkMetric(
            name="test_metric",
            category="test_cat",
            status="measured",
            repetition_count=3,
            scale="10 pages",
            cold_latency_ms=100.5,
            warm_latency_ms=50.2,
            avg_latency_ms=66.9,
            min_latency_ms=50.0,
            max_latency_ms=100.5,
            throughput_value=150.0,
            throughput_unit="pages/sec",
            heap_peak_mb=1.25,
            notes="Test notes",
            limitations="Test limitations"
        )
        d = metric.to_dict()
        self.assertEqual(d["name"], "test_metric")
        self.assertEqual(d["category"], "test_cat")
        self.assertEqual(d["status"], "measured")
        self.assertEqual(d["avg_latency_ms"], 66.9)


class TestSyntheticData(unittest.TestCase):
    """Test synthetic PDF generation and reproducibility."""

    def test_generate_synthetic_pdf_page_counts(self):
        for pages in [1, 3, 5]:
            pdf_bytes = generate_synthetic_pdf(pages)
            self.assertIsInstance(pdf_bytes, bytes)
            self.assertGreater(len(pdf_bytes), 0)

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            self.assertEqual(len(doc), pages)
            first_page_text = doc[0].get_text()
            self.assertIn("Photosynthesis", first_page_text)
            doc.close()

    def test_generate_synthetic_pdf_stream(self):
        stream = generate_synthetic_pdf_stream(2)
        doc = fitz.open(stream=stream, filetype="pdf")
        self.assertEqual(len(doc), 2)
        doc.close()


class TestBenchmarkSuite(unittest.TestCase):
    """Test benchmark suite components in offline/isolated environments."""

    def setUp(self):
        self.suite = BenchmarkSuite(repetitions=1, live_llm=False, quick=True)

    def test_run_startup_benchmarks(self):
        metrics = self.suite.run_startup_benchmarks()
        self.assertGreater(len(metrics), 0)
        self.assertEqual(metrics[0].category, "startup")
        self.assertGreater(metrics[0].avg_latency_ms, 0.0)

    def test_run_pdf_ingestion_benchmarks_quick(self):
        metrics = self.suite.run_pdf_ingestion_benchmarks()
        self.assertGreaterEqual(len(metrics), 2)
        for m in metrics:
            self.assertEqual(m.category, "pdf_ingestion")
            self.assertIn("pages/sec", m.throughput_unit)
            self.assertGreater(m.throughput_value, 0.0)

    def test_run_chunking_benchmarks_quick(self):
        metrics = self.suite.run_chunking_benchmarks()
        self.assertGreaterEqual(len(metrics), 2)
        for m in metrics:
            self.assertEqual(m.category, "chunking")
            self.assertIn("tokens/sec", m.throughput_unit)
            self.assertGreater(m.throughput_value, 0.0)

    def test_run_faiss_benchmarks_isolated(self):
        # Create a mock embedding service to avoid loading SentenceTransformers weights in quick test
        mock_embed = MagicMock()
        mock_embed.model_name = "all-MiniLM-L6-v2"
        mock_embed.get_dimension.return_value = 384
        mock_embed.embed_texts.side_effect = lambda texts, **kw: np.ones((len(texts), 384), dtype=np.float32)
        mock_embed.embed_query.return_value = np.ones((384,), dtype=np.float32)

        metrics = self.suite.run_faiss_benchmarks(embed_service=mock_embed)
        self.assertGreater(len(metrics), 0)
        categories = [m.category for m in metrics]
        self.assertTrue(all(c == "vector_db" for c in categories))

    def test_run_ollama_benchmarks_mocked_provider(self):
        mock_provider = MagicMock()
        mock_provider.check_connection.return_value = (True, "Connected")
        mock_provider.check_model_available.return_value = (True, "Available")
        mock_provider.model_name = "qwen2.5:7b"

        suite = BenchmarkSuite(repetitions=1, live_llm=False, quick=True)
        metrics = suite.run_ollama_benchmarks(provider=mock_provider)
        self.assertGreaterEqual(len(metrics), 3)

        conn_m = next(m for m in metrics if m.name == "ollama_connection_check")
        self.assertEqual(conn_m.status, "measured")

    def test_run_conversational_rag_offline_deterministic(self):
        mock_embed = MagicMock()
        mock_embed.model_name = "all-MiniLM-L6-v2"
        mock_embed.get_dimension.return_value = 384
        mock_embed.embed_texts.side_effect = lambda texts, **kw: np.ones((len(texts), 384), dtype=np.float32)
        mock_embed.embed_query.return_value = np.ones((384,), dtype=np.float32)

        mock_provider = MagicMock()
        mock_provider.check_connection.return_value = (False, "Offline")
        mock_provider.check_model_available.return_value = (False, "Offline")

        suite = BenchmarkSuite(repetitions=1, live_llm=False, quick=True)
        metrics = suite.run_conversational_rag_benchmarks(embed_service=mock_embed, provider=mock_provider)
        self.assertGreaterEqual(len(metrics), 1)
        self.assertEqual(metrics[0].category, "conversational_rag")


class TestReportGenerator(unittest.TestCase):
    """Test JSON export and Markdown generation from benchmark metrics."""

    def test_json_and_markdown_generation(self):
        mock_data = {
            "timestamp": "2026-11-20T12:00:00Z",
            "system_info": {
                "os_name": "Windows",
                "os_release": "11",
                "architecture": "AMD64",
                "python_version": "3.14.6",
                "python_compiler": "MSC v.1944",
                "cpu_count_logical": 16,
                "total_ram_gb": 16.0,
                "available_ram_gb": 4.5,
                "torch_version": "2.10.0+cpu",
                "cuda_available": False,
                "gpu_device_name": "N/A (PyTorch running on CPU)",
                "cuda_version": "N/A",
                "gpu_vram_total_gb": 0.0,
            },
            "config": {"repetitions": 1, "live_llm": False, "quick": True},
            "metrics": [
                {
                    "name": "pdf_extraction_10p",
                    "category": "pdf_ingestion",
                    "status": "measured",
                    "scale": "10 pages",
                    "cold_latency_ms": 15.0,
                    "warm_latency_ms": 12.0,
                    "avg_latency_ms": 13.5,
                    "throughput_value": 740.0,
                    "throughput_unit": "pages/sec",
                    "heap_peak_mb": 0.02,
                    "notes": "PyMuPDF extraction",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp_path = Path(tmp_str)
            json_file = tmp_path / "test_out.json"
            md_file = tmp_path / "test_out.md"

            ReportGenerator.save_json(mock_data, json_file)
            self.assertTrue(json_file.exists())
            with open(json_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["timestamp"], "2026-11-20T12:00:00Z")

            ReportGenerator.save_markdown(mock_data, md_file)
            self.assertTrue(md_file.exists())
            with open(md_file, "r", encoding="utf-8") as f:
                md_text = f.read()
            self.assertIn("# StudyAstra System Benchmark & Profiling Report", md_text)
            self.assertIn("pdf_extraction_10p", md_text)
            self.assertIn("PyMuPDF extraction", md_text)


if __name__ == "__main__":
    unittest.main()
