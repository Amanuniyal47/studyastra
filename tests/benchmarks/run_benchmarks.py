"""Standalone CLI runner for Stage 11.1 performance benchmarks.

Usage:
  python -m tests.benchmarks.run_benchmarks [--live-llm] [--quick] [--reps N] [--output-json PATH] [--output-md PATH]
"""

import argparse
import os
import sys
from pathlib import Path

# Enable offline loading for cached HuggingFace models
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Add project root to path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tests.benchmarks.benchmark_suite import BenchmarkSuite
from tests.benchmarks.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="StudyAstra Stage 11.1 Benchmarking & Profiling Harness")
    parser.add_argument("--live-llm", action="store_true", help="Execute live LLM queries against local Ollama instance")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark pass with fewer repetitions and scales")
    parser.add_argument("--reps", type=int, default=3, help="Number of repetitions for latency measurements (default: 3)")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(WORKSPACE_ROOT / "benchmarks" / "benchmark_results.json"),
        help="Path to save machine-readable JSON results",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=str(WORKSPACE_ROOT / "benchmarks" / "benchmark_report.md"),
        help="Path to save human-readable Markdown report",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("STUDYASTRA SYSTEM BENCHMARK & PERFORMANCE PROFILING (STAGE 11.1)")
    print("=" * 72)
    print(f"Repetitions : {args.reps}")
    print(f"Quick Mode  : {args.quick}")
    print(f"Live LLM    : {args.live_llm}")
    print(f"Output JSON : {args.output_json}")
    print(f"Output MD   : {args.output_md}")
    print("=" * 72)

    suite = BenchmarkSuite(
        repetitions=args.reps,
        live_llm=args.live_llm,
        quick=args.quick,
    )

    print("\nRunning benchmark suite across all subsystems...")
    results = suite.run_all()

    # Save reports
    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    ReportGenerator.save_json(results, json_path)
    ReportGenerator.save_markdown(results, md_path)

    print("\n" + "=" * 72)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 72)
    print(f"Total metrics recorded : {len(results.get('metrics', []))}")
    print(f"Results saved to JSON  : {json_path}")
    print(f"Report saved to MD     : {md_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
