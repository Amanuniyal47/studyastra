#!/usr/bin/env python
"""StudyAstra Environment and Deployment Preflight Diagnostics.

Performs read-only system and dependency checks to verify host readiness
for Windows 11 production deployment with Ollama (Qwen2.5 7B) and offline
SentenceTransformers embeddings.

Usage:
    python scripts/check_env.py [--strict]
"""

import argparse
import os
from pathlib import Path
import platform
import sys
import tempfile
import urllib.parse

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DiagnosticResult:
    def __init__(self, name: str, status: str, message: str, is_warning: bool = False):
        self.name = name
        self.status = status  # "PASS", "WARN", "FAIL"
        self.message = message
        self.is_warning = is_warning


def check_python_environment() -> DiagnosticResult:
    v = sys.version_info
    is_64bit = sys.maxsize > 2**32
    arch = platform.architecture()[0]

    if (v.major, v.minor) < (3, 14):
        return DiagnosticResult(
            "Python Version",
            "WARN",
            f"Python {v.major}.{v.minor}.{v.micro} ({arch}) detected. Target is Python 3.14.0+ 64-bit.",
            is_warning=True,
        )
    if not is_64bit:
        return DiagnosticResult(
            "Architecture",
            "FAIL",
            f"32-bit Python detected ({arch}). 64-bit Python is strictly required for FAISS and PyTorch.",
            is_warning=False,
        )

    return DiagnosticResult(
        "Python Runtime",
        "PASS",
        f"Python {v.major}.{v.minor}.{v.micro} 64-bit ({platform.system()} {platform.release()})",
    )


def check_platform() -> DiagnosticResult:
    if sys.platform != "win32":
        return DiagnosticResult(
            "Operating System",
            "WARN",
            f"Non-Windows OS detected: '{sys.platform}'. Production target is Windows 10/11 64-bit.",
            is_warning=True,
        )
    return DiagnosticResult(
        "Operating System",
        "PASS",
        f"Windows platform verified ({platform.platform()})",
    )


def check_required_packages() -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = []
    required = [
        ("streamlit", "Streamlit"),
        ("pymupdf", "PyMuPDF"),
        ("sentence_transformers", "SentenceTransformers"),
        ("faiss", "FAISS (CPU)"),
        ("torch", "PyTorch"),
        ("httpx", "HTTPX"),
        ("numpy", "NumPy"),
        ("dotenv", "python-dotenv"),
    ]

    for mod_name, label in required:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            results.append(DiagnosticResult(f"Package: {label}", "PASS", f"v{ver}"))
        except ImportError:
            results.append(
                DiagnosticResult(
                    f"Package: {label}",
                    "FAIL",
                    f"Package '{mod_name}' is not installed.",
                )
            )

    return results


def check_configuration() -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = []
    try:
        from src.config import validate_production_config, LLM_PROVIDER, OLLAMA_MODEL

        is_valid, errors, warnings = validate_production_config()

        if is_valid:
            results.append(
                DiagnosticResult(
                    "Configuration Validation",
                    "PASS",
                    f"Configuration valid (LLM_PROVIDER={LLM_PROVIDER}, OLLAMA_MODEL={OLLAMA_MODEL})",
                )
            )
        else:
            for err in errors:
                results.append(
                    DiagnosticResult("Configuration Validation", "FAIL", err)
                )

        for warn in warnings:
            results.append(
                DiagnosticResult("Configuration Warning", "WARN", warn, is_warning=True)
            )

    except Exception as exc:
        results.append(
            DiagnosticResult(
                "Configuration Load",
                "FAIL",
                f"Failed to import or validate src.config: {exc}",
            )
        )

    return results


def check_ollama_service() -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = []
    try:
        import httpx
        from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL

        # 1. Probe version endpoint
        version_url = f"{OLLAMA_BASE_URL}/api/version"
        try:
            resp = httpx.get(version_url, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                results.append(
                    DiagnosticResult(
                        "Ollama Service",
                        "PASS",
                        f"Ollama daemon reachable at {OLLAMA_BASE_URL} (v{data.get('version', 'unknown')})",
                    )
                )
            else:
                results.append(
                    DiagnosticResult(
                        "Ollama Service",
                        "WARN",
                        f"Ollama endpoint responded with HTTP {resp.status_code}.",
                        is_warning=True,
                    )
                )
        except Exception as exc:
            results.append(
                DiagnosticResult(
                    "Ollama Service",
                    "WARN",
                    f"Cannot reach Ollama at {OLLAMA_BASE_URL} ({type(exc).__name__}). Ensure Ollama is running.",
                    is_warning=True,
                )
            )
            return results

        # 2. Probe tags endpoint for required model
        tags_url = f"{OLLAMA_BASE_URL}/api/tags"
        try:
            tags_resp = httpx.get(tags_url, timeout=2.0)
            if tags_resp.status_code == 200:
                models = [m.get("name", "") for m in tags_resp.json().get("models", [])]
                # Check exact model or model without tag (e.g. qwen2.5:7b or qwen2.5:7b-latest)
                model_base = OLLAMA_MODEL.split(":")[0]
                matched = any(OLLAMA_MODEL in m or model_base in m for m in models)
                if matched:
                    results.append(
                        DiagnosticResult(
                            "Ollama Model",
                            "PASS",
                            f"Configured model '{OLLAMA_MODEL}' is pulled and ready.",
                        )
                    )
                else:
                    results.append(
                        DiagnosticResult(
                            "Ollama Model",
                            "WARN",
                            f"Model '{OLLAMA_MODEL}' not found in Ollama. Run: ollama pull {OLLAMA_MODEL}",
                            is_warning=True,
                        )
                    )
        except Exception as exc:
            results.append(
                DiagnosticResult(
                    "Ollama Model",
                    "WARN",
                    f"Failed to query Ollama tags: {exc}",
                    is_warning=True,
                )
            )

    except Exception as exc:
        results.append(
            DiagnosticResult(
                "Ollama Diagnostic",
                "WARN",
                f"Diagnostic probe error: {exc}",
                is_warning=True,
            )
        )

    return results


def check_embedding_cache() -> DiagnosticResult:
    try:
        from src.config import EMBEDDING_MODEL
        from src.retrieval.embeddings import verify_offline_cache

        is_complete, missing = verify_offline_cache(EMBEDDING_MODEL)
        if is_complete:
            return DiagnosticResult(
                "Offline Embedding Cache",
                "PASS",
                f"All 6 required cache files verified for '{EMBEDDING_MODEL}'.",
            )
        else:
            missing_str = ", ".join(missing)
            return DiagnosticResult(
                "Offline Embedding Cache",
                "FAIL",
                f"Cache incomplete for '{EMBEDDING_MODEL}' (Missing: {missing_str}). "
                "Run: python scripts/prepare_model.py",
                is_warning=False,
            )
    except Exception as exc:
        return DiagnosticResult(
            "Offline Embedding Cache",
            "FAIL",
            f"Failed to audit embedding cache: {exc}",
            is_warning=False,
        )


def check_faiss_subsystem() -> DiagnosticResult:
    try:
        import faiss

        dim = 384
        index = faiss.IndexFlatL2(dim)
        if index.d != dim:
            return DiagnosticResult(
                "FAISS In-Memory Probe",
                "FAIL",
                f"FAISS index dimension mismatch (expected {dim}, got {index.d}).",
            )
        return DiagnosticResult(
            "FAISS In-Memory Probe",
            "PASS",
            f"FAISS IndexFlatL2 initialized with expected dimension {dim}.",
        )
    except Exception as exc:
        return DiagnosticResult(
            "FAISS In-Memory Probe",
            "FAIL",
            f"FAISS probe failed: {exc}",
        )


def check_storage_permissions() -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = []
    try:
        from src.config import VECTOR_STORE_DIR, STUDY_STATS_FILE

        paths_to_test = [
            ("Vector Store Dir", VECTOR_STORE_DIR if VECTOR_STORE_DIR.exists() else VECTOR_STORE_DIR.parent),
            ("Study Stats Parent", STUDY_STATS_FILE.parent),
        ]

        for label, path in paths_to_test:
            target = path
            while not target.exists() and target != target.parent:
                target = target.parent

            if os.access(target, os.W_OK):
                results.append(
                    DiagnosticResult(
                        f"Permissions: {label}",
                        "PASS",
                        f"Path '{target}' is writable.",
                    )
                )
            else:
                results.append(
                    DiagnosticResult(
                        f"Permissions: {label}",
                        "FAIL",
                        f"Path '{target}' lacks write permissions.",
                    )
                )

        # Check temporary probe outside durable application data
        with tempfile.TemporaryDirectory() as temp_dir:
            probe_file = Path(temp_dir) / ".probe.tmp"
            probe_file.write_text("probe", encoding="utf-8")
            probe_file.unlink()

        results.append(
            DiagnosticResult(
                "Permissions: Temp Directory",
                "PASS",
                "System temp storage is functional and writable.",
            )
        )

    except Exception as exc:
        results.append(
            DiagnosticResult(
                "Storage Permissions",
                "FAIL",
                f"Permission check error: {exc}",
            )
        )

    return results


def run_all_checks() -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = []
    results.append(check_python_environment())
    results.append(check_platform())
    results.extend(check_required_packages())
    results.extend(check_configuration())
    results.extend(check_ollama_service())
    results.append(check_embedding_cache())
    results.append(check_faiss_subsystem())
    results.extend(check_storage_permissions())
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="StudyAstra Preflight Environment Diagnostics")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as critical failures (returns exit code 1 if any warning occurs).",
    )
    args = parser.parse_args()

    print("=" * 76)
    print("           StudyAstra Deployment & Environment Diagnostics")
    print("=" * 76)

    results = run_all_checks()

    pass_count = 0
    warn_count = 0
    fail_count = 0

    for r in results:
        status_tag = f"[{r.status}]"
        if r.status == "PASS":
            pass_count += 1
            print(f"  {status_tag:8} {r.name:28} : {r.message}")
        elif r.status == "WARN":
            warn_count += 1
            print(f"  {status_tag:8} {r.name:28} : {r.message}")
        else:
            fail_count += 1
            print(f"  {status_tag:8} {r.name:28} : {r.message}")

    print("-" * 76)
    print(f"Summary: {pass_count} passed, {warn_count} warning(s), {fail_count} failure(s)")
    print("=" * 76)

    if fail_count > 0:
        print("[RESULT] FAILED: One or more critical requirements failed.")
        return 1
    elif warn_count > 0 and args.strict:
        print("[RESULT] FAILED (Strict Mode): Warnings treated as failure.")
        return 1
    elif warn_count > 0:
        print("[RESULT] PASSED WITH WARNINGS: Ready for development/testing.")
        return 0
    else:
        print("[RESULT] PASSED: Environment is 100% production ready.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
