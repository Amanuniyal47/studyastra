#!/usr/bin/env python
"""StudyAstra Operator Utility: Prepare Offline Embedding Model Cache.

Explicitly downloads and caches the configured SentenceTransformer embedding
model (default: all-MiniLM-L6-v2) for offline operation.

NOTE: Initial download requires an active internet connection.
Normal application startup will NEVER download weights automatically.

Usage:
    python scripts/prepare_model.py [--model <model_name>] [--force]
"""

import argparse
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly download and cache the embedding model for offline use."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to download (defaults to config.EMBEDDING_MODEL).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model is already present in cache.",
    )
    args = parser.parse_args()

    from src import config
    from src.retrieval.embeddings import verify_offline_cache

    model_name = args.model or config.EMBEDDING_MODEL

    print("=" * 76)
    print("      StudyAstra Offline Embedding Model Cache Preparation")
    print("=" * 76)
    print(f"Target Model       : {model_name}")
    print("Internet Access    : REQUIRED when downloading new weights.")
    print("-" * 76)

    is_cached, missing = verify_offline_cache(model_name)
    if is_cached and not args.force:
        print("[INFO] All 6 mandatory cache files already exist locally. Skipping download.")
        print("[INFO] (Use --force to re-download from Hugging Face Hub).")
    else:
        # Temporarily save and unset offline flags to permit download
        old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")

        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        try:
            from sentence_transformers import SentenceTransformer

            print(f"[INFO] Downloading model '{model_name}' to local cache...")
            _ = SentenceTransformer(model_name, local_files_only=False)
            print("[INFO] Download completed.")
        except Exception as exc:
            print(f"[FAIL] Failed to download model '{model_name}': {exc}")
            return 1
        finally:
            # Restore original offline flags
            if old_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = old_hf_offline
            else:
                os.environ["HF_HUB_OFFLINE"] = "1"

            if old_tf_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Verify all 6 mandatory files are present in the offline cache
    print("-" * 76)
    print("[INFO] Verifying cache completeness...")
    is_complete, missing = verify_offline_cache(model_name)
    if not is_complete:
        print(f"[FAIL] Cache verification failed. Missing files: {missing}")
        return 1

    print("[PASS] All 6 mandatory cache files verified successfully.")

    # Verify local-only offline loading and smoke-test encoding
    print("[INFO] Verifying offline model initialization and smoke encoding...")
    try:
        from sentence_transformers import SentenceTransformer

        offline_model = SentenceTransformer(model_name, local_files_only=True)
        dim = getattr(offline_model, "get_embedding_dimension", None) or getattr(
            offline_model, "get_sentence_embedding_dimension", None
        )
        actual_dim = dim() if callable(dim) else 384

        if actual_dim != 384:
            print(f"[FAIL] Dimension mismatch: expected 384, got {actual_dim}.")
            return 1

        test_vec = offline_model.encode(["StudyAstra offline verification"])
        if test_vec.shape != (1, 384):
            print(f"[FAIL] Output vector shape mismatch: expected (1, 384), got {test_vec.shape}.")
            return 1

        print(f"[PASS] Offline model loaded and encoded successfully (dimension: {actual_dim}).")
    except Exception as exc:
        print(f"[FAIL] Offline loading smoke test failed: {exc}")
        return 1

    print("=" * 76)
    print(f"[SUCCESS] Model '{model_name}' is fully cached and ready for offline use.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
