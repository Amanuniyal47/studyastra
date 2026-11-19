# Testing Guide

## Test Runners
Uses Python's built-in `unittest` module:

```bash
# Full offline unit + Streamlit UI integration suite
./venv/bin/python -m unittest discover -s tests -v

# Opt-in real all-MiniLM-L6-v2 smoke test
RUN_REAL_EMBEDDING_TESTS=1 ./venv/bin/python -m unittest tests.test_embeddings.TestRealEmbeddingModelSmoke -v
```

## Test Modules
- **`tests/test_chunker.py`**: Stage 1 config defaults (`220/40`, `all-MiniLM-L6-v2`), paragraph/sentence boundaries, overlap, multi-page topic isolation, multi-PDF unique `chunk_id`s, deduplication, and large-doc bounds.
- **`tests/test_embeddings.py`**: Stage 2 preprocessing consistency, strict blank/empty rejection, batch shape/dtype/L2-norm, query encoding, single-load caching, error wrapping, plus opt-in real `all-MiniLM-L6-v2` smoke test.
- **`tests/test_vector_store.py`**: Stage 3 multi-PDF persistence & restart, duplicate skipping, empty/repeated/large PDFs, filtered retrieval, isolated deletion + monotonic non-reused `vector_id`s, rollback on persistence failure, model/dimension compatibility, and corruption detection.
- **`tests/test_llm_provider.py`**: Stage 6.1a `GroqLLMProvider` success & `max_tokens`, missing key & prompt validation, Groq API error classes, provider registry/factory (including unimplemented `ollama` error), and `generate_answer()` provider injection.
- **`tests/test_summarizer.py`**: Stage 6.1b `get_document_chunks()` isolation/immutability/ordering, `short`/`medium`/`detailed` lengths, page-range validation, exact heading vs. semantic section fallback, single-PDF topic scoping, multi-level hierarchical prompt-overhead budget enforcement + page reference preservation, monotonic progress callbacks, and mid-batch failure safety.
- **`tests/test_question_validation.py`**: Stage 6.2a config defaults, balanced `mixed` allocation (`10 -> 4/3/3`), type/difficulty aliases, JSON extraction (plain/fenced/preamble/malformed), strict `source_chunk_ids` validation (rejecting unknown/empty/cross-document chunk IDs and ignoring LLM-hallucinated page guesses), MCQ/Short-Answer/True-False schema checks, Jaccard deduplication, and bounded previous-stem prompt builders.
- **`tests/test_question_generator.py`**: Stage 6.2b mocked-provider orchestration across all 4 types × 3 difficulties, `mixed` 10-question `4/3/3` balance, Page Range & Topic single-PDF isolation, 12-page large-PDF strict token-budget enforcement + start-to-end coverage, duplicate filtering + bounded top-up + shortfall notice, malformed JSON retry/failure, mid-batch `LLMProviderError` safety, and zero FAISS mutation.
- **`tests/test_pipeline.py`**: Core `file_hash`, `extract_pdf_text`, `chunk_text`, `index_pdf`, `retrieve_chunks`, `generate_answer`, and initial Streamlit render smoke test.
- **`tests/test_app_ui.py`**: Stage 4, 6.1c, and 6.2c Streamlit `AppTest` integration flows covering empty KB, explicit indexing, duplicate prevention, multi-PDF restart persistence, retrieval testing, intentional deletion, reset confirmation, Quick Q&A, missing `GROQ_API_KEY` independence, **PDF Summary** tab, and **Practice Questions** tab (all scopes & types, Mixed distribution, explicit generation, Show/Hide Answers toggle without regeneration, single-PDF cache isolation, shortfall notice, delete/reset cache invalidation, and API error stale-result clearing).
