# Study Bot — Project Overview

## Goal
Study Bot is a local Python + Streamlit AI study assistant that ingests PDF study materials into a persistent multi-PDF FAISS vector database, provides semantic retrieval testing, generates grounded single-PDF summaries, and generates chunk-grounded practice questions with interactive quiz scoring alongside single-turn Q&A.

## Implemented Capabilities
1. **PDF Extraction:** In-memory page-by-page text extraction via PyMuPDF (`src/ingestion/pdf_loader.py`).
2. **Token-Aware Chunking:** Paragraph-first, sentence-aware chunking (`220` tokens / `40` overlap) with full document/page/chunk metadata and within-document deduplication (`src/ingestion/chunker.py`).
3. **LLM-Independent Local Embeddings:** Local `all-MiniLM-L6-v2` (`384`-dim) L2-normalized batch and query embeddings cached once per Streamlit session (`src/retrieval/embeddings.py`).
4. **Persistent Multi-PDF FAISS Vector Store & Document Manager:** `IndexIDMap2(IndexFlatIP)` with monotonically increasing `int64` IDs, SHA-256 duplicate detection, generation-based transactional commits (`manifest.json`), isolated per-document deletion, read-only document chunk getters, and strict compatibility/corruption validation (`src/retrieval/vector_store.py`, `src/database/document_manager.py`).
5. **Reusable LLM Provider Abstraction:** `BaseLLMProvider`, `GroqLLMProvider`, and provider registry/factory ready for future local providers (`src/generation/llm_provider.py`).
6. **Hierarchical Single-PDF Summarization (Stage 6.1):** Full-document, page-range, section (exact heading vs. transparent semantic fallback), and topic summaries across `short`, `medium`, and `detailed` lengths (`src/generation/summarizer.py`, `src/generation/summary_prompts.py`).
7. **Practice Question Generation (Stage 6.2):** Single-PDF Full Document, Page Range (actual indexed pages), and Topic generation across `MCQ`, `Short Answer`, `True / False`, and balanced `Mixed` types and `Easy`, `Medium`, `Hard` difficulties, with strict `source_chunk_ids` validation, deterministic page derivation, full prompt-overhead token budgeting, bounded retries/top-up, shortfall warnings, and Show/Hide Answers toggle (`src/generation/question_generator.py`, `question_validation.py`, `question_prompts.py`).
8. **Interactive Quiz Evaluation Engine (Stage 6.3):** Discrete scoring (`0.0`, `0.5`, `1.0` points per question), deterministic grading for MCQ and True/False, batched LLM short-answer grading strictly grounded in cited chunks with fallback/offline deterministic token-overlap evaluator (`src/generation/quiz_evaluator.py`).
9. **Streamlit UI (`app.py`):** KB sidebar with intentional deletion, explicit **Process & Index PDF** workflow, explicit reset confirmation for corrupted/incompatible stores, **Vector Retrieval Testing** tab (LLM-independent), **PDF Summary** tab, **Practice Questions & Interactive Quiz** tab (Study View vs. Interactive Quiz, answer persistence across reruns/failures, zero generation calls on retry, missed-pages review panel, session-only caches invalidated on delete/reset), and preserved **Quick Q&A (Groq)** tab.

## How to Run & Test
```bash
# Launch app
./venv/bin/streamlit run app.py

# Full offline test suite (109 passed, 1 opt-in skipped)
./venv/bin/python -m unittest discover -s tests -v

# Opt-in real embedding model smoke test (1 passed)
RUN_REAL_EMBEDDING_TESTS=1 ./venv/bin/python -m unittest tests.test_embeddings.TestRealEmbeddingModelSmoke -v
```
