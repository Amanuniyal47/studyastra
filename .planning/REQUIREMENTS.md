# Requirements

## Implemented & Verified
- [x] **REQ-01**: In-memory PDF text extraction with page numbers (`src/ingestion/pdf_loader.py`).
- [x] **REQ-02**: Paragraph- & sentence-aware token-budget chunker (`220`/`40` default, configurable) preserving `chunk_id`, `document_id`, `filename`, `page`, `page_range`, `chunk_index`, and deduplicating identical chunks (`src/ingestion/chunker.py`).
- [x] **REQ-03**: LLM-independent local embedding service (`all-MiniLM-L6-v2`, `384`-dim, L2-normalized, batched, cached once) (`src/retrieval/embeddings.py`).
- [x] **REQ-04**: Persistent multi-PDF FAISS vector database (`IndexIDMap2(IndexFlatIP)`) with generation-based transactional persistence, monotonic non-reused `vector_id`s, read-only document chunk getters, model/dimension compatibility guards, and corruption/desync detection (`src/retrieval/vector_store.py`).
- [x] **REQ-05**: Document registry with SHA-256 duplicate prevention and isolated per-document vector ownership & safe deletion (`src/database/document_manager.py`).
- [x] **REQ-06**: Reusable LLM provider abstraction (`BaseLLMProvider`, `GroqLLMProvider`, registry/factory) decoupled from embeddings & FAISS and preserving `generate_answer()` compatibility (`src/generation/llm_provider.py`, `src/generation/answer_generator.py`).
- [x] **REQ-07**: Hierarchical single-PDF summarization (`full`, `page_range`, `section`, `topic`; `short`, `medium`, `detailed`) with prompt-overhead-aware token budgeting, bounded multi-level reduction, verified page metadata, and progress callbacks (`src/generation/summarizer.py`, `src/generation/summary_prompts.py`).
- [x] **REQ-08**: Practice question generation (`full`, `page_range`, `topic`; `mcq`, `short_answer`, `true_false`, balanced `mixed`; `easy`, `medium`, `hard`) with strict `source_chunk_ids` grounding, deterministic page derivation, full prompt token budgets, bounded retries/top-up, shortfall notices, and session-only UI caching (`src/generation/question_generator.py`, `question_validation.py`, `question_prompts.py`).
- [x] **REQ-09**: Streamlit UI integrating KB sidebar, explicit indexing/deletion/reset, **Vector Retrieval Testing**, **PDF Summary**, **Practice Questions** (with Show/Hide Answers toggle and delete/reset cache invalidation), and preserved **Quick Q&A (Groq)** (`app.py`).
- [x] **REQ-10**: Full automated offline test suite + opt-in real-model smoke test + live Streamlit health check (`HTTP 200`) + updated `README.md` and `.planning/` docs.
- [x] **REQ-12**: Interactive practice & quiz evaluation (Stage 6.3) with discrete 0.0/0.5/1.0 scoring, deterministic MCQ/True-False grading, batched LLM or deterministic short-answer grading, answer preservation across reruns/failures, zero question re-generation on retry, and missed questions review (`src/generation/quiz_evaluator.py`, `app.py`).

## Planned Future Work (Not Yet Implemented)
- [ ] **REQ-11**: Ollama / Qwen local LLM integration (`OllamaLLMProvider`).
- [ ] **REQ-13**: Flashcards (spaced-repetition study cards).
- [ ] **REQ-14**: Multi-turn PDF chatbot with persistent chat history.
- [ ] **REQ-15**: OCR support for scanned/image-only PDFs.
- [ ] **REQ-16**: Timers and timed quiz challenge mode.
- [ ] **REQ-17**: Initialize Git repository with `.gitignore` protecting `.env`, `venv/`, and local `data/vector_store/`.
