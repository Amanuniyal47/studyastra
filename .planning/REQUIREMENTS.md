# Requirements

## Implemented & Verified
- [x] **REQ-01**: In-memory PDF text extraction with page numbers (`src/ingestion/pdf_loader.py`).
- [x] **REQ-02**: Paragraph- & sentence-aware token-budget chunker (`220`/`40` default, configurable) preserving `chunk_id`, `document_id`, `filename`, `page`, `page_range`, `chunk_index`, and deduplicating identical chunks (`src/ingestion/chunker.py`).
- [x] **REQ-03**: LLM-independent local embedding service (`all-MiniLM-L6-v2`, `384`-dim, L2-normalized, batched, cached once) (`src/retrieval/embeddings.py`).
- [x] **REQ-04**: Persistent multi-PDF FAISS vector database (`IndexIDMap2(IndexFlatIP)`) with generation-based transactional persistence, monotonic non-reused `vector_id`s, read-only document chunk getters, model/dimension compatibility guards, and corruption/desync detection (`src/retrieval/vector_store.py`).
- [x] **REQ-05**: Document registry with SHA-256 duplicate prevention and isolated per-document vector ownership & safe deletion (`src/database/document_manager.py`).
- [x] **REQ-06**: Reusable LLM provider abstraction (`BaseLLMProvider`, `GroqLLMProvider`, `OllamaLLMProvider`, registry/factory) decoupled from embeddings & FAISS and preserving `generate_answer()` compatibility (`src/generation/llm_provider.py`, `src/generation/answer_generator.py`).
- [x] **REQ-07**: Hierarchical single-PDF summarization (`full`, `page_range`, `section`, `topic`; `short`, `medium`, `detailed`) with prompt-overhead-aware token budgeting, bounded multi-level reduction, verified page metadata, and progress callbacks (`src/generation/summarizer.py`, `src/generation/summary_prompts.py`).
- [x] **REQ-08**: Practice question generation (`full`, `page_range`, `topic`; `mcq`, `short_answer`, `true_false`, balanced `mixed`; `easy`, `medium`, `hard`) with strict `source_chunk_ids` grounding, deterministic page derivation, full prompt token budgets, bounded retries/top-up, shortfall notices, and session-only UI caching (`src/generation/question_generator.py`, `question_validation.py`, `question_prompts.py`).
- [x] **REQ-09**: Streamlit UI integrating KB sidebar, explicit indexing/deletion/reset, **Vector Retrieval Testing**, **PDF Summary**, **Practice Questions** (with Show/Hide Answers toggle and delete/reset cache invalidation), and **Quick Q&A (Local Qwen2.5 7B)** (`app.py`).
- [x] **REQ-10**: Full automated offline test suite + opt-in real-model smoke test + live Streamlit health check (`HTTP 200`) + updated `README.md` and `.planning/` docs.
- [x] **REQ-11**: Centralized Ollama / Qwen2.5 7B local LLM integration across all AI capabilities (`OllamaLLMProvider` at `http://localhost:11434`).
- [x] **REQ-12**: Interactive practice & quiz evaluation (Stage 6.3) with discrete 0.0/0.5/1.0 scoring, deterministic MCQ/True-False grading, batched LLM or deterministic short-answer grading, answer preservation across reruns/failures, zero question re-generation on retry, and missed questions review (`src/generation/quiz_evaluator.py`, `app.py`).
- [x] **REQ-13**: Spaced-Repetition Study Flashcard Deck & Viewer (Stage 7 — Resolves BUG-02) with multi-batch chunk distribution, single-card carousel, front/back flip, in-place editing, persistence across reruns, Markdown and Anki CSV exports (`src/generation/flashcard_generator.py`, `app.py`).
- [x] **REQ-14**: Hierarchical Academic Study Notes Generation & Markdown/Text Export (Stage 8 — Resolves BUG-01) with multi-batch prompt packing within `NOTES_MAX_BATCH_TOKENS = 2400`, recursive map-reduce reduction, citation filtering, dedicated Study Notes tab with Full/Page Range/Topic scopes, and single-click Markdown export (`src/generation/study_notes_generator.py`, `app.py`).
- [x] **REQ-15**: Timers & Timed Study Sessions / Countdown Quizzes (Stage 9) including Pomodoro Focus Timer with cycles and break sequences, Countdown Study Challenges with preset and custom durations, and Timed Quiz Mode with monotonic deadline tracking, auto/manual submissions, answer persistence, and isolated statistics persistence (`src/utils/timer_manager.py`, `src/utils/study_statistics.py`, `app.py`).
- [x] **REQ-16**: Multi-Turn Conversational PDF Chatbot with Sliding Context Memory (Stage 10) featuring sliding-window token management, non-destructive transcript preservation, conversational follow-up query rewriting, document-scoped FAISS retrieval, strict prompt context budgeting, centralized Ollama Qwen2.5 7B generation, citation validation & repair, thread-safe request concurrency via reference-counted lock leases, and native Streamlit multi-thread chat interface with live streaming, retry handling, and Markdown export (`src/chat/`, `app.py`).
- [x] **REQ-17**: Initialized repository hygiene and `.gitignore` strictly protecting `.env`, `data/`, `venv/`, and build/cache artifacts.

---

## Active Defect Register & Target Stages (See .planning/BUGS.md)
- [x] **BUG-01**: Notes Generation not working -> **Resolved in Stage 8**
- [x] **BUG-02**: Flashcard Generation not working -> **Resolved in Stage 7**

---

## Planned Future Stages (Not Yet Implemented)
- [ ] **REQ-18**: OCR Engine for Scanned / Image-Only PDFs (Stage 11).
