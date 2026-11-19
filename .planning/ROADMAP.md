# Roadmap

## Milestone 1: Onboarding & Baseline Stabilization — *Completed 2026-11-18*
- Analyzed codebase, fixed deprecated Groq model & `fitz` import, added baseline tests, created `.planning/` structure.

## Milestone 2: Chunking, Local Embeddings & Persistent FAISS Vector Database (Stages 1–5) — *Completed 2026-11-18*
- Configurable `220/40` token chunking, local `all-MiniLM-L6-v2` `EmbeddingService`, persistent transactional `FAISSVectorStore` (`IndexIDMap2(IndexFlatIP)` + `manifest.json`), `DocumentManager`, Streamlit Retrieval Testing UI, tests, and docs.

## Milestone 3 — Stage 6.1: PDF Summarization — *Completed 2026-11-18*
- `BaseLLMProvider` / `GroqLLMProvider` abstraction, read-only document chunk getters, grounded hierarchical single-PDF summarizer (`full`, `page_range`, `section`, `topic`; `short`/`medium`/`detailed`), Streamlit **PDF Summary** tab, tests, and docs.

## Milestone 3 — Stage 6.2: Practice Question Generation — *Completed 2026-11-18*
- **Stage 6.2a:** Config defaults, `PracticeQuestion`/`QuestionSetResult`, balanced `mixed` allocation, JSON extraction, strict `source_chunk_ids` grounding, deterministic page derivation, deduplication, and prompt templates (`question_validation.py`, `question_prompts.py`).
- **Stage 6.2b:** Single-PDF batch question orchestrator (`full`, `page_range`, `topic`; `mcq`, `short_answer`, `true_false`, `mixed`; `easy`, `medium`, `hard`) with full prompt token budgeting, evenly spaced coverage, bounded retries/top-up, shortfall warnings, and progress callbacks (`question_generator.py`).
- **Stage 6.2c:** Streamlit **Practice Questions** tab with actual indexed page controls, Show/Hide Answers toggle (zero rerun LLM calls), session-only caching invalidated on delete/reset, and stale clearing on API error (`app.py`).
- **Stage 6.2d:** Final offline + real-model verification, `requirements.txt` audit, live Streamlit `/_stcore/health` HTTP `200` check, and documentation updates.

## Milestone 3 — Stage 6.3: Interactive Practice & Quiz Evaluation — *Completed 2026-11-19*
- **Stage 6.3a:** Config thresholds and quiz evaluation engine (`src/generation/quiz_evaluator.py`, `tests/test_quiz_evaluator.py`) with discrete 0.0/0.5/1.0 scoring, deterministic MCQ/True-False grading, batched LLM short-answer grading bounded to source chunks, and deterministic negation-aware token-overlap fallback evaluator.
- **Stage 6.3b:** Streamlit Interactive Quiz UI and state management in `app.py` under **Practice Questions** tab:
  - Practice Mode selector (Study View vs. Interactive Quiz).
  - Answers hidden until explicit submission; discrete scoring badges and per-type score breakdowns.
  - Missed-questions review panel with verified source chunk citations and page links.
  - Zero question re-generation on Retry Quiz (reuses exact cached questions).
  - Preservation of user-entered answers across Streamlit reruns and transient grading/API failures.
  - Document isolation and automatic invalidation on document deletion or vector-store reset.
- **Stage 6.3c:** Final offline + real-model verification, `requirements.txt` audit, live Streamlit `/_stcore/health` HTTP `200` check, and documentation updates.

## Planned Future Stages (Not Started)
- **Ollama / Qwen Local LLM Integration** (`OllamaLLMProvider`).
- **Timers & Timed Study Sessions** (countdown clocks, timed quiz challenges).
- **Flashcards** (spaced-repetition study cards).
- **Multi-Turn PDF Chatbot** (conversational RAG with chat history).
- **OCR** for scanned/image-only PDFs.
- **Git Repository Initialization** (with `.gitignore` for `.env`, `venv/`, and `data/vector_store/`).
