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

## Milestone 3 — Stage 6.4: Local Ollama & Qwen2.5 7B Centralized Integration — *Completed 2026-11-19*
- **Centralized LLM Provider:** Implemented `OllamaLLMProvider` in `src/generation/llm_provider.py` conforming to `BaseLLMProvider`, defaulting to `qwen2.5:7b` at `http://localhost:11434` with configurable temperature, token limits, and robust timeout/error wrapping.
- **Single Model Centralization:** Audited and migrated all 11 AI features across the application to local Qwen via `get_llm_provider("ollama")` (Summaries, Practice Questions, Quizzes, Answer Evaluation, Q&A, Notes, Flashcards, Explanations, Follow-ups).
- **Architecture Preservation:** Maintained local `all-MiniLM-L6-v2` embeddings and FAISS index isolation; updated Streamlit UI headers and references to reflect local Qwen2.5 7B.
- **Verification:** 149/149 test suite passes.

---

## Milestone 3 — Stage 7: Flashcard Generation & Interactive Study Deck — *Completed 2026-11-19* (Resolves BUG-02)
- **Sub-stage 7.1: Flashcard Core Engine & Validation:** Multi-batch chunk distribution and prompt budgeting (`_build_flashcard_batches`, `_plan_flashcard_batches`) for large PDFs, resilient JSON parsing (root arrays `[...]`, synonym keys `question`/`answer` and `term`/`definition`, string `source_chunk_ids`), concise valid answer thresholds (`len(back) >= 1`), bounded retries, and fallback chunk attribution (`src/generation/flashcard_generator.py`).
- **Sub-stage 7.2: Interactive Streamlit Flashcard Deck UI:** Single-card carousel with Front/Back flip, previous/next navigation, `Card X of Y` indicator, in-place question and answer revision with session persistence across reruns, full list viewer, and one-click export controls (`app.py`).
- **Sub-stage 7.3: Export & Comprehensive Verification:** Added Markdown (`export_flashcards_to_markdown`) and RFC 4180 Anki CSV (`export_flashcards_to_anki_csv`) export utilities with download buttons, 12 targeted unit tests (`tests/test_flashcard_generator.py`), 5 AppTest integration tests (`tests/test_app_ui.py`), and live verification against local `qwen2.5:7b` through Ollama. All 161 tests passing.

---

## Milestone 3 — Stage 8: Hierarchical Academic Study Notes Generation — *Completed 2026-11-19* (Resolves BUG-01)
- **Sub-stage 8.1: Hierarchical Notes Engine & Chunk Synthesis:**
  - Implemented prompt-overhead-aware chunk packing in `_build_notes_batches()` with `NOTES_MAX_BATCH_TOKENS = 2400`.
  - Implemented recursive multi-level map-reduce reduction in `_reduce_notes_hierarchically()`, safely combining multi-batch notes into coherent academic structures while preventing prompt truncation.
  - Implemented `validate_and_filter_page_citations()` stripping hallucinated page references not present in source chunks.
  - Implemented `validate_notes_structure()` ensuring academic heading hierarchy, definitions, and takeaways.
  - Centralized LLM integration exclusively via `get_llm_provider("ollama")` with local Qwen2.5 7B (`http://localhost:11434`).
- **Sub-stage 8.2: Streamlit Study Notes UI & Export:**
  - Built dedicated **Study Notes** tab in `app.py`, completely decoupling study notes from the PDF Summary tab.
  - Provided scope controls: Full Document, Page Range (with start/end validation), and Topic focus (with top-k chunks).
  - Designed academic presentation with executive overview, chapters/topics, definitions, mechanisms, and review takeaways.
  - Implemented single-click Markdown export (`export_notes_to_markdown()`) via `st.download_button`.
  - Session-state caching in `st.session_state["cached_notes"]` preserved across reruns and automatically invalidated on document deletion or vector-store reset.
- **Sub-stage 8.3: Comprehensive Verification & Test Suite:**
  - 11 unit tests in `tests/test_study_notes_generator.py` covering multi-batch packing, recursive reduction, citation filtering, and Markdown export.
  - 5 Streamlit AppTest integration tests in `tests/test_app_ui.py` covering tab rendering, page range validation, topic search, persistence, and Markdown export.
  - Full project test suite: 172 tests passing (`OK (skipped=1)`).
  - Live end-to-end integration test against local Qwen2.5 7B via Ollama confirmed working with 3,050-character structured notes generation.

---

## Milestone 3 — Stage 9: Timers & Timed Study Sessions / Countdown Quizzes
- **Stage 9.1: Pomodoro Focus Timer — Completed 2026-11-19**
  - Designed monotonic deadline-based `PomodoroTimer` state machine in `src/utils/timer_manager.py` (Focus, Short Break, Long Break, Cycle transitions).
  - Built atomic persistent `StudyStatisticsManager` in `src/utils/study_statistics.py` writing to `data/study_stats.json` with streak calculation and history.
  - Built isolated Streamlit Pomodoro UI in `app.py` with metrics, cycle banner, live ticking display (`@st.fragment(run_every=1)`), controls, duration customizer, and history.
  - 17 unit and AppTest UI tests in `tests/test_timer_manager.py`, `tests/test_study_statistics.py`, and `tests/test_app_ui.py`.
- **Stage 9.2: Countdown Study Challenges — Completed 2026-11-20**
  - Designed `ChallengeTimer` in `src/utils/timer_manager.py` with custom titles, preset durations (5, 10, 15, 25, 30, 45, 60 mins), custom durations (1–180 mins), start, pause, resume, reset, cancel with elapsed time tracking, and exactly-once completion.
  - Extended `StudyStatisticsManager` in `src/utils/study_statistics.py` with `record_challenge`, total challenge minutes, and independent challenge history without cross-contaminating Pomodoro stats.
  - Added dedicated **⚡ Study Challenges** sub-tab inside **Study Timer** tab in `app.py`, featuring setup controls, live countdown container (`@st.fragment(run_every=1)`), safe cancellation confirmation dialog, and recent challenge history.
  - 16 additional tests (8 ChallengeTimer tests, 3 study statistics tests, 5 AppTest integration tests).
  - Full project test suite: 205 passed, 0 failures, 1 skipped.
- **Stage 9.3: Timed Quiz Mode — Completed 2026-11-20**
  - Designed monotonic deadline-anchored `QuizTimer` in `src/utils/timer_manager.py` scoped strictly by `attempt_key` (`document_id`, `question_set_id`, `attempt_number`) with idle, running, submitted, and expired states, low-time warning detection (`<= 60s`), and clean serialization.
  - Extended atomic `StudyStatisticsManager` in `src/utils/study_statistics.py` with `record_timed_quiz` tracking completed quizzes, total attempts, total minutes, average score percentage, best score percentage, and detailed attempt history (with auto-submission flags and attempt deduplication).
  - Integrated optional Timed Quiz Challenge in `app.py` under the **Practice Questions** tab's Interactive Quiz view:
    - Overall quiz countdown timer with duration presets (5, 10, 15, 20, 30 mins) and custom duration (1–60 mins). Untimed quiz remains default.
    - Isolated live ticking countdown fragment (`@st.fragment(run_every=1)`) with visual color warnings when remaining time is low (`<= 60s`).
    - Robust answer preservation across Streamlit reruns.
    - Manual submission stops timer and records exact elapsed time.
    - Automatic submission on timer expiration with unanswered questions evaluated as 0.0.
    - Strict submission state machine (`not_submitted` -> `submitting` -> `submitted`) preventing duplicate submissions.
    - Evaluator failure protection: catches LLM or evaluation exceptions, preserves answers, allows retry, and does not record invalid completions in statistics.
    - Seamless retry workflow: Attempt #2 reuses identical questions with a fresh timer without regenerating questions.
    - Added dedicated **📝 Timed Quiz Stats** sub-tab inside the **Study Timer** tab displaying high-level performance metrics, best scores, and detailed attempt history.
  - 17 new tests (8 `QuizTimer` unit tests, 4 study statistics unit tests, 5 Streamlit AppTest integration tests).
  - Full project test suite: 222 tests passing, 0 failures, 1 skipped.

---

## Milestone 4 — Stage 10: Multi-Turn Conversational PDF Chatbot with Sliding Context Memory
- **Stage 10.1: Conversation Memory & Sliding-Window Engine — Completed 2026-11-20**
  - Created independent conversation memory subsystem in `src/chat/conversation_memory.py` with `ChatMessage`, `ConversationThread`, `ConversationMemoryConfig`, `SlidingWindowMemoryManager`, and `ConversationManager`.
  - Implemented full 36-character UUIDv4 identifiers, ISO 8601 timestamps, role validation, and token estimation using `estimate_tokens()`.
  - Configurable context budgeting (`CHAT_CONTEXT_WINDOW_TOKENS`, `CHAT_MAX_HISTORY_TOKENS`, `CHAT_OUTPUT_RESERVE_TOKENS`, `CHAT_SAFETY_MARGIN_RATIO`) reserving output and context tokens before computing remaining prompt capacity.
  - Sliding-window prompt builder prioritizing recent turns in chronological order while guaranteeing memory limits are never exceeded.
  - Injectable out-of-window summarization with deterministic pure-Python extractive fallback (`deterministic_extractive_summary()`) that preserves historical continuity even if LLM summarization fails.
  - Multi-conversation thread management with document isolation, title auto-generation, schema versioning, JSON serialization, and Markdown transcript export.
  - 20 unit tests in `tests/test_conversation_memory.py` covering all 14 required aspects; full project test discovery passes (242 tests passing, 0 failures, 1 skipped).
- **Stage 10.2: Conversational Retrieval & Query Rewriting — Completed & Audited 2026-11-20**
  - Implemented selective query rewriting in `src/chat/query_rewriter.py`:
    - `is_follow_up_candidate` fast regex heuristic detecting pronouns ("it", "they", "this", "that", "its", "their"), anaphoric phrases ("the previous method", "that topic", "the former"), continuation connectives, and short ellipsis questions while skipping self-contained standalone questions to eliminate unnecessary LLM calls.
    - `rewrite_query` converting conversational follow-ups into explicit, standalone search queries using local Qwen2.5 7B.
    - Bounded execution pool (`MAX_REWRITE_WORKERS = 2`) with non-blocking semaphore saturation fallback (`0.000s`), completely eliminating thread accumulation and memory leaks under slow or hanging provider conditions.
    - Dual-layer timeout protection: Layer 1 passes native HTTP socket timeouts (`timeout=effective_timeout`) down to `httpx.Client` terminating TCP connections at the OS socket level; Layer 2 enforces caller return guarantees via bounded pool futures.
    - Post-processing cleaning (`clean_rewritten_query`) stripping labels (`Rewritten query:`), outer quotes, and code backticks.
    - Documented Python thread termination architecture limitations in module docstring.
  - Implemented document-scoped retrieval in `src/chat/conversational_retriever.py`:
    - Scopes semantic search strictly to `document_id` via MiniLM embeddings and FAISS vector store, preventing cross-document vector leakage.
    - Defensive chunk filtering ensuring zero cross-document chunk leakage.
  - Implemented grounded chat prompt construction & citation validation in `src/chat/chat_prompts.py`:
    - Grounded system prompt (`CHAT_GROUNDED_SYSTEM_PROMPT`) mandating strict context adherence, explicit [Source X, Page Y] citations, and refusal when information is absent.
    - Formats deterministic document evidence blocks (`[Source 1, Page X] (File: Y)`) and generates structured `source_map`.
    - Strict Prompt-Context Budget Enforcement & Inviolable Question Preservation (Area 2):
      - Complete user question and system instructions are treated as irreducible; questions are **never** truncated or damaged.
      - Defined `PromptBudgetExceededError` raised if system prompt + complete user question + output reserve cannot fit within the context window.
      - Strict prompt reduction hierarchy: history reduction (turn-by-turn) -> summary memory drop -> lower-ranked evidence chunk exclusion (with dynamic note accommodation) -> Chunk 1 safe truncation -> controlled error.
      - Output tokens (`output_reserve_tokens`) are strictly preserved.
      - Final hard ceiling clamp guarantees `total_prompt_tokens + output_reserve_tokens <= context_window_tokens`.
    - Token-Budget Guarantees Documentation (Area 3): Documented that ceilings are computed against regex-based `estimate_tokens`, with BPE token variance (~10-15%) absorbed by `output_reserve_tokens` and 10% safety margin.
    - Citation extraction, validation, and repair (`validate_chat_citations`, `sanitize_and_repair_citations`) verifying source IDs, page numbers, and stripping hallucinated citations.
  - 43 comprehensive unit tests in `tests/test_chat_retrieval_and_prompts.py` (covering follow-up detection, timeouts, bounded pool saturation, leakage prevention, prompt assembly, inviolable question preservation, budget ceiling clamps, and citation validation).
  - Full project test suite: 289 passed, 0 failures, 1 skipped.
- **Stage 10.3: Multi-Turn RAG Pipeline Orchestrator — Completed & Audited 2026-11-20**
  - Implemented central coordination layer in `src/chat/chat_orchestrator.py` (`ChatOrchestrator`, `ChatOrchestratorResult`, `orchestrate_chat`):
    - 10-step end-to-end execution sequence integrating memory, selective query rewriting, document-scoped FAISS retrieval, prompt budgeting, centralized Ollama Qwen2.5 7B generation, and citation hygiene.
    - Safe request and document existence validation preventing cross-document leakage and missing document crashes.
    - Non-destructive sliding-window memory integration preserving full transcript fidelity.
    - Selective follow-up rewriting with native HTTP socket timeouts and bounded pool saturation fallback (`0.000s`).
    - Document-scoped MiniLM/FAISS retrieval with empty result handling.
    - Strict token-budget ceiling enforcement handling `PromptBudgetExceededError`.
    - Centralized Qwen2.5 7B generation supporting streaming via `on_token` callback and `stream_turn` generator.
    - Rigorous citation extraction, validation, and repair sanitizing hallucinated references and verifying true page numbers.
  - Final Focused Reliability Audit Corrections:
    - **Area 1: Streaming Failure & Transcript Consistency:**
      - Tokens yielded live via `stream_turn` generator with complete mid-stream exception containment.
      - Assistant message committed to `thread.messages` strictly after successful generation and citation validation.
      - On generation errors or mid-stream disconnection, assistant message is **never** committed; user question is preserved with `turn_status="failed"` and `last_error`.
      - UI rendering exceptions inside `on_token` callbacks are caught and logged without aborting LLM generation or corrupting transcript state.
      - `self.last_result` consistently updated on both success and failure paths.
    - **Area 2: Retry Idempotency & Thread-Safe Request Concurrency:**
      - Clear retry vs new-turn differentiation using `request_id` and `turn_status`.
      - Reuses unanswered user message on retry without duplicating user messages.
      - Cleanly replaces abandoned unanswered questions if user changes the topic after a failure, avoiding consecutive orphan user nodes.
      - Creates legitimate separate user-assistant pairs for consecutive identical questions when the previous turn succeeded.
      - Per-thread reentrant lock registry (`_get_thread_lock`) safely serializes concurrent requests targeting the same conversation thread while allowing requests for independent threads to run in parallel.
    - **Area 3: Summary-Memory Failure Isolation:**
      - Sliding-window summarization update (`memory_manager.update_memory`) is treated as a secondary operation that never fails the turn or invalidates the generated answer.
      - Exceptions during summary update are isolated into `result.summary_error` while keeping `result.success = True`.
      - Pre-existing summary memory and complete message history are preserved intact upon summarization errors.
  - Final Transcript Integrity Audit Corrections:
    - **Area 1: Non-Destructive Failed Turn Preservation:**
      - Eliminated all `messages.pop()` operations across orchestrator execution paths.
      - User questions that encounter retrieval, budgeting, or generation failures are permanently retained in `thread.messages` with `turn_status="failed"` and `last_error`.
      - Original message ID, timestamp, and request ID are guaranteed immutable.
      - When the learner asks a different question, a new independent turn is appended without deleting the previous failed question.
      - When the learner retries, the retry is associated with the existing failed user node, preserving original identity while recording `initial_request_id` and incrementing `retry_count`.
      - Consecutive identical questions after a successful turn create independent turns with distinct message IDs.
      - Transcript export (`to_markdown`, `to_dict`, `to_json`) and sliding-window memory manager handle failed turns cleanly without crashing or producing orphan assistant messages.
    - **Area 2: Thread Lock Lifecycle & Reference-Counted Lease Mechanism:**
      - Eliminated lock acquisition vs eviction race condition via `_ConversationLockEntry` and `ConversationLockLease`.
      - Atomically registers and increments reference count under `_thread_locks_guard` before lock acquisition begins; eviction is strictly blocked while any caller holds a lease reference even prior to lock acquisition.
      - Guarantees concurrent requests targeting the same conversation always receive the identical lock instance.
      - Decrements reference count reliably in a `finally` block in `ConversationLockLease.release()`, executing deferred eviction automatically on final reference release if deletion occurred while the lock was in active use.
      - Unheld locks for deleted threads are evicted safely without manual cleanup calls.
      - Independent conversations use distinct lock instances, preserving parallel execution.
  - 42 comprehensive unit tests in `tests/test_chat_orchestrator.py` (21 core orchestration tests + 21 focused reliability, transcript integrity, and lock lease concurrency audit tests).
  - Full project test suite: 331 passed, 0 failures, 1 skipped.
- **Stage 10.4: Streamlit Conversational Chat Interface & Verification — Completed & Verified 2026-11-20**
  - Integrated complete conversational RAG subsystem into Streamlit interface (`app.py`):
    - Added "Chat with PDF" tab as the primary first tab in `st.tabs(...)`.
    - Centralized integration directly via `ChatOrchestrator` (`src/chat/chat_orchestrator.py`); zero duplicate RAG or LLM logic in UI code.
    - Document selector (`chat_doc_select`) and thread selector (`chat_thread_select`) with clean title format mapping and auto-initialization of default thread.
    - Sidebar conversation thread management (`_render_sidebar_conversations`) displaying thread list per document with message counts and timestamps.
    - Thread action controls:
      - ➕ New Chat: creates a fresh conversation thread for the active document.
      - ✏️ Rename: inline text input with validation, "Save", and "Cancel".
      - 🗑️ Delete: inline confirmation dialog with "Confirm Delete" and "Cancel".
      - 📥 Export Chat: single-click Markdown transcript download button.
    - Full conversational turn history rendering using native `st.chat_message("user")` and `st.chat_message("assistant")`.
    - Citation expander (`📚 Sources ({count} cited)`) rendering `[Source X] — Page Y *(Document: doc_name)*` with text snippet previews.
    - Conversational query rewrite expander (`🔍 Conversational Query Rewrite`) displaying original query vs rewritten retrieval query.
    - Preserves failed user turns with warning banner (`⚠️ Generation failed: ...`) and dedicated `[🔄 Retry Question]` button without duplicating user nodes in transcript.
    - Live streaming token delivery using `st.write_stream(orchestrator.stream_turn(...))` with mid-stream exception containment and non-destructive failed turn marking.
    - Document deletion and vector store reset cache invalidation (`_invalidate_cached_chat`) with safe lock cleanup (`clear_document_threads`, `clear_all_thread_locks`).
    - Non-blocking Ollama status check banner informing user of daemon status.
    - Multi-turn live hardware verification with Ollama `qwen2.5:7b` confirmed.
  - 16 comprehensive unit & AppTest tests in `tests/test_chat_ui.py` covering all 16 required areas (empty states, creation, switching, isolation, rename, delete with confirmation, message history, streaming, citations, query rewrite display, failed turn preservation, retry idempotency, cache invalidation, and export).
  - Full project test suite: 347 passed, 0 failures, 1 skipped.
- **Phase 10: Multi-Turn Conversational PDF Chatbot with Sliding Context Memory — FULLY COMPLETED & VERIFIED 2026-11-20**

---

## Milestone 5 — Phase 11: Performance Optimization, Reliability & Production Readiness

- **Stage 11.1: Automated Benchmarking Harness & Latency Profiling — Completed 2026-11-20**
  - Built standalone reproducible benchmarking suite under `tests/benchmarks/`:
    - `profiler.py`: System info, hardware detection (OS, CPU, RAM via Windows PSAPI, PyTorch, CUDA, Ollama), memory tracking (heap via `tracemalloc`, process RSS/peak RSS via `GetProcessMemoryInfo`), and `BenchmarkMetric` data structures.
    - `synthetic_data.py`: In-memory synthetic multi-page PDF generation (10, 50, 100 pages) for zero-dependency reproducible benchmarks.
    - `benchmark_suite.py`: Multi-repetition performance measurement across 7 components (PDF Ingestion, Token Chunking, Embeddings, FAISS Vector Store, Ollama LLM, Conversational RAG, Application Startup).
    - `report_generator.py`: Generates machine-readable `benchmarks/benchmark_results.json` and human-readable `benchmarks/benchmark_report.md`.
    - `run_benchmarks.py`: Standalone CLI with `--live-llm`, `--quick`, `--reps`, `--output-json`, and `--output-md` flags.
  - Executed full live hardware benchmark run (24 metrics recorded across live Ollama Qwen2.5 7B and CPU PyTorch).
  - 15 unit and integration tests in `tests/test_benchmarks.py` covering system info, memory tracking, synthetic data, isolated vector store execution, mocked Ollama/RAG, JSON serialization, and Markdown report generation.
  - Full project test suite: 362 passed, 0 failures, 1 skipped.

- **Stage 11.2: UI Rerun Optimization, Health Check TTL & Memory Lifecycle — Completed 2026-11-20**
  - **Work Area A (Ollama Health Check TTL Caching):**
    - Implemented thread-safe class-level TTL caches (`_connection_cache`, `_model_cache`) with `threading.Lock()` in `OllamaLLMProvider`.
    - Added configurable `OLLAMA_HEALTH_CHECK_TTL_SECONDS` (defaulting to 20.0s) in `src/config.py` and `.env.example`.
    - Implemented `force` bypass parameter and fine-grained `invalidate_health_cache()`.
    - Wired automatic cache invalidation upon connection errors or HTTP 404 in `generate()` and streaming paths.
    - Verified latency reduction: cold HTTP connection + model check takes ~76 ms (mock) / ~4,273 ms (live Ollama) vs warm cached check taking **0.0026 ms** (>29,000x speedup), eliminating blocking pauses on Streamlit user interactions.
    - 11 comprehensive unit tests in `tests/test_health_cache.py`.
  - **Work Area B (FAISS Vector Store Transaction Memory Optimization):**
    - Implemented `_clone_chunk_metadata()` for deep recursive cloning of nested dicts, lists, and sets, guaranteeing 100% caller mutation isolation.
    - Replaced `copy.deepcopy(self._chunks)` with `self._chunks.copy()` for outer staging list replication in `FAISSVectorStore._commit_transaction()`, `add_document_chunks()`, and `delete_document()`.
    - Preserved full ACID transaction guarantees: generation manifests, rollback isolation on write failure, and disk integrity.
    - Verified benchmarks across chunk scales:
      - 100 chunks: 2.92 ms (161 KB RAM) -> 0.005 ms (1.6 KB RAM) — **624x speedup, 103x RAM reduction**.
      - 1,000 chunks: 23.85 ms (1,569 KB RAM) -> 0.041 ms (15.6 KB RAM) — **582x speedup, 100x RAM reduction**.
      - 5,000 chunks: 111.60 ms (7,702 KB RAM) -> 0.123 ms (78.1 KB RAM) — **907x speedup, 98.6x RAM reduction**.
    - 8 comprehensive unit tests in `tests/test_vector_store_optimization.py` covering multi-stage write failure and deletion rollback.
  - **Work Area C (Streamlit Session-State Memory Management):**
    - Implemented `src/utils/session_lifecycle.py` with bounded FIFO/LRU eviction (`prune_lru_cache`, default max 5 entries, with protected keys exception accurately documented).
    - Established strict protection for essential user data (conversations, quiz answers, flashcard edits, study timer).
    - Implemented automatic pruning and stale error banner cleanup (`clean_stale_ephemeral_state`) across Summary, Notes, Questions, Flashcards, and Chat tabs.
    - 9 comprehensive unit tests in `tests/test_session_lifecycle.py`.
  - Full project test suite: 390 executed — 389 passed, 1 skipped (`TestRealEmbeddingModelSmoke`), 0 failures, 0 errors.

- **Stage 11.3: Large PDF Processing & Scalable Map-Reduce Pipeline — Completed 2026-11-20**
  - **Work Area A (Scalable Map-Reduce Summarization):**
    - Updated `src/generation/summarizer.py` with multi-batch convergence guarantees, configurable reduction limits (`MAX_REDUCTION_LEVELS = 6`), and token budgeting (`SUMMARY_MAX_BATCH_TOKENS = 2400`).
    - Added source coverage tracking in `SummaryResult`: `status` ("complete" | "partial" | "failed"), `total_eligible_chunks`, `processed_chunks`, `failed_chunks`, and `failed_batch_indices`. Zero silent chunk dropping.
    - Added `_intermediate_reduce_builder` with mathematical convergence guarantee (pairing $\ge 2$ items per group).
    - Hardened reduction level limit handling: exceeding `MAX_REDUCTION_LEVELS` properly sets `status="partial"` with clear explanatory notice.
    - 6 targeted tests in `tests/test_summarizer_map_reduce.py`.
  - **Work Area B (Scalable Hierarchical Study Notes):**
    - Updated `src/generation/study_notes_generator.py` with multi-level map-reduce reduction and configurable limits (`MAX_REDUCTION_LEVELS = 6`).
    - Added source coverage tracking in `StudyNotesResult`: `status`, `total_eligible_chunks`, `processed_chunks`, `failed_chunks`, and `failed_batch_indices`.
    - Hardened partial coverage accounting: failed map batches and reduction level limits are recorded as `status="partial"` with accurate failed chunk counts.
    - Preserved 6-section academic structure, definition deduplication across reduction levels, and page reference filtering.
    - 14 comprehensive tests in `tests/test_study_notes_generator.py`.
  - **Work Area C (Layout-Aware Academic PDF Text Extraction):**
    - Rewrote `src/ingestion/pdf_loader.py` with `extract_page_text_layout_aware(page, sort_layout=True)`: classifies full-width spanning headers/abstracts vs multi-column blocks, sorts columns left-to-right to preserve logical reading order, and handles single-column/fallback pages gracefully.
    - Hardened multi-column detection to handle uneven block counts (e.g. 2 left, 1 right) and eliminate horizontal interleaving.
    - Implemented gapless vertical slice band partitioning around spanning figures/tables, ensuring 100% block preservation.
    - Added `progress_callback` to `extract_pdf_text` for granular page-level reporting.
    - 7 targeted tests in `tests/test_pdf_loader_layout.py`.
  - **Work Area D (Batched Embedding & Incremental Progress):**
    - Updated `src/retrieval/embeddings.py` `embed_texts` with bounded batching and `progress_callback` reporting completion percentage.
    - Updated `src/retrieval/vector_store.py` `add_pdf` to report multi-stage ingestion progress (Extraction, Chunking, Batched Embeddings, ACID Commit).
    - Verified transaction safety and rollback preservation of previous generation under mid-batch failures and pre-commit failures.
    - 4 targeted tests in `tests/test_embedding_batching.py`.
  - **Work Area E (Streamlit UI Responsiveness & Progress Indicators):**
    - Updated `app.py` with multi-stage progress bars and captions during PDF ingestion, summarization, and study notes generation.
    - Added warning banners for partial coverage and failure indicators.
    - 44 tests passing in `tests/test_app_ui.py`.
  - **Work Area F (Configuration & Defaults):**
    - Added `MAX_REDUCTION_LEVELS = 6` and `PDF_LAYOUT_SORT_ENABLED = True` in `src/config.py` and `.env.example`.
  - **Large-PDF Scale Verification (10, 50, 100, 250, 500 Pages):**
    - Executed scale benchmark suite `tests/benchmarks/verify_stage11_3_large_pdf.py` measuring both incremental heap (`tracemalloc`) and process Working Set (`RSS`):
      - 10 pages: 10 chunks, 0.01s extract, 0.002s chunk, 1 embed batch, 1 sum batch (0 levels), heap peak: 0.11 MB, RSS: 496.36 MB.
      - 50 pages: 50 chunks, 0.05s extract, 0.010s chunk, 2 embed batches, 5 sum batches (1 level), heap peak: 0.34 MB, RSS: 497.21 MB.
      - 100 pages: 100 chunks, 0.10s extract, 0.020s chunk, 4 embed batches, 9 sum batches (1 level), heap peak: 0.61 MB, RSS: 498.77 MB.
      - 250 pages: 250 chunks, 0.23s extract, 0.045s chunk, 8 embed batches, 21 sum batches (1 level), heap peak: 1.44 MB, RSS: 503.79 MB.
      - 500 pages: 500 chunks, 0.45s extract, 0.095s chunk, 16 embed batches, 42 sum batches (2 levels), heap peak: 2.82 MB, RSS: 508.01 MB (peak 512.05 MB).
  - Full project test suite: 410 executed — 409 passed, 1 skipped (`TestRealEmbeddingModelSmoke`), 0 failures, 0 errors.

- **Stage 11.4: Resilient Fault Recovery, Staging Cleanup & Timeout Hardening — Completed & Verified 2026-11-20**
  - **Work Area A (Ollama Connection & Timeout Recovery):**
    - Hardened `OllamaLLMProvider` with granular timeouts (`connect_timeout`, `read_timeout`, `stream_idle_timeout`).
    - Implemented bounded exponential backoff retries (`max_retries`, `retry_backoff_factor`, `retry_backoff_max`) on transient connection errors, read timeouts, and HTTP 429/500/502/503/504 errors.
    - Added strict non-retry fail-fast on permanent errors: HTTP 404 (model missing), HTTP 400 (bad request), configuration errors, and invalid parameters.
    - Hardened streaming generator (`generate_stream`): allows transient connection retries before the first token, but strictly forbids retrying once tokens have begun streaming to prevent transcript interleaving or duplicate chunks.
    - Guaranteed HTTP response socket cleanup in `_process_stream` via `finally: response.close()`.
  - **Work Area B (FAISS Transaction & Staging Recovery):**
    - Added `cleanup_staging_artifacts(max_age_seconds)` in `FAISSVectorStore` to safely remove orphaned `*.staging` directories and `manifest.json.tmp`.
    - Integrated automated staging cleanup during store startup/initialization (`load_or_initialize`) guarded by `VECTOR_STORE_CLEANUP_STAGING` and `VECTOR_STORE_STAGING_MAX_AGE_SECONDS`.
    - Preserved active generation directory and committed manifest at all times.
    - Implemented `_safe_replace` with exponential retry backoff on Windows file locking errors during directory replacement and manifest flips.
    - Added automatic manifest recovery in `load_or_initialize` from valid leftover `manifest.json.tmp` if process was killed during atomic commit.
  - **Work Area C & D (PDF Ingestion Interruption & Resource Lifecycle):**
    - Verified complete transactional rollback on PDF extraction or embedding failure, leaving zero partial chunks or document records.
    - Guaranteed PyMuPDF document handle cleanup using context managers (`with pymupdf.open(...) as document:`) across normal and exception paths.
    - Verified re-indexing idempotency: failed initial indexing leaves store clean, subsequent attempt succeeds, and repeat attempt cleanly detects duplicate without index bloat.
  - **Work Area E (Locking & Concurrency Safety):**
    - Introduced `self._write_lock = threading.RLock()` across mutating `FAISSVectorStore` operations (`add_document_chunks`, `delete_document`, `reset_store`, `load_or_initialize`, `cleanup_staging_artifacts`).
    - Confirmed that long-running computations (`extract_pdf_text` and `embed_texts`) execute outside the vector store write lock, preventing search starvation and UI blocking.
    - Verified thread safety of multi-threaded concurrent indexing without vector ID collisions or data corruption.
  - **Work Area F (Streamlit UI Error Recovery & Session State Preservation):**
    - Wrapped UI indexing handler in `app.py` with broad exception containment and guaranteed progress cleanup (`finally: progress_bar.empty()`, `progress_status.empty()`).
    - Verified that indexing and LLM failures display actionable error messages without crashing the Streamlit app or clearing session state (chat history, quiz answers, flashcards, study notes, pomodoro timer).
  - **Work Area G (Configuration & Defaults):**
    - Added `OLLAMA_CONNECT_TIMEOUT_SECONDS = 10.0`, `OLLAMA_READ_TIMEOUT_SECONDS = 120.0`, `OLLAMA_MAX_RETRIES = 2`, `OLLAMA_RETRY_BACKOFF_FACTOR = 0.5`, `OLLAMA_RETRY_BACKOFF_MAX_SECONDS = 3.0`, `OLLAMA_STREAM_IDLE_TIMEOUT_SECONDS = 30.0`, `VECTOR_STORE_CLEANUP_STAGING = True`, and `VECTOR_STORE_STAGING_MAX_AGE_SECONDS = 3600.0` in `src/config.py` and `.env.example`.
  - **Work Area H (Fault Injection & Automated Tests):**
    - Created dedicated test suite `tests/test_fault_recovery.py` with 27 focused fault recovery and transaction consistency tests covering all failure injection scenarios.
    - Verified reader snapshot isolation under concurrent write/delete commits via atomic `_GenerationState` publishing.
    - Verified commit boundary rollback and staging cleanup under injected Windows replacement failures.
  - Full project test suite: 437 executed — 436 passed, 1 skipped (`TestRealEmbeddingModelSmoke`), 0 failures, 0 errors.

- **Stage 11.5: Production Deployment, Environment Diagnostics & Packaging (Planned)**
  - Audit and pin explicit direct dependencies in `requirements.txt` (add `httpx`).
  - Create pre-flight setup verification script (`check_env.py`).
  - Update deployment documentation, README, and `.env.example`.
