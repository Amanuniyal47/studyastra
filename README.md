# Study Bot — AI PDF Knowledge Base, Retrieval, Summarization & Practice Questions

Study Bot is a Python and Streamlit application for uploading PDF study materials, extracting text, chunking content with page metadata, generating local dense embeddings, storing them in a persistent multi-PDF FAISS vector database, testing semantic retrieval, generating grounded single-PDF summaries, and generating chunk-grounded practice questions.

---

## 1. Currently Implemented Features

* **PDF Upload & In-Memory Text Extraction:** Extracts page-by-page text from uploaded PDFs via PyMuPDF (`pymupdf`) without leaving temporary files on disk.
* **Paragraph- & Sentence-Aware Chunking:** Splits extracted text into token-budgeted chunks (default `220` tokens, `40`-token overlap) while preserving paragraph boundaries, sentence boundaries, exact page attribution, unique `chunk_id`s, and deduplicating identical chunks within a document.
* **Local Embedding Service (LLM-Independent):** Generates L2-normalized `384`-dimensional `float32` embeddings locally using `SentenceTransformer("all-MiniLM-L6-v2")` with batch processing and single-load Streamlit caching.
* **Persistent Multi-PDF FAISS Vector Database:** Stores vectors in `faiss.IndexIDMap2(faiss.IndexFlatIP(384))` with monotonically increasing `int64` vector IDs (never reused after deletion) and generation-based transactional persistence under `data/vector_store/`.
* **Document Management & Safe Deletion:** Tracks indexed PDFs by SHA-256 content hash (`document_id`), prevents duplicate indexing before embedding, supports incremental multi-PDF indexing, and supports isolated per-document deletion with confirmation.
* **Vector Retrieval Testing Tab (LLM-Independent):** Query FAISS across all documents or filtered to a single PDF, adjust `Top K`, and inspect ranked chunks with document filename, page number, `chunk_id`, cosine similarity score, and chunk text—works even without `GROQ_API_KEY`.
* **PDF Summary Tab (Stage 6.1):**
  * Strictly single-PDF summarization across **Full Document**, **Page Range** (actual indexed pages), **Section** (exact heading match vs. transparent semantic fallback), and **Topic** (single-PDF FAISS retrieval).
  * **Short**, **Medium**, and **Detailed** lengths with prompt-aware hierarchical map-reduce (`SUMMARY_MAX_BATCH_TOKENS=2400`, bounded `MAX_REDUCTION_LEVELS=3`), live progress updates, verified source page metadata, and session-only caching invalidated on delete/reset.
* **Practice Questions & Interactive Quiz Tab (Stages 6.2 & 6.3):**
  * Strictly single-PDF practice question generation across **Full Document**, **Page Range** (using actual indexed page numbers), and **Topic** (using document-scoped FAISS retrieval with configurable `Top K`).
  * Four question types: **MCQ** (4 distinct options, unambiguous single correct answer, explanation), **Short Answer** (expected answer + explanation), **True / False** (`True`/`False` + grounded `correct_statement` + explanation), and **Mixed** (balanced distribution across all three types, e.g., `10 -> 4 MCQ, 3 Short Answer, 3 True/False`, displaying actual distribution).
  * Three difficulty levels: **Easy**, **Medium**, and **Hard**, with configurable question count (`1..QUESTION_MAX_COUNT`).
  * **Strict Source-Chunk Grounding:** Every question must cite valid `source_chunk_ids` from the selected document and batch; unknown or cross-document chunk IDs are rejected, and `source_pages`, `filename`, and `document_id` are derived deterministically from verified vector-store chunks (never trusting LLM-generated page guesses).
  * **Batch Generation, Token Budgets, Retries & Shortfall Handling:** Full prompt token budgeting (`QUESTION_MAX_BATCH_TOKENS=2400`, including system/user prompts, schema, headers, source text, and bounded previous-stem context), bounded output-token allowance (`QUESTION_MAX_OUTPUT_TOKENS=1400`), evenly spaced multi-batch coverage for large PDFs, duplicate filtering, bounded retries (`QUESTION_MAX_RETRIES=1`), bounded top-up (max 1 call), and graceful shortfall warnings (returning only valid unique questions without inventing facts or exceeding requested count).
  * **Two Practice Modes:**
    * **Study View:** Immediate question display with collapsible **Show Answers & Explanations** toggle (zero LLM calls on toggle or rerun) and Markdown export.
    * **Interactive Quiz (Stage 6.3):** Answers, explanations, and scores remain strictly hidden until submission. Provides interactive inputs (`st.radio` for MCQ/TF, `st.text_area` for Short Answer), candidate progress counts, and explicit submission.
  * **Discrete Scoring & Dual Evaluator Engine (Stage 6.3):**
    * Discrete scoring: exactly `0.0`, `0.5`, or `1.0` points per question (1.0 point max). Unanswered questions score `0.0`.
    * Deterministic evaluation for MCQ and True/False.
    * Configurable Short Answer evaluation (`QUIZ_SHORT_ANSWER_EVALUATOR`): either **LLM** (single batched call strictly grounded in retrieved source chunks) or **Deterministic** (local token-overlap and negation-aware scoring with zero LLM calls).
    * Summary metrics: Total score, percentage, breakdown by question type (MCQ, Short Answer, True/False), detailed per-question feedback, and an expandable review panel highlighting missed pages with verified source citations.
  * **Robust State Management:** Streamlit reruns preserve answers; grading failures preserve typed responses so users can resubmit without re-typing; clicking **Retry Quiz** clears prior answers and results while reusing the exact cached question set with **zero LLM generation calls**; switching documents isolates quiz states; deleting a document or resetting the vector store immediately clears all quiz sessions and widget keys.
* **Reusable LLM Provider Abstraction:** `BaseLLMProvider`, `GroqLLMProvider`, and provider registry/factory (`get_llm_provider`, `register_llm_provider`) keeping ingestion, embeddings, and FAISS strictly decoupled from LLMs.
* **Quick Q&A (Groq) Tab:** Preserved single-turn grounded Q&A using the shared provider abstraction.

---

## 2. Architecture & Folder Structure

```text
Study Bot/
├── app.py                              # Streamlit UI: KB sidebar, Upload/Index, Retrieval Testing, PDF Summary, Practice Questions, Quick Q&A
├── requirements.txt                    # Verified Python dependencies (7/7 exact match)
├── .env.example                        # Template environment variables
├── README.md                           # Project documentation
├── data/
│   └── vector_store/                   # Created automatically on first index commit
│       ├── manifest.json               # Atomic pointer to the active committed generation
│       └── generations/
│           └── gen_<N>/
│               ├── index.faiss         # FAISS IndexIDMap2(IndexFlatIP) binary index
│               ├── metadata.json       # Chunk text & metadata keyed by vector_id
│               └── documents.json      # Document registry & per-document vector_id ownership
├── src/
│   ├── config.py                       # Centralized env & hyperparameter configuration
│   ├── utils/
│   │   └── file_utils.py               # SHA-256 file hashing for document deduplication
│   ├── ingestion/
│   │   ├── pdf_loader.py               # PyMuPDF in-memory page-by-page PDF text extraction
│   │   └── chunker.py                  # Paragraph/sentence-aware token-budget chunker
│   ├── retrieval/
│   │   ├── embeddings.py               # Local SentenceTransformer EmbeddingService (LLM-independent)
│   │   ├── vector_store.py             # Persistent transactional FAISSVectorStore + read-only doc chunk getters
│   │   └── retriever.py                # High-level query embedding & FAISS similarity search
│   ├── database/
│   │   └── document_manager.py         # Document registry & per-document vector ownership validation
│   └── generation/
│       ├── llm_provider.py             # BaseLLMProvider, GroqLLMProvider, provider registry & factory
│       ├── prompts.py                  # Grounded Q&A prompt templates
│       ├── answer_generator.py         # Single-turn Q&A using LLM provider abstraction
│       ├── summary_prompts.py          # Grounded single-pass, map-batch & reduce summary prompts
│       ├── summarizer.py               # Hierarchical single-PDF summarizer (full, page_range, section, topic)
│       ├── question_validation.py      # PracticeQuestion/QuestionSetResult, JSON parsing, chunk grounding, dedup
│       ├── question_prompts.py         # Difficulty rubrics, JSON schemas & bounded previous-stem prompts
│       ├── question_generator.py       # Batch practice-question orchestrator (full, page_range, topic, mixed)
│       └── quiz_evaluator.py           # Stage 6.3 quiz scoring engine (deterministic MCQ/TF, batched LLM short answer)
└── tests/
    ├── test_chunker.py                 # Stage 1 config & chunker unit tests
    ├── test_embeddings.py              # Stage 2 mocked embedding tests + opt-in real-model smoke test
    ├── test_vector_store.py            # Stage 3 persistence, restart, deletion, rollback, corruption tests
    ├── test_llm_provider.py            # Stage 6.1a provider factory, Groq error handling & extensibility tests
    ├── test_summarizer.py              # Stage 6.1b modes, lengths, prompt-aware hierarchical budget & isolation tests
    ├── test_question_validation.py     # Stage 6.2a config, mixed allocation, JSON parsing, chunk grounding & dedup
    ├── test_question_generator.py      # Stage 6.2b batch orchestration, token budget, shortfall, retry & isolation
    ├── test_quiz_evaluator.py          # Stage 6.3a quiz scoring, deterministic safeguards & batched LLM grading tests
    ├── test_pipeline.py                # Core pipeline & Q&A unit tests
    └── test_app_ui.py                  # Stages 4, 6.1c, 6.2c & 6.3b Streamlit AppTest end-to-end UI tests
```

---

## 3. Data Flow & Key Design Choices

```text
PDF Upload (app.py)
  │
  ├─► file_hash()              SHA-256 content hash -> unique document_id
  ├─► DocumentManager          Duplicate check BEFORE extraction or embedding
  ├─► extract_pdf_text()       PyMuPDF -> [(page_number, page_text), ...]
  ├─► chunk_text()             Paragraph-first, sentence-aware, 220/40 tokens
  ├─► EmbeddingService         all-MiniLM-L6-v2 -> L2-normalized float32 (N, 384)
  └─► FAISSVectorStore         IndexIDMap2(IndexFlatIP) + monotonic int64 vector_ids
                               Transactional commit: gen_<N>/{index,metadata,documents} -> atomic manifest.json

Vector Retrieval Testing (LLM-Independent)
  └─► preprocess_text -> EmbeddingService.embed_query -> FAISSVectorStore.search (optional document_id)

PDF Summary & Practice Questions (Strictly Single-PDF, Session-Cached, Never Indexed in FAISS)
  ├─► Full / Page Range / Exact Section -> store.get_document_chunks(document_id, ...)
  ├─► Topic / Semantic Section Fallback -> retrieve_chunks(..., document_id=document_id)
  ├─► Summarizer                        -> Prompt-aware bounded hierarchical map-reduce
  ├─► Question Generator                -> Evenly spaced token-budgeted batches -> JSON parse ->
  │                                        strict source_chunk_ids verification -> deterministic page derivation ->
  │                                        deduplication -> bounded top-up / shortfall notice
  └─► Quiz Evaluator (Stage 6.3)        -> MCQ / True-False (deterministic) + Short Answer (batched LLM or deterministic)
                                           -> discrete scoring (0.0/0.5/1.0) -> breakdown & missed page review
```

* **`all-MiniLM-L6-v2` + `220/40` Token Chunking:** CPU-friendly `384`-dim embeddings with default chunk sizes below the model's `256`-token sequence limit.
* **`IndexIDMap2(IndexFlatIP)` + Transactional Generations:** Exact cosine similarity on L2-normalized vectors, stable non-reused `int64` vector IDs for isolated per-document deletion, and atomic `manifest.json` swaps.
* **Chunk-ID Grounding for Practice Questions:** Rather than trusting LLM-reported page numbers, every question must cite exact `source_chunk_ids` from the batch. `question_validation.py` verifies ownership against the selected `document_id`, rejects unknown/cross-document IDs, and derives `source_pages`, `filename`, and `document_id` deterministically from the vector store.
* **Grounded Quiz Scoring & Resubmission Safeguards:** Short-answer questions evaluated by LLM are strictly bounded to retrieved chunks of the cited document; MCQ/True-False grading is completely deterministic. Unanswered questions receive 0.0 points. If an API call fails during submission, user-typed answers remain intact in session state so submission can be retried immediately. Retrying a quiz increments the attempt counter and clears prior answers while reusing the exact cached questions with zero generation calls.

---

## 4. LLM Architecture: Current Groq & Future Local LLM Readiness

* **Current Provider:** `GroqLLMProvider` (default `GROQ_MODEL=openai/gpt-oss-120b`) powers the **PDF Summary**, **Practice Questions**, and **Quick Q&A (Groq)** tabs.
* **Strict Separation:** `src/ingestion/`, `src/retrieval/`, and `src/database/` have **zero imports or dependencies on Groq**. Uploading, chunking, embedding, persisting, deleting, and searching PDFs work without `GROQ_API_KEY`.
* **Future Ollama / Qwen Readiness (Not Implemented Yet):** `src/generation/llm_provider.py` provides `BaseLLMProvider` and `register_llm_provider()`. Adding an Ollama/Qwen provider in a future phase will not require rewriting chunking, embeddings, FAISS, summarization, question generation, or the Streamlit UI. Requesting `LLM_PROVIDER=ollama` currently raises a clear configuration error.

---

## 5. Installation & Setup

### Prerequisites
* Python 3.11+ (verified on Python 3.13.15)

### Virtual Environment & Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verified dependencies in `requirements.txt` (exact 7/7 match with third-party imports in `app.py` and `src/`):
* `streamlit` — Web UI, session state, resource caching
* `PyMuPDF` (`pymupdf`) — In-memory PDF text extraction
* `faiss-cpu` — Local FAISS vector index
* `sentence-transformers` — Local `all-MiniLM-L6-v2` embeddings
* `numpy` — Contiguous `float32` vector arrays
* `python-dotenv` — `.env` configuration loader
* `groq` — Hosted LLM client for `GroqLLMProvider`

---

## 6. Environment Configuration

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model name (validated against `manifest.json`) |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size for chunk embedding |
| `CHUNK_SIZE_TOKENS` | `220` | Target max tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `40` | Token overlap between adjacent chunks on the same page |
| `TOP_K` | `3` | Default number of retrieved chunks |
| `VECTOR_STORE_DIR` | `data/vector_store` | Local directory for persisted FAISS generations & manifest |
| `LLM_PROVIDER` | `groq` | Active LLM provider (`groq`; `ollama` planned for a future phase) |
| `LLM_TIMEOUT_SECONDS` | `60` | LLM request timeout in seconds |
| `GROQ_API_KEY` | *(none)* | Required for **PDF Summary**, **Practice Questions**, and **Quick Q&A**; optional for indexing & retrieval |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Active Groq chat completion model |
| `SUMMARY_MAX_BATCH_TOKENS` | `2400` | Max input tokens per summary LLM call (includes prompts, headers & text) |
| `SUMMARY_TOPIC_TOP_K` | `6` | Default retrieved chunks for Topic / Section fallback summaries |
| `QUESTION_MAX_BATCH_TOKENS` | `2400` | Max input tokens per question-generation LLM call (includes prompts, schema, headers, text & previous stems) |
| `QUESTION_DEFAULT_COUNT` | `5` | Default number of practice questions |
| `QUESTION_MAX_COUNT` | `20` | Maximum allowed questions per generation request |
| `QUESTION_TOPIC_TOP_K` | `6` | Default retrieved chunks for Topic-scoped practice questions |
| `QUESTION_MAX_RETRIES` | `1` | Max retries per batch on malformed JSON or zero valid questions |
| `QUESTION_MAX_OUTPUT_TOKENS` | `1400` | Upper bound on reserved output tokens per question batch |
| `QUESTION_MAX_PREVIOUS_STEMS` | `15` | Max prior question stems passed to subsequent batches to prevent duplicates |
| `QUIZ_SHORT_ANSWER_EVALUATOR` | `llm` | Short answer evaluator mode (`llm` or `deterministic`) |
| `QUIZ_SHORT_ANSWER_CORRECT_THRESHOLD` | `0.75` | Deterministic token-overlap threshold for full credit (1.0 pt) |
| `QUIZ_SHORT_ANSWER_PARTIAL_THRESHOLD` | `0.40` | Deterministic token-overlap threshold for partial credit (0.5 pt) |

---

## 7. Running the Application

```bash
./venv/bin/streamlit run app.py
```

1. **Upload & Index PDF:** Upload PDFs and click **Process & Index PDF** (duplicates blocked before indexing; persisted across restarts).
2. **Vector Retrieval Testing:** Search across all PDFs or filter to one PDF and inspect ranked chunks with filename, page, `chunk_id`, and cosine similarity score.
3. **PDF Summary:** Choose one PDF, pick Full Document / Page Range / Section / Topic and Short / Medium / Detailed, and click **Generate Summary**.
4. **Practice Questions & Interactive Quiz:** Choose one PDF, pick Full Document / Page Range / Topic, choose question type (MCQ, Short Answer, True / False, Mixed), select Easy / Medium / Hard, choose question count, and click **Generate Practice Questions**.
   * In **Study View**, toggle **Show Answers & Explanations** to reveal answers and explanations without calling the LLM.
   * In **Interactive Quiz**, choose or type your answers with correct answers hidden. Select LLM or Deterministic grading for short answers, click **Submit Answers** to view discrete scores (`0.0`, `0.5`, `1.0`), type breakdown, per-question badges, and missed pages review. Click **Retry Quiz** to start a fresh attempt without re-calling question generation.
5. **Quick Q&A (Groq):** Ask a grounded single-turn question across your indexed PDFs.

---

## 8. Running Automated Tests

```bash
# Full offline unit + Streamlit UI test suite (109 passed, 1 opt-in skipped)
./venv/bin/python -m unittest discover -s tests -v

# Opt-in real all-MiniLM-L6-v2 embedding smoke test (1 passed)
RUN_REAL_EMBEDDING_TESTS=1 ./venv/bin/python -m unittest tests.test_embeddings.TestRealEmbeddingModelSmoke -v
```

---

## 9. Current Limitations & Troubleshooting

### Known Limitations
* **Scanned / Image-Only PDFs (No OCR):** PyMuPDF extracts embedded text only. Image-only PDFs raise `"No readable text was found in this PDF."` OCR is not implemented yet.
* **Lightweight Token Estimation:** `estimate_tokens()` uses a fast regex estimator (~10–15% variance vs. exact model tokenizers, safely absorbed by configured batch token budgets).
* **Session-Only Summary, Question & Quiz Caches:** Generated summaries, questions, and active quiz sessions stay in Streamlit session state (never indexed in FAISS) and clear on browser restart or document deletion/store reset.
* **Small-PDF Question Shortfalls:** Narrow page ranges or very brief PDFs may yield fewer unique grounded questions than requested; Study Bot displays a clear shortfall warning rather than fabricating facts or duplicates.
* **Single-User Local Persistence:** Designed for single-user local execution rather than concurrent multi-writer servers.

### Troubleshooting
* **Incompatible Embedding Model / Corrupted Store:** Study Bot leaves existing local index files untouched and requires explicit confirmation before resetting.
* **Missing `GROQ_API_KEY` Warning:** Only affects LLM tabs (**PDF Summary**, **Practice Questions**, **Quick Q&A**); upload, indexing, deletion, and **Vector Retrieval Testing** remain fully functional.
* **Groq Rate Limits on Large PDFs or Quiz Grading:** Lower `SUMMARY_MAX_BATCH_TOKENS` or `QUESTION_MAX_BATCH_TOKENS` in `.env`, narrow the Page Range / Topic, or switch quiz short-answer evaluation to `deterministic` mode.

---

## 10. Planned Future Work (Not Yet Implemented)

* **Ollama / Qwen Local LLM Integration** (`OllamaLLMProvider` for local Qwen models such as Qwen 2.5 3B).
* **Timers & Timed Study Sessions** (timed quiz challenges, countdown clocks).
* **Flashcards** (spaced-repetition study flashcards).
* **Multi-Turn PDF Chatbot** (conversational RAG chat with persistent chat history).
* **OCR** for scanned / image-only PDFs.
* **Multi-Turn PDF Chatbot** (conversational RAG chat with persistent chat history).
* **OCR** for scanned / image-only PDFs.
