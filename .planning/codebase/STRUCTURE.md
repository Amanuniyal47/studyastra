# Directory Structure

```text
Study Bot/
├── app.py                              # Streamlit UI (KB sidebar, Upload/Index, Retrieval Testing, PDF Summary, Practice Questions, Quick Q&A)
├── README.md                           # Full architecture, setup, testing, and troubleshooting guide
├── requirements.txt                    # Verified Python dependencies (7/7 exact match)
├── .env                                # Local secrets (GROQ_API_KEY) — do not commit
├── .env.example                        # Template environment variables
├── .vscode/
│   └── settings.json                   # VS Code python.terminal.useEnvFile
├── data/
│   └── vector_store/                   # Persisted FAISS generations & manifest.json (created on index)
├── src/
│   ├── __init__.py
│   ├── config.py                       # Env loading, chunk/embedding/store/LLM/summary/question config
│   ├── database/
│   │   ├── __init__.py
│   │   └── document_manager.py         # Document registry & per-document vector ID ownership
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_loader.py               # PyMuPDF in-memory page-by-page text extraction
│   │   └── chunker.py                  # Paragraph/sentence-aware token-budget chunker
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embeddings.py               # LLM-independent EmbeddingService + cached model loader
│   │   ├── vector_store.py             # Persistent transactional FAISSVectorStore + read-only doc chunk getters
│   │   └── retriever.py                # Top-K query retrieval with optional document filtering
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_provider.py             # BaseLLMProvider, GroqLLMProvider, provider registry & factory
│   │   ├── prompts.py                  # System & user Q&A prompt formatting
│   │   ├── answer_generator.py         # Single-turn Q&A via LLM provider abstraction
│   │   ├── summary_prompts.py          # Grounded single-pass, map-batch & reduce summary prompts
│   │   ├── summarizer.py               # Hierarchical single-PDF summarizer (full, page_range, section, topic)
│   │   ├── question_validation.py      # PracticeQuestion/QuestionSetResult, JSON parsing, chunk grounding, dedup
│   │   ├── question_prompts.py         # Difficulty rubrics, JSON schemas & bounded previous-stem prompts
│   │   └── question_generator.py       # Batch single-PDF practice question generator (full, page_range, topic, mixed)
│   └── utils/
│       ├── __init__.py
│       └── file_utils.py               # SHA-256 file hashing
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py                 # Stage 1 config & chunking tests
│   ├── test_embeddings.py              # Stage 2 mocked + opt-in real-model embedding tests
│   ├── test_vector_store.py            # Stage 3 persistence, restart, deletion, rollback & corruption tests
│   ├── test_llm_provider.py            # Stage 6.1a provider factory, Groq error handling & registry tests
│   ├── test_summarizer.py              # Stage 6.1b modes, lengths, prompt-aware hierarchical budget & isolation tests
│   ├── test_question_validation.py     # Stage 6.2a config, mixed allocation, JSON parsing, chunk grounding & dedup tests
│   ├── test_question_generator.py      # Stage 6.2b batch orchestration, budget, shortfall, retry & isolation tests
│   ├── test_pipeline.py                # Core pipeline & Q&A unit tests
│   └── test_app_ui.py                  # Stage 4, 6.1c & 6.2c Streamlit AppTest UI integration tests
└── venv/                               # Local Python 3.13 virtual environment
```
