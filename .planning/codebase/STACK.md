# Technology Stack

**Project:** Study Bot (AI PDF Knowledge Base, Retrieval, Summarization & Practice Questions)
**Last Updated:** 2026-11-18

## Runtime & Environment
- **Language:** Python 3.13.15 (`./venv/bin/python`)
- **Virtual Environment:** `./venv/`
- **Configuration:** `python-dotenv` 1.2.4 loading `.env` via `src/config.py`

## Core Libraries (Exact 7/7 Match Against `requirements.txt`)
| Package | Version (venv) | Role |
|---|---|---|
| `streamlit` | 1.67.0 | Web UI, session state, model caching (`@st.cache_resource`) |
| `PyMuPDF` (`pymupdf`) | 2.1 | In-memory PDF text extraction |
| `sentence-transformers` | 6.2.0 | Local dense text embeddings (`all-MiniLM-L6-v2`, 384-dim) |
| `faiss-cpu` | 1.15.1 | Persistent `IndexIDMap2(IndexFlatIP)` cosine similarity store |
| `numpy` | 2.5.4 | Contiguous `float32` L2-normalized embedding arrays |
| `python-dotenv` | 1.2.4 | `.env` configuration management |
| `groq` | 1.7.0 | Hosted LLM client wrapped in `GroqLLMProvider` (PDF Summary, Practice Questions & Quick Q&A tabs) |

## Configuration Parameters (`src/config.py`)
- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"`, `EMBEDDING_BATCH_SIZE = 32`
- `CHUNK_SIZE_TOKENS = 220`, `CHUNK_OVERLAP_TOKENS = 40`
- `TOP_K = 3`, `VECTOR_STORE_DIR = data/vector_store`
- `LLM_PROVIDER = "groq"`, `LLM_TIMEOUT_SECONDS = 60`
- `GROQ_API_KEY`, `GROQ_MODEL = "openai/gpt-oss-120b"`
- `SUMMARY_MAX_BATCH_TOKENS = 2400`, `SUMMARY_TOPIC_TOP_K = 6`
- `QUESTION_MAX_BATCH_TOKENS = 2400`, `QUESTION_DEFAULT_COUNT = 5`, `QUESTION_MAX_COUNT = 20`, `QUESTION_TOPIC_TOP_K = 6`, `QUESTION_MAX_RETRIES = 1`, `QUESTION_MAX_OUTPUT_TOKENS = 1400`, `QUESTION_MAX_PREVIOUS_STEMS = 15`
