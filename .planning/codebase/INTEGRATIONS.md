# External Integrations

## Hugging Face Hub (Local Embedding Model)
- **SDK:** `sentence-transformers` 6.2.0 (`all-MiniLM-L6-v2`, `384`-dim)
- **Usage:** `src/retrieval/embeddings.py::EmbeddingService` + unified `@st.cache_resource` loader
- **Independence:** Completely independent of Groq or any LLM provider; runs locally on CPU.
- **Auth:** Public download works without authentication; optional `HF_TOKEN` suppresses Hub rate-limit notices.

## Groq Cloud API (Via `BaseLLMProvider` / `GroqLLMProvider`)
- **SDK:** `groq` 1.7.0 (`src/generation/llm_provider.py`)
- **Auth:** `GROQ_API_KEY` loaded via `src/config.py` (optional — indexing and Vector Retrieval Testing work without it)
- **Model:** `GROQ_MODEL` (default `openai/gpt-oss-120b`)
- **Usage:** Powers the **PDF Summary** (`src/generation/summarizer.py`), **Practice Questions** (`src/generation/question_generator.py`), and **Quick Q&A (Groq)** (`src/generation/answer_generator.py`) tabs.

## Planned Future Integration (Not Yet Implemented)
- **Ollama + Local Qwen (e.g., Qwen 2.5 3B):** `src/generation/llm_provider.py` provides `BaseLLMProvider` and `register_llm_provider()` so Ollama can be added in a future phase without modifying ingestion, embeddings, FAISS, summarization, question generation, or UI logic. Not implemented yet.
