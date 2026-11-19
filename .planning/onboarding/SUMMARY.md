# Onboarding & Stage 6.2 Summary — Study Bot

**Last Updated:** 2026-11-18

## Current Project State
Study Bot includes a verified, LLM-independent PDF-to-FAISS vector pipeline plus single-PDF hierarchical summarization and chunk-grounded practice question generation:
- **Ingestion:** `src/ingestion/pdf_loader.py` (PyMuPDF in-memory) + `src/ingestion/chunker.py` (paragraph/sentence-aware `220/40` token chunker with rich metadata & deduplication).
- **Embeddings:** `src/retrieval/embeddings.py` (`EmbeddingService` using local `all-MiniLM-L6-v2`, `384`-dim, L2-normalized, cached once per session).
- **Persistent Vector DB & Document Registry:** `src/retrieval/vector_store.py` (`FAISSVectorStore` with `IndexIDMap2(IndexFlatIP)`, generation-based transactional persistence, monotonic `int64` IDs, read-only document chunk helpers, compatibility & corruption guards) + `src/database/document_manager.py` + `src/retrieval/retriever.py`.
- **LLM Provider & Generation:** `src/generation/llm_provider.py` (`BaseLLMProvider`, `GroqLLMProvider`, registry/factory), `src/generation/summarizer.py` + `summary_prompts.py` (Stage 6.1), `src/generation/question_generator.py` + `question_validation.py` + `question_prompts.py` (Stage 6.2), and `src/generation/answer_generator.py`.
- **Streamlit UI:** `app.py` with KB sidebar, explicit indexing/deletion/reset controls, **Vector Retrieval Testing** (LLM-independent), **PDF Summary**, **Practice Questions** (session-only caches invalidated on delete/reset), and preserved **Quick Q&A (Groq)**.
- **Documentation:** Updated `README.md` and synchronized `.planning/` docs.

## Next Recommended Phase (Planned, Not Started)
- Stage 6.3+: Interactive quizzes (scoring/timers), flashcards, multi-turn PDF chatbot, Ollama/Qwen local LLM integration, or OCR.
