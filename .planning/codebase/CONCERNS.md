# Known Concerns & Technical Debt

## Resolved (Stages 1–6.2, 2026-11-18)
1. **Deprecated Groq model (`llama-3.3-70b-versatile`)**: Replaced with configurable `GROQ_MODEL` (`openai/gpt-oss-120b`) wrapped in `GroqLLMProvider`.
2. **Ephemeral single-PDF store**: Replaced with persistent generation-based transactional `FAISSVectorStore` (`IndexIDMap2(IndexFlatIP)`) and `DocumentManager`.
3. **Naïve character chunking**: Replaced with paragraph-first, sentence-aware `220/40` token chunking preserving page/chunk metadata.
4. **LLM coupling**: Kept ingestion, chunking, embeddings, document registry, FAISS storage, and retrieval testing 100% independent of Groq; introduced `BaseLLMProvider` + registry for future local providers.
5. **Large-PDF summarization limits (Stage 6.1)**: Added prompt-overhead-aware hierarchical map-reduce (`SUMMARY_MAX_BATCH_TOKENS=2400`, `MAX_REDUCTION_LEVELS=3`) with verified source page metadata and session-only caching.
6. **Hallucination & ungrounded practice questions (Stage 6.2)**: Added strict `source_chunk_ids` validation, deterministic page/filename derivation, full prompt-overhead token budgeting with bounded previous-stem context, MCQ/Short-Answer/True-False/Mixed validators, deduplication, bounded retries/top-up, shortfall reporting, and session-only caching invalidated on delete/reset.

## Remaining Limitations & Future Work (Not Yet Implemented)
- **No OCR for scanned/image-only PDFs**: Image-only PDFs raise a clear `ValueError`; OCR is planned for a future phase.
- **Regex token estimator variance**: `estimate_tokens()` has ~10–15% variance vs. exact model tokenizers (mitigated by configured chunk headroom and batch token budgets).
- **Session-only summary & question caches**: Generated summaries and practice questions clear on browser/server restart (intentionally kept separate from FAISS).
- **Small-PDF question shortfalls**: Narrow scopes or short PDFs may yield fewer valid unique questions than requested (surfaced transparently via shortfall notice without inventing content).
- **No Git repository initialized**: No commits, pushes, or publications have been made.
- **Planned future phases (not implemented)**: Ollama + local Qwen provider, interactive quiz scoring, timers, flashcards, multi-turn PDF chatbot, and OCR.
