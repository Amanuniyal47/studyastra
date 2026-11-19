# Architecture

**Pattern:** Modular, LLM-independent Streamlit RAG foundation with persistent multi-PDF FAISS storage and a decoupled, provider-abstracted generation layer.

## Data Flow
```text
Uploaded PDF (Streamlit)
  │
  ├─► src/utils/file_utils.py::file_hash()          [SHA-256 document_id]
  ├─► src/database/document_manager.py              [Duplicate check & per-document vector ownership]
  ├─► src/ingestion/pdf_loader.py::extract_pdf_text() [PyMuPDF in-memory page extraction]
  ├─► src/ingestion/chunker.py::chunk_text()        [Paragraph/sentence-aware 220/40 token chunker]
  └─► src/retrieval/embeddings.py::EmbeddingService [Cached all-MiniLM-L6-v2, L2-normalized float32 (N, 384)]
        └─► src/retrieval/vector_store.py::FAISSVectorStore
              ├─► IndexIDMap2(IndexFlatIP(384))     [Exact cosine similarity + monotonic int64 vector_ids]
              ├─► Read-only doc helpers             [get_document_chunks(), get_document_page_numbers()]
              └─► data/vector_store/                [generations/gen_<N>/ + atomic manifest.json]

Vector Retrieval Testing (LLM-Independent)
  └─► src/retrieval/retriever.py::retrieve_chunks() [Query validation -> embed_query -> filtered/global top-K search]

Generation Layer (Decoupled via BaseLLMProvider / GroqLLMProvider)
  ├─► src/generation/llm_provider.py                [BaseLLMProvider, GroqLLMProvider, registry & factory]
  ├─► src/generation/summarizer.py + summary_prompts.py
  │     [Single-PDF Full / Page Range / Section / Topic + prompt-aware bounded hierarchical map-reduce]
  ├─► src/generation/question_generator.py + question_validation.py + question_prompts.py
  │     [Single-PDF Full / Page Range / Topic; MCQ, Short Answer, True/False, Mixed; strict source_chunk_ids grounding]
  └─► src/generation/answer_generator.py + prompts.py
        [Existing single-turn Quick Q&A]
```

## State & Persistence
- **Model Cache:** `@st.cache_resource` caches `SentenceTransformer` once per resolved `EMBEDDING_MODEL`.
- **Session Store:** `st.session_state["vector_store"]` loads `FAISSVectorStore` once per session from `data/vector_store/`.
- **Session-Only Generation Caches:** `st.session_state["cached_summaries"]` and `st.session_state["cached_questions"]` stay strictly separate from FAISS and are invalidated immediately on document deletion or store reset.
- **LLM Decoupling:** `src/ingestion/`, `src/retrieval/`, and `src/database/` have zero imports from `src/generation/` or `groq`.
