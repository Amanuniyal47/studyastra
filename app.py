import logging
import shutil

import streamlit as st

import src.config as config
from src.generation.answer_generator import generate_answer
import src.generation.llm_provider as llm_provider
from src.generation.llm_provider import (
    LLMConfigurationError,
    LLMProviderError,
)
import src.generation.question_generator as question_generator
from src.generation.question_validation import QuestionGenerationError
import src.generation.quiz_evaluator as quiz_evaluator
from src.generation.quiz_evaluator import QuizEvaluationError
from src.generation.summarizer import (
    SummarizationError,
    format_page_range,
    summarize_document,
    summarize_page_range,
    summarize_section,
    summarize_topic,
)
from src.retrieval.embeddings import (
    EmbeddingGenerationError,
    EmbeddingModelError,
    EmbeddingService,
)
from src.retrieval.retriever import retrieve_chunks
from src.retrieval.vector_store import (
    FAISSVectorStore,
    VectorStoreCompatibilityError,
    VectorStoreCorruptionError,
    VectorStoreError,
    VectorStorePersistenceError,
)
from src.utils.file_utils import file_hash

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PDF Chat",
    page_icon="📄",
    layout="wide",
)


def _get_embedding_service() -> EmbeddingService:
    service = st.session_state.get("embedding_service")
    if service is None or service.model_name != config.EMBEDDING_MODEL:
        service = EmbeddingService(model_name=config.EMBEDDING_MODEL)
        st.session_state["embedding_service"] = service
    return service


def _invalidate_cached_summaries(document_id: str | None = None) -> None:
    """Clear session-cached summaries for one deleted document or all documents."""
    cached = st.session_state.get("cached_summaries")
    if document_id is None:
        st.session_state.pop("cached_summaries", None)
        st.session_state.pop("summary_doc_select", None)
        st.session_state.pop("summary_error", None)
        st.session_state.pop("summary_error_doc_id", None)
        return

    if isinstance(cached, dict):
        cached.pop(document_id, None)
    if st.session_state.get("summary_doc_select") == document_id:
        st.session_state.pop("summary_doc_select", None)
    if st.session_state.get("summary_error_doc_id") == document_id:
        st.session_state.pop("summary_error", None)
        st.session_state.pop("summary_error_doc_id", None)


def _clear_quiz_widget_keys(document_id: str | None = None) -> None:
    """Remove quiz answer/view widget keys from session state for one or all documents."""
    prefixes = (
        ("quiz_ans_", "questions_view_mode_", "quiz_eval_mode_")
        if document_id is None
        else (
            f"quiz_ans_{document_id}_",
            f"questions_view_mode_{document_id}",
            f"quiz_eval_mode_{document_id}",
        )
    )
    stale_keys = [
        key for key in list(st.session_state.keys()) if isinstance(key, str) and key.startswith(prefixes)
    ]
    for key in stale_keys:
        st.session_state.pop(key, None)


def _invalidate_cached_questions(document_id: str | None = None) -> None:
    """Clear session-cached practice questions and quiz state for one or all documents."""
    cached = st.session_state.get("cached_questions")
    quiz_sessions = st.session_state.get("quiz_sessions")

    if document_id is None:
        st.session_state.pop("cached_questions", None)
        st.session_state.pop("quiz_sessions", None)
        st.session_state.pop("questions_doc_select", None)
        st.session_state.pop("questions_error", None)
        st.session_state.pop("questions_error_doc_id", None)
        _clear_quiz_widget_keys(None)
        return

    if isinstance(cached, dict):
        cached.pop(document_id, None)
    if isinstance(quiz_sessions, dict):
        quiz_sessions.pop(document_id, None)
    if st.session_state.get("questions_doc_select") == document_id:
        st.session_state.pop("questions_doc_select", None)
    if st.session_state.get("questions_error_doc_id") == document_id:
        st.session_state.pop("questions_error", None)
        st.session_state.pop("questions_error_doc_id", None)
    st.session_state.pop(f"show_answers_{document_id}", None)
    _clear_quiz_widget_keys(document_id)


def _get_or_init_quiz_session(document_id: str, question_set_id: str) -> dict:
    """Return active quiz session for (document_id, question_set_id), initializing if needed."""
    sessions = st.session_state.setdefault("quiz_sessions", {})
    existing = sessions.get(document_id)
    if (
        isinstance(existing, dict)
        and existing.get("document_id") == document_id
        and existing.get("question_set_id") == question_set_id
    ):
        return existing

    fresh = {
        "document_id": document_id,
        "question_set_id": question_set_id,
        "attempt_number": 1,
        "submitted": False,
        "answers": {},
        "result": None,
        "error": None,
    }
    sessions[document_id] = fresh
    return fresh


def _load_vector_store() -> tuple[FAISSVectorStore | None, str | None, bool]:
    """Load the persisted FAISS store once per Streamlit session.

    Returns:
        (store, error_message, recoverable_by_reset)
    """
    existing = st.session_state.get("vector_store")
    if existing is not None:
        return existing, None, False

    try:
        store = FAISSVectorStore(
            store_dir=config.VECTOR_STORE_DIR,
            embedding_model=config.EMBEDDING_MODEL,
        )
        st.session_state["vector_store"] = store
        return store, None, False
    except VectorStoreCompatibilityError as exc:
        logger.exception("Incompatible vector store at %s", config.VECTOR_STORE_DIR)
        return (
            None,
            f"Incompatible local vector index: {exc} "
            "Switch EMBEDDING_MODEL back to the original model or explicitly reset the local store below.",
            True,
        )
    except VectorStoreCorruptionError as exc:
        logger.exception("Corrupted vector store at %s", config.VECTOR_STORE_DIR)
        return (
            None,
            f"Local vector index is corrupted or incomplete: {exc} "
            "Existing files were left untouched. You may explicitly reset the local store below.",
            True,
        )
    except VectorStoreError as exc:
        logger.exception("Vector store initialization error at %s", config.VECTOR_STORE_DIR)
        return None, f"Failed to load local vector store: {exc}", False


def _render_recovery_panel(load_error: str, can_reset: bool) -> None:
    st.error(load_error)
    if not can_reset:
        return

    st.warning(
        "Your existing files have NOT been deleted automatically. "
        "Confirm below only if you want to permanently clear the local vector store."
    )
    confirm_reset = st.checkbox(
        "I understand resetting will permanently delete all locally indexed PDFs and vectors.",
        key="confirm_reset_vector_store",
    )
    if st.button(
        "Reset Local Vector Store",
        type="secondary",
        disabled=not confirm_reset,
        key="reset_vector_store_btn",
    ):
        try:
            if config.VECTOR_STORE_DIR.exists():
                shutil.rmtree(config.VECTOR_STORE_DIR)
            st.session_state.pop("vector_store", None)
            st.session_state.pop("last_index_result", None)
            st.session_state.pop("last_index_error", None)
            _invalidate_cached_summaries(None)
            _invalidate_cached_questions(None)
            st.session_state["last_action_notice"] = "Local vector store has been reset."
            st.rerun()
        except Exception as exc:
            logger.exception("Failed to reset local vector store at %s", config.VECTOR_STORE_DIR)
            st.error(f"Could not reset local vector store: {exc}")


def _render_sidebar(store: FAISSVectorStore) -> None:
    st.sidebar.header("Knowledge Base")
    st.sidebar.caption(f"Embedding model: `{store.embedding_model}`")
    if store.embedding_dimension is not None:
        st.sidebar.caption(f"Vector dimension: `{store.embedding_dimension}`")

    col1, col2 = st.sidebar.columns(2)
    col1.metric("Documents", store.total_documents)
    col2.metric("FAISS Vectors", store.total_vectors)

    st.sidebar.divider()
    st.sidebar.subheader("Indexed Documents")

    documents = store.list_documents()
    if not documents:
        st.sidebar.caption("No PDFs indexed yet.")
        return

    pending_delete_id = st.session_state.get("pending_delete_doc_id")

    for doc in documents:
        doc_id = doc["document_id"]
        short_id = doc_id[:12]
        with st.sidebar.container(border=True):
            st.markdown(f"**{doc['filename']}**")
            st.caption(
                f"Pages: {doc['page_count']} | Chunks: {doc['chunk_count']} | ID: `{short_id}`"
            )

            if pending_delete_id == doc_id:
                st.warning(f"Delete '{doc['filename']}' from the index?")
                c_confirm, c_cancel = st.columns(2)
                if c_confirm.button(
                    "Confirm Delete",
                    key=f"confirm_delete_{doc_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    try:
                        deleted = store.delete_document(doc_id)
                        st.session_state.pop("pending_delete_doc_id", None)
                        st.session_state.pop("last_index_result", None)
                        _invalidate_cached_summaries(doc_id)
                        _invalidate_cached_questions(doc_id)
                        if st.session_state.get("retrieval_doc_filter") == doc_id:
                            st.session_state["retrieval_doc_filter"] = None
                        st.session_state["last_action_notice"] = (
                            f"Deleted '{deleted['filename']}' "
                            f"({deleted['removed_vectors']} vectors removed)."
                        )
                        st.rerun()
                    except (KeyError, VectorStoreError) as exc:
                        logger.exception("Failed to delete document %s", doc_id)
                        st.sidebar.error(f"Could not delete document: {exc}")
                if c_cancel.button(
                    "Cancel",
                    key=f"cancel_delete_{doc_id}",
                    use_container_width=True,
                ):
                    st.session_state.pop("pending_delete_doc_id", None)
                    st.rerun()
            else:
                if st.button(
                    "Delete",
                    key=f"delete_doc_{doc_id}",
                    use_container_width=True,
                ):
                    st.session_state["pending_delete_doc_id"] = doc_id
                    st.rerun()


def _render_upload_and_indexing(store: FAISSVectorStore) -> None:
    st.subheader("Upload & Index PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        doc_id = file_hash(uploaded_file)
        filename = getattr(uploaded_file, "name", None) or f"document_{doc_id[:8]}.pdf"
        already_indexed = store.is_document_indexed(doc_id)

        if already_indexed:
            existing_doc = store.get_document(doc_id)
            st.warning(
                f"Duplicate detected: '{filename}' is already indexed "
                f"(ID: `{doc_id[:12]}`, {existing_doc['chunk_count']} chunks). "
                "Duplicate indexing is prevented."
            )
        else:
            st.caption(f"Ready to process: **{filename}** (ID: `{doc_id[:12]}`)")

        if st.button(
            "Process & Index PDF",
            type="primary",
            disabled=already_indexed,
            key="process_index_pdf_btn",
        ):
            with st.spinner("Extracting text, chunking, generating embeddings, and updating FAISS..."):
                try:
                    service = _get_embedding_service()
                    result = store.add_pdf(
                        uploaded_file,
                        filename=filename,
                        embedding_service=service,
                        skip_duplicates=True,
                    )
                    st.session_state["last_index_result"] = result
                    st.session_state.pop("last_index_error", None)
                    st.rerun()
                except ValueError as exc:
                    st.session_state["last_index_error"] = str(exc)
                    st.session_state.pop("last_index_result", None)
                except (EmbeddingModelError, EmbeddingGenerationError) as exc:
                    logger.exception("Embedding error while indexing '%s'", filename)
                    st.session_state["last_index_error"] = str(exc)
                    st.session_state.pop("last_index_result", None)
                except (
                    VectorStoreCompatibilityError,
                    VectorStoreCorruptionError,
                    VectorStorePersistenceError,
                    VectorStoreError,
                ) as exc:
                    logger.exception("Vector store error while indexing '%s'", filename)
                    st.session_state["last_index_error"] = str(exc)
                    st.session_state.pop("last_index_result", None)

    if st.session_state.get("last_index_error"):
        st.error(st.session_state["last_index_error"])

    last_result = st.session_state.get("last_index_result")
    if last_result:
        if last_result.get("already_indexed"):
            st.info(
                f"Skipped duplicate PDF '{last_result['filename']}' — already indexed in FAISS."
            )
        else:
            st.success(
                f"Indexed '{last_result['filename']}' successfully "
                f"({last_result['page_count']} page(s), ID: `{last_result['document_id'][:12]}`)."
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks Generated", last_result["chunks_generated"])
        c2.metric("Embeddings Generated", last_result["embeddings_generated"])
        c3.metric("Total Vectors in FAISS", last_result["total_vectors"])


def _render_retrieval_testing_tab(store: FAISSVectorStore) -> None:
    st.subheader("Vector Retrieval Testing")

    if store.total_vectors == 0:
        st.info("Upload a PDF to start asking questions.")
        return

    documents = store.list_documents()
    doc_options: list[str | None] = [None] + [doc["document_id"] for doc in documents]
    doc_labels = {
        None: "All documents",
        **{doc["document_id"]: f"{doc['filename']} ({doc['document_id'][:8]})" for doc in documents},
    }

    if st.session_state.get("retrieval_doc_filter") not in doc_options:
        st.session_state["retrieval_doc_filter"] = None

    with st.form("retrieval_test_form"):
        query = st.text_input(
            "Enter a test search query",
            placeholder="e.g., What are the main stages of photosynthesis?",
            key="retrieval_query_input",
        )
        col_k, col_doc = st.columns([1, 2])
        with col_k:
            top_k = st.number_input(
                "Top K results",
                min_value=1,
                max_value=20,
                value=min(config.TOP_K, max(1, store.total_vectors)),
                step=1,
                key="retrieval_top_k",
            )
        with col_doc:
            selected_doc_id = st.selectbox(
                "Filter by document",
                options=doc_options,
                format_func=lambda doc_id: doc_labels[doc_id],
                key="retrieval_doc_filter",
            )
        submitted = st.form_submit_button("Search Vectors", type="primary")

    if submitted:
        if not query or not query.strip():
            st.warning("Enter a non-empty search query.")
            return

        with st.spinner("Embedding query and searching FAISS..."):
            try:
                service = _get_embedding_service()
                results = retrieve_chunks(
                    query,
                    top_k=int(top_k),
                    document_id=selected_doc_id,
                    store=store,
                    embedding_service=service,
                )
            except (
                ValueError,
                EmbeddingModelError,
                EmbeddingGenerationError,
                VectorStoreError,
            ) as exc:
                logger.exception("Retrieval search failed")
                st.error(f"Retrieval failed: {exc}")
                return

        if not results:
            st.info("No matching chunks were found for this filter.")
            return

        st.caption(f"Showing top {len(results)} retrieved chunk(s) by cosine similarity.")
        for rank, item in enumerate(results, start=1):
            meta = item["metadata"]
            header = (
                f"#{rank} — {meta['filename']} | "
                f"Page {meta['page_range']} | "
                f"Chunk ID: {meta['chunk_id']} (Chunk {meta['chunk_index']}) | "
                f"Cosine Similarity: {item['score']:.4f}"
            )
            with st.expander(header, expanded=(rank == 1)):
                st.caption(
                    f"Filename: {meta['filename']} | Page: {meta['page']} | "
                    f"Chunk ID: `{meta['chunk_id']}` | Similarity: `{item['score']:.4f}`"
                )
                st.write(item["text"])


def _render_summary_tab(store: FAISSVectorStore) -> None:
    st.subheader("PDF Summary")
    st.caption(
        "Generate grounded summaries strictly for one indexed PDF at a time. "
        "Generated summaries stay in session state and are never indexed into FAISS."
    )

    documents = store.list_documents()
    if not documents:
        st.info("Index at least one PDF first to generate a summary.")
        return

    try:
        provider = llm_provider.get_llm_provider()
    except LLMConfigurationError as exc:
        st.warning(
            f"LLM provider is not available for summarization: {exc} "
            "PDF indexing and Vector Retrieval Testing continue working normally."
        )
        return

    if not provider.is_configured():
        st.warning(
            "GROQ_API_KEY is not configured. Add it to your .env file to generate PDF summaries. "
            "PDF indexing and Vector Retrieval Testing continue working without it."
        )
        return

    doc_ids = [doc["document_id"] for doc in documents]
    doc_labels = {
        doc["document_id"]: f"{doc['filename']} ({doc['page_count']} page(s), {doc['document_id'][:8]})"
        for doc in documents
    }

    if st.session_state.get("summary_doc_select") not in doc_ids:
        st.session_state["summary_doc_select"] = doc_ids[0]

    col_doc, col_mode, col_len = st.columns([2, 1.3, 1])
    with col_doc:
        selected_doc_id = st.selectbox(
            "Select PDF to summarize",
            options=doc_ids,
            format_func=lambda doc_id: doc_labels[doc_id],
            key="summary_doc_select",
        )
    with col_mode:
        mode_label = st.selectbox(
            "Summary mode",
            options=["Full Document", "Page Range", "Section", "Topic"],
            key="summary_mode_select",
        )
    with col_len:
        length_label = st.selectbox(
            "Summary length",
            options=["Short", "Medium", "Detailed"],
            index=1,
            key="summary_length_select",
        )

    available_pages = store.get_document_page_numbers(selected_doc_id)
    start_page = available_pages[0]
    end_page = available_pages[-1]
    section_heading = ""
    topic_query = ""
    topic_top_k = config.SUMMARY_TOPIC_TOP_K

    if mode_label == "Page Range":
        st.caption(
            f"Actual indexed pages in this PDF: **{format_page_range(available_pages)}**"
        )
        if st.session_state.get("summary_start_page") not in available_pages:
            st.session_state["summary_start_page"] = available_pages[0]
        if st.session_state.get("summary_end_page") not in available_pages:
            st.session_state["summary_end_page"] = available_pages[-1]

        col_start, col_end = st.columns(2)
        with col_start:
            start_page = int(
                st.selectbox(
                    "Start page",
                    options=available_pages,
                    key="summary_start_page",
                )
            )
        with col_end:
            end_page = int(
                st.selectbox(
                    "End page",
                    options=available_pages,
                    key="summary_end_page",
                )
            )
    elif mode_label == "Section":
        section_heading = st.text_input(
            "Section heading or title",
            placeholder="e.g., Chapter 2: Cellular Respiration",
            key="summary_section_input",
        )
    elif mode_label == "Topic":
        col_topic, col_k = st.columns([3, 1])
        with col_topic:
            topic_query = st.text_input(
                "Topic to summarize",
                placeholder="e.g., ATP synthesis in mitochondria",
                key="summary_topic_input",
            )
        with col_k:
            topic_top_k = int(
                st.number_input(
                    "Top K chunks",
                    min_value=1,
                    max_value=20,
                    value=min(config.SUMMARY_TOPIC_TOP_K, max(1, store.total_vectors)),
                    step=1,
                    key="summary_topic_top_k",
                )
            )

    if st.button("Generate Summary", type="primary", key="generate_summary_btn"):
        # Clear any stale summary or previous error BEFORE starting a new generation attempt
        cached_summaries = st.session_state.setdefault("cached_summaries", {})
        cached_summaries.pop(selected_doc_id, None)
        st.session_state.pop("summary_error", None)
        st.session_state.pop("summary_error_doc_id", None)

        progress_bar = st.progress(0.0)
        progress_status = st.empty()

        def _on_progress(step: int, total: int, message: str) -> None:
            ratio = min(1.0, max(0.0, step / max(1, total)))
            progress_bar.progress(ratio)
            progress_status.caption(f"Step {step}/{total}: {message}")

        length_key = length_label.lower()
        try:
            if mode_label == "Full Document":
                result = summarize_document(
                    store,
                    selected_doc_id,
                    length=length_key,
                    provider=provider,
                    progress_callback=_on_progress,
                )
            elif mode_label == "Page Range":
                result = summarize_page_range(
                    store,
                    selected_doc_id,
                    start_page=start_page,
                    end_page=end_page,
                    length=length_key,
                    provider=provider,
                    progress_callback=_on_progress,
                )
            elif mode_label == "Section":
                result = summarize_section(
                    store,
                    selected_doc_id,
                    section_heading,
                    length=length_key,
                    embedding_service=_get_embedding_service(),
                    provider=provider,
                    progress_callback=_on_progress,
                )
            else:
                result = summarize_topic(
                    store,
                    selected_doc_id,
                    topic_query,
                    length=length_key,
                    top_k=topic_top_k,
                    embedding_service=_get_embedding_service(),
                    provider=provider,
                    progress_callback=_on_progress,
                )

            cached_summaries[selected_doc_id] = result.to_dict()
            progress_bar.progress(1.0)
            progress_status.caption("Summary complete.")
        except (
            ValueError,
            KeyError,
            LLMConfigurationError,
            LLMProviderError,
            SummarizationError,
            EmbeddingModelError,
            EmbeddingGenerationError,
            VectorStoreError,
        ) as exc:
            logger.exception("PDF summarization failed for document %s", selected_doc_id)
            cached_summaries.pop(selected_doc_id, None)
            st.session_state["summary_error"] = f"Summary generation failed: {exc}"
            st.session_state["summary_error_doc_id"] = selected_doc_id
            progress_bar.empty()
            progress_status.empty()

    if (
        st.session_state.get("summary_error")
        and st.session_state.get("summary_error_doc_id") == selected_doc_id
    ):
        st.error(st.session_state["summary_error"])
        return

    cached_summaries = st.session_state.get("cached_summaries", {})
    summary_data = cached_summaries.get(selected_doc_id)
    if not summary_data or not store.is_document_indexed(selected_doc_id):
        return

    st.divider()
    st.markdown(f"### Summary — {summary_data['filename']}")
    st.caption(
        f"Document: **{summary_data['filename']}** | "
        f"Pages Covered: **{summary_data['page_range_label']}** | "
        f"Length: **{summary_data['length'].title()}** | "
        f"Mode: **{summary_data['mode'].replace('_', ' ').title()}** | "
        f"Provider: `{summary_data['provider']}` (`{summary_data['model']}`)"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks Used", summary_data["chunks_used"])
    m2.metric("LLM Batches", summary_data["batches_processed"])
    m3.metric(
        "Reduction Levels",
        summary_data["reduction_levels"] if summary_data["hierarchical"] else 0,
    )

    if summary_data.get("match_notice"):
        st.info(summary_data["match_notice"])

    st.markdown(summary_data["summary_text"])

    with st.expander(
        f"Source Chunks Referenced ({len(summary_data['source_chunks'])})",
        expanded=False,
    ):
        for idx, chunk in enumerate(summary_data["source_chunks"], start=1):
            meta = chunk["metadata"]
            score_suffix = (
                f" | Similarity: {chunk['score']:.4f}"
                if chunk.get("score") is not None
                else ""
            )
            st.markdown(
                f"**{idx}. {meta['filename']} — Page {meta['page_range']}, "
                f"Chunk {meta['chunk_index']} (`{meta['chunk_id']}`){score_suffix}**"
            )
            st.write(chunk["text"])
            if idx < len(summary_data["source_chunks"]):
                st.divider()


def _render_practice_questions_tab(store: FAISSVectorStore) -> None:
    st.subheader("Practice Questions")
    st.caption(
        "Generate chunk-grounded practice questions strictly for one indexed PDF at a time. "
        "Generated questions stay in session state and are never indexed into FAISS."
    )

    documents = store.list_documents()
    if not documents:
        st.info("Index at least one PDF first to generate practice questions.")
        return

    try:
        provider = llm_provider.get_llm_provider()
    except LLMConfigurationError as exc:
        st.warning(
            f"LLM provider is not available for practice questions: {exc} "
            "PDF indexing and Vector Retrieval Testing continue working normally."
        )
        return

    if not provider.is_configured():
        st.warning(
            "GROQ_API_KEY is not configured. Add it to your .env file to generate practice questions. "
            "PDF indexing and Vector Retrieval Testing continue working without it."
        )
        return

    doc_ids = [doc["document_id"] for doc in documents]
    doc_labels = {
        doc["document_id"]: f"{doc['filename']} ({doc['page_count']} page(s), {doc['document_id'][:8]})"
        for doc in documents
    }

    if st.session_state.get("questions_doc_select") not in doc_ids:
        st.session_state["questions_doc_select"] = doc_ids[0]

    col_doc, col_scope = st.columns([2, 1.2])
    with col_doc:
        selected_doc_id = st.selectbox(
            "Select PDF for practice questions",
            options=doc_ids,
            format_func=lambda doc_id: doc_labels[doc_id],
            key="questions_doc_select",
        )
    with col_scope:
        scope_label = st.selectbox(
            "Content scope",
            options=["Full Document", "Page Range", "Topic"],
            key="questions_scope_select",
        )

    col_type, col_diff, col_count = st.columns([1.3, 1, 1])
    with col_type:
        type_label = st.selectbox(
            "Question type",
            options=["MCQ", "Short Answer", "True / False", "Mixed"],
            key="questions_type_select",
        )
    with col_diff:
        difficulty_label = st.selectbox(
            "Difficulty",
            options=["Easy", "Medium", "Hard"],
            index=1,
            key="questions_difficulty_select",
        )
    with col_count:
        question_count = int(
            st.number_input(
                "Number of questions",
                min_value=1,
                max_value=config.QUESTION_MAX_COUNT,
                value=min(config.QUESTION_DEFAULT_COUNT, config.QUESTION_MAX_COUNT),
                step=1,
                key="questions_count_input",
            )
        )

    available_pages = store.get_document_page_numbers(selected_doc_id)
    start_page: int | None = None
    end_page: int | None = None
    topic_query: str | None = None
    topic_top_k: int | None = None

    if scope_label == "Page Range":
        st.caption(
            f"Actual indexed pages in this PDF: **{format_page_range(available_pages)}**"
        )
        if st.session_state.get("questions_start_page") not in available_pages:
            st.session_state["questions_start_page"] = available_pages[0]
        if st.session_state.get("questions_end_page") not in available_pages:
            st.session_state["questions_end_page"] = available_pages[-1]

        col_start, col_end = st.columns(2)
        with col_start:
            start_page = int(
                st.selectbox(
                    "Start page",
                    options=available_pages,
                    key="questions_start_page",
                )
            )
        with col_end:
            end_page = int(
                st.selectbox(
                    "End page",
                    options=available_pages,
                    key="questions_end_page",
                )
            )
    elif scope_label == "Topic":
        col_topic, col_k = st.columns([3, 1])
        with col_topic:
            topic_query = st.text_input(
                "Topic for practice questions",
                placeholder="e.g., Photosynthesis light reactions",
                key="questions_topic_input",
            )
        with col_k:
            topic_top_k = int(
                st.number_input(
                    "Top K chunks",
                    min_value=1,
                    max_value=20,
                    value=min(config.QUESTION_TOPIC_TOP_K, max(1, store.total_vectors)),
                    step=1,
                    key="questions_topic_top_k",
                )
            )

    scope_map = {
        "Full Document": "full",
        "Page Range": "page_range",
        "Topic": "topic",
    }
    type_map = {
        "MCQ": "mcq",
        "Short Answer": "short_answer",
        "True / False": "true_false",
        "Mixed": "mixed",
    }

    if st.button(
        "Generate Practice Questions",
        type="primary",
        key="generate_questions_btn",
    ):
        # Clear any stale cached questions, quiz state, or prior error BEFORE starting generation
        cached_questions = st.session_state.setdefault("cached_questions", {})
        cached_questions.pop(selected_doc_id, None)
        quiz_sessions = st.session_state.get("quiz_sessions")
        if isinstance(quiz_sessions, dict):
            quiz_sessions.pop(selected_doc_id, None)
        _clear_quiz_widget_keys(selected_doc_id)
        st.session_state.pop("questions_error", None)
        st.session_state.pop("questions_error_doc_id", None)

        progress_bar = st.progress(0.0)
        progress_status = st.empty()

        def _on_progress(step: int, total: int, message: str) -> None:
            ratio = min(1.0, max(0.0, step / max(1, total)))
            progress_bar.progress(ratio)
            progress_status.caption(f"Step {step}/{total}: {message}")

        scope_key = scope_map[scope_label]
        type_key = type_map[type_label]
        difficulty_key = difficulty_label.lower()

        try:
            result = question_generator.generate_practice_questions(
                store,
                selected_doc_id,
                question_type=type_key,
                difficulty=difficulty_key,
                count=question_count,
                scope=scope_key,
                start_page=start_page,
                end_page=end_page,
                topic=topic_query,
                top_k=topic_top_k,
                embedding_service=_get_embedding_service() if scope_key == "topic" else None,
                provider=provider,
                progress_callback=_on_progress,
            )
            seq = int(st.session_state.get("question_set_seq", 0)) + 1
            st.session_state["question_set_seq"] = seq
            payload = result.to_dict()
            payload["question_set_id"] = f"qs_{seq}"
            cached_questions[selected_doc_id] = payload
            progress_bar.progress(1.0)
            progress_status.caption("Practice questions ready.")
        except (
            ValueError,
            KeyError,
            LLMConfigurationError,
            LLMProviderError,
            QuestionGenerationError,
            EmbeddingModelError,
            EmbeddingGenerationError,
            VectorStoreError,
        ) as exc:
            logger.exception("Practice question generation failed for document %s", selected_doc_id)
            cached_questions.pop(selected_doc_id, None)
            if isinstance(quiz_sessions, dict):
                quiz_sessions.pop(selected_doc_id, None)
            st.session_state["questions_error"] = f"Question generation failed: {exc}"
            st.session_state["questions_error_doc_id"] = selected_doc_id
            progress_bar.empty()
            progress_status.empty()

    if (
        st.session_state.get("questions_error")
        and st.session_state.get("questions_error_doc_id") == selected_doc_id
    ):
        st.error(st.session_state["questions_error"])
        return

    cached_questions = st.session_state.get("cached_questions", {})
    question_data = cached_questions.get(selected_doc_id)
    if (
        not question_data
        or question_data.get("document_id") != selected_doc_id
        or not store.is_document_indexed(selected_doc_id)
    ):
        return

    st.divider()
    st.markdown(f"### Practice Questions — {question_data['filename']}")
    st.caption(
        f"Document: **{question_data['filename']}** | "
        f"Pages Covered: **{question_data['page_range_label']}** | "
        f"Type: **{question_data['question_type'].replace('_', ' ').title()}** | "
        f"Difficulty: **{question_data['difficulty'].title()}** | "
        f"Scope: **{question_data['scope'].replace('_', ' ').title()}** | "
        f"Provider: `{question_data['provider']}` (`{question_data['model']}`)"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Generated / Requested",
        f"{question_data['generated_count']} / {question_data['requested_count']}",
    )
    m2.metric("Source Chunks", question_data["chunks_used"])
    m3.metric("LLM Batches", question_data["batches_processed"])

    if question_data["question_type"] == "mixed":
        dist = question_data.get("type_distribution", {})
        st.caption(
            "Actual type distribution: "
            f"**MCQ:** {dist.get('mcq', 0)} | "
            f"**Short Answer:** {dist.get('short_answer', 0)} | "
            f"**True / False:** {dist.get('true_false', 0)}"
        )

    if question_data.get("notice"):
        st.warning(question_data["notice"])

    type_display_names = {
        "mcq": "MCQ",
        "short_answer": "Short Answer",
        "true_false": "True / False",
    }
    option_letters = ("A", "B", "C", "D")

    active_qs = st.session_state.get("quiz_sessions", {}).get(selected_doc_id, {})
    default_mode_idx = 1 if active_qs.get("submitted") else 0
    view_mode = st.radio(
        "Practice Mode",
        options=["Study View", "Interactive Quiz"],
        index=default_mode_idx,
        horizontal=True,
        key=f"questions_view_mode_{selected_doc_id}",
    )

    if view_mode == "Study View":
        show_answers = st.checkbox(
            "Show Answers & Explanations",
            value=False,
            key=f"show_answers_{selected_doc_id}",
        )

        for idx, q in enumerate(question_data["questions"], start=1):
            badge = type_display_names.get(q["question_type"], q["question_type"])
            with st.container(border=True):
                st.markdown(
                    f"**Q{idx} ({badge} · {q['difficulty'].title()})** — {q['question']}"
                )
                st.caption(
                    f"Source: **{q['filename']}** | {q['page_range_label']}"
                )

                if q["question_type"] == "mcq" and q.get("options"):
                    for opt_letter, opt_text in zip(option_letters, q["options"]):
                        st.markdown(f"- **{opt_letter}.** {opt_text}")
                elif q["question_type"] == "true_false":
                    st.caption("Determine whether this statement is True or False.")
                else:
                    st.caption("Formulate a concise answer before revealing the expected response.")

                if show_answers:
                    st.divider()
                    if q["question_type"] == "short_answer":
                        st.markdown(f"**Expected Answer:** {q['correct_answer']}")
                    else:
                        st.markdown(f"**Correct Answer:** {q['correct_answer']}")
                    if q["question_type"] == "true_false" and q.get("correct_statement"):
                        st.markdown(f"**Correct Statement:** {q['correct_statement']}")
                    st.markdown(f"**Explanation:** {q['explanation']}")
                    st.caption(
                        f"Verified chunks: `{', '.join(q['source_chunk_ids'])}` ({q['page_range_label']})"
                    )
    else:
        _render_interactive_quiz_view(
            store=store,
            selected_doc_id=selected_doc_id,
            question_data=question_data,
            provider=provider,
            type_display_names=type_display_names,
            option_letters=option_letters,
        )

    with st.expander(
        f"Source Chunks Referenced ({len(question_data['source_chunks'])})",
        expanded=False,
    ):
        for idx, chunk in enumerate(question_data["source_chunks"], start=1):
            meta = chunk["metadata"]
            score_suffix = (
                f" | Similarity: {chunk['score']:.4f}"
                if chunk.get("score") is not None
                else ""
            )
            st.markdown(
                f"**{idx}. {meta['filename']} — Page {meta['page_range']}, "
                f"Chunk {meta['chunk_index']} (`{meta['chunk_id']}`){score_suffix}**"
            )
            st.write(chunk["text"])
            if idx < len(question_data["source_chunks"]):
                st.divider()


def _render_interactive_quiz_view(
    *,
    store: FAISSVectorStore,
    selected_doc_id: str,
    question_data: dict,
    provider,
    type_display_names: dict[str, str],
    option_letters: tuple[str, ...],
) -> None:
    """Render the stateful interactive quiz attempt, submission, and graded review."""
    qset_id = question_data.get("question_set_id") or "qs_default"
    quiz_session = _get_or_init_quiz_session(selected_doc_id, qset_id)
    attempt_num = int(quiz_session["attempt_number"])
    submitted = bool(quiz_session["submitted"])
    saved_answers: dict = quiz_session["answers"]

    questions = question_data["questions"]
    has_short_answer = any(q["question_type"] == "short_answer" for q in questions)

    default_eval_label = (
        "Deterministic"
        if (config.QUIZ_SHORT_ANSWER_EVALUATOR or "").strip().lower() == "deterministic"
        else "LLM"
    )
    eval_mode_label = default_eval_label
    if has_short_answer:
        eval_mode_label = st.selectbox(
            "Short-answer grading mode",
            options=["LLM", "Deterministic"],
            index=0 if default_eval_label == "LLM" else 1,
            disabled=submitted,
            key=f"quiz_eval_mode_{selected_doc_id}",
            help="LLM uses one batched call grounded in source chunks; Deterministic grades locally offline.",
        )

    # Render interactive question inputs (answers stay strictly hidden until Submit Answers)
    for idx, q in enumerate(questions, start=1):
        qid = q["question_id"]
        qtype = q["question_type"]
        badge = type_display_names.get(qtype, qtype)
        widget_key = f"quiz_ans_{selected_doc_id}_{qset_id}_a{attempt_num}_{qid}"

        with st.container(border=True):
            st.markdown(
                f"**Q{idx} ({badge} · {q['difficulty'].title()})** — {q['question']}"
            )
            st.caption(f"Source: **{q['filename']}** | {q['page_range_label']}")

            if qtype == "mcq" and q.get("options"):
                options = list(q["options"])
                if widget_key not in st.session_state and saved_answers.get(qid) in options:
                    st.session_state[widget_key] = saved_answers[qid]
                option_display = {
                    opt: f"{letter}. {opt}" for letter, opt in zip(option_letters, options)
                }
                choice = st.radio(
                    "Select one option",
                    options=options,
                    index=None,
                    format_func=lambda opt, mapping=option_display: mapping.get(opt, opt),
                    disabled=submitted,
                    key=widget_key,
                    label_visibility="collapsed",
                )
                saved_answers[qid] = choice
            elif qtype == "true_false":
                tf_options = ["True", "False"]
                if widget_key not in st.session_state and saved_answers.get(qid) in tf_options:
                    st.session_state[widget_key] = saved_answers[qid]
                choice = st.radio(
                    "Select True or False",
                    options=tf_options,
                    index=None,
                    horizontal=True,
                    disabled=submitted,
                    key=widget_key,
                    label_visibility="collapsed",
                )
                saved_answers[qid] = choice
            else:
                if widget_key not in st.session_state and isinstance(saved_answers.get(qid), str):
                    st.session_state[widget_key] = saved_answers[qid]
                typed = st.text_area(
                    "Your answer",
                    placeholder="Type your concise answer here...",
                    disabled=submitted,
                    key=widget_key,
                    height=88,
                )
                saved_answers[qid] = typed

    answered_count = sum(
        1
        for q in questions
        if isinstance(saved_answers.get(q["question_id"]), str)
        and saved_answers[q["question_id"]].strip()
    )
    st.caption(
        f"Attempt #{attempt_num} | Answered **{answered_count} / {len(questions)}** question(s). "
        "Answers and explanations remain hidden until you submit."
    )

    col_submit, col_retry = st.columns([1.2, 1])
    with col_submit:
        submit_clicked = st.button(
            "Submit Answers",
            type="primary",
            disabled=submitted,
            key=f"submit_quiz_btn_{selected_doc_id}",
            use_container_width=True,
        )
    with col_retry:
        retry_clicked = st.button(
            "Retry Quiz",
            type="secondary",
            key=f"retry_quiz_btn_{selected_doc_id}",
            use_container_width=True,
        )

    if retry_clicked:
        _clear_quiz_widget_keys(selected_doc_id)
        quiz_session["attempt_number"] = attempt_num + 1
        quiz_session["answers"] = {}
        quiz_session["submitted"] = False
        quiz_session["result"] = None
        quiz_session["error"] = None
        st.session_state[f"questions_view_mode_{selected_doc_id}"] = "Interactive Quiz"
        st.rerun()

    if submit_clicked and not submitted:
        quiz_session["error"] = None
        eval_mode_key = eval_mode_label.lower()
        try:
            with st.spinner("Evaluating your answers..."):
                eval_result = quiz_evaluator.evaluate_quiz(
                    question_data,
                    dict(saved_answers),
                    evaluator_mode=eval_mode_key,
                    provider=provider if eval_mode_key == "llm" else None,
                    store=store,
                )
            quiz_session["result"] = eval_result.to_dict()
            quiz_session["submitted"] = True
            st.rerun()
        except (
            ValueError,
            QuizEvaluationError,
            LLMConfigurationError,
            LLMProviderError,
            VectorStoreError,
        ) as exc:
            logger.exception("Quiz evaluation failed for document %s", selected_doc_id)
            # Preserve typed/selected answers intact so the user can retry submission!
            quiz_session["error"] = f"Quiz evaluation failed: {exc}"
            quiz_session["submitted"] = False
            quiz_session["result"] = None

    if quiz_session.get("error"):
        st.error(quiz_session["error"])
        st.info("Your answers have been preserved above. Fix the issue or switch grading mode and submit again.")
        return

    eval_data = quiz_session.get("result")
    if not submitted or not eval_data or eval_data.get("document_id") != selected_doc_id:
        return

    st.divider()
    st.markdown(f"### Quiz Results — {eval_data['filename']}")
    grader_label = eval_data["short_answer_evaluator"].upper()
    if eval_data.get("provider") and eval_data.get("model"):
        grader_detail = f"`{eval_data['provider']}` (`{eval_data['model']}`)"
    else:
        grader_detail = "`deterministic`"
    st.caption(
        f"Attempt #{attempt_num} | Short-Answer Grader: **{grader_label}** ({grader_detail})"
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Score", f"{eval_data['earned_points']} / {eval_data['total_points']}")
    r2.metric("Percentage", f"{eval_data['percentage']}%")
    r3.metric("Correct / Partial", f"{eval_data['correct_count']} / {eval_data['partial_count']}")
    r4.metric(
        "Incorrect / Unanswered",
        f"{eval_data['incorrect_count']} / {eval_data['unanswered_count']}",
    )

    breakdown = eval_data.get("type_breakdown", {})
    if breakdown:
        parts = []
        for qtype, br in breakdown.items():
            label = type_display_names.get(qtype, qtype)
            parts.append(
                f"**{label}:** {br['earned_points']}/{br['total_points']} ({br['percentage']}%)"
            )
        st.caption("Breakdown by question type: " + " | ".join(parts))

    status_badges = {
        "correct": "Correct (+1.0)",
        "partial": "Partially Correct (+0.5)",
        "incorrect": "Incorrect (0.0)",
        "unanswered": "Unanswered (0.0)",
    }

    st.markdown("#### Detailed Feedback")
    for idx, ev in enumerate(eval_data["evaluations"], start=1):
        badge = type_display_names.get(ev["question_type"], ev["question_type"])
        status_label = status_badges.get(ev["status"], ev["status"])
        with st.container(border=True):
            st.markdown(
                f"**Q{idx} ({badge}) — {status_label}**"
            )
            st.markdown(ev["question"])
            st.markdown(f"**Your Answer:** {ev['user_answer'] if ev['user_answer'] else '*(Unanswered)*'}")
            if ev["question_type"] == "short_answer":
                st.markdown(f"**Expected Answer:** {ev['correct_answer']}")
            else:
                st.markdown(f"**Correct Answer:** {ev['correct_answer']}")
            if ev["question_type"] == "true_false" and ev.get("correct_statement"):
                st.markdown(f"**Correct Statement:** {ev['correct_statement']}")
            st.markdown(f"**Feedback:** {ev['feedback']}")
            st.markdown(f"**Explanation:** {ev['explanation']}")
            st.caption(
                f"Verified source: **{ev['filename']}** — {ev['page_range_label']} "
                f"(Chunks: `{', '.join(ev['source_chunk_ids'])}` | Graded by: `{ev['evaluated_by']}`)"
            )

    missed = eval_data.get("missed_questions", [])
    if missed:
        with st.expander(
            f"Pages to Review ({len(missed)} missed or partially correct question(s))",
            expanded=True,
        ):
            for ev in missed:
                st.markdown(
                    f"- **{ev['question_id'].upper()}** ({ev['status'].title()}) — "
                    f"Review **{ev['filename']}** ({ev['page_range_label']}): {ev['question']}"
                )
    else:
        st.success("Perfect score! All questions answered correctly.")


def _render_groq_qa_tab(store: FAISSVectorStore) -> None:
    st.subheader("Quick Q&A (Groq)")
    st.caption(
        "Existing single-turn Q&A powered by the configured LLM provider. "
        "PDF indexing and vector retrieval operate independently of Groq."
    )

    if not config.GROQ_API_KEY:
        st.warning(
            "GROQ_API_KEY is not configured. Add it to your .env file to use Groq Q&A. "
            "PDF indexing and Vector Retrieval Testing continue working without it."
        )
        return

    if store.total_vectors == 0:
        st.info("Index at least one PDF first to ask a question.")
        return

    question = st.text_input(
        "Ask a question about your indexed PDFs",
        key="groq_qa_question",
    )

    if question:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                service = _get_embedding_service()
                source_chunks = retrieve_chunks(
                    question,
                    top_k=config.TOP_K,
                    store=store,
                    embedding_service=service,
                )
                answer = generate_answer(question, source_chunks)

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Source Chunks")
                for index, chunk in enumerate(source_chunks, start=1):
                    metadata = chunk["metadata"]
                    with st.expander(
                        f"Source {index} — {metadata['filename']} "
                        f"(Page {metadata['page_range']}, Chunk {metadata['chunk_index']})"
                    ):
                        st.write(chunk["text"])
            except Exception as error:
                logger.exception("Groq Q&A failed")
                st.error(f"Failed to answer question: {error}")


def main() -> None:
    st.title("PDF Chat - RAG Question Answering System")

    if st.session_state.get("last_action_notice"):
        st.success(st.session_state.pop("last_action_notice"))

    store, load_error, can_reset = _load_vector_store()
    if load_error is not None or store is None:
        _render_recovery_panel(
            load_error or "Unable to initialize the local vector store.",
            can_reset,
        )
        return

    _render_sidebar(store)
    _render_upload_and_indexing(store)
    st.divider()

    tab_retrieval, tab_summary, tab_questions, tab_qa = st.tabs(
        ["Vector Retrieval Testing", "PDF Summary", "Practice Questions", "Quick Q&A (Groq)"]
    )
    with tab_retrieval:
        _render_retrieval_testing_tab(store)
    with tab_summary:
        _render_summary_tab(store)
    with tab_questions:
        _render_practice_questions_tab(store)
    with tab_qa:
        _render_groq_qa_tab(store)


main()
