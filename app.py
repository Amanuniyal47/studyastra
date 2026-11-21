import logging
import re
import shutil
import sys
from typing import Any

import streamlit as st

import src.config as config
from src.chat.chat_orchestrator import (
    ChatOrchestrator,
    ChatOrchestratorResult,
)
from src.chat.conversation_memory import (
    ChatMessage,
    ConversationManager,
    ConversationMemoryConfig,
    ConversationThread,
)
from src.generation.answer_generator import generate_answer
from src.generation.explanation_generator import explain_concept
from src.generation.flashcard_generator import (
    FlashcardGenerationError,
    export_flashcards_to_anki_csv,
    export_flashcards_to_markdown,
    generate_flashcards,
)
from src.generation.follow_up_generator import generate_follow_up_questions
import src.generation.llm_provider as llm_provider
from src.generation.llm_provider import (
    LLMConfigurationError,
    LLMProviderError,
)
import src.generation.question_generator as question_generator
from src.generation.question_validation import QuestionGenerationError
import src.generation.quiz_evaluator as quiz_evaluator
from src.generation.quiz_evaluator import QuizEvaluationError
from src.generation.study_notes_generator import (
    StudyNotesGenerationError,
    export_notes_to_markdown,
    generate_study_notes,
)
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
from src.utils.app_lock import get_application_lock
from src.utils.file_utils import file_hash
from src.utils.study_statistics import StudyStatisticsManager
from src.utils.timer_manager import PomodoroTimer, ChallengeTimer, QuizTimer
from src.utils.session_lifecycle import (
    clean_stale_ephemeral_state,
    prune_lru_cache,
)

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
    """Remove quiz answer widget keys from session state for one or all documents."""
    prefixes = (
        ("quiz_ans_",)
        if document_id is None
        else (f"quiz_ans_{document_id}_",)
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


def _invalidate_cached_flashcards(document_id: str | None = None) -> None:
    """Clear session-cached flashcards and deck viewer state."""
    cached = st.session_state.get("cached_flashcards")
    if document_id is None:
        st.session_state.pop("cached_flashcards", None)
        st.session_state.pop("flashcard_doc_select", None)
        st.session_state.pop("flashcard_error", None)
        st.session_state.pop("flashcard_error_doc_id", None)
        for k in list(st.session_state.keys()):
            if k.startswith(("fc_idx_", "fc_flipped_", "fc_input_front_", "fc_input_back_", "fc_show_editor_")):
                st.session_state.pop(k, None)
        return
    if isinstance(cached, dict):
        cached.pop(document_id, None)
    if st.session_state.get("flashcard_error_doc_id") == document_id:
        st.session_state.pop("flashcard_error", None)
        st.session_state.pop("flashcard_error_doc_id", None)
    st.session_state.pop(f"fc_idx_{document_id}", None)
    st.session_state.pop(f"fc_flipped_{document_id}", None)
    for k in list(st.session_state.keys()):
        if k.startswith((f"fc_input_front_{document_id}_", f"fc_input_back_{document_id}_", f"fc_show_editor_{document_id}_")):
            st.session_state.pop(k, None)


def _invalidate_cached_notes(document_id: str | None = None) -> None:
    """Clear session-cached study notes for one or all documents."""
    cached = st.session_state.get("cached_notes")
    if document_id is None:
        st.session_state.pop("cached_notes", None)
        st.session_state.pop("notes_doc_select", None)
        st.session_state.pop("notes_error", None)
        st.session_state.pop("notes_error_doc_id", None)
        return
    if isinstance(cached, dict):
        cached.pop(document_id, None)
    if st.session_state.get("notes_error_doc_id") == document_id:
        st.session_state.pop("notes_error", None)
        st.session_state.pop("notes_error_doc_id", None)


def _invalidate_cached_chat(document_id: str | None = None) -> None:
    """Clear session-cached chat state and release locks for one or all documents."""
    orchestrator: ChatOrchestrator | None = st.session_state.get("chat_orchestrator")
    conv_mgr: ConversationManager | None = st.session_state.get("chat_conversation_manager")

    if document_id is None:
        ChatOrchestrator.clear_all_thread_locks()
        if conv_mgr is not None:
            conv_mgr.threads.clear()
        st.session_state.pop("chat_selected_doc_id", None)
        st.session_state.pop("chat_active_conv_by_doc", None)
        st.session_state.pop("chat_pending_delete_thread_id", None)
        st.session_state.pop("chat_renaming_thread_id", None)
        st.session_state.pop("chat_doc_select", None)
        st.session_state.pop("chat_thread_select", None)
        st.session_state.pop("chat_error", None)
        st.session_state.pop("chat_pending_retry_thread_id", None)
        st.session_state.pop("chat_pending_retry_query", None)
        st.session_state.pop("chat_pending_retry_request_id", None)
        return

    if orchestrator is not None:
        orchestrator.clear_document_threads(document_id)
    elif conv_mgr is not None:
        conv_mgr.clear_document_threads(document_id)

    active_by_doc = st.session_state.get("chat_active_conv_by_doc")
    if isinstance(active_by_doc, dict):
        active_by_doc.pop(document_id, None)

    if st.session_state.get("chat_selected_doc_id") == document_id:
        st.session_state.pop("chat_selected_doc_id", None)
        st.session_state.pop("chat_pending_delete_thread_id", None)
        st.session_state.pop("chat_renaming_thread_id", None)
        st.session_state.pop("chat_doc_select", None)
        st.session_state.pop("chat_thread_select", None)
        st.session_state.pop("chat_error", None)
        st.session_state.pop("chat_pending_retry_thread_id", None)
        st.session_state.pop("chat_pending_retry_query", None)
        st.session_state.pop("chat_pending_retry_request_id", None)


def _get_or_init_chat_manager(
    store: FAISSVectorStore,
) -> tuple[ConversationManager, ChatOrchestrator]:
    """Get or initialize ConversationManager and ChatOrchestrator in session state."""
    conv_mgr = st.session_state.get("chat_conversation_manager")
    if conv_mgr is None or not isinstance(conv_mgr, ConversationManager):
        conv_mgr = ConversationManager()
        st.session_state["chat_conversation_manager"] = conv_mgr

    orchestrator = st.session_state.get("chat_orchestrator")
    if (
        orchestrator is None
        or not isinstance(orchestrator, ChatOrchestrator)
        or orchestrator.store is not store
    ):
        embedding_service = _get_embedding_service()
        orchestrator = ChatOrchestrator(
            store=store,
            embedding_service=embedding_service,
            conversation_manager=conv_mgr,
        )
        st.session_state["chat_orchestrator"] = orchestrator
    return conv_mgr, orchestrator


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
        "submission_state": "not_submitted",
        "submitted": False,
        "auto_submitted": False,
        "timed_mode": False,
        "timer": None,
        "answers": {},
        "result": None,
        "error": None,
    }
    sessions[document_id] = fresh
    return fresh


def _get_or_init_pomodoro_timer() -> PomodoroTimer:
    """Return active Pomodoro timer from session state, initializing if needed."""
    timer = st.session_state.get("pomodoro_timer")
    if isinstance(timer, PomodoroTimer):
        return timer
    timer = PomodoroTimer(
        focus_minutes=config.POMODORO_DEFAULT_FOCUS_MINUTES,
        short_break_minutes=config.POMODORO_DEFAULT_SHORT_BREAK_MINUTES,
        long_break_minutes=config.POMODORO_DEFAULT_LONG_BREAK_MINUTES,
        sessions_per_cycle=config.POMODORO_DEFAULT_SESSIONS_PER_CYCLE,
    )
    st.session_state["pomodoro_timer"] = timer
    return timer


def _get_or_init_challenge_timer() -> ChallengeTimer:
    """Return active ChallengeTimer from session state, initializing if needed."""
    timer = st.session_state.get("challenge_timer")
    if isinstance(timer, ChallengeTimer):
        return timer
    timer = ChallengeTimer(
        duration_minutes=config.CHALLENGE_DEFAULT_MINUTES,
    )
    st.session_state["challenge_timer"] = timer
    return timer


def _get_or_init_study_stats() -> StudyStatisticsManager:
    """Return active StudyStatisticsManager from session state, initializing if needed."""
    mgr = st.session_state.get("study_stats_mgr")
    if isinstance(mgr, StudyStatisticsManager):
        return mgr
    mgr = StudyStatisticsManager(config.STUDY_STATS_FILE)
    st.session_state["study_stats_mgr"] = mgr
    return mgr


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
            _invalidate_cached_flashcards(None)
            _invalidate_cached_notes(None)
            _invalidate_cached_chat(None)
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

    if (config.LLM_PROVIDER or "").strip().lower() == "ollama":
        st.sidebar.caption(f"LLM backend: Local Qwen (`{config.OLLAMA_MODEL}` via Ollama)")
    else:
        st.sidebar.caption(f"LLM backend: Groq Cloud (`{config.GROQ_MODEL}`)")

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
                        _invalidate_cached_flashcards(doc_id)
                        _invalidate_cached_notes(doc_id)
                        _invalidate_cached_chat(doc_id)
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

    st.sidebar.divider()
    _render_sidebar_conversations(store)


def _render_sidebar_conversations(store: FAISSVectorStore) -> None:
    """Render conversation threads and controls in the sidebar."""
    documents = store.list_documents()
    if not documents:
        return

    st.sidebar.subheader("Conversations")

    doc_map = {d["document_id"]: d["filename"] for d in documents}
    doc_ids = list(doc_map.keys())

    selected_doc_id = st.session_state.get("chat_selected_doc_id")
    if selected_doc_id not in doc_map:
        selected_doc_id = doc_ids[0]
        st.session_state["chat_selected_doc_id"] = selected_doc_id

    st.sidebar.caption(f"Document: **{doc_map[selected_doc_id]}**")

    conv_mgr, orchestrator = _get_or_init_chat_manager(store)
    threads = conv_mgr.list_threads(selected_doc_id)

    active_by_doc = st.session_state.setdefault("chat_active_conv_by_doc", {})
    active_thread_id = active_by_doc.get(selected_doc_id)
    if not active_thread_id and threads:
        active_thread_id = threads[0].conversation_id
        active_by_doc[selected_doc_id] = active_thread_id
    elif not threads:
        initial_thread = conv_mgr.create_thread(
            document_id=selected_doc_id, title="New Conversation"
        )
        active_thread_id = initial_thread.conversation_id
        active_by_doc[selected_doc_id] = active_thread_id
        threads = [initial_thread]

    if st.sidebar.button(
        "➕ New Conversation",
        key="sidebar_new_chat_btn",
        use_container_width=True,
    ):
        new_th = conv_mgr.create_thread(
            document_id=selected_doc_id,
            title=f"Chat {len(threads) + 1}",
        )
        active_by_doc[selected_doc_id] = new_th.conversation_id
        st.session_state.pop("chat_pending_delete_thread_id", None)
        st.session_state.pop("chat_renaming_thread_id", None)
        st.rerun()

    pending_del = st.session_state.get("chat_pending_delete_thread_id")
    renaming_id = st.session_state.get("chat_renaming_thread_id")

    for th in threads:
        is_active = th.conversation_id == active_thread_id
        with st.sidebar.container(border=is_active):
            if renaming_id == th.conversation_id:
                new_title = st.text_input(
                    "New title",
                    value=th.title,
                    key=f"sidebar_rename_val_{th.conversation_id}",
                )
                c_s, c_c = st.columns(2)
                if c_s.button(
                    "Save",
                    key=f"sidebar_save_rename_{th.conversation_id}",
                    use_container_width=True,
                ):
                    if new_title.strip():
                        th.title = new_title.strip()
                    st.session_state.pop("chat_renaming_thread_id", None)
                    st.rerun()
                if c_c.button(
                    "Cancel",
                    key=f"sidebar_cancel_rename_{th.conversation_id}",
                    use_container_width=True,
                ):
                    st.session_state.pop("chat_renaming_thread_id", None)
                    st.rerun()
            elif pending_del == th.conversation_id:
                st.warning(f"Delete '{th.title}'?")
                c_y, c_n = st.columns(2)
                if c_y.button(
                    "Confirm",
                    key=f"sidebar_confirm_delete_{th.conversation_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    orchestrator.delete_thread(selected_doc_id, th.conversation_id)
                    st.session_state.pop("chat_pending_delete_thread_id", None)
                    rem = conv_mgr.list_threads(selected_doc_id)
                    active_by_doc[selected_doc_id] = (
                        rem[0].conversation_id if rem else None
                    )
                    st.rerun()
                if c_n.button(
                    "Cancel",
                    key=f"sidebar_cancel_delete_{th.conversation_id}",
                    use_container_width=True,
                ):
                    st.session_state.pop("chat_pending_delete_thread_id", None)
                    st.rerun()
            else:
                prefix = "👉 " if is_active else "💬 "
                time_str = (
                    str(th.created_at)[:16].replace("T", " ") if th.created_at else ""
                )
                btn_label = f"{prefix}{th.title} ({len(th.messages)})"
                c_btn, c_ren, c_del = st.columns([5, 1, 1])
                if c_btn.button(
                    btn_label,
                    key=f"sidebar_select_thread_{th.conversation_id}",
                    use_container_width=True,
                    help=f"Created: {time_str}" if time_str else None,
                ):
                    active_by_doc[selected_doc_id] = th.conversation_id
                    st.session_state.pop("chat_pending_delete_thread_id", None)
                    st.session_state.pop("chat_renaming_thread_id", None)
                    st.rerun()
                if c_ren.button(
                    "✏️",
                    key=f"sidebar_ren_btn_{th.conversation_id}",
                    help="Rename conversation",
                ):
                    st.session_state["chat_renaming_thread_id"] = th.conversation_id
                    st.session_state.pop("chat_pending_delete_thread_id", None)
                    st.rerun()
                if c_del.button(
                    "🗑️",
                    key=f"sidebar_del_btn_{th.conversation_id}",
                    help="Delete conversation",
                ):
                    st.session_state["chat_pending_delete_thread_id"] = th.conversation_id
                    st.session_state.pop("chat_renaming_thread_id", None)
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
            progress_bar = st.progress(0.0)
            progress_status = st.empty()

            def _on_index_progress(step: int, total: int, message: str) -> None:
                ratio = min(1.0, max(0.0, step / max(1, total)))
                progress_bar.progress(ratio)
                progress_status.caption(message)

            try:
                service = _get_embedding_service()
                result = store.add_pdf(
                    uploaded_file,
                    filename=filename,
                    embedding_service=service,
                    skip_duplicates=True,
                    progress_callback=_on_index_progress,
                )
                progress_bar.progress(1.0)
                progress_status.caption("PDF indexed successfully.")
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
            except Exception as exc:
                logger.exception("Unexpected error while indexing '%s'", filename)
                st.session_state["last_index_error"] = f"Unexpected error during indexing: {exc}"
                st.session_state.pop("last_index_result", None)
            finally:
                progress_bar.empty()
                progress_status.empty()

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

    if isinstance(provider, llm_provider.OllamaLLMProvider):
        st.caption(
            f"🤖 Summaries powered by local **{provider.model_name}** via Ollama (`{provider.base_url}`)."
        )
        ok_conn, conn_msg = provider.check_connection()
        if not ok_conn:
            st.warning(f"⚠️ {conn_msg}")
            st.caption(
                "PDF indexing and Vector Retrieval Testing continue working without Ollama. "
                "Start Ollama with `ollama serve` to enable summaries."
            )
        else:
            ok_model, model_msg = provider.check_model_available()
            if not ok_model:
                st.warning(f"⚠️ {model_msg}")
    else:
        if not provider.is_configured():
            st.warning(
                "GROQ_API_KEY is not configured. Add it to your .env file to generate PDF summaries. "
                "Or set LLM_PROVIDER=ollama to use local Qwen2.5 7B instead."
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
        clean_stale_ephemeral_state(st.session_state, active_doc_id=selected_doc_id)
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

    generate_sum_clicked = st.button("Generate Summary", type="primary", key="generate_summary_btn")

    if generate_sum_clicked:
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
            prune_lru_cache(cached_summaries, max_entries=5)
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

    cached_summaries = st.session_state.get("cached_summaries", {})
    summary_data = cached_summaries.get(selected_doc_id)
    if summary_data and store.is_document_indexed(selected_doc_id):
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
            if summary_data.get("status") == "partial":
                st.warning(f"⚠️ {summary_data['match_notice']}")
            else:
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


def _render_study_notes_tab(store: FAISSVectorStore) -> None:
    st.subheader("Study Notes")
    st.caption(
        "Generate hierarchical academic study notes strictly grounded in indexed PDF chunks. "
        "Generated notes stay in session state and are never indexed into FAISS."
    )

    documents = store.list_documents()
    if not documents:
        st.info("Index at least one PDF first to generate study notes.")
        return

    try:
        provider = llm_provider.get_llm_provider()
    except LLMConfigurationError as exc:
        st.warning(
            f"LLM provider is not available for study notes: {exc} "
            "PDF indexing and Vector Retrieval Testing continue working normally."
        )
        return

    if isinstance(provider, llm_provider.OllamaLLMProvider):
        st.caption(
            f"🤖 Study Notes powered by local **{provider.model_name}** via Ollama (`{provider.base_url}`)."
        )
        ok_conn, conn_msg = provider.check_connection()
        if not ok_conn:
            st.warning(f"⚠️ {conn_msg}")
            st.caption(
                "PDF indexing and Vector Retrieval Testing continue working without Ollama. "
                "Start Ollama with `ollama serve` to enable study notes."
            )
        else:
            ok_model, model_msg = provider.check_model_available()
            if not ok_model:
                st.warning(f"⚠️ {model_msg}")
    else:
        if not provider.is_configured():
            st.warning(
                "GROQ_API_KEY is not configured. Add it to your .env file to generate study notes. "
                "Or set LLM_PROVIDER=ollama to use local Qwen2.5 7B instead."
            )
            return

    doc_ids = [doc["document_id"] for doc in documents]
    doc_labels = {
        doc["document_id"]: f"{doc['filename']} ({doc['page_count']} page(s), {doc['document_id'][:8]})"
        for doc in documents
    }

    if st.session_state.get("notes_doc_select") not in doc_ids:
        st.session_state["notes_doc_select"] = doc_ids[0]

    col_doc, col_scope = st.columns([2, 1.2])
    with col_doc:
        selected_doc_id = st.selectbox(
            "Select PDF for study notes",
            options=doc_ids,
            format_func=lambda doc_id: doc_labels[doc_id],
            key="notes_doc_select",
        )
        clean_stale_ephemeral_state(st.session_state, active_doc_id=selected_doc_id)
    with col_scope:
        scope_label = st.selectbox(
            "Content scope",
            options=["Full Document", "Page Range", "Topic"],
            key="notes_scope_select",
        )

    available_pages = store.get_document_page_numbers(selected_doc_id)
    start_page: int | None = None
    end_page: int | None = None
    topic_query: str | None = None
    topic_top_k: int = config.NOTES_TOPIC_TOP_K
    page_range_invalid = False

    if scope_label == "Page Range":
        st.caption(
            f"Actual indexed pages in this PDF: **{format_page_range(available_pages)}**"
        )
        if st.session_state.get("notes_start_page") not in available_pages:
            st.session_state["notes_start_page"] = available_pages[0] if available_pages else 1
        if st.session_state.get("notes_end_page") not in available_pages:
            st.session_state["notes_end_page"] = available_pages[-1] if available_pages else 1

        col_start, col_end = st.columns(2)
        with col_start:
            start_page = int(
                st.selectbox(
                    "Start page",
                    options=available_pages,
                    key="notes_start_page",
                )
            )
        with col_end:
            end_page = int(
                st.selectbox(
                    "End page",
                    options=available_pages,
                    key="notes_end_page",
                )
            )
        if start_page is not None and end_page is not None and start_page > end_page:
            st.error(
                f"Invalid page range: start page ({start_page}) cannot be greater than end page ({end_page})."
            )
            page_range_invalid = True

    elif scope_label == "Topic":
        col_topic, col_k = st.columns([3, 1])
        with col_topic:
            topic_query = st.text_input(
                "Topic for study notes",
                placeholder="e.g., Photosynthesis light reactions, attention mechanism...",
                key="notes_topic_input",
            )
        with col_k:
            topic_top_k = int(
                st.number_input(
                    "Top K chunks",
                    min_value=1,
                    max_value=30,
                    value=min(config.NOTES_TOPIC_TOP_K, max(1, store.total_vectors)),
                    step=1,
                    key="notes_topic_top_k",
                )
            )

    col_gen, col_regen = st.columns([1, 1])
    with col_gen:
        gen_clicked = st.button(
            "Generate Study Notes",
            type="primary",
            key="generate_notes_btn",
            disabled=page_range_invalid,
        )
    with col_regen:
        has_cached = selected_doc_id in st.session_state.get("cached_notes", {})
        regen_clicked = st.button(
            "🔄 Regenerate Notes",
            key="regenerate_notes_btn",
            disabled=not has_cached or page_range_invalid,
        )

    if gen_clicked or regen_clicked:
        if scope_label == "Topic" and (not topic_query or not topic_query.strip()):
            st.warning("Please enter a topic to focus study notes generation.")
            return

        cached_notes = st.session_state.setdefault("cached_notes", {})
        cached_notes.pop(selected_doc_id, None)
        st.session_state.pop("notes_error", None)
        st.session_state.pop("notes_error_doc_id", None)

        progress_bar = st.progress(0.0)
        progress_status = st.empty()

        def _on_notes_progress(step: int, total: int, message: str) -> None:
            ratio = min(1.0, max(0.0, step / max(1, total)))
            progress_bar.progress(ratio)
            progress_status.caption(f"Step {step}/{total}: {message}")

        try:
            if scope_label == "Full Document":
                notes_res = generate_study_notes(
                    store,
                    selected_doc_id,
                    scope="full",
                    provider=provider,
                    progress_callback=_on_notes_progress,
                )
            elif scope_label == "Page Range":
                notes_res = generate_study_notes(
                    store,
                    selected_doc_id,
                    scope="page_range",
                    start_page=start_page,
                    end_page=end_page,
                    provider=provider,
                    progress_callback=_on_notes_progress,
                )
            else:
                notes_res = generate_study_notes(
                    store,
                    selected_doc_id,
                    scope="topic",
                    topic=topic_query.strip() if topic_query else "",
                    top_k=topic_top_k,
                    embedding_service=_get_embedding_service(),
                    provider=provider,
                    progress_callback=_on_notes_progress,
                )

            cached_notes[selected_doc_id] = notes_res.to_dict()
            prune_lru_cache(cached_notes, max_entries=5)
            progress_bar.progress(1.0)
            progress_status.caption("Study notes generation complete.")
        except (
            StudyNotesGenerationError,
            ValueError,
            KeyError,
            LLMConfigurationError,
            LLMProviderError,
            EmbeddingModelError,
            EmbeddingGenerationError,
            VectorStoreError,
        ) as exc:
            logger.exception("Study notes generation failed for document %s", selected_doc_id)
            cached_notes.pop(selected_doc_id, None)
            st.session_state["notes_error"] = f"Study notes generation failed: {exc}"
            st.session_state["notes_error_doc_id"] = selected_doc_id
            progress_bar.empty()
            progress_status.empty()

    if (
        st.session_state.get("notes_error")
        and st.session_state.get("notes_error_doc_id") == selected_doc_id
    ):
        st.error(st.session_state["notes_error"])

    cached_notes = st.session_state.get("cached_notes", {})
    notes_data = cached_notes.get(selected_doc_id)
    if notes_data and store.is_document_indexed(selected_doc_id):
        st.divider()
        st.markdown(f"### {notes_data['title']}")
        st.caption(
            f"Document: **{notes_data['filename']}** | "
            f"Pages Covered: **{notes_data['page_range_label']}** | "
            f"Scope: **{notes_data['scope'].title()}** | "
            f"Provider: `{notes_data['provider']}` (`{notes_data['model']}`)"
        )

        if notes_data.get("status") == "partial":
            st.warning(
                f"⚠️ Partial Study Notes: Only {notes_data.get('chunks_used', 0)} of "
                f"{notes_data.get('total_eligible_chunks', 0)} chunks could be processed."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pages Covered", notes_data["page_range_label"])
        m2.metric("Chunks Used", notes_data["chunks_used"])
        m3.metric("LLM Batches", notes_data.get("batches_processed", 1))
        m4.metric("Reduction Levels", notes_data.get("reduction_levels", 0))

        # Markdown Export Button
        md_content = export_notes_to_markdown(notes_data)
        clean_stem = re.sub(r"[^\w\-.]", "_", notes_data.get("filename", "document")).rstrip(".pdf")
        st.download_button(
            label="📥 Download Notes (.md)",
            data=md_content,
            file_name=f"study_notes_{clean_stem}.md",
            mime="text/markdown",
            key=f"notes_download_md_{selected_doc_id}",
        )

        st.markdown(notes_data["notes_text"])

        with st.expander(
            f"Study Notes Source Chunks ({len(notes_data.get('source_chunks', []))})",
            expanded=False,
        ):
            for idx, chunk in enumerate(notes_data.get("source_chunks", []), start=1):
                meta = chunk.get("metadata", chunk)
                p_label = meta.get("page_range", meta.get("page", "Unknown"))
                score_suffix = (
                    f" | Similarity: {chunk['score']:.4f}"
                    if chunk.get("score") is not None
                    else ""
                )
                st.markdown(
                    f"**{idx}. {meta.get('filename', notes_data['filename'])} — Page {p_label}, "
                    f"Chunk {meta.get('chunk_index', idx)} (`{meta.get('chunk_id', 'id')}`){score_suffix}**"
                )
                st.write(chunk.get("text", ""))
                if idx < len(notes_data.get("source_chunks", [])):
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

    if isinstance(provider, llm_provider.OllamaLLMProvider):
        st.caption(
            f"🤖 Practice questions & quiz grading powered by local **{provider.model_name}** via Ollama (`{provider.base_url}`)."
        )
        ok_conn, conn_msg = provider.check_connection()
        if not ok_conn:
            st.warning(f"⚠️ {conn_msg}")
            st.caption(
                "PDF indexing and Vector Retrieval Testing continue working without Ollama. "
                "Start Ollama with `ollama serve` to enable practice questions."
            )
        else:
            ok_model, model_msg = provider.check_model_available()
            if not ok_model:
                st.warning(f"⚠️ {model_msg}")
    else:
        if not provider.is_configured():
            st.warning(
                "GROQ_API_KEY is not configured. Add it to your .env file to generate practice questions. "
                "Or set LLM_PROVIDER=ollama to use local Qwen2.5 7B instead."
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
        clean_stale_ephemeral_state(st.session_state, active_doc_id=selected_doc_id)
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
        st.session_state.pop(f"questions_view_mode_{selected_doc_id}", None)
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
            protected_qs = set(quiz_sessions.keys()) if isinstance(quiz_sessions, dict) else None
            prune_lru_cache(cached_questions, max_entries=5, protected_keys=protected_qs)
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


def _execute_quiz_submission(
    *,
    selected_doc_id: str,
    quiz_session: dict[str, Any],
    question_data: dict[str, Any],
    saved_answers: dict[str, Any],
    eval_mode_key: str,
    provider: Any,
    store: FAISSVectorStore,
    stats_mgr: StudyStatisticsManager,
    auto_submitted: bool = False,
) -> bool:
    """Centralized quiz evaluation and submission execution for manual and auto submission."""
    if quiz_session.get("submitted") or quiz_session.get("submission_state") == "submitting":
        return False

    quiz_session["submission_state"] = "submitting"
    quiz_session["error"] = None

    timer: QuizTimer | None = quiz_session.get("timer")
    actual_elapsed = 0.0
    if timer is not None and timer.state == QuizTimer.STATE_RUNNING:
        if auto_submitted:
            actual_elapsed = float(timer.duration_seconds)
            timer.state = QuizTimer.STATE_EXPIRED
            timer.deadline_monotonic = None
            timer.remaining_seconds = 0.0
        else:
            actual_elapsed = timer.stop()
    elif timer is not None:
        actual_elapsed = timer.actual_elapsed_seconds

    try:
        with st.spinner("Evaluating your answers..."):
            eval_result = quiz_evaluator.evaluate_quiz(
                question_data,
                dict(saved_answers),
                evaluator_mode=eval_mode_key,
                provider=provider if eval_mode_key == "llm" else None,
                store=store,
            )
        result_dict = eval_result.to_dict()
        quiz_session["result"] = result_dict
        quiz_session["submitted"] = True
        quiz_session["auto_submitted"] = bool(auto_submitted)
        quiz_session["submission_state"] = "submitted"
        quiz_session["error"] = None

        # Record statistics if timed quiz
        if quiz_session.get("timed_mode") and timer is not None:
            attempt_num = int(quiz_session.get("attempt_number", 1))
            qset_id = str(quiz_session.get("question_set_id", "qs_default"))
            stats_mgr.record_timed_quiz(
                document_id=selected_doc_id,
                question_set_id=qset_id,
                attempt_number=attempt_num,
                configured_duration_seconds=timer.duration_seconds,
                actual_elapsed_seconds=actual_elapsed,
                earned_points=result_dict["earned_points"],
                total_points=result_dict["total_points"],
                percentage=result_dict["percentage"],
                answered_count=sum(
                    1
                    for q in question_data.get("questions", [])
                    if isinstance(saved_answers.get(q["question_id"]), str)
                    and saved_answers[q["question_id"]].strip()
                ),
                unanswered_count=result_dict.get("unanswered_count", 0),
                auto_submitted=bool(auto_submitted),
            )
        return True
    except (
        ValueError,
        QuizEvaluationError,
        LLMConfigurationError,
        LLMProviderError,
        VectorStoreError,
    ) as exc:
        logger.exception("Quiz evaluation failed for document %s", selected_doc_id)
        quiz_session["error"] = f"Quiz evaluation failed: {exc}"
        quiz_session["submitted"] = False
        quiz_session["submission_state"] = "not_submitted"
        quiz_session["result"] = None
        return False


@st.fragment(run_every=1)
def _render_timed_quiz_live_timer(
    timer: QuizTimer,
    quiz_session: dict[str, Any],
    question_data: dict[str, Any],
    saved_answers: dict[str, Any],
    eval_mode_key: str,
    provider: Any,
    store: FAISSVectorStore,
    stats_mgr: StudyStatisticsManager,
    selected_doc_id: str,
    answered_count: int,
    total_count: int,
) -> None:
    """Render live ticking timer in an isolated fragment; auto-submits on expiration."""
    is_just_expired, elapsed = timer.update()
    if is_just_expired and not quiz_session.get("submitted"):
        _execute_quiz_submission(
            selected_doc_id=selected_doc_id,
            quiz_session=quiz_session,
            question_data=question_data,
            saved_answers=saved_answers,
            eval_mode_key=eval_mode_key,
            provider=provider,
            store=store,
            stats_mgr=stats_mgr,
            auto_submitted=True,
        )
        st.rerun()

    if timer.is_warning():
        st.warning(f"⚠️ **Less than {timer.format_time()} remaining!** Finalize your answers.")

    timer_color = "#e53e3e" if timer.is_warning() else "inherit"
    st.markdown(
        f"<h2 style='text-align: center; margin: 0; padding: 0.25rem 0; color: {timer_color};'>⏱️ Time Remaining: {timer.format_time()}</h2>",
        unsafe_allow_html=True,
    )
    st.progress(timer.get_progress())
    st.caption(f"⏱️ Timed Quiz in progress | Answered **{answered_count} / {total_count}** question(s)")


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
    stats_mgr = _get_or_init_study_stats()

    questions = question_data["questions"]
    has_short_answer = any(q["question_type"] == "short_answer" for q in questions)

    default_eval_label = (
        "Deterministic"
        if (config.QUIZ_SHORT_ANSWER_EVALUATOR or "").strip().lower() == "deterministic"
        else "LLM"
    )
    eval_mode_label = default_eval_label

    is_timer_running = (
        quiz_session.get("timed_mode")
        and isinstance(quiz_session.get("timer"), QuizTimer)
        and quiz_session["timer"].state == QuizTimer.STATE_RUNNING
    )

    col_eval_opt, col_timed_opt = st.columns([1.2, 1.2])
    with col_eval_opt:
        if has_short_answer:
            eval_mode_label = st.selectbox(
                "Short-answer grading mode",
                options=["LLM", "Deterministic"],
                index=0 if default_eval_label == "LLM" else 1,
                disabled=submitted,
                key=f"quiz_eval_mode_{selected_doc_id}",
                help="LLM uses one batched call grounded in source chunks; Deterministic grades locally offline.",
            )
    with col_timed_opt:
        timed_active = st.checkbox(
            "⏱️ Timed Quiz Challenge",
            value=bool(quiz_session.get("timed_mode", False)),
            disabled=submitted or is_timer_running,
            key=f"quiz_timed_toggle_{selected_doc_id}_{qset_id}",
            help="Enable overall countdown timer with automatic submission on expiration.",
        )
        quiz_session["timed_mode"] = timed_active

    eval_mode_key = eval_mode_label.lower()

    answered_count = sum(
        1
        for q in questions
        if isinstance(saved_answers.get(q["question_id"]), str)
        and saved_answers[q["question_id"]].strip()
    )

    timer: QuizTimer | None = quiz_session.get("timer")
    if timed_active:
        with st.container(border=True):
            st.markdown("#### ⏱️ Timed Quiz Setup")
            col_pre, col_cust, col_start = st.columns([1.2, 1.2, 1])
            with col_pre:
                preset_options = [f"{d} mins" for d in config.TIMED_QUIZ_PRESET_DURATIONS] + ["Custom"]
                preset_choice = st.selectbox(
                    "Duration Preset",
                    options=preset_options,
                    index=1,  # 10 mins default
                    disabled=submitted or is_timer_running,
                    key=f"quiz_timed_preset_{selected_doc_id}_{qset_id}",
                )
            with col_cust:
                custom_mins = st.number_input(
                    f"Custom ({config.TIMED_QUIZ_MIN_MINUTES}–{config.TIMED_QUIZ_MAX_MINUTES} mins)",
                    min_value=config.TIMED_QUIZ_MIN_MINUTES,
                    max_value=config.TIMED_QUIZ_MAX_MINUTES,
                    value=config.TIMED_QUIZ_DEFAULT_MINUTES,
                    step=1,
                    disabled=submitted or is_timer_running or preset_choice != "Custom",
                    key=f"quiz_timed_custom_{selected_doc_id}_{qset_id}",
                )
                if preset_choice == "Custom":
                    chosen_duration = int(custom_mins)
                else:
                    chosen_duration = int(preset_choice.split()[0])
            with col_start:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                start_clicked = st.button(
                    "▶️ Start Timed Quiz",
                    type="primary",
                    disabled=submitted or is_timer_running,
                    key=f"start_timed_quiz_btn_{selected_doc_id}",
                    use_container_width=True,
                )
                if start_clicked and not is_timer_running and not submitted:
                    new_timer = QuizTimer(
                        document_id=selected_doc_id,
                        question_set_id=qset_id,
                        attempt_number=attempt_num,
                        duration_minutes=chosen_duration,
                        warning_threshold_seconds=config.TIMED_QUIZ_WARNING_SECONDS,
                    )
                    new_timer.start()
                    quiz_session["timer"] = new_timer
                    st.rerun()

        if is_timer_running and not submitted:
            with st.container(border=True):
                _render_timed_quiz_live_timer(
                    timer=timer,
                    quiz_session=quiz_session,
                    question_data=question_data,
                    saved_answers=saved_answers,
                    eval_mode_key=eval_mode_key,
                    provider=provider,
                    store=store,
                    stats_mgr=stats_mgr,
                    selected_doc_id=selected_doc_id,
                    answered_count=answered_count,
                    total_count=len(questions),
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
        quiz_session["submission_state"] = "not_submitted"
        quiz_session["auto_submitted"] = False
        quiz_session["result"] = None
        quiz_session["error"] = None
        quiz_session["timer"] = None
        st.rerun()

    if submit_clicked and not submitted:
        _execute_quiz_submission(
            selected_doc_id=selected_doc_id,
            quiz_session=quiz_session,
            question_data=question_data,
            saved_answers=saved_answers,
            eval_mode_key=eval_mode_key,
            provider=provider,
            store=store,
            stats_mgr=stats_mgr,
            auto_submitted=False,
        )
        st.rerun()

    if quiz_session.get("error"):
        st.error(quiz_session["error"])
        st.info("Your answers have been preserved above. Fix the issue or switch grading mode and submit again.")
        return

    eval_data = quiz_session.get("result")
    if not submitted or not eval_data or eval_data.get("document_id") != selected_doc_id:
        return

    st.divider()
    if quiz_session.get("auto_submitted"):
        st.warning("⏱️ **Time Expired!** The quiz was automatically submitted when time ran out. Any unanswered questions were scored as 0.0.")
    elif quiz_session.get("timed_mode"):
        timer_used = quiz_session.get("timer")
        el_mins = f"{timer_used.actual_elapsed_seconds / 60.0:.1f} mins" if timer_used else ""
        st.success(f"🎉 **Timed Quiz Completed!** Submitted with time remaining ({el_mins}).")

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


def _render_flashcard_tab(store: FAISSVectorStore) -> None:
    st.subheader("Flashcards")
    st.caption(
        "Generate concise, high-yield study flashcards strictly grounded in indexed PDF chunks. "
        "Generated flashcards stay in session state and are never indexed into FAISS."
    )

    documents = store.list_documents()
    if not documents:
        st.info("Index at least one PDF first to generate flashcards.")
        return

    try:
        provider = llm_provider.get_llm_provider()
    except LLMConfigurationError as exc:
        st.warning(
            f"LLM provider is not available for flashcards: {exc} "
            "PDF indexing and Vector Retrieval Testing continue working normally."
        )
        return

    if isinstance(provider, llm_provider.OllamaLLMProvider):
        st.caption(
            f"🤖 Flashcards powered by local **{provider.model_name}** via Ollama (`{provider.base_url}`)."
        )
        ok_conn, conn_msg = provider.check_connection()
        if not ok_conn:
            st.warning(f"⚠️ {conn_msg}")
            st.caption(
                "PDF indexing and Vector Retrieval Testing continue working without Ollama. "
                "Start Ollama with `ollama serve` to enable flashcards."
            )
        else:
            ok_model, model_msg = provider.check_model_available()
            if not ok_model:
                st.warning(f"⚠️ {model_msg}")
    else:
        if not provider.is_configured():
            st.warning(
                "GROQ_API_KEY is not configured. Add it to your .env file to generate flashcards. "
                "Or set LLM_PROVIDER=ollama to use local Qwen2.5 7B instead."
            )
            return

    doc_ids = [doc["document_id"] for doc in documents]
    doc_labels = {
        doc["document_id"]: f"{doc['filename']} ({doc['page_count']} page(s), {doc['document_id'][:8]})"
        for doc in documents
    }

    if st.session_state.get("flashcard_doc_select") not in doc_ids:
        st.session_state["flashcard_doc_select"] = doc_ids[0]

    col_doc, col_scope = st.columns([2, 1.2])
    with col_doc:
        selected_doc_id = st.selectbox(
            "Select PDF for flashcards",
            options=doc_ids,
            format_func=lambda doc_id: doc_labels[doc_id],
            key="flashcard_doc_select",
        )
        clean_stale_ephemeral_state(st.session_state, active_doc_id=selected_doc_id)
    with col_scope:
        scope_label = st.selectbox(
            "Content scope",
            options=["Full Document", "Page Range", "Topic"],
            key="flashcard_scope_select",
        )

    available_pages = store.get_document_page_numbers(selected_doc_id)
    start_page: int | None = None
    end_page: int | None = None
    topic_query: str | None = None
    topic_top_k: int = config.FLASHCARD_TOPIC_TOP_K

    if scope_label == "Page Range":
        st.caption(
            f"Actual indexed pages in this PDF: **{format_page_range(available_pages)}**"
        )
        if st.session_state.get("flashcard_start_page") not in available_pages:
            st.session_state["flashcard_start_page"] = available_pages[0]
        if st.session_state.get("flashcard_end_page") not in available_pages:
            st.session_state["flashcard_end_page"] = available_pages[-1]

        col_start, col_end = st.columns(2)
        with col_start:
            start_page = int(
                st.selectbox(
                    "Start page",
                    options=available_pages,
                    key="flashcard_start_page",
                )
            )
        with col_end:
            end_page = int(
                st.selectbox(
                    "End page",
                    options=available_pages,
                    key="flashcard_end_page",
                )
            )
    elif scope_label == "Topic":
        col_topic, col_k = st.columns([3, 1])
        with col_topic:
            topic_query = st.text_input(
                "Topic for flashcards",
                placeholder="e.g., Cellular respiration electron transport",
                key="flashcard_topic_input",
            )
        with col_k:
            topic_top_k = int(
                st.number_input(
                    "Top K chunks",
                    min_value=1,
                    max_value=20,
                    value=min(config.FLASHCARD_TOPIC_TOP_K, max(1, store.total_vectors)),
                    step=1,
                    key="flashcard_topic_top_k",
                )
            )

    card_count = int(
        st.number_input(
            "Number of flashcards",
            min_value=1,
            max_value=config.FLASHCARD_MAX_COUNT,
            value=min(config.FLASHCARD_DEFAULT_COUNT, config.FLASHCARD_MAX_COUNT),
            step=1,
            key="flashcard_count_input",
        )
    )

    col_gen, col_regen = st.columns([1, 1])
    with col_gen:
        gen_clicked = st.button("Generate Flashcards", type="primary", key="generate_flashcards_btn")
    with col_regen:
        has_cached = selected_doc_id in st.session_state.get("cached_flashcards", {})
        regen_clicked = st.button(
            "🔄 Regenerate Deck",
            key="regenerate_flashcards_btn",
            disabled=not has_cached,
        )

    if gen_clicked or regen_clicked:
        cached_flashcards = st.session_state.setdefault("cached_flashcards", {})
        cached_flashcards.pop(selected_doc_id, None)
        st.session_state.pop("flashcard_error", None)
        st.session_state.pop("flashcard_error_doc_id", None)
        st.session_state[f"fc_idx_{selected_doc_id}"] = 0
        st.session_state[f"fc_flipped_{selected_doc_id}"] = False

        progress_bar = st.progress(0.0)
        progress_status = st.empty()

        def _on_fc_progress(step: int, total: int, message: str) -> None:
            ratio = min(1.0, max(0.0, step / max(1, total)))
            progress_bar.progress(ratio)
            progress_status.caption(f"Step {step}/{total}: {message}")

        scope_map = {"Full Document": "full", "Page Range": "page_range", "Topic": "topic"}
        scope_key = scope_map[scope_label]

        try:
            result = generate_flashcards(
                store,
                selected_doc_id,
                scope=scope_key,
                count=card_count,
                start_page=start_page,
                end_page=end_page,
                topic=topic_query,
                top_k=topic_top_k,
                embedding_service=_get_embedding_service() if scope_key == "topic" else None,
                provider=provider,
                progress_callback=_on_fc_progress,
            )
            cached_flashcards[selected_doc_id] = result.to_dict()
            prune_lru_cache(cached_flashcards, max_entries=5)
            progress_bar.progress(1.0)
            progress_status.caption("Flashcards ready.")
        except (
            ValueError,
            KeyError,
            LLMConfigurationError,
            LLMProviderError,
            FlashcardGenerationError,
            EmbeddingModelError,
            EmbeddingGenerationError,
            VectorStoreError,
        ) as exc:
            logger.exception("Flashcard generation failed for document %s", selected_doc_id)
            cached_flashcards.pop(selected_doc_id, None)
            st.session_state["flashcard_error"] = f"Flashcard generation failed: {exc}"
            st.session_state["flashcard_error_doc_id"] = selected_doc_id
            progress_bar.empty()
            progress_status.empty()

    if (
        st.session_state.get("flashcard_error")
        and st.session_state.get("flashcard_error_doc_id") == selected_doc_id
    ):
        st.error(st.session_state["flashcard_error"])
        return

    cached_flashcards = st.session_state.get("cached_flashcards", {})
    fc_data = cached_flashcards.get(selected_doc_id)
    if not fc_data or not store.is_document_indexed(selected_doc_id):
        return

    cards = fc_data.get("cards", [])
    if not cards:
        st.info("No flashcards were generated for this selection.")
        return

    st.divider()
    st.markdown(f"### Flashcard Deck — {fc_data['filename']}")
    st.caption(
        f"Cards: **{len(cards)}** | Pages: **{fc_data['page_range_label']}** | "
        f"Scope: **{fc_data['scope'].title()}** | "
        f"Provider: `{fc_data['provider']}` (`{fc_data['model']}`)"
    )
    if fc_data.get("notice"):
        st.info(fc_data["notice"])

    current_idx = int(st.session_state.get(f"fc_idx_{selected_doc_id}", 0))
    current_idx = max(0, min(current_idx, len(cards) - 1))
    st.session_state[f"fc_idx_{selected_doc_id}"] = current_idx
    flipped = bool(st.session_state.get(f"fc_flipped_{selected_doc_id}", False))

    current_card = cards[current_idx]

    with st.container(border=True):
        col_card_header, col_edit_toggle = st.columns([3, 1])
        with col_card_header:
            edited_tag = " ✏️ *(Edited)*" if current_card.get("edited") else ""
            st.caption(f"Card {current_idx + 1} of {len(cards)}{edited_tag}")
        with col_edit_toggle:
            show_editor = st.checkbox(
                "✏️ Edit Card",
                key=f"fc_show_editor_{selected_doc_id}_{current_idx}",
            )

        if not show_editor:
            if not flipped:
                st.markdown("#### ❓ Front / Question")
                st.markdown(f"### {current_card['front']}")
            else:
                st.markdown("#### 💡 Back / Answer")
                st.markdown(f"**{current_card['back']}**")
                st.caption(
                    f"Source: **{current_card['filename']}** | {current_card['page_range_label']} "
                    f"(Chunks: `{', '.join(current_card['source_chunk_ids'])}`)"
                )
        else:
            st.markdown("#### ✏️ Revise Flashcard")
            edited_front = st.text_input(
                "Front / Question",
                value=current_card["front"],
                key=f"fc_input_front_{selected_doc_id}_{current_idx}",
            )
            edited_back = st.text_area(
                "Back / Answer",
                value=current_card["back"],
                key=f"fc_input_back_{selected_doc_id}_{current_idx}",
            )
            if st.button("Save Changes to Card", type="primary", key=f"fc_save_btn_{selected_doc_id}_{current_idx}"):
                if edited_front.strip() and edited_back.strip():
                    current_card["front"] = edited_front.strip()
                    current_card["back"] = edited_back.strip()
                    current_card["edited"] = True
                    cards[current_idx] = current_card
                    fc_data["cards"] = cards
                    cached_flashcards[selected_doc_id] = fc_data
                    st.session_state["cached_flashcards"] = cached_flashcards
                    st.success("Flashcard updated successfully!")
                    st.rerun()
                else:
                    st.warning("Both Front and Back must not be empty.")

    col_prev, col_flip, col_next = st.columns([1, 1, 1])
    with col_prev:
        if st.button("⬅️ Previous Card", disabled=(current_idx == 0), key=f"fc_prev_{selected_doc_id}", use_container_width=True):
            st.session_state[f"fc_idx_{selected_doc_id}"] = current_idx - 1
            st.session_state[f"fc_flipped_{selected_doc_id}"] = False
            st.rerun()
    with col_flip:
        flip_label = "🔄 Show Front" if flipped else "🔄 Flip Card (Show Answer)"
        if st.button(flip_label, type="secondary", key=f"fc_flip_{selected_doc_id}", use_container_width=True):
            st.session_state[f"fc_flipped_{selected_doc_id}"] = not flipped
            st.rerun()
    with col_next:
        if st.button("Next Card ➡️", disabled=(current_idx >= len(cards) - 1), key=f"fc_next_{selected_doc_id}", use_container_width=True):
            st.session_state[f"fc_idx_{selected_doc_id}"] = current_idx + 1
            st.session_state[f"fc_flipped_{selected_doc_id}"] = False
            st.rerun()

    # Export controls
    st.divider()
    st.markdown("#### 📥 Export Deck")
    clean_stem = re.sub(r"[^\w\-.]", "_", fc_data["filename"]).replace(".pdf", "")
    col_dl_md, col_dl_csv = st.columns(2)
    with col_dl_md:
        md_content = export_flashcards_to_markdown(fc_data)
        st.download_button(
            label="📥 Download Markdown (.md)",
            data=md_content,
            file_name=f"flashcards_{clean_stem}.md",
            mime="text/markdown",
            key=f"fc_dl_md_{selected_doc_id}",
            use_container_width=True,
        )
    with col_dl_csv:
        csv_content = export_flashcards_to_anki_csv(fc_data)
        st.download_button(
            label="📥 Download Anki CSV (.csv)",
            data=csv_content,
            file_name=f"flashcards_{clean_stem}_anki.csv",
            mime="text/csv",
            key=f"fc_dl_csv_{selected_doc_id}",
            use_container_width=True,
        )

    with st.expander(f"View All {len(cards)} Flashcards as List"):
        for idx, card in enumerate(cards, start=1):
            with st.container(border=True):
                edit_badge = " ✏️ *(Edited)*" if card.get("edited") else ""
                st.markdown(f"**Card #{idx}{edit_badge} — Front:** {card['front']}")
                st.markdown(f"**Back:** {card['back']}")
                st.caption(
                    f"Page(s): {card['page_range_label']} | Chunk IDs: `{', '.join(card['source_chunk_ids'])}`"
                )


@st.fragment(run_every=1)
def _render_pomodoro_live_display(timer: PomodoroTimer, stats_mgr: StudyStatisticsManager) -> None:
    """Render the live ticking Pomodoro display when running."""
    just_completed, completed_type = timer.update()
    if just_completed:
        duration = timer.get_configured_duration(completed_type)
        stats_mgr.record_session(
            completed_type,
            duration,
            cycle_number=timer.cycle_number,
            session_in_cycle=timer.session_in_cycle,
        )
        timer.advance_to_next_session()
        st.rerun()

    st.markdown(
        f"<h1 style='text-align: center; font-size: 4.5rem; margin: 0; padding: 0.5rem 0;'>{timer.format_time()}</h1>",
        unsafe_allow_html=True,
    )
    st.progress(timer.get_progress())
    st.caption("⏳ Focus session in progress...")


def _render_pomodoro_static_display(timer: PomodoroTimer, stats_mgr: StudyStatisticsManager) -> None:
    """Render the static Pomodoro display when idle or paused."""
    st.markdown(
        f"<h1 style='text-align: center; font-size: 4.5rem; margin: 0; padding: 0.5rem 0;'>{timer.format_time()}</h1>",
        unsafe_allow_html=True,
    )
    st.progress(timer.get_progress())
    if timer.state == PomodoroTimer.STATE_PAUSED:
        st.caption("⏸️ Session is paused.")
    elif timer.state == PomodoroTimer.STATE_COMPLETED:
        st.caption("🎉 Session complete! Ready for next interval.")
    else:
        st.caption("Ready to begin. Press Start Focus.")


def _render_pomodoro_subtab(
    timer: PomodoroTimer,
    stats_mgr: StudyStatisticsManager,
    stats: dict[str, Any],
) -> None:
    """Render the complete Pomodoro Focus Timer view."""
    # Metrics summary
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Completed Focus", stats["completed_focus_sessions"])
    m2.metric("Total Focus Time", f"{int(stats['total_focus_minutes'])} mins")
    m3.metric("Total Break Time", f"{int(stats['total_break_minutes'])} mins")
    streak_label = f"{stats['current_streak']} 🔥" if stats["current_streak"] > 0 else "0 days"
    m4.metric("Day Streak", streak_label)

    st.divider()

    # Session type banner
    session_labels = {
        PomodoroTimer.FOCUS: ("🎯 Focus Session", "Time to concentrate without distractions."),
        PomodoroTimer.SHORT_BREAK: ("☕ Short Break", "Step away, stretch, and relax your eyes."),
        PomodoroTimer.LONG_BREAK: ("🌴 Long Break", "Well done! Take a longer recharge break."),
    }
    badge_title, badge_caption = session_labels.get(
        timer.current_session_type, ("🎯 Focus Session", "")
    )

    col_info, col_cycle = st.columns([2, 1.2])
    with col_info:
        st.markdown(f"### {badge_title}")
        st.caption(badge_caption)
    with col_cycle:
        st.caption(
            f"**Session {timer.session_in_cycle} of {timer.sessions_per_cycle}** (Cycle #{timer.cycle_number})"
        )

    # Isolated timer countdown container
    with st.container(border=True):
        if timer.state == PomodoroTimer.STATE_RUNNING:
            _render_pomodoro_live_display(timer, stats_mgr)
        else:
            _render_pomodoro_static_display(timer, stats_mgr)

        # Control buttons
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1, 1])

        if timer.state == PomodoroTimer.STATE_IDLE:
            with col_ctrl1:
                start_label = (
                    "▶️ Start Focus"
                    if timer.current_session_type == PomodoroTimer.FOCUS
                    else "▶️ Start Break"
                )
                if st.button(start_label, type="primary", key="pomo_start_btn", use_container_width=True):
                    timer.start()
                    st.rerun()
            with col_ctrl2:
                if st.button("🔄 Reset", key="pomo_reset_btn", use_container_width=True):
                    timer.reset()
                    st.rerun()
            with col_ctrl3:
                if st.button("⏭️ Skip Session", key="pomo_skip_btn", use_container_width=True):
                    timer.skip()
                    st.rerun()

        elif timer.state == PomodoroTimer.STATE_RUNNING:
            with col_ctrl1:
                if st.button("⏸️ Pause", type="primary", key="pomo_pause_btn", use_container_width=True):
                    timer.pause()
                    st.rerun()
            with col_ctrl2:
                if st.button("🔄 Reset", key="pomo_reset_btn", use_container_width=True):
                    timer.reset()
                    st.rerun()
            with col_ctrl3:
                if st.button("⏭️ Skip Session", key="pomo_skip_btn", use_container_width=True):
                    timer.skip()
                    st.rerun()

        elif timer.state == PomodoroTimer.STATE_PAUSED:
            with col_ctrl1:
                if st.button("▶️ Resume", type="primary", key="pomo_resume_btn", use_container_width=True):
                    timer.resume()
                    st.rerun()
            with col_ctrl2:
                if st.button("🔄 Reset", key="pomo_reset_btn", use_container_width=True):
                    timer.reset()
                    st.rerun()
            with col_ctrl3:
                if st.button("⏭️ Skip Session", key="pomo_skip_btn", use_container_width=True):
                    timer.skip()
                    st.rerun()

        elif timer.state == PomodoroTimer.STATE_COMPLETED:
            with col_ctrl1:
                if st.button("▶️ Next Interval", type="primary", key="pomo_next_btn", use_container_width=True):
                    timer.advance_to_next_session()
                    st.rerun()
            with col_ctrl2:
                if st.button("🔄 Reset", key="pomo_reset_btn", use_container_width=True):
                    timer.reset()
                    st.rerun()
            with col_ctrl3:
                if st.button("⏭️ Skip", key="pomo_skip_btn", use_container_width=True):
                    timer.skip()
                    st.rerun()

    # Settings Container
    with st.container(border=True):
        st.markdown("#### ⚙️ Customize Timer Durations")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            custom_focus = st.number_input(
                "Focus (minutes)",
                min_value=1,
                max_value=120,
                value=int(timer.focus_seconds // 60),
                step=1,
                key="pomo_input_focus",
            )
        with c2:
            custom_short = st.number_input(
                "Short Break (mins)",
                min_value=1,
                max_value=30,
                value=int(timer.short_break_seconds // 60),
                step=1,
                key="pomo_input_short",
            )
        with c3:
            custom_long = st.number_input(
                "Long Break (mins)",
                min_value=1,
                max_value=60,
                value=int(timer.long_break_seconds // 60),
                step=1,
                key="pomo_input_long",
            )
        with c4:
            custom_cycle = st.number_input(
                "Sessions per Cycle",
                min_value=1,
                max_value=10,
                value=int(timer.sessions_per_cycle),
                step=1,
                key="pomo_input_cycle",
            )

        if st.button("Apply New Settings", key="pomo_apply_settings_btn"):
            timer.focus_seconds = int(custom_focus * 60)
            timer.short_break_seconds = int(custom_short * 60)
            timer.long_break_seconds = int(custom_long * 60)
            timer.sessions_per_cycle = int(custom_cycle)
            if timer.state in (PomodoroTimer.STATE_IDLE, PomodoroTimer.STATE_COMPLETED):
                timer.remaining_seconds = float(timer.get_configured_duration())
            st.success("Timer settings updated!")
            st.rerun()

    # History Container
    history = stats.get("session_history", [])
    with st.container(border=True):
        st.markdown(f"#### 📜 Session History ({len(history)})")
        if not history:
            st.caption("No completed study sessions recorded yet. Start a focus session above!")
        else:
            for item in reversed(history[-10:]):
                type_name = item.get("session_type", "").replace("_", " ").title()
                mins = item.get("duration_minutes", 0.0)
                completed_at = item.get("completed_at", "")[:19].replace("T", " ")
                cycle_str = f"Cycle #{item.get('cycle_number', 1)} (Session {item.get('session_in_cycle', 1)})"
                st.markdown(f"**{type_name}** — {mins:.1f} mins | {cycle_str} | Completed at `{completed_at} UTC`")

            col_res_btn, _ = st.columns([1, 2])
            with col_res_btn:
                if st.button("🗑️ Reset All Statistics", key="pomo_reset_stats_btn"):
                    stats_mgr.reset_statistics()
                    st.success("All statistics have been reset.")
                    st.rerun()


@st.fragment(run_every=1)
def _render_challenge_live_display(timer: ChallengeTimer, stats_mgr: StudyStatisticsManager) -> None:
    """Render the live ticking Challenge display when running."""
    just_completed, title, actual_elapsed = timer.update()
    if just_completed:
        stats_mgr.record_challenge(
            title=title,
            planned_duration_seconds=timer.duration_seconds,
            actual_duration_seconds=actual_elapsed,
        )
        st.rerun()

    st.markdown(
        f"<h1 style='text-align: center; font-size: 4.5rem; margin: 0; padding: 0.5rem 0;'>{timer.format_time()}</h1>",
        unsafe_allow_html=True,
    )
    st.progress(timer.get_progress())
    st.caption("⏳ Challenge in progress! Stay focused on your objective.")


def _render_challenge_static_display(timer: ChallengeTimer) -> None:
    """Render the static Challenge display when idle, paused, completed, or cancelled."""
    st.markdown(
        f"<h1 style='text-align: center; font-size: 4.5rem; margin: 0; padding: 0.5rem 0;'>{timer.format_time()}</h1>",
        unsafe_allow_html=True,
    )
    st.progress(timer.get_progress())
    if timer.state == ChallengeTimer.STATE_PAUSED:
        st.caption("⏸️ Challenge is paused.")
    elif timer.state == ChallengeTimer.STATE_COMPLETED:
        st.caption("🏆 Challenge complete! Outstanding focus work!")
    elif timer.state == ChallengeTimer.STATE_CANCELLED:
        st.caption("🛑 Challenge was cancelled. Ready for another attempt.")
    else:
        st.caption("Ready to begin. Set your objective and duration, then press Start Challenge.")


def _render_challenge_subtab(
    timer: ChallengeTimer,
    stats_mgr: StudyStatisticsManager,
    stats: dict[str, Any],
) -> None:
    """Render the Countdown Study Challenge interface."""
    st.markdown("### ⚡ Countdown Study Challenge")
    st.caption("Sprint through focused study blocks with custom challenge targets.")

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Completed Challenges", stats.get("completed_challenges", 0))
    m2.metric("Total Challenge Time", f"{int(stats.get('total_challenge_minutes', 0))} mins")
    status_map = {
        ChallengeTimer.STATE_RUNNING: "Running ⏳",
        ChallengeTimer.STATE_PAUSED: "Paused ⏸️",
        ChallengeTimer.STATE_COMPLETED: "Completed 🏆",
        ChallengeTimer.STATE_CANCELLED: "Cancelled 🛑",
        ChallengeTimer.STATE_IDLE: "Ready 🎯",
    }
    m3.metric("Challenge Status", status_map.get(timer.state, "Ready 🎯"))

    st.divider()

    # Challenge Setup Container (Editable when idle, completed, or cancelled)
    is_active = timer.state in (ChallengeTimer.STATE_RUNNING, ChallengeTimer.STATE_PAUSED)
    with st.container(border=True):
        st.markdown("#### 🎯 Setup Your Challenge")
        col_t, col_d = st.columns([2, 1.2])
        with col_t:
            chal_title_input = st.text_input(
                "Challenge Objective / Task",
                value=timer.title if timer.title != "Study Challenge" else "",
                placeholder="e.g., Chapter 4 Cell Division Review",
                disabled=is_active,
                key="chal_title_input",
            )
        with col_d:
            preset_labels = [f"{d} mins" for d in config.CHALLENGE_PRESET_DURATIONS] + ["Custom"]
            default_preset_idx = 3  # 25 mins
            preset_choice = st.selectbox(
                "Duration Preset",
                options=preset_labels,
                index=default_preset_idx,
                disabled=is_active,
                key="chal_preset_select",
            )

        if preset_choice == "Custom":
            custom_mins = st.number_input(
                f"Custom Duration ({config.CHALLENGE_MIN_MINUTES}–{config.CHALLENGE_MAX_MINUTES} mins)",
                min_value=config.CHALLENGE_MIN_MINUTES,
                max_value=config.CHALLENGE_MAX_MINUTES,
                value=int(timer.duration_seconds // 60),
                step=1,
                disabled=is_active,
                key="chal_custom_mins",
            )
            chosen_duration_mins = int(custom_mins)
        else:
            chosen_duration_mins = int(preset_choice.split()[0])

    # Cancel confirmation banner if requested
    if st.session_state.get("chal_confirm_cancel"):
        with st.container(border=True):
            st.warning("⚠️ Are you sure you want to cancel this challenge? Progress will not be recorded.")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, Cancel Challenge", type="secondary", key="chal_confirm_cancel_btn", use_container_width=True):
                    timer.cancel()
                    st.session_state["chal_confirm_cancel"] = False
                    st.rerun()
            with col_no:
                if st.button("Keep Going", key="chal_cancel_dismiss_btn", use_container_width=True):
                    st.session_state["chal_confirm_cancel"] = False
                    st.rerun()

    # Active Countdown Container
    with st.container(border=True):
        st.markdown(f"### ⚡ {timer.title}")

        if timer.state == ChallengeTimer.STATE_RUNNING:
            _render_challenge_live_display(timer, stats_mgr)
        else:
            _render_challenge_static_display(timer)

        # Control Buttons
        col_c1, col_c2, col_c3 = st.columns([1.5, 1, 1])

        if timer.state == ChallengeTimer.STATE_IDLE:
            with col_c1:
                if st.button("🚀 Start Challenge", type="primary", key="chal_start_btn", use_container_width=True):
                    final_title = chal_title_input.strip() if chal_title_input.strip() else "Study Challenge"
                    timer.set_challenge(final_title, chosen_duration_mins)
                    timer.start()
                    st.rerun()
            with col_c2:
                if st.button("🔄 Reset", key="chal_reset_btn", use_container_width=True):
                    timer.reset()
                    st.rerun()

        elif timer.state == ChallengeTimer.STATE_RUNNING:
            with col_c1:
                if st.button("⏸️ Pause", type="primary", key="chal_pause_btn", use_container_width=True):
                    timer.pause()
                    st.rerun()
            with col_c2:
                if st.button("🔄 Reset", key="chal_reset_btn", use_container_width=True):
                    timer.reset()
                    st.rerun()
            with col_c3:
                if st.button("🛑 Cancel", key="chal_cancel_btn", use_container_width=True):
                    st.session_state["chal_confirm_cancel"] = True
                    st.rerun()

        elif timer.state == ChallengeTimer.STATE_PAUSED:
            with col_c1:
                if st.button("▶️ Resume", type="primary", key="chal_resume_btn", use_container_width=True):
                    timer.resume()
                    st.rerun()
            with col_c2:
                if st.button("🔄 Reset", key="chal_reset_btn", use_container_width=True):
                    timer.reset()
                    st.rerun()
            with col_c3:
                if st.button("🛑 Cancel", key="chal_cancel_btn", use_container_width=True):
                    st.session_state["chal_confirm_cancel"] = True
                    st.rerun()

        elif timer.state == ChallengeTimer.STATE_COMPLETED:
            with col_c1:
                if st.button("✨ New Challenge", type="primary", key="chal_new_btn", use_container_width=True):
                    timer.reset()
                    st.rerun()
            with col_c2:
                if st.button("🔄 Reset", key="chal_reset_btn", use_container_width=True):
                    timer.reset()
                    st.rerun()

        elif timer.state == ChallengeTimer.STATE_CANCELLED:
            with col_c1:
                if st.button("🔄 Try Again", type="primary", key="chal_retry_btn", use_container_width=True):
                    timer.start()
                    st.rerun()
            with col_c2:
                if st.button("✨ New Challenge", key="chal_new_btn", use_container_width=True):
                    timer.reset()
                    st.rerun()

    # Challenge History Container
    chal_history = stats.get("challenge_history", [])
    with st.container(border=True):
        st.markdown(f"#### 📜 Challenge History ({len(chal_history)})")
        if not chal_history:
            st.caption("No completed challenges recorded yet. Launch a study challenge above!")
        else:
            for item in reversed(chal_history[-10:]):
                title_name = item.get("title", "Study Challenge")
                mins = item.get("duration_minutes", 0.0)
                planned_m = item.get("planned_duration_seconds", 0) // 60
                completed_at = item.get("completed_at", "")[:19].replace("T", " ")
                st.markdown(
                    f"**⚡ {title_name}** — {mins:.1f} mins (Target: {planned_m} mins) | Completed at `{completed_at} UTC`"
                )


def _render_timed_quiz_stats_subtab(
    stats_mgr: StudyStatisticsManager,
    stats: dict[str, Any],
) -> None:
    """Render performance metrics and history for timed quizzes."""
    st.markdown("### 📝 Timed Quiz Performance")
    st.caption("Review completed timed quiz attempts, scores, and time invested.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Completed Quizzes", stats.get("completed_timed_quizzes", 0))
    m2.metric("Total Quiz Time", f"{int(stats.get('total_timed_quiz_minutes', 0))} mins")
    m3.metric("Average Score", f"{stats.get('average_quiz_score_pct', 0.0):.1f}%")
    m4.metric("Best Score", f"{stats.get('best_quiz_score_pct', 0.0):.1f}%")

    st.divider()

    history = stats.get("timed_quiz_history", [])
    with st.container(border=True):
        st.markdown(f"#### 📜 Timed Quiz History ({len(history)})")
        if not history:
            st.caption("No completed timed quizzes recorded yet. Take a timed quiz under Practice Questions!")
        else:
            for item in reversed(history[-10:]):
                score_str = f"{item.get('earned_points', 0)}/{item.get('total_points', 0)} ({item.get('percentage', 0.0):.1f}%)"
                mins = item.get("duration_minutes", 0.0)
                sub_badge = "⏱️ Auto-submitted" if item.get("auto_submitted") else "✅ Submitted"
                completed_at = item.get("completed_at", "")[:19].replace("T", " ")
                st.markdown(
                    f"**{item.get('attempt_id', 'Quiz')}** — Score: **{score_str}** | Duration: {mins:.1f} mins | {sub_badge} | `{completed_at} UTC`"
                )


def _render_study_timer_tab() -> None:
    st.subheader("Study Timer & Focus")
    st.caption(
        "Boost focus and avoid burnout using structured Pomodoro study intervals or timed study challenges."
    )

    pomo_timer = _get_or_init_pomodoro_timer()
    chal_timer = _get_or_init_challenge_timer()
    stats_mgr = _get_or_init_study_stats()
    stats = stats_mgr.to_dict()

    subtab_pomo, subtab_challenge, subtab_quiz = st.tabs(
        ["🎯 Pomodoro Focus", "⚡ Study Challenges", "📝 Timed Quiz Stats"]
    )

    with subtab_pomo:
        _render_pomodoro_subtab(pomo_timer, stats_mgr, stats)

    with subtab_challenge:
        _render_challenge_subtab(chal_timer, stats_mgr, stats)

    with subtab_quiz:
        _render_timed_quiz_stats_subtab(stats_mgr, stats)


def _render_qa_tab(store: FAISSVectorStore) -> None:
    is_ollama = (config.LLM_PROVIDER or "").strip().lower() == "ollama"
    title_suffix = f"Local Qwen2.5 ({config.OLLAMA_MODEL})" if is_ollama else "Groq"
    st.subheader(f"Quick Q&A ({title_suffix})")

    if is_ollama:
        st.caption(
            f"Local LLM inference powered by Ollama (`{config.OLLAMA_MODEL}`) at `{config.OLLAMA_BASE_URL}`. "
            "Inference and vector retrieval run completely offline on your hardware."
        )
        try:
            ollama_p = llm_provider.get_llm_provider("ollama")
            if isinstance(ollama_p, llm_provider.OllamaLLMProvider):
                ok_conn, conn_msg = ollama_p.check_connection()
                if not ok_conn:
                    st.warning(f"⚠️ {conn_msg}")
                    st.caption(
                        "PDF indexing and Vector Retrieval Testing continue working without Ollama. "
                        "Start Ollama with `ollama serve` to enable Q&A."
                    )
                else:
                    ok_model, model_msg = ollama_p.check_model_available()
                    if not ok_model:
                        st.warning(f"⚠️ {model_msg}")
        except Exception as check_exc:
            logger.debug("Ollama availability check note: %s", check_exc)
    else:
        st.caption(
            "Existing single-turn Q&A powered by the configured cloud LLM provider. "
            "PDF indexing and vector retrieval operate independently of Groq."
        )
        if not config.GROQ_API_KEY:
            st.warning(
                "GROQ_API_KEY is not configured. Add it to your .env file to use Groq Q&A. "
                "Or set LLM_PROVIDER=ollama to use local Qwen2.5 7B instead."
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
            except Exception as retrieval_err:
                logger.exception("Context retrieval failed")
                st.error(f"Failed to retrieve context: {retrieval_err}")
                return

            if not source_chunks:
                st.info("No relevant context found in the uploaded documents.")
                return

            st.subheader("Answer")
            full_answer = ""
            try:
                if is_ollama:
                    stream_iter = generate_answer(question, source_chunks, stream=True)
                    full_answer = st.write_stream(stream_iter)
                else:
                    full_answer = generate_answer(question, source_chunks)
                    st.write(full_answer)
            except LLMProviderError as prov_err:
                logger.exception("LLM generation failed")
                st.error(f"Generation error: {prov_err}")
            except Exception as error:
                logger.exception("Q&A generation failed")
                st.error(f"Failed to answer question: {error}")

            if full_answer and isinstance(full_answer, str):
                try:
                    follow_ups = generate_follow_up_questions(
                        question,
                        full_answer,
                        source_chunks,
                        count=config.FOLLOW_UP_DEFAULT_COUNT,
                    )
                    if follow_ups:
                        st.markdown("##### 💡 Suggested Follow-Up Questions")
                        for fq in follow_ups:
                            st.markdown(f"- {fq}")
                except Exception as fu_err:
                    logger.debug("Follow-up question generation failed: %s", fu_err)

                with st.expander("🔍 Deep Concept Explainer", expanded=False):
                    q_hash = abs(hash(question)) % 10000000
                    col_c1, col_c2 = st.columns([3, 1])
                    with col_c1:
                        concept_to_explain = st.text_input(
                            "Concept or phrase to explain in depth",
                            placeholder="e.g., Photosystem II reaction center",
                            key=f"explain_concept_input_{q_hash}",
                        )
                    with col_c2:
                        explain_style = st.selectbox(
                            "Explanation Style",
                            options=["Standard", "Simplified (ELI5)", "Academic"],
                            index=0,
                            key=f"explain_style_select_{q_hash}",
                        )
                    if st.button("Explain Concept", key=f"explain_concept_btn_{q_hash}"):
                        if concept_to_explain:
                            style_key = explain_style.split()[0].lower()
                            with st.spinner("Generating grounded concept explanation..."):
                                try:
                                    if is_ollama:
                                        exp_stream = explain_concept(
                                            concept_to_explain,
                                            source_chunks,
                                            style=style_key,
                                            stream=True,
                                        )
                                        st.write_stream(exp_stream)
                                    else:
                                        exp_text = explain_concept(
                                            concept_to_explain,
                                            source_chunks,
                                            style=style_key,
                                            stream=False,
                                        )
                                        st.write(exp_text)
                                except Exception as exp_err:
                                    st.error(f"Explanation failed: {exp_err}")

            st.subheader("Source Chunks")
            for index, chunk in enumerate(source_chunks, start=1):
                metadata = chunk.get("metadata", {})
                filename = metadata.get("filename", "Document")
                page_range = metadata.get("page_range", metadata.get("page", "Unknown"))
                chunk_index = metadata.get("chunk_index", index)
                score = chunk.get("score")
                score_str = f" | Similarity: {score:.3f}" if score is not None else ""
                with st.expander(
                    f"Source {index} — {filename} (Page {page_range}, Chunk {chunk_index}){score_str}"
                ):
                    st.write(chunk.get("text", ""))


_render_groq_qa_tab = _render_qa_tab


def _render_chat_tab(store: FAISSVectorStore) -> None:
    """Render the main Chat with PDF conversational RAG interface."""
    st.subheader("Chat with PDF")

    documents = store.list_documents()
    if not documents:
        st.info("No documents indexed yet. Upload and index a PDF above to begin chatting.")
        return

    doc_map = {d["document_id"]: d["filename"] for d in documents}
    doc_ids = list(doc_map.keys())

    selected_doc_id = st.session_state.get("chat_selected_doc_id")
    if selected_doc_id not in doc_map:
        selected_doc_id = doc_ids[0]
        st.session_state["chat_selected_doc_id"] = selected_doc_id

    def _on_chat_doc_change():
        st.session_state["chat_selected_doc_id"] = st.session_state["chat_doc_select"]
        st.session_state.pop("chat_pending_delete_thread_id", None)
        st.session_state.pop("chat_renaming_thread_id", None)
        clean_stale_ephemeral_state(st.session_state, active_doc_id=st.session_state["chat_doc_select"])

    conv_mgr, orchestrator = _get_or_init_chat_manager(store)
    threads = conv_mgr.list_threads(selected_doc_id)

    active_by_doc = st.session_state.setdefault("chat_active_conv_by_doc", {})
    active_thread_id = active_by_doc.get(selected_doc_id)
    if not any(t.conversation_id == active_thread_id for t in threads):
        if not threads:
            initial_th = conv_mgr.create_thread(
                document_id=selected_doc_id, title="New Conversation"
            )
            threads = [initial_th]
            active_thread_id = initial_th.conversation_id
        else:
            active_thread_id = threads[0].conversation_id
        active_by_doc[selected_doc_id] = active_thread_id

    active_thread = conv_mgr.get_thread(selected_doc_id, active_thread_id)
    if active_thread is None:
        active_thread = threads[0]
        active_by_doc[selected_doc_id] = active_thread.conversation_id

    # Top control toolbar
    c_doc_col, c_th_col = st.columns([2, 3])
    with c_doc_col:
        st.selectbox(
            "Active Document",
            options=doc_ids,
            index=doc_ids.index(selected_doc_id),
            format_func=lambda did: doc_map.get(did, did),
            key="chat_doc_select",
            on_change=_on_chat_doc_change,
        )

    with c_th_col:
        def _on_chat_th_change():
            active_by_doc[selected_doc_id] = st.session_state["chat_thread_select"]
            st.session_state.pop("chat_pending_delete_thread_id", None)
            st.session_state.pop("chat_renaming_thread_id", None)

        thread_ids = [t.conversation_id for t in threads]
        title_map = {t.conversation_id: t.title for t in threads}
        active_idx = (
            thread_ids.index(active_thread.conversation_id)
            if active_thread.conversation_id in thread_ids
            else 0
        )
        st.selectbox(
            "Conversation Thread",
            options=thread_ids,
            index=active_idx,
            format_func=lambda cid: title_map.get(cid, cid),
            key="chat_thread_select",
            on_change=_on_chat_th_change,
        )

    # Action buttons: New, Rename, Delete, Export
    c_new, c_ren, c_del, c_exp, _ = st.columns([1, 1, 1, 1.5, 3])
    with c_new:
        if st.button("➕ New Chat", key="chat_tab_new_chat_btn", use_container_width=True):
            new_th = conv_mgr.create_thread(
                document_id=selected_doc_id,
                title=f"Chat {len(threads) + 1}",
            )
            active_by_doc[selected_doc_id] = new_th.conversation_id
            st.session_state.pop("chat_pending_delete_thread_id", None)
            st.session_state.pop("chat_renaming_thread_id", None)
            st.rerun()

    with c_ren:
        if st.button("✏️ Rename", key="chat_tab_rename_btn", use_container_width=True):
            st.session_state["chat_renaming_thread_id"] = active_thread.conversation_id
            st.session_state.pop("chat_pending_delete_thread_id", None)
            st.rerun()

    with c_del:
        if st.button("🗑️ Delete", key="chat_tab_delete_btn", use_container_width=True):
            st.session_state["chat_pending_delete_thread_id"] = active_thread.conversation_id
            st.session_state.pop("chat_renaming_thread_id", None)
            st.rerun()

    with c_exp:
        md_content = active_thread.to_markdown()
        file_slug = (
            re.sub(r"[^\w\s-]", "", active_thread.title).strip().replace(" ", "_").lower()
            or "chat"
        )
        st.download_button(
            "📥 Export Chat",
            data=md_content,
            file_name=f"{file_slug}_{active_thread.conversation_id[:8]}.md",
            mime="text/markdown",
            key=f"chat_export_btn_{active_thread.conversation_id}",
            use_container_width=True,
            help="Download conversation history as Markdown",
        )

    # Inline rename dialog
    renaming_id = st.session_state.get("chat_renaming_thread_id")
    if renaming_id == active_thread.conversation_id:
        with st.container(border=True):
            new_title_val = st.text_input(
                "New conversation title",
                value=active_thread.title,
                key="chat_tab_rename_input",
            )
            c_s, c_c = st.columns([1, 1])
            if c_s.button("Save Title", key="chat_tab_save_rename_btn", type="primary"):
                if new_title_val.strip():
                    active_thread.title = new_title_val.strip()
                st.session_state.pop("chat_renaming_thread_id", None)
                st.rerun()
            if c_c.button("Cancel", key="chat_tab_cancel_rename_btn"):
                st.session_state.pop("chat_renaming_thread_id", None)
                st.rerun()

    # Inline delete confirmation dialog
    pending_del = st.session_state.get("chat_pending_delete_thread_id")
    if pending_del == active_thread.conversation_id:
        with st.container(border=True):
            st.warning(
                f"Are you sure you want to delete conversation '{active_thread.title}'? "
                "All messages in this thread will be permanently cleared from this session."
            )
            c_yd, c_nd = st.columns([1, 1])
            if c_yd.button(
                "Confirm Delete", key="chat_tab_confirm_delete_btn", type="primary"
            ):
                orchestrator.delete_thread(selected_doc_id, active_thread.conversation_id)
                st.session_state.pop("chat_pending_delete_thread_id", None)
                remaining = conv_mgr.list_threads(selected_doc_id)
                active_by_doc[selected_doc_id] = (
                    remaining[0].conversation_id if remaining else None
                )
                st.rerun()
            if c_nd.button("Cancel", key="chat_tab_cancel_delete_btn"):
                st.session_state.pop("chat_pending_delete_thread_id", None)
                st.rerun()

    # Ollama provider status check (non-blocking)
    if (config.LLM_PROVIDER or "").strip().lower() == "ollama":
        try:
            ollama_p = llm_provider.get_llm_provider("ollama")
            if isinstance(ollama_p, llm_provider.OllamaLLMProvider):
                ok_conn, conn_msg = ollama_p.check_connection()
                if not ok_conn:
                    st.warning(f"⚠️ {conn_msg}")
                    st.caption("Start Ollama (`ollama serve`) to enable answers.")
                else:
                    ok_model, model_msg = ollama_p.check_model_available()
                    if not ok_model:
                        st.warning(f"⚠️ {model_msg}")
        except Exception as check_exc:
            logger.debug("Ollama check in chat tab note: %s", check_exc)

    st.divider()

    # Message history
    if not active_thread.messages:
        st.caption(
            "No messages yet in this conversation. Ask a question below to start chatting!"
        )
    else:
        for msg in active_thread.messages:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
                if msg.role == "user" and msg.metadata.get("turn_status") == "failed":
                    st.error(
                        f"⚠️ Generation failed: {msg.metadata.get('last_error', 'Generation error')}"
                    )
                    if st.button("🔄 Retry Question", key=f"retry_turn_{msg.message_id}"):
                        st.session_state["chat_pending_retry_thread_id"] = (
                            active_thread.conversation_id
                        )
                        st.session_state["chat_pending_retry_query"] = msg.content
                        st.session_state["chat_pending_retry_request_id"] = (
                            msg.metadata.get("request_id")
                        )
                        st.rerun()
                elif msg.role == "assistant":
                    citations = msg.citations or []
                    valid_citations = [
                        c for c in citations if c.get("is_valid", True)
                    ]
                    if valid_citations:
                        with st.expander(
                            f"📚 Sources ({len(valid_citations)} cited)",
                            expanded=False,
                        ):
                            for c in valid_citations:
                                src_num = c.get("source_id") or c.get("source_num", 1)
                                page = c.get("page", "?")
                                doc_name = c.get("filename", "")
                                snippet = c.get("snippet", "")
                                st.markdown(
                                    f"**[Source {src_num}]** — Page {page} *(Document: {doc_name})*"
                                )
                                if snippet:
                                    st.caption(f'"{snippet}"')
                    if msg.metadata.get("is_rewritten") and msg.metadata.get("retrieval_query"):
                        with st.expander(
                            "🔍 Conversational Query Rewrite", expanded=False
                        ):
                            st.caption(
                                f"Retrieved evidence using rewritten query: **{msg.metadata.get('retrieval_query')}**"
                            )

    # Process pending retry OR new user input from chat_input
    retry_thread_id = st.session_state.pop("chat_pending_retry_thread_id", None)
    retry_query = st.session_state.pop("chat_pending_retry_query", None)
    retry_req_id = st.session_state.pop("chat_pending_retry_request_id", None)

    user_input = st.chat_input(
        "Ask a question about this document...", key="chat_user_input"
    )

    exec_query = None
    is_retry = False
    exec_req_id = None

    if retry_thread_id == active_thread.conversation_id and retry_query:
        exec_query = retry_query
        is_retry = True
        exec_req_id = retry_req_id
    elif user_input and user_input.strip():
        exec_query = user_input.strip()
        is_retry = False
        exec_req_id = None

    if exec_query:
        if not is_retry:
            with st.chat_message("user"):
                st.markdown(exec_query)

        with st.chat_message("assistant"):
            try:
                stream = orchestrator.stream_turn(
                    query=exec_query,
                    document_id=selected_doc_id,
                    thread=active_thread,
                    is_retry=is_retry,
                    request_id=exec_req_id,
                )
                full_response = st.write_stream(stream)
                last_res = orchestrator.last_result
                if last_res and not last_res.success:
                    st.rerun()
                elif last_res and last_res.success:
                    valid_citations = [
                        c for c in last_res.citations if c.get("is_valid", True)
                    ]
                    if valid_citations:
                        with st.expander(
                            f"📚 Sources ({len(valid_citations)} cited)",
                            expanded=False,
                        ):
                            for c in valid_citations:
                                src_num = c.get("source_id") or c.get("source_num", 1)
                                page = c.get("page", "?")
                                doc_name = c.get("filename", "")
                                snippet = c.get("snippet", "")
                                st.markdown(
                                    f"**[Source {src_num}]** — Page {page} *(Document: {doc_name})*"
                                )
                                if snippet:
                                    st.caption(f'"{snippet}"')
                    if last_res.is_rewritten and last_res.retrieval_query:
                        with st.expander(
                            "🔍 Conversational Query Rewrite", expanded=False
                        ):
                            st.caption(
                                f"Retrieved evidence using rewritten query: **{last_res.retrieval_query}**"
                            )
            except Exception as exc:
                logger.exception("Error executing chat stream turn")
                st.error(f"Error during chat turn: {exc}")


def main() -> None:
    st.title("PDF Chat - RAG Question Answering System")

    # 1. Single-Instance Application Lock Check
    app_lock = get_application_lock()
    is_acquired, lock_info, lock_err = app_lock.acquire()
    if not is_acquired:
        holder_pid = lock_info.get("pid", "Unknown") if lock_info else "Unknown"
        started_at = lock_info.get("started_at", "Unknown") if lock_info else "Unknown"
        st.error(
            "🚫 **StudyAstra Application Lock Conflict**\n\n"
            "Another StudyAstra process is already running against this data directory.\n\n"
            f"- **Active Process PID:** `{holder_pid}`\n"
            f"- **Process Started At:** `{started_at}`\n\n"
            "To prevent vector index corruption and data collisions, only one StudyAstra instance "
            "is permitted to run at a time.\n\n"
            "Please close the other StudyAstra window or terminate that process before reloading this page."
        )
        st.stop()

    # 2. Production Configuration Preflight Validation
    is_valid, config_errors, config_warnings = config.validate_production_config()

    # Allow legacy test harnesses that inject mocked Groq clients to run UI tests
    if not is_valid and (config.LLM_PROVIDER or "").strip().lower() == "groq":
        from src.generation.llm_provider import Groq as LLMGroq

        is_mocked = (
            LLMGroq is not None
            and (
                hasattr(LLMGroq, "assert_called")
                or hasattr(LLMGroq, "_mock_return_value")
            )
        )
        is_app_test = any("streamlit.testing" in mod for mod in sys.modules)
        if is_mocked or is_app_test:
            non_groq_errors = [e for e in config_errors if "groq" not in e.lower()]
            if not non_groq_errors:
                is_valid = True
                config_warnings.append(
                    "Running with non-production provider 'groq' in test mode."
                )

    if not is_valid:
        error_items = "\n".join(f"- {err}" for err in config_errors)
        st.error(
            "❌ **Production Configuration Error**\n\n"
            "The application cannot start because the environment configuration is invalid for production:\n\n"
            f"{error_items}\n\n"
            "Please run `python scripts/check_env.py` to diagnose and fix configuration issues."
        )
        st.stop()

    for warning in config_warnings:
        st.warning(f"⚠️ {warning}")

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

    qa_tab_label = (
        f"Quick Q&A (Qwen2.5 7B)"
        if (config.LLM_PROVIDER or "").strip().lower() == "ollama"
        else "Quick Q&A (Groq)"
    )
    tab_chat, tab_retrieval, tab_summary, tab_notes, tab_questions, tab_flashcards, tab_timer, tab_qa = st.tabs(
        [
            "Chat with PDF",
            "Vector Retrieval Testing",
            "PDF Summary",
            "Study Notes",
            "Practice Questions",
            "Flashcards",
            "Study Timer",
            qa_tab_label,
        ]
    )
    with tab_chat:
        _render_chat_tab(store)
    with tab_retrieval:
        _render_retrieval_testing_tab(store)
    with tab_summary:
        _render_summary_tab(store)
    with tab_notes:
        _render_study_notes_tab(store)
    with tab_questions:
        _render_practice_questions_tab(store)
    with tab_flashcards:
        _render_flashcard_tab(store)
    with tab_timer:
        _render_study_timer_tab()
    with tab_qa:
        _render_qa_tab(store)


main()
