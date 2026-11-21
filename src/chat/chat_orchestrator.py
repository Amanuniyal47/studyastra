"""Multi-Turn RAG Pipeline Orchestrator for Conversational PDF Chatbot.

Stage 10.3 Implementation & Reliability Audit:
Connects:
1. Conversation memory (ChatMessage, ConversationThread, SlidingWindowMemoryManager).
2. Conversational query rewriting (selective rewriting for follow-ups with bounded timeout).
3. Document-scoped FAISS retrieval (strictly isolated by document_id via MiniLM embeddings).
4. Grounded prompt construction (strict estimated-token budgeting and ceiling guarantees).
5. Centralized Qwen2.5 7B generation (via get_llm_provider("ollama")).
6. Citation validation and repair (sanitizing hallucinated references, matching true pages).
7. Conversation persistence (thread-level message tracking, metadata, and failure policies).

Concurrency & Thread Safety Model:
- Thread safety is enforced per conversation thread via reentrant locks (threading.RLock).
- Concurrent requests for the SAME conversation thread are serialized safely, preventing
  race conditions during transcript updates, sliding-window management, and retry handling.
- Requests targeting DISTINCT conversation threads run concurrently without contention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Callable, Iterator
import uuid

import streamlit as st

import src.config as config
from src.chat.chat_prompts import (
    GroundedChatPrompt,
    PromptBudgetExceededError,
    build_grounded_chat_prompt,
    sanitize_and_repair_citations,
    validate_chat_citations,
)
from src.chat.conversation_memory import (
    ChatMessage,
    ConversationManager,
    ConversationMemoryConfig,
    ConversationThread,
    SlidingWindowMemoryManager,
)
from src.chat.conversational_retriever import retrieve_chat_context
from src.chat.query_rewriter import is_follow_up_candidate, rewrite_query
from src.generation.llm_provider import (
    BaseLLMProvider,
    LLMConfigurationError,
    LLMProviderError,
    get_llm_provider,
)
from src.ingestion.chunker import estimate_tokens
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vector_store import FAISSVectorStore, get_default_vector_store

logger = logging.getLogger(__name__)


@dataclass
class ChatOrchestratorResult:
    """Structured output from a multi-turn chat orchestration execution."""

    success: bool
    answer: str
    original_query: str
    retrieval_query: str
    is_rewritten: bool
    document_id: str
    conversation_id: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    source_map: dict[int, dict[str, Any]] = field(default_factory=dict)
    raw_chunks: list[dict[str, Any]] = field(default_factory=list)
    user_message: ChatMessage | None = None
    assistant_message: ChatMessage | None = None
    prompt_tokens: int = 0
    answer_tokens: int = 0
    history_tokens: int = 0
    evidence_tokens: int = 0
    total_prompt_tokens: int = 0
    error_message: str | None = None
    summary_error: str | None = None
    request_id: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)
    evidence_truncated: bool = False
    excluded_chunks: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to a dictionary."""
        return {
            "success": self.success,
            "answer": self.answer,
            "original_query": self.original_query,
            "retrieval_query": self.retrieval_query,
            "is_rewritten": self.is_rewritten,
            "document_id": self.document_id,
            "conversation_id": self.conversation_id,
            "citations": self.citations,
            "source_map": self.source_map,
            "evidence_count": len(self.raw_chunks),
            "prompt_tokens": self.prompt_tokens,
            "answer_tokens": self.answer_tokens,
            "history_tokens": self.history_tokens,
            "evidence_tokens": self.evidence_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "error_message": self.error_message,
            "summary_error": self.summary_error,
            "request_id": self.request_id,
            "timing_ms": self.timing_ms,
            "evidence_truncated": self.evidence_truncated,
            "excluded_chunks": self.excluded_chunks,
        }


class _ConversationLockEntry:
    """Internal thread-safe registry entry for a conversation lock with active lease tracking."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id: str = conversation_id
        self.lock: threading.RLock = threading.RLock()
        self.ref_count: int = 0
        self.pending_eviction: bool = False


class ConversationLockLease:
    """A reference-counted lease holding an active conversation lock entry.

    Guarantees:
    - Atomically increments ref_count upon lease creation under _thread_locks_guard.
    - Prevents premature registry eviction while a caller holds this lease, even before
      lock acquisition begins.
    - Reliably releases the lock and decrements ref_count in release() / __exit__().
    - Triggers deferred eviction on the final reference release if eviction was requested
      while the lock was in active use.
    """

    def __init__(self, entry: _ConversationLockEntry, orchestrator_cls: Any) -> None:
        self._entry = entry
        self._orchestrator_cls = orchestrator_cls
        self._acquired = False
        self._released = False

    @property
    def lock(self) -> threading.RLock:
        return self._entry.lock

    @property
    def conversation_id(self) -> str:
        return self._entry.conversation_id

    @property
    def ref_count(self) -> int:
        return self._entry.ref_count

    def __enter__(self) -> threading.RLock:
        self._entry.lock.acquire()
        self._acquired = True
        return self._entry.lock

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    def release(self) -> None:
        """Release underlying RLock if acquired and decrement ref_count atomically."""
        if self._released:
            return
        self._released = True
        try:
            if self._acquired:
                self._entry.lock.release()
                self._acquired = False
        finally:
            self._orchestrator_cls._release_lock_entry(self._entry)


class ChatOrchestrator:
    """Central pipeline coordinator for multi-turn conversational PDF chat.

    Encapsulates all RAG stages, ensuring document isolation, prompt ceiling enforcement,
    citation hygiene, failure containment, and retry idempotency.
    """

    # Global registry of conversation lock entries keyed by conversation_id
    _thread_locks: dict[str, _ConversationLockEntry] = {}
    _thread_locks_guard: threading.Lock = threading.Lock()

    def __init__(
        self,
        *,
        store: FAISSVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        provider: BaseLLMProvider | None = None,
        memory_manager: SlidingWindowMemoryManager | None = None,
        conversation_manager: ConversationManager | None = None,
        memory_config: ConversationMemoryConfig | None = None,
        top_k: int | None = None,
        rewrite_enabled: bool | None = None,
        rewrite_timeout_seconds: float | None = None,
    ) -> None:
        self.store = store
        self.embedding_service = embedding_service
        self.provider = provider
        self.memory_config = memory_config or ConversationMemoryConfig()
        self.memory_manager = (
            memory_manager
            or SlidingWindowMemoryManager(config=self.memory_config)
        )
        self.conversation_manager = (
            conversation_manager or ConversationManager()
        )
        self.top_k = (
            config.CHAT_TOP_K if top_k is None else max(1, int(top_k))
        )
        self.rewrite_enabled = (
            config.CHAT_REWRITE_ENABLED
            if rewrite_enabled is None
            else bool(rewrite_enabled)
        )
        self.rewrite_timeout_seconds = (
            config.CHAT_REWRITE_TIMEOUT_SECONDS
            if rewrite_timeout_seconds is None
            else float(rewrite_timeout_seconds)
        )
        self.last_result: ChatOrchestratorResult | None = None

    @classmethod
    def acquire_conversation_lock(cls, conversation_id: str) -> ConversationLockLease:
        """Atomically register/retrieve lock entry and increment reference count, returning a lease."""
        with cls._thread_locks_guard:
            entry = cls._thread_locks.get(conversation_id)
            if entry is None:
                entry = _ConversationLockEntry(conversation_id)
                cls._thread_locks[conversation_id] = entry
            entry.ref_count += 1
            # Reset pending_eviction when a new active request arrives
            entry.pending_eviction = False
            return ConversationLockLease(entry, cls)

    @classmethod
    def _release_lock_entry(cls, entry: _ConversationLockEntry) -> None:
        """Atomically decrement reference count and complete deferred eviction if applicable."""
        with cls._thread_locks_guard:
            entry.ref_count = max(0, entry.ref_count - 1)
            if entry.ref_count == 0 and entry.pending_eviction:
                # Evict only if this entry is still the registered instance
                if cls._thread_locks.get(entry.conversation_id) is entry:
                    if entry.lock.acquire(blocking=False):
                        try:
                            cls._thread_locks.pop(entry.conversation_id, None)
                            logger.debug(
                                "Deferred eviction completed for conversation %s upon final lease release",
                                entry.conversation_id,
                            )
                        finally:
                            entry.lock.release()

    @classmethod
    def _get_thread_lock(cls, conversation_id: str) -> threading.RLock:
        """Retrieve or initialize a reentrant lock for a specific conversation thread (backwards-compatible)."""
        with cls._thread_locks_guard:
            entry = cls._thread_locks.get(conversation_id)
            if entry is None:
                entry = _ConversationLockEntry(conversation_id)
                cls._thread_locks[conversation_id] = entry
            return entry.lock

    @classmethod
    def release_thread_lock(cls, conversation_id: str) -> bool:
        """Safely evict a lock from registry if it has zero active references and is unheld.

        If the lock currently has active references (ref_count > 0) or is held by a caller,
        immediate eviction is refused and the entry is marked pending_eviction = True.
        When the final active lease releases, deferred eviction will remove the entry automatically.

        Returns:
            True if immediately evicted; False if not found or deferred.
        """
        with cls._thread_locks_guard:
            entry = cls._thread_locks.get(conversation_id)
            if entry is None:
                return False

            if entry.ref_count > 0:
                entry.pending_eviction = True
                logger.debug(
                    "Conversation %s has %d active reference(s); eviction deferred",
                    conversation_id,
                    entry.ref_count,
                )
                return False

            # Check if lock can be acquired non-blockingly (not held)
            acquired = entry.lock.acquire(blocking=False)
            if not acquired:
                entry.pending_eviction = True
                logger.debug(
                    "Conversation lock %s is currently held; eviction deferred",
                    conversation_id,
                )
                return False

            try:
                cls._thread_locks.pop(conversation_id, None)
                logger.debug("Immediate eviction completed for conversation %s", conversation_id)
                return True
            finally:
                entry.lock.release()

    @classmethod
    def cleanup_inactive_locks(cls, active_conversation_ids: Any) -> int:
        """Prune locks for conversation threads that no longer exist, provided they are unreferenced and unheld."""
        active_set = set(active_conversation_ids)
        removed_count = 0
        with cls._thread_locks_guard:
            candidates = [cid for cid in cls._thread_locks if cid not in active_set]
            for cid in candidates:
                entry = cls._thread_locks[cid]
                if entry.ref_count > 0:
                    entry.pending_eviction = True
                    continue
                if entry.lock.acquire(blocking=False):
                    try:
                        cls._thread_locks.pop(cid, None)
                        removed_count += 1
                    finally:
                        entry.lock.release()
                else:
                    entry.pending_eviction = True
        return removed_count

    @classmethod
    def clear_all_thread_locks(cls) -> int:
        """Testing and session reset utility: evict all unreferenced and unheld locks."""
        removed_count = 0
        with cls._thread_locks_guard:
            candidates = list(cls._thread_locks.keys())
            for cid in candidates:
                entry = cls._thread_locks[cid]
                if entry.ref_count > 0:
                    entry.pending_eviction = True
                    continue
                if entry.lock.acquire(blocking=False):
                    try:
                        cls._thread_locks.pop(cid, None)
                        removed_count += 1
                    finally:
                        entry.lock.release()
                else:
                    entry.pending_eviction = True
        return removed_count

    @classmethod
    def get_active_lock_count(cls) -> int:
        """Return the number of lock entries currently registered."""
        with cls._thread_locks_guard:
            return len(cls._thread_locks)

    @classmethod
    def get_lock_ref_count(cls, conversation_id: str) -> int:
        """Return current active reference count for a conversation lock entry."""
        with cls._thread_locks_guard:
            entry = cls._thread_locks.get(conversation_id)
            return entry.ref_count if entry is not None else 0

    @classmethod
    def is_lock_pending_eviction(cls, conversation_id: str) -> bool:
        """Return whether eviction is currently deferred for a conversation lock entry."""
        with cls._thread_locks_guard:
            entry = cls._thread_locks.get(conversation_id)
            return entry.pending_eviction if entry is not None else False

    def delete_thread(self, document_id: str, conversation_id: str) -> bool:
        """Delete a conversation thread from storage and clean up its lock safely."""
        deleted = self.conversation_manager.delete_thread(document_id, conversation_id)
        if deleted:
            self.release_thread_lock(conversation_id)
        return deleted

    def clear_document_threads(self, document_id: str) -> None:
        """Remove all conversations for a document and release their unheld locks."""
        threads = self.conversation_manager.list_threads(document_id)
        cids = [t.conversation_id for t in threads]
        self.conversation_manager.clear_document_threads(document_id)
        for cid in cids:
            self.release_thread_lock(cid)

    def _resolve_store(
        self, store_override: FAISSVectorStore | None = None
    ) -> FAISSVectorStore | None:
        """Resolve active vector store from override, instance, session state, or default."""
        if store_override is not None:
            return store_override
        if self.store is not None:
            return self.store
        try:
            session_store = st.session_state.get("vector_store")
            if session_store is not None:
                return session_store
        except Exception:
            pass
        try:
            return get_default_vector_store(
                embedding_service=self.embedding_service
            )
        except Exception as exc:
            logger.debug("Could not resolve default vector store: %s", exc)
            return None

    def _resolve_provider(
        self, provider_override: BaseLLMProvider | None = None
    ) -> BaseLLMProvider:
        """Resolve centralized LLM provider, defaulting to Ollama Qwen2.5 7B."""
        if provider_override is not None:
            return provider_override
        if self.provider is not None:
            return self.provider
        return get_llm_provider("ollama")

    def get_or_create_thread(
        self,
        document_id: str,
        conversation_id: str | None = None,
        title: str | None = None,
    ) -> ConversationThread:
        """Retrieve or initialize a conversation thread scoped to document_id."""
        return self.conversation_manager.get_or_create_thread(
            document_id=document_id,
            conversation_id=conversation_id,
            title=title,
        )

    def orchestrate_turn(
        self,
        query: str,
        document_id: str,
        *,
        thread: ConversationThread | None = None,
        conversation_id: str | None = None,
        store: FAISSVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        provider: BaseLLMProvider | None = None,
        top_k: int | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
        is_retry: bool = False,
        rollback_user_message_on_failure: bool = False,
        request_id: str | None = None,
    ) -> ChatOrchestratorResult:
        """Execute a complete multi-turn conversational chat turn with RAG grounding.

        Execution Steps:
        1. Validate request and document availability.
        2. Store or reuse user message with unambiguous turn identity.
        3. Assemble conversation context (sliding window).
        4. Rewrite follow-up query selectively (retrieval only).
        5. Retrieve document evidence scoped strictly to document_id.
        6. Build grounded prompt with token ceiling enforcement.
        7. Generate answer via centralized Ollama Qwen2.5 7B.
        8. Validate and repair source citations.
        9. Store assistant response with metadata (never store on failure).
        10. Return structured result with timing, token, and summary error metrics.
        """
        timing: dict[str, float] = {}
        turn_start_time = time.perf_counter()
        req_id = request_id or str(uuid.uuid4())

        # -------------------------------------------------------------
        # Step 1: Validate Request & Target Document
        # -------------------------------------------------------------
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            res = ChatOrchestratorResult(
                success=False,
                answer="",
                original_query=str(query or ""),
                retrieval_query=str(query or ""),
                is_rewritten=False,
                document_id="",
                conversation_id=conversation_id or "",
                request_id=req_id,
                error_message="document_id must be a non-empty string.",
            )
            self.last_result = res
            return res

        clean_doc_id = document_id.strip()

        if not query or not isinstance(query, str) or not query.strip():
            res = ChatOrchestratorResult(
                success=False,
                answer="",
                original_query=str(query or ""),
                retrieval_query=str(query or ""),
                is_rewritten=False,
                document_id=clean_doc_id,
                conversation_id=conversation_id or "",
                request_id=req_id,
                error_message="User query cannot be empty.",
            )
            self.last_result = res
            return res

        clean_query = query.strip()

        # Resolve or validate ConversationThread
        active_thread = thread
        if active_thread is None:
            active_thread = self.conversation_manager.get_or_create_thread(
                document_id=clean_doc_id,
                conversation_id=conversation_id,
            )
        else:
            if active_thread.document_id != clean_doc_id:
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=clean_query,
                    is_rewritten=False,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    request_id=req_id,
                    error_message=(
                        f"Thread document_id '{active_thread.document_id}' does not match "
                        f"requested document_id '{clean_doc_id}'."
                    ),
                )
                self.last_result = res
                return res

        # Synchronize execution for this specific conversation thread with ref-counted lease
        with self.acquire_conversation_lock(active_thread.conversation_id):
            # Validate document in vector store registry
            active_store = self._resolve_store(store)
            if active_store is None:
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=clean_query,
                    is_rewritten=False,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    request_id=req_id,
                    error_message="No vector store is available or configured.",
                )
                self.last_result = res
                return res

            if active_store.total_vectors == 0:
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=clean_query,
                    is_rewritten=False,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    request_id=req_id,
                    error_message=f"Document '{clean_doc_id}' is not indexed or vector store is empty.",
                )
                self.last_result = res
                return res

            if not active_store.is_document_indexed(clean_doc_id):
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=clean_query,
                    is_rewritten=False,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    request_id=req_id,
                    error_message=f"Document '{clean_doc_id}' not found in document registry.",
                )
                self.last_result = res
                return res

            # -------------------------------------------------------------
            # Step 2: Store or Reuse User Message (Turn & Retry Identity)
            # -------------------------------------------------------------
            # An unanswered tail message exists if the last message in thread is a user message
            has_unanswered_user_tail = (
                bool(active_thread.messages)
                and active_thread.messages[-1].role == "user"
            )

            user_message_created = False
            if has_unanswered_user_tail:
                tail_msg = active_thread.messages[-1]
                if is_retry or tail_msg.content.strip() == clean_query:
                    # Retry of the existing unanswered question
                    user_msg = tail_msg
                    if "initial_request_id" not in user_msg.metadata:
                        user_msg.metadata["initial_request_id"] = user_msg.metadata.get(
                            "request_id", req_id
                        )
                    user_msg.metadata["request_id"] = req_id
                    user_msg.metadata["retry_count"] = (
                        user_msg.metadata.get("retry_count", 0) + 1
                    )
                    user_msg.metadata["turn_status"] = "retrying"
                    logger.info(
                        "Reusing unanswered user message %s for retry of query '%s'",
                        user_msg.message_id,
                        clean_query,
                    )
                else:
                    # Learner asked a different question: preserve failed turn in transcript
                    # and append a new independent user turn (Area 1 Transcript Integrity)
                    logger.info(
                        "Preserving failed user message %s and appending new independent turn '%s'",
                        tail_msg.message_id,
                        clean_query,
                    )
                    user_msg = active_thread.add_user_message(clean_query)
                    user_msg.metadata["turn_status"] = "pending"
                    user_msg.metadata["request_id"] = req_id
                    user_message_created = True
            else:
                # Normal new turn (even if consecutive identical query after a completed turn)
                user_msg = active_thread.add_user_message(clean_query)
                user_msg.metadata["turn_status"] = "pending"
                user_msg.metadata["request_id"] = req_id
                user_message_created = True

            # Extract prior messages (excluding the current user message) for context assembly
            prior_messages = [
                m for m in active_thread.messages if m.message_id != user_msg.message_id
            ]
            prior_thread_view = ConversationThread(
                document_id=clean_doc_id,
                conversation_id=active_thread.conversation_id,
                title=active_thread.title,
                messages=prior_messages,
                summary_memory=active_thread.summary_memory,
            )

            # -------------------------------------------------------------
            # Step 3 & 4: Selective Query Rewriting for Follow-Up Questions
            # -------------------------------------------------------------
            t_rewrite_start = time.perf_counter()
            retrieval_query = clean_query
            is_rewritten = False

            active_provider = self._resolve_provider(provider)

            if self.rewrite_enabled and prior_messages and is_follow_up_candidate(clean_query, prior_messages):
                logger.info("Follow-up detected for query '%s'; invoking rewriter", clean_query)
                rewritten_text, was_rewritten = rewrite_query(
                    clean_query,
                    prior_messages,
                    provider=active_provider,
                    timeout=self.rewrite_timeout_seconds,
                    enabled=self.rewrite_enabled,
                )
                if was_rewritten and rewritten_text and rewritten_text.strip():
                    retrieval_query = rewritten_text.strip()
                    is_rewritten = True
                    logger.info("Rewrote query for retrieval: '%s' -> '%s'", clean_query, retrieval_query)
            else:
                logger.debug("Query '%s' classified as standalone; skipping rewrite", clean_query)

            timing["rewrite_ms"] = (time.perf_counter() - t_rewrite_start) * 1000

            # -------------------------------------------------------------
            # Step 5: Retrieve Document Evidence
            # -------------------------------------------------------------
            t_retrieval_start = time.perf_counter()
            effective_top_k = self.top_k if top_k is None else max(1, int(top_k))

            try:
                chunks = retrieve_chat_context(
                    retrieval_query,
                    clean_doc_id,
                    store=active_store,
                    embedding_service=embedding_service or self.embedding_service,
                    top_k=effective_top_k,
                )
            except Exception as exc:
                logger.exception("Retrieval failed for doc '%s': %s", clean_doc_id, exc)
                user_msg.metadata["turn_status"] = "failed"
                user_msg.metadata["last_error"] = str(exc)
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=retrieval_query,
                    is_rewritten=is_rewritten,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    user_message=user_msg,
                    request_id=req_id,
                    error_message=f"Retrieval error: {exc}",
                    timing_ms=timing,
                )
                self.last_result = res
                return res

            timing["retrieval_ms"] = (time.perf_counter() - t_retrieval_start) * 1000

            # -------------------------------------------------------------
            # Step 6: Build Grounded Prompt with Token Ceiling Enforcement
            # -------------------------------------------------------------
            t_prompt_start = time.perf_counter()
            try:
                grounded_prompt = build_grounded_chat_prompt(
                    clean_query,
                    chunks,
                    thread=prior_thread_view,
                    memory_manager=self.memory_manager,
                    config_obj=self.memory_config,
                )
            except PromptBudgetExceededError as exc:
                logger.warning("Prompt budget exceeded for query '%s': %s", clean_query, exc)
                user_msg.metadata["turn_status"] = "failed"
                user_msg.metadata["last_error"] = str(exc)
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=retrieval_query,
                    is_rewritten=is_rewritten,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    raw_chunks=chunks,
                    user_message=user_msg,
                    request_id=req_id,
                    error_message=str(exc),
                    timing_ms=timing,
                )
                self.last_result = res
                return res
            except Exception as exc:
                logger.exception("Unexpected error building prompt: %s", exc)
                user_msg.metadata["turn_status"] = "failed"
                user_msg.metadata["last_error"] = str(exc)
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=retrieval_query,
                    is_rewritten=is_rewritten,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    raw_chunks=chunks,
                    user_message=user_msg,
                    request_id=req_id,
                    error_message=f"Prompt build error: {exc}",
                    timing_ms=timing,
                )
                self.last_result = res
                return res

            timing["prompt_build_ms"] = (time.perf_counter() - t_prompt_start) * 1000

            # -------------------------------------------------------------
            # Step 7: Generate Answer via Centralized LLM Provider
            # -------------------------------------------------------------
            t_gen_start = time.perf_counter()
            raw_answer = ""

            try:
                if on_token is not None:
                    token_chunks: list[str] = []
                    for token in active_provider.generate_stream(
                        grounded_prompt.system_prompt,
                        grounded_prompt.user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        token_chunks.append(token)
                        try:
                            on_token(token)
                        except Exception as cb_exc:
                            # Callback errors do not interrupt transcript handling or fail generation
                            logger.warning("on_token callback exception: %s", cb_exc)
                    raw_answer = "".join(token_chunks).strip()
                else:
                    raw_answer = active_provider.generate(
                        grounded_prompt.system_prompt,
                        grounded_prompt.user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ).strip()

                if not raw_answer:
                    raise LLMProviderError("Ollama returned an empty completion text.")

            except (LLMProviderError, LLMConfigurationError) as exc:
                logger.error("LLM generation failed: %s", exc)
                user_msg.metadata["turn_status"] = "failed"
                user_msg.metadata["last_error"] = str(exc)
                timing["generation_ms"] = (time.perf_counter() - t_gen_start) * 1000
                timing["total_ms"] = (time.perf_counter() - turn_start_time) * 1000
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=retrieval_query,
                    is_rewritten=is_rewritten,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    source_map=grounded_prompt.source_map,
                    raw_chunks=chunks,
                    user_message=user_msg,
                    prompt_tokens=grounded_prompt.total_prompt_tokens,
                    history_tokens=grounded_prompt.history_tokens,
                    evidence_tokens=grounded_prompt.evidence_tokens,
                    total_prompt_tokens=grounded_prompt.total_prompt_tokens,
                    request_id=req_id,
                    error_message=str(exc),
                    timing_ms=timing,
                    evidence_truncated=grounded_prompt.evidence_truncated,
                    excluded_chunks=grounded_prompt.excluded_chunks,
                )
                self.last_result = res
                return res
            except Exception as exc:
                logger.exception("Unexpected error during LLM generation: %s", exc)
                user_msg.metadata["turn_status"] = "failed"
                user_msg.metadata["last_error"] = str(exc)
                timing["generation_ms"] = (time.perf_counter() - t_gen_start) * 1000
                timing["total_ms"] = (time.perf_counter() - turn_start_time) * 1000
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=retrieval_query,
                    is_rewritten=is_rewritten,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    source_map=grounded_prompt.source_map,
                    raw_chunks=chunks,
                    user_message=user_msg,
                    prompt_tokens=grounded_prompt.total_prompt_tokens,
                    history_tokens=grounded_prompt.history_tokens,
                    evidence_tokens=grounded_prompt.evidence_tokens,
                    total_prompt_tokens=grounded_prompt.total_prompt_tokens,
                    request_id=req_id,
                    error_message=f"Generation failed: {exc}",
                    timing_ms=timing,
                    evidence_truncated=grounded_prompt.evidence_truncated,
                    excluded_chunks=grounded_prompt.excluded_chunks,
                )
                self.last_result = res
                return res

            timing["generation_ms"] = (time.perf_counter() - t_gen_start) * 1000

            # -------------------------------------------------------------
            # Step 8: Validate and Sanitize Citations BEFORE Committing
            # -------------------------------------------------------------
            t_cite_start = time.perf_counter()
            validated_citations = validate_chat_citations(
                raw_answer, grounded_prompt.source_map
            )
            repaired_answer = sanitize_and_repair_citations(
                raw_answer, grounded_prompt.source_map
            )
            timing["citation_validation_ms"] = (
                time.perf_counter() - t_cite_start
            ) * 1000

            # -------------------------------------------------------------
            # Step 9: Store Assistant Response in Conversation Thread
            # -------------------------------------------------------------
            persisted_citations = [c for c in validated_citations if c.get("is_valid")]

            assistant_msg = active_thread.add_assistant_message(
                content=repaired_answer,
                citations=persisted_citations,
                metadata={
                    "retrieval_query": retrieval_query,
                    "is_rewritten": is_rewritten,
                    "evidence_count": len(chunks),
                    "excluded_chunks": grounded_prompt.excluded_chunks,
                    "evidence_truncated": grounded_prompt.evidence_truncated,
                    "prompt_tokens": grounded_prompt.total_prompt_tokens,
                    "model": active_provider.model_name,
                    "request_id": req_id,
                    "turn_status": "completed",
                },
            )
            user_msg.metadata["turn_status"] = "completed"

            # -------------------------------------------------------------
            # Step 9b: Secondary Summary-Memory Update (Isolated from Answer)
            # -------------------------------------------------------------
            summary_error: str | None = None
            try:
                self.memory_manager.update_memory(active_thread)
            except Exception as mem_exc:
                logger.warning("Secondary memory manager update notice: %s", mem_exc)
                summary_error = str(mem_exc)

            # -------------------------------------------------------------
            # Step 10: Return Structured Result
            # -------------------------------------------------------------
            timing["total_ms"] = (time.perf_counter() - turn_start_time) * 1000
            answer_tokens = estimate_tokens(repaired_answer)

            result = ChatOrchestratorResult(
                success=True,
                answer=repaired_answer,
                original_query=clean_query,
                retrieval_query=retrieval_query,
                is_rewritten=is_rewritten,
                document_id=clean_doc_id,
                conversation_id=active_thread.conversation_id,
                citations=validated_citations,
                source_map=grounded_prompt.source_map,
                raw_chunks=chunks,
                user_message=user_msg,
                assistant_message=assistant_msg,
                prompt_tokens=grounded_prompt.total_prompt_tokens,
                answer_tokens=answer_tokens,
                history_tokens=grounded_prompt.history_tokens,
                evidence_tokens=grounded_prompt.evidence_tokens,
                total_prompt_tokens=grounded_prompt.total_prompt_tokens,
                request_id=req_id,
                error_message=None,
                summary_error=summary_error,
                timing_ms=timing,
                evidence_truncated=grounded_prompt.evidence_truncated,
                excluded_chunks=grounded_prompt.excluded_chunks,
            )
            self.last_result = result
            return result

    def stream_turn(
        self,
        query: str,
        document_id: str,
        *,
        thread: ConversationThread | None = None,
        conversation_id: str | None = None,
        store: FAISSVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        provider: BaseLLMProvider | None = None,
        top_k: int | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        is_retry: bool = False,
        rollback_user_message_on_failure: bool = False,
        request_id: str | None = None,
    ) -> Iterator[str]:
        """Stream generated text chunks in real-time with strict failure containment.

        Tokens are yielded live as they arrive. If generation fails mid-stream:
        - The assistant message is NEVER committed to thread.messages.
        - self.last_result is updated with success=False and the error.
        - The user message is preserved in thread for retry.

        If generation succeeds:
        - The complete assistant message is sanitized and committed to thread.messages.
        - self.last_result contains the full ChatOrchestratorResult with success=True.
        """
        timing: dict[str, float] = {}
        turn_start_time = time.perf_counter()
        req_id = request_id or str(uuid.uuid4())

        # Validate request
        if not document_id or not isinstance(document_id, str) or not document_id.strip():
            res = ChatOrchestratorResult(
                success=False,
                answer="",
                original_query=str(query or ""),
                retrieval_query=str(query or ""),
                is_rewritten=False,
                document_id="",
                conversation_id=conversation_id or "",
                request_id=req_id,
                error_message="document_id must be a non-empty string.",
            )
            self.last_result = res
            return

        clean_doc_id = document_id.strip()

        if not query or not isinstance(query, str) or not query.strip():
            res = ChatOrchestratorResult(
                success=False,
                answer="",
                original_query=str(query or ""),
                retrieval_query=str(query or ""),
                is_rewritten=False,
                document_id=clean_doc_id,
                conversation_id=conversation_id or "",
                request_id=req_id,
                error_message="User query cannot be empty.",
            )
            self.last_result = res
            return

        clean_query = query.strip()

        active_thread = thread
        if active_thread is None:
            active_thread = self.conversation_manager.get_or_create_thread(
                document_id=clean_doc_id,
                conversation_id=conversation_id,
            )
        else:
            if active_thread.document_id != clean_doc_id:
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=clean_query,
                    is_rewritten=False,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    request_id=req_id,
                    error_message=(
                        f"Thread document_id '{active_thread.document_id}' does not match "
                        f"requested document_id '{clean_doc_id}'."
                    ),
                )
                self.last_result = res
                return

        # Synchronize execution for this specific conversation thread with ref-counted lease
        with self.acquire_conversation_lock(active_thread.conversation_id):
            active_store = self._resolve_store(store)
            if active_store is None:
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=clean_query,
                    is_rewritten=False,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    request_id=req_id,
                    error_message="No vector store is available or configured.",
                )
                self.last_result = res
                return

            if active_store.total_vectors == 0:
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=clean_query,
                    is_rewritten=False,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    request_id=req_id,
                    error_message=f"Document '{clean_doc_id}' is not indexed or vector store is empty.",
                )
                self.last_result = res
                return

            if not active_store.is_document_indexed(clean_doc_id):
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=clean_query,
                    is_rewritten=False,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    request_id=req_id,
                    error_message=f"Document '{clean_doc_id}' not found in document registry.",
                )
                self.last_result = res
                return

            # Store or reuse user message
            has_unanswered_user_tail = (
                bool(active_thread.messages)
                and active_thread.messages[-1].role == "user"
            )

            user_message_created = False
            if has_unanswered_user_tail:
                tail_msg = active_thread.messages[-1]
                if is_retry or tail_msg.content.strip() == clean_query:
                    user_msg = tail_msg
                    if "initial_request_id" not in user_msg.metadata:
                        user_msg.metadata["initial_request_id"] = user_msg.metadata.get(
                            "request_id", req_id
                        )
                    user_msg.metadata["request_id"] = req_id
                    user_msg.metadata["retry_count"] = (
                        user_msg.metadata.get("retry_count", 0) + 1
                    )
                    user_msg.metadata["turn_status"] = "retrying"
                else:
                    user_msg = active_thread.add_user_message(clean_query)
                    user_msg.metadata["turn_status"] = "pending"
                    user_msg.metadata["request_id"] = req_id
                    user_message_created = True
            else:
                user_msg = active_thread.add_user_message(clean_query)
                user_msg.metadata["turn_status"] = "pending"
                user_msg.metadata["request_id"] = req_id
                user_message_created = True

            prior_messages = [
                m for m in active_thread.messages if m.message_id != user_msg.message_id
            ]
            prior_thread_view = ConversationThread(
                document_id=clean_doc_id,
                conversation_id=active_thread.conversation_id,
                title=active_thread.title,
                messages=prior_messages,
                summary_memory=active_thread.summary_memory,
            )

            t_rewrite_start = time.perf_counter()
            retrieval_query = clean_query
            is_rewritten = False
            active_provider = self._resolve_provider(provider)

            if self.rewrite_enabled and prior_messages and is_follow_up_candidate(clean_query, prior_messages):
                rewritten_text, was_rewritten = rewrite_query(
                    clean_query,
                    prior_messages,
                    provider=active_provider,
                    timeout=self.rewrite_timeout_seconds,
                    enabled=self.rewrite_enabled,
                )
                if was_rewritten and rewritten_text and rewritten_text.strip():
                    retrieval_query = rewritten_text.strip()
                    is_rewritten = True

            timing["rewrite_ms"] = (time.perf_counter() - t_rewrite_start) * 1000

            t_retrieval_start = time.perf_counter()
            effective_top_k = self.top_k if top_k is None else max(1, int(top_k))

            try:
                chunks = retrieve_chat_context(
                    retrieval_query,
                    clean_doc_id,
                    store=active_store,
                    embedding_service=embedding_service or self.embedding_service,
                    top_k=effective_top_k,
                )
            except Exception as exc:
                user_msg.metadata["turn_status"] = "failed"
                user_msg.metadata["last_error"] = str(exc)
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=retrieval_query,
                    is_rewritten=is_rewritten,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    user_message=user_msg,
                    request_id=req_id,
                    error_message=f"Retrieval error: {exc}",
                    timing_ms=timing,
                )
                self.last_result = res
                return

            timing["retrieval_ms"] = (time.perf_counter() - t_retrieval_start) * 1000

            t_prompt_start = time.perf_counter()
            try:
                grounded_prompt = build_grounded_chat_prompt(
                    clean_query,
                    chunks,
                    thread=prior_thread_view,
                    memory_manager=self.memory_manager,
                    config_obj=self.memory_config,
                )
            except PromptBudgetExceededError as exc:
                user_msg.metadata["turn_status"] = "failed"
                user_msg.metadata["last_error"] = str(exc)
                res = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=retrieval_query,
                    is_rewritten=is_rewritten,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    raw_chunks=chunks,
                    user_message=user_msg,
                    request_id=req_id,
                    error_message=str(exc),
                    timing_ms=timing,
                )
                self.last_result = res
                return

            timing["prompt_build_ms"] = (time.perf_counter() - t_prompt_start) * 1000

            # Live streaming loop with mid-stream exception containment
            token_chunks = []
            t_gen_start = time.perf_counter()

            try:
                for token in active_provider.generate_stream(
                    grounded_prompt.system_prompt,
                    grounded_prompt.user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    token_chunks.append(token)
                    yield token

                raw_answer = "".join(token_chunks).strip()
                if not raw_answer:
                    raise LLMProviderError("Ollama returned an empty completion text.")

            except Exception as exc:
                logger.error("Streaming failed mid-stream: %s", exc)
                # DO NOT COMMIT ASSISTANT MESSAGE
                user_msg.metadata["turn_status"] = "failed"
                user_msg.metadata["last_error"] = str(exc)
                timing["generation_ms"] = (time.perf_counter() - t_gen_start) * 1000
                timing["total_ms"] = (time.perf_counter() - turn_start_time) * 1000
                self.last_result = ChatOrchestratorResult(
                    success=False,
                    answer="",
                    original_query=clean_query,
                    retrieval_query=retrieval_query,
                    is_rewritten=is_rewritten,
                    document_id=clean_doc_id,
                    conversation_id=active_thread.conversation_id,
                    user_message=user_msg,
                    request_id=req_id,
                    error_message=f"Streaming failed: {exc}",
                    timing_ms=timing,
                )
                return

            timing["generation_ms"] = (time.perf_counter() - t_gen_start) * 1000

            # Citations validation and repair
            t_cite_start = time.perf_counter()
            validated_citations = validate_chat_citations(
                raw_answer, grounded_prompt.source_map
            )
            repaired_answer = sanitize_and_repair_citations(
                raw_answer, grounded_prompt.source_map
            )
            timing["citation_validation_ms"] = (
                time.perf_counter() - t_cite_start
            ) * 1000

            # Commit assistant message to thread
            persisted_citations = [c for c in validated_citations if c.get("is_valid")]
            assistant_msg = active_thread.add_assistant_message(
                content=repaired_answer,
                citations=persisted_citations,
                metadata={
                    "retrieval_query": retrieval_query,
                    "is_rewritten": is_rewritten,
                    "evidence_count": len(chunks),
                    "excluded_chunks": grounded_prompt.excluded_chunks,
                    "evidence_truncated": grounded_prompt.evidence_truncated,
                    "prompt_tokens": grounded_prompt.total_prompt_tokens,
                    "model": active_provider.model_name,
                    "request_id": req_id,
                    "turn_status": "completed",
                },
            )
            user_msg.metadata["turn_status"] = "completed"

            # Secondary summary update
            summary_error = None
            try:
                self.memory_manager.update_memory(active_thread)
            except Exception as mem_exc:
                logger.warning("Secondary memory manager update notice: %s", mem_exc)
                summary_error = str(mem_exc)

            timing["total_ms"] = (time.perf_counter() - turn_start_time) * 1000
            answer_tokens = estimate_tokens(repaired_answer)

            self.last_result = ChatOrchestratorResult(
                success=True,
                answer=repaired_answer,
                original_query=clean_query,
                retrieval_query=retrieval_query,
                is_rewritten=is_rewritten,
                document_id=clean_doc_id,
                conversation_id=active_thread.conversation_id,
                citations=validated_citations,
                source_map=grounded_prompt.source_map,
                raw_chunks=chunks,
                user_message=user_msg,
                assistant_message=assistant_msg,
                prompt_tokens=grounded_prompt.total_prompt_tokens,
                answer_tokens=answer_tokens,
                history_tokens=grounded_prompt.history_tokens,
                evidence_tokens=grounded_prompt.evidence_tokens,
                total_prompt_tokens=grounded_prompt.total_prompt_tokens,
                request_id=req_id,
                error_message=None,
                summary_error=summary_error,
                timing_ms=timing,
                evidence_truncated=grounded_prompt.evidence_truncated,
                excluded_chunks=grounded_prompt.excluded_chunks,
            )


def orchestrate_chat(
    query: str,
    document_id: str,
    *,
    thread: ConversationThread | None = None,
    conversation_id: str | None = None,
    store: FAISSVectorStore | None = None,
    embedding_service: EmbeddingService | None = None,
    provider: BaseLLMProvider | None = None,
    memory_manager: SlidingWindowMemoryManager | None = None,
    conversation_manager: ConversationManager | None = None,
    memory_config: ConversationMemoryConfig | None = None,
    top_k: int | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    rewrite_timeout_seconds: float | None = None,
    rewrite_enabled: bool | None = None,
    on_token: Callable[[str], None] | None = None,
    is_retry: bool = False,
    rollback_user_message_on_failure: bool = False,
    request_id: str | None = None,
) -> ChatOrchestratorResult:
    """Convenience functional interface for executing a complete multi-turn chat turn.

    Instantiates a temporary ChatOrchestrator or uses provided managers and returns
    the structured ChatOrchestratorResult.
    """
    orchestrator = ChatOrchestrator(
        store=store,
        embedding_service=embedding_service,
        provider=provider,
        memory_manager=memory_manager,
        conversation_manager=conversation_manager,
        memory_config=memory_config,
        top_k=top_k,
        rewrite_enabled=rewrite_enabled,
        rewrite_timeout_seconds=rewrite_timeout_seconds,
    )
    return orchestrator.orchestrate_turn(
        query=query,
        document_id=document_id,
        thread=thread,
        conversation_id=conversation_id,
        store=store,
        embedding_service=embedding_service,
        provider=provider,
        top_k=top_k,
        temperature=temperature,
        max_tokens=max_tokens,
        on_token=on_token,
        is_retry=is_retry,
        rollback_user_message_on_failure=rollback_user_message_on_failure,
        request_id=request_id,
    )
