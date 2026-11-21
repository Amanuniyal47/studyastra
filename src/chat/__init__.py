"""Conversational PDF Chatbot module (Stage 10)."""

from src.chat.chat_orchestrator import (
    ChatOrchestrator,
    ChatOrchestratorResult,
    ConversationLockLease,
    orchestrate_chat,
)
from src.chat.chat_prompts import (
    CHAT_GROUNDED_SYSTEM_PROMPT,
    GroundedChatPrompt,
    PromptBudgetExceededError,
    build_document_evidence_block,
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
from src.chat.query_rewriter import (
    MAX_REWRITE_WORKERS,
    build_rewrite_prompt,
    clean_rewritten_query,
    get_active_rewrite_worker_count,
    is_follow_up_candidate,
    reset_rewrite_pool,
    rewrite_query,
)

__all__ = [
    "ChatOrchestrator",
    "ChatOrchestratorResult",
    "ConversationLockLease",
    "orchestrate_chat",
    "ChatMessage",
    "ConversationMemoryConfig",
    "ConversationThread",
    "ConversationManager",
    "SlidingWindowMemoryManager",
    "is_follow_up_candidate",
    "build_rewrite_prompt",
    "clean_rewritten_query",
    "rewrite_query",
    "retrieve_chat_context",
    "CHAT_GROUNDED_SYSTEM_PROMPT",
    "GroundedChatPrompt",
    "PromptBudgetExceededError",
    "build_document_evidence_block",
    "build_grounded_chat_prompt",
    "validate_chat_citations",
    "sanitize_and_repair_citations",
    "MAX_REWRITE_WORKERS",
    "reset_rewrite_pool",
    "get_active_rewrite_worker_count",
]
