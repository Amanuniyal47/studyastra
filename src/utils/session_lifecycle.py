"""Streamlit session state memory lifecycle management (Stage 11.2 Work Area C).

Provides:
- LRU bounding for recomputable cache dictionaries (e.g., cached_summaries, cached_questions).
- Protection for essential user data (conversations, quiz answers, flashcard edits, notes).
- Cleanup of stale ephemeral UI state (e.g., error banners from previous documents).
- Document deletion cache clearing.
"""

from typing import Any, Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_CACHE_MAX_ENTRIES = 5

ESSENTIAL_STATE_KEYS = {
    "chat_conversation_manager",
    "chat_active_conv_by_doc",
    "quiz_sessions",
    "study_stats_mgr",
    "pomodoro_timer",
    "challenge_timer",
}

RECOMPUTABLE_CACHE_KEYS = {
    "cached_summaries",
    "cached_questions",
    "cached_notes",
    "cached_flashcards",
}

EPHEMERAL_DOC_ERROR_PAIRS = [
    ("summary_error", "summary_error_doc_id"),
    ("notes_error", "notes_error_doc_id"),
    ("questions_error", "questions_error_doc_id"),
    ("flashcard_error", "flashcard_error_doc_id"),
]


def prune_lru_cache(
    cache_dict: Dict[str, Any],
    max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
    protected_keys: Optional[Set[str]] = None,
) -> List[str]:
    """Bound a recomputable cache dictionary to max_entries using FIFO/LRU eviction.

    Args:
        cache_dict: The dictionary to prune in-place.
        max_entries: Maximum allowable items in the cache. Must be >= 1.
        protected_keys: Optional set of document IDs that must NEVER be evicted
            (e.g., documents with active user quiz submissions or flashcard edits).

    Returns:
        List[str]: Keys that were evicted from the cache.
    """
    if not isinstance(cache_dict, dict) or max_entries <= 0:
        return []

    protected = protected_keys or set()
    evicted: List[str] = []

    # If already within budget, no eviction needed
    if len(cache_dict) <= max_entries:
        return evicted

    # Evict oldest un-protected items (in insertion order)
    keys_in_order = list(cache_dict.keys())
    for key in keys_in_order:
        if len(cache_dict) <= max_entries:
            break
        if key not in protected:
            cache_dict.pop(key, None)
            evicted.append(key)
            logger.debug("Pruned recomputable cache entry '%s' (max: %d)", key, max_entries)

    return evicted


def clean_stale_ephemeral_state(
    session_dict: Dict[str, Any],
    active_doc_id: Optional[str] = None,
) -> List[str]:
    """Remove ephemeral error messages if they were generated for a different document.

    Args:
        session_dict: The session state dictionary to clean.
        active_doc_id: The currently active document ID. If None, cleans all doc-tied errors.

    Returns:
        List[str]: Keys that were removed from session state.
    """
    if not isinstance(session_dict, dict):
        return []

    cleaned: List[str] = []
    for err_key, doc_id_key in EPHEMERAL_DOC_ERROR_PAIRS:
        err_doc_id = session_dict.get(doc_id_key)
        if err_doc_id is not None and err_doc_id != active_doc_id:
            if err_key in session_dict:
                session_dict.pop(err_key, None)
                cleaned.append(err_key)
            if doc_id_key in session_dict:
                session_dict.pop(doc_id_key, None)
                cleaned.append(doc_id_key)

    return cleaned


def classify_session_keys(keys: List[str]) -> Dict[str, List[str]]:
    """Classify session state keys into architectural lifecycle tiers."""
    categorized: Dict[str, List[str]] = {
        "essential": [],
        "recomputable_cache": [],
        "ephemeral_ui": [],
        "system": [],
        "other": [],
    }

    for key in keys:
        if (
            key in ESSENTIAL_STATE_KEYS
            or key.startswith("quiz_ans_")
            or key.startswith(("fc_input_front_", "fc_input_back_"))
            or key.startswith("study_timer_")
        ):
            categorized["essential"].append(key)
        elif key in RECOMPUTABLE_CACHE_KEYS:
            categorized["recomputable_cache"].append(key)
        elif (
            key in {"chat_error", "chat_doc_select", "chat_thread_select"}
            or any(key.startswith(p) for p in ("fc_", "chat_pending_", "chal_", "summary_", "notes_", "questions_", "flashcard_"))
        ):
            categorized["ephemeral_ui"].append(key)
        elif key in {"vector_store", "embedding_service"}:
            categorized["system"].append(key)
        else:
            categorized["other"].append(key)

    return categorized
