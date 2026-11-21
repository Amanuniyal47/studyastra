"""Conversational query rewriting for follow-up questions and conversational references.

Stage 10.2 Implementation:
- Selective rewriting: Heuristic detection (is_follow_up_candidate) avoids unnecessary
  LLM calls when questions are already standalone and self-contained.
- Resolves anaphoric references ("it", "that topic", "the previous method", "why?", etc.)
  using recent conversation turns.
- Native HTTP request timeouts on OllamaLLMProvider as the primary termination mechanism.
- Bounded worker pool (MAX_REWRITE_WORKERS = 2) with non-blocking saturation detection,
  guaranteeing no unbounded thread growth or caller blocking.
- Graceful non-blocking fallback to original query on timeout, network error, or disabled setting.
- Clean post-processing removing conversational prefixes and quotation marks.

ARCHITECTURAL NOTE ON PYTHON THREAD LIFECYCLE:
Python's runtime does not support forcibly terminating or preempting running threads
(Thread.stop() does not exist in standard Python). If an external provider lacks
native socket timeouts and blocks permanently, its background thread will continue
running until it completes or the process exits.
To eliminate unbounded thread accumulation, Studyastra enforces:
1. Native HTTP request timeouts on OllamaLLMProvider as the primary termination mechanism.
2. A bounded thread pool (MAX_REWRITE_WORKERS = 2) with non-blocking saturation detection.
3. Immediate non-blocking fallback to the original query if the pool is saturated.
"""

from __future__ import annotations

from collections.abc import Callable
import concurrent.futures
import logging
import re
import threading
from typing import Any

import src.config as config
from src.chat.conversation_memory import ChatMessage
from src.generation.llm_provider import BaseLLMProvider, get_llm_provider

logger = logging.getLogger(__name__)

# Bounded worker pool configuration
MAX_REWRITE_WORKERS = 2

_rewrite_lock = threading.Lock()
_rewrite_pool: concurrent.futures.ThreadPoolExecutor | None = None
_rewrite_semaphore: threading.Semaphore | None = None


def get_rewrite_pool() -> tuple[concurrent.futures.ThreadPoolExecutor, threading.Semaphore]:
    """Retrieve or lazily initialize the bounded query-rewrite worker pool."""
    global _rewrite_pool, _rewrite_semaphore
    with _rewrite_lock:
        if _rewrite_pool is None:
            _rewrite_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_REWRITE_WORKERS,
                thread_name_prefix="studyastra_rewriter",
            )
            _rewrite_semaphore = threading.Semaphore(MAX_REWRITE_WORKERS)
        return _rewrite_pool, _rewrite_semaphore


def reset_rewrite_pool() -> None:
    """Reset the rewrite pool (used in unit testing for lifecycle verification)."""
    global _rewrite_pool, _rewrite_semaphore
    with _rewrite_lock:
        if _rewrite_pool is not None:
            _rewrite_pool.shutdown(wait=False, cancel_futures=True)
            _rewrite_pool = None
            _rewrite_semaphore = None


def get_active_rewrite_worker_count() -> int:
    """Return the number of currently active rewrite workers (for test verification)."""
    with _rewrite_lock:
        if _rewrite_semaphore is None:
            return 0
        return MAX_REWRITE_WORKERS - getattr(_rewrite_semaphore, "_value", MAX_REWRITE_WORKERS)


# Patterns identifying follow-up / anaphoric phrases that point to earlier context
_ANAPHORIC_PHRASES_PATTERN = re.compile(
    r"\b("
    r"what about|how about|tell me more|explain that|explain more|expand on|elaborate|"
    r"continue|go on|give an example|give me an example|how so|why so|and then|what else|"
    r"the former|the latter|the previous|the next|the above|the earlier|the same|"
    r"as you said|you mentioned|mentioned earlier|earlier you said|what does that mean|"
    r"this topic|that topic|this concept|that concept|this method|that method|"
    r"these methods|those methods|this approach|that approach|the first one|the second one|"
    r"the third one|the last one"
    r")\b",
    re.IGNORECASE,
)

# Connectives at start of query indicating continuation
_START_CONNECTIVE_PATTERN = re.compile(
    r"^(and|also|so|but|or|then)\s+",
    re.IGNORECASE,
)

# Questions starting with a pronoun or demonstrative referring to prior entity
_PRONOUN_STARTER_PATTERN = re.compile(
    r"^(how|why|what|where|when|can|could|does|did|do|is|are|was|were|will|would|explain|describe)"
    r"\s+(?:does\s+|did\s+|do\s+|is\s+|are\s+|was\s+|were\s+|can\s+|could\s+|will\s+|would\s+)?"
    r"(it|its|this|that|these|those|they|them|their|he|she|him|her)\b",
    re.IGNORECASE,
)

# Demonstratives / pronouns anywhere in query referring backward
_BACKWARD_PRONOUN_PATTERN = re.compile(
    r"\b(it|its|they|them|their|theirs|this|that|these|those)\b",
    re.IGNORECASE,
)

# Direct standalone question pattern: "What is X?", "Define X", etc. with no pronouns
_STANDALONE_DEFINITIONAL_PATTERN = re.compile(
    r"^(?:what is|what are|define|explain|describe)\s+([a-zA-Z0-9_\-\s]{2,})\??$",
    re.IGNORECASE,
)


def is_follow_up_candidate(query: str, history: list[ChatMessage]) -> bool:
    """Determine whether a query is a conversational follow-up requiring rewriting.

    Returns False immediately if:
    - The query is empty or whitespace.
    - History is empty or contains no user/assistant messages (nothing to rewrite against).
    - The query is already a standalone question with self-contained entities and no anaphora.

    Returns True if:
    - The query contains anaphoric pointers ("the previous method", "that topic", etc.).
    - The query begins with a continuation connective ("And...", "Also...", "So...").
    - The query contains possessive pronouns referring to earlier context ("its", "their").
    - The query is a short ellipsis question (e.g. "Why?", "How come?", "What else?").
    - The query uses pronouns ("it", "they", "this", "that") in subject/referential positions
      without a locally established noun antecedent.
    """
    if not query or not isinstance(query, str):
        return False

    q_clean = query.strip()
    if not q_clean:
        return False

    # Without prior conversational turns, a query cannot be a follow-up
    if not history or not any(m.role in ("user", "assistant") for m in history):
        return False

    # Check for explicit anaphoric phrases
    if _ANAPHORIC_PHRASES_PATTERN.search(q_clean):
        return True

    # Check for starting continuation connectives
    if _START_CONNECTIVE_PATTERN.search(q_clean):
        return True

    # Check for pronoun starters ("Why does it...", "What are its...", "Can that...")
    if _PRONOUN_STARTER_PATTERN.search(q_clean):
        return True

    # Check for possessive anaphoric pronouns ("its", "their", "theirs")
    if re.search(r"\b(its|their|theirs)\b", q_clean, re.IGNORECASE):
        return True

    words = re.findall(r"\b\w+\b", q_clean)
    word_count = len(words)

    # Short query heuristic:
    # Very short queries (<= 4 words) in the presence of history are almost always follow-ups
    # (e.g. "Why?", "How?", "Any examples?", "What causes it?"), UNLESS it is a standalone
    # definitional question like "What is ATP?" or "Define entropy"
    if word_count <= 4:
        # Check if it's a standalone definition without pronouns
        match = _STANDALONE_DEFINITIONAL_PATTERN.match(q_clean)
        if match and not _BACKWARD_PRONOUN_PATTERN.search(q_clean):
            return False
        return True

    # For longer queries (>= 5 words):
    # If it contains pronouns ("it", "they", "this", "that"), check if it refers backward
    if _BACKWARD_PRONOUN_PATTERN.search(q_clean):
        # If the query contains pronouns in referential positions ("about it", "like this", "that method")
        referential_match = re.search(
            r"\b(about (?:it|this|that|them)|for (?:it|this|that|them)|with (?:it|this|that|them)|"
            r"of (?:it|this|that|them)|why (?:it|this|that)|how (?:it|this|that)|"
            r"(?:this|that) (?:one|method|concept|topic|step|stage|result|finding|technique|process))\b",
            q_clean,
            re.IGNORECASE,
        )
        if referential_match:
            return True

        # Check for isolated subject pronouns
        subject_pronoun_match = re.search(
            r"\b(it|they|this|that)\s+(?:can|could|will|would|should|is|are|was|were|has|have|does|did|do)\b",
            q_clean,
            re.IGNORECASE,
        )
        if subject_pronoun_match:
            return True

    # Otherwise, the query is standalone and self-contained
    return False


REWRITE_SYSTEM_PROMPT = (
    "You are an expert query rewriting assistant for an academic study platform. "
    "Your job is to rewrite conversational follow-up questions into clear, standalone, "
    "self-contained search queries.\n\n"
    "CRITICAL RULES:\n"
    "1. Replace ambiguous pronouns ('it', 'they', 'this', 'that', 'the previous method') "
    "with the specific concept, term, or entity discussed in the conversation history.\n"
    "2. Output ONLY the standalone rewritten question. Nothing else.\n"
    "3. Do NOT answer the question.\n"
    "4. Do NOT include greetings, preamble, quotes, explanations, or labels like 'Rewritten query:'.\n"
    "5. If the question is already clear and self-contained, return it verbatim."
)


def build_rewrite_prompt(
    query: str,
    history: list[ChatMessage],
    max_turns: int = 3,
) -> str:
    """Build the user prompt for the query rewriter including recent conversation turns."""
    # Select last max_turns user-assistant pairs
    max_msgs = max_turns * 2
    recent_msgs = history[-max_msgs:] if len(history) > max_msgs else history

    history_lines: list[str] = []
    for msg in recent_msgs:
        if msg.role == "user":
            history_lines.append(f"User: {msg.content.strip()}")
        elif msg.role == "assistant":
            # Truncate assistant answer if lengthy to keep rewrite prompt concise
            content = msg.content.strip()
            if len(content) > 300:
                content = content[:300].rsplit(" ", 1)[0] + "..."
            history_lines.append(f"Assistant: {content}")

    history_block = "\n".join(history_lines)

    return (
        f"Conversation History:\n"
        f"{history_block}\n\n"
        f"Follow-up Question: {query.strip()}\n\n"
        f"Standalone Rewritten Query:"
    )


def clean_rewritten_query(raw_text: str, original_query: str) -> str:
    """Strip preamble, label prefixes, quotation marks, and backticks from rewritten query."""
    if not raw_text or not isinstance(raw_text, str):
        return original_query.strip()

    cleaned = raw_text.strip()

    # Strip code block or backtick wrappers
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()
    cleaned = cleaned.strip("`").strip()

    # Strip common label prefixes
    label_patterns = [
        r"^Rewritten\s*Query\s*:\s*",
        r"^Rewritten\s*Question\s*:\s*",
        r"^Standalone\s*Query\s*:\s*",
        r"^Standalone\s*Question\s*:\s*",
        r"^Query\s*:\s*",
        r"^Question\s*:\s*",
        r"^Search\s*Query\s*:\s*",
    ]
    for pattern in label_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # Strip surrounding quotes
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        cleaned = cleaned[1:-1].strip()

    # If the output is empty or whitespace, fall back to original
    if not cleaned:
        return original_query.strip()

    return cleaned


def _run_with_bounded_timeout(
    target_func: Callable[[], str],
    timeout_seconds: float,
) -> str:
    """Execute target_func with a strict timeout using the bounded worker pool.

    Guarantees:
    - Bounded worker pool (MAX_REWRITE_WORKERS = 2) eliminates unbounded thread creation.
    - If the pool is saturated (all workers busy), non-blockingly returns fallback immediately.
    - Caller is never blocked beyond timeout_seconds.
    - Captures and re-raises exceptions raised inside target_func.
    """
    pool, sem = get_rewrite_pool()

    # Non-blocking acquisition of worker permit
    acquired = sem.acquire(blocking=False)
    if not acquired:
        logger.warning(
            "Query rewriter pool is saturated (%d active workers); falling back immediately.",
            MAX_REWRITE_WORKERS,
        )
        raise concurrent.futures.TimeoutError("Rewrite worker pool is saturated")

    def _worker_wrapper() -> str:
        try:
            return target_func()
        finally:
            sem.release()

    future = pool.submit(_worker_wrapper)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError:
        raise


def rewrite_query(
    query: str,
    history: list[ChatMessage],
    *,
    provider: BaseLLMProvider | None = None,
    timeout: float | None = None,
    enabled: bool | None = None,
) -> tuple[str, bool]:
    """Rewrite a follow-up query into a standalone search query if needed.

    Args:
        query: The user's input question.
        history: Complete or recent message history of the conversation thread.
        provider: Optional BaseLLMProvider. If None, acquires the centralized Ollama provider.
        timeout: Seconds to wait for rewrite completion. Defaults to CHAT_REWRITE_TIMEOUT_SECONDS.
        enabled: Whether query rewriting is enabled. Defaults to CHAT_REWRITE_ENABLED.

    Returns:
        tuple[str, bool]: (effective_query, was_rewritten)
        - effective_query: Rewritten query if rewritten, else original query.
        - was_rewritten: True if an LLM rewrite occurred and changed the query; False otherwise.
    """
    is_enabled = (
        config.CHAT_REWRITE_ENABLED if enabled is None else bool(enabled)
    )
    if not is_enabled:
        return query.strip(), False

    # Check if the query is a follow-up candidate
    if not is_follow_up_candidate(query, history):
        return query.strip(), False

    effective_timeout = (
        config.CHAT_REWRITE_TIMEOUT_SECONDS
        if timeout is None
        else max(0.01, float(timeout))
    )

    user_prompt = build_rewrite_prompt(query, history)

    def _execute_llm_call() -> str:
        active_provider = provider
        if active_provider is None:
            # Use native provider timeout to terminate the underlying socket at effective_timeout
            active_provider = get_llm_provider("ollama", timeout=effective_timeout)
        elif hasattr(active_provider, "timeout") and getattr(active_provider, "timeout") is None:
            try:
                active_provider.timeout = effective_timeout
            except Exception:
                pass

        return active_provider.generate(
            REWRITE_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,
            max_tokens=100,
        )

    try:
        # Strictly enforce timeout with bounded worker pool execution to prevent caller blocking and unbounded threads
        raw_response = _run_with_bounded_timeout(_execute_llm_call, effective_timeout)

        rewritten = clean_rewritten_query(raw_response, query)
        if rewritten and rewritten.lower() != query.strip().lower():
            logger.info("Rewrote query from '%s' to '%s'", query, rewritten)
            return rewritten, True

        return query.strip(), False

    except concurrent.futures.TimeoutError:
        logger.warning(
            "Query rewriting timed out after %.2fs; falling back to original query: '%s'",
            effective_timeout,
            query,
        )
        return query.strip(), False
    except Exception as exc:
        logger.warning(
            "Query rewriting failed (%s); gracefully falling back to original query: '%s'",
            exc,
            query,
        )
        return query.strip(), False
