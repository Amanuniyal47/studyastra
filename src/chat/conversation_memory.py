"""Conversation memory and sliding-window context engine for multi-turn PDF chatbot.

Stage 10.1 Implementation:
- Independent conversation memory subsystem.
- ChatMessage and ConversationThread dataclasses with full UUID identifiers.
- ISO 8601 timestamps and token estimation via estimate_tokens().
- Configurable sliding-window token budgets with conservative margins.
- Out-of-window turn summarization (injectable with deterministic extractive fallback).
- Multi-thread document isolation and serialization (JSON & Markdown exports).
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import re
from typing import Any
import uuid

import src.config as config
from src.ingestion.chunker import estimate_tokens

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def is_valid_uuid(val: Any) -> bool:
    """Validate whether val is a standard 36-character hexadecimal UUID string."""
    if not isinstance(val, str) or len(val) != 36:
        return False
    try:
        parsed = uuid.UUID(val)
        return str(parsed).lower() == val.lower()
    except (ValueError, AttributeError, TypeError):
        return False


def new_uuid() -> str:
    """Generate a full, standard 36-character UUIDv4 string."""
    return str(uuid.uuid4())


@dataclass
class ChatMessage:
    """Single message in a conversational thread."""

    role: str  # "user" | "assistant" | "system"
    content: str
    message_id: str = field(default_factory=new_uuid)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    token_count: int = 0
    citations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_roles = {"user", "assistant", "system"}
        if self.role not in valid_roles:
            raise ValueError(
                f"Invalid role '{self.role}'. Must be one of: {sorted(valid_roles)}"
            )

        if not isinstance(self.content, str):
            self.content = str(self.content or "")

        if not is_valid_uuid(self.message_id):
            self.message_id = new_uuid()

        if self.token_count <= 0 and self.content:
            self.token_count = estimate_tokens(self.content)

        if not isinstance(self.citations, list):
            self.citations = []

        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize message to dictionary."""
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "citations": deepcopy(self.citations),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatMessage:
        """Construct ChatMessage from dictionary with backward compatibility and defaults."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for ChatMessage, got {type(data)}")

        role = str(data.get("role", "user")).lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"

        content = str(data.get("content", ""))
        mid = str(data.get("message_id", ""))
        if not is_valid_uuid(mid):
            mid = new_uuid()

        ts = str(
            data.get("timestamp") or datetime.now(timezone.utc).isoformat()
        )
        tc = int(data.get("token_count", 0))
        if tc <= 0 and content:
            tc = estimate_tokens(content)

        raw_cites = data.get("citations", [])
        citations = [c for c in raw_cites if isinstance(c, dict)] if isinstance(raw_cites, list) else []

        raw_meta = data.get("metadata", {})
        metadata = deepcopy(raw_meta) if isinstance(raw_meta, dict) else {}

        return cls(
            role=role,
            content=content,
            message_id=mid,
            timestamp=ts,
            token_count=tc,
            citations=citations,
            metadata=metadata,
        )


@dataclass
class ConversationMemoryConfig:
    """Configurable token budgeting and sliding-window parameters."""

    context_window_tokens: int = field(
        default_factory=lambda: getattr(config, "CHAT_CONTEXT_WINDOW_TOKENS", 4096)
    )
    max_history_tokens: int = field(
        default_factory=lambda: getattr(config, "CHAT_MAX_HISTORY_TOKENS", 1500)
    )
    output_reserve_tokens: int = field(
        default_factory=lambda: getattr(config, "CHAT_OUTPUT_RESERVE_TOKENS", 1000)
    )
    system_prompt_reserve_tokens: int = field(
        default_factory=lambda: getattr(
            config, "CHAT_SYSTEM_PROMPT_RESERVE_TOKENS", 400
        )
    )
    doc_context_reserve_tokens: int = field(
        default_factory=lambda: getattr(
            config, "CHAT_DOC_CONTEXT_RESERVE_TOKENS", 1000
        )
    )
    min_recent_turns: int = field(
        default_factory=lambda: getattr(config, "CHAT_MIN_RECENT_TURNS", 2)
    )
    summary_max_tokens: int = field(
        default_factory=lambda: getattr(config, "CHAT_SUMMARY_MAX_TOKENS", 300)
    )
    safety_margin_ratio: float = field(
        default_factory=lambda: getattr(config, "CHAT_SAFETY_MARGIN_RATIO", 0.10)
    )

    def compute_available_history_budget(
        self,
        *,
        doc_context_tokens: int | None = None,
        system_prompt_tokens: int | None = None,
        user_query_tokens: int | None = None,
    ) -> int:
        """Calculate dynamic prompt capacity available for conversation history.

        Reserves output tokens first, then accounts for system prompt, user query,
        and retrieved PDF context, applying a conservative safety margin.

        Explicit None checks distinguish between unsupplied parameters (which fall back
        to configured reserves) and explicitly supplied zeros (which allocate 0 tokens).
        """
        sys_tokens = (
            self.system_prompt_reserve_tokens
            if system_prompt_tokens is None
            else max(0, int(system_prompt_tokens))
        )
        doc_tokens = (
            self.doc_context_reserve_tokens
            if doc_context_tokens is None
            else max(0, int(doc_context_tokens))
        )
        query_tokens = (
            0
            if user_query_tokens is None
            else max(0, int(user_query_tokens))
        )
        non_history_tokens = (
            sys_tokens
            + doc_tokens
            + query_tokens
            + self.output_reserve_tokens
        )

        raw_remaining = max(0, self.context_window_tokens - non_history_tokens)
        safe_remaining = int(
            raw_remaining * (1.0 - max(0.0, min(0.5, self.safety_margin_ratio)))
        )
        return max(0, min(self.max_history_tokens, safe_remaining))


@dataclass
class ConversationThread:
    """Document-scoped conversational thread with message history and summary memory."""

    document_id: str
    conversation_id: str = field(default_factory=new_uuid)
    title: str = "New Conversation"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    messages: list[ChatMessage] = field(default_factory=list)
    summary_memory: str = ""
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.document_id = str(self.document_id).strip()
        if not self.document_id:
            raise ValueError("document_id cannot be empty")

        if not is_valid_uuid(self.conversation_id):
            self.conversation_id = new_uuid()

        if not isinstance(self.title, str) or not self.title.strip():
            self.title = "New Conversation"
        else:
            self.title = self.title.strip()

        if not isinstance(self.messages, list):
            self.messages = []

        if not isinstance(self.summary_memory, str):
            self.summary_memory = str(self.summary_memory or "")

    def add_message(
        self,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Append a message and update conversation metadata."""
        msg = ChatMessage(
            role=role,
            content=content,
            citations=citations or [],
            metadata=metadata or {},
        )
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc).isoformat()

        # Automatically derive title from first user query if still generic
        if (
            self.title == "New Conversation"
            and role == "user"
            and content.strip()
        ):
            clean_title = re.sub(r"\s+", " ", content).strip()
            self.title = (
                clean_title[:45] + "..." if len(clean_title) > 48 else clean_title
            )

        return msg

    def add_user_message(self, content: str) -> ChatMessage:
        """Helper to append a user message."""
        return self.add_message("user", content)

    def add_assistant_message(
        self,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Helper to append an assistant message with citations."""
        return self.add_message(
            "assistant", content, citations=citations, metadata=metadata
        )

    def get_total_tokens(self) -> int:
        """Return total estimated tokens in messages plus summary memory."""
        msg_tokens = sum(m.token_count for m in self.messages)
        sum_tokens = (
            estimate_tokens(self.summary_memory) if self.summary_memory else 0
        )
        return msg_tokens + sum_tokens

    def get_turn_count(self) -> int:
        """Return number of user turns in thread."""
        return sum(1 for m in self.messages if m.role == "user")

    def clear(self) -> None:
        """Reset conversation messages and summary memory while preserving identity."""
        self.messages.clear()
        self.summary_memory = ""
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize thread to JSON-compatible dictionary."""
        return {
            "version": self.version,
            "conversation_id": self.conversation_id,
            "document_id": self.document_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "summary_memory": self.summary_memory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationThread:
        """Construct ConversationThread from dictionary, handling corrupted or missing keys."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict for ConversationThread, got {type(data)}")

        doc_id = str(data.get("document_id", "")).strip()
        if not doc_id:
            raise ValueError("Missing or empty 'document_id' in serialized conversation")

        cid = str(data.get("conversation_id", ""))
        if not is_valid_uuid(cid):
            cid = new_uuid()

        title = str(data.get("title", "New Conversation"))
        created_at = str(
            data.get("created_at") or datetime.now(timezone.utc).isoformat()
        )
        updated_at = str(
            data.get("updated_at") or datetime.now(timezone.utc).isoformat()
        )
        summary = str(data.get("summary_memory", ""))
        version = int(data.get("version", SCHEMA_VERSION))

        raw_msgs = data.get("messages", [])
        messages: list[ChatMessage] = []
        if isinstance(raw_msgs, list):
            for item in raw_msgs:
                if isinstance(item, dict):
                    try:
                        messages.append(ChatMessage.from_dict(item))
                    except Exception as err:
                        logger.warning("Skipping malformed message entry: %s", err)

        return cls(
            document_id=doc_id,
            conversation_id=cid,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            messages=messages,
            summary_memory=summary,
            version=version,
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize thread to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        """Generate a clean, readable Markdown transcript for study export."""
        lines = [
            f"# Study Chat Transcript — {self.title}",
            "",
            f"- **Document ID:** `{self.document_id}`",
            f"- **Conversation ID:** `{self.conversation_id}`",
            f"- **Created:** {self.created_at}",
            f"- **Last Updated:** {self.updated_at}",
            f"- **Total Messages:** {len(self.messages)}",
            "",
            "---",
            "",
        ]

        if self.summary_memory:
            lines.extend(
                [
                    "### 🧠 Summary of Earlier Discussion",
                    "",
                    self.summary_memory.strip(),
                    "",
                    "---",
                    "",
                ]
            )

        if not self.messages:
            lines.append("*No messages in this conversation yet.*")
            return "\n".join(lines)

        for idx, msg in enumerate(self.messages, start=1):
            role_badge = (
                "👤 **User**"
                if msg.role == "user"
                else ("🤖 **Assistant**" if msg.role == "assistant" else "⚙️ **System**")
            )
            lines.append(f"### Turn {idx} — {role_badge}")
            lines.append(f"*{msg.timestamp}*")
            lines.append("")
            lines.append(msg.content.strip())
            lines.append("")

            if msg.citations:
                lines.append("**Sources Cited:**")
                for cite in msg.citations:
                    page = cite.get("page", cite.get("page_range", "Unknown"))
                    fname = cite.get("filename", "")
                    score = cite.get("score")
                    score_str = f" (similarity: {score:.3f})" if score is not None else ""
                    chunk_id = cite.get("chunk_id", "")
                    lines.append(f"- **Source (Page {page}, {fname}){score_str}:** `{chunk_id}`")
                    text_preview = cite.get("text", "").strip()
                    if text_preview:
                        quote = "\n> ".join(text_preview.splitlines()[:3])
                        lines.append(f"> {quote}")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)


def deterministic_extractive_summary(
    messages: list[ChatMessage],
    existing_summary: str = "",
    max_tokens: int = 300,
) -> str:
    """Pure-Python deterministic extractive summarizer for older conversation turns.

    Used when no LLM summarizer is supplied or if LLM summarization fails.
    Extracts core question themes, key terminology, and unresolved queries
    without losing historical continuity or throwing exceptions.
    """
    if not messages and not existing_summary:
        return ""

    key_points: list[str] = []
    seen_points: set[str] = set()

    # Retain points from existing summary if already present
    if existing_summary.strip():
        for line in existing_summary.strip().splitlines():
            clean = line.strip().lstrip("-* ").strip()
            if clean and clean not in seen_points:
                seen_points.add(clean)
                key_points.append(clean)

    for msg in messages:
        content = msg.content.strip()
        if not content:
            continue

        if msg.role == "user":
            # Extract question or core inquiry
            q_clean = re.sub(r"^(can you|please|tell me|what is|how do|explain)\s+", "", content, flags=re.IGNORECASE).strip()
            if q_clean:
                q_point = f"User inquired about: {q_clean[:100]}"
                if q_point not in seen_points:
                    seen_points.add(q_point)
                    key_points.append(q_point)
        elif msg.role == "assistant":
            # Extract major capitalized concepts, terms, and cited pages
            cited_pages: list[str] = []
            for c in msg.citations:
                p = c.get("page")
                if p is not None and str(p) not in cited_pages:
                    cited_pages.append(str(p))

            # Look for definition lines or core sentences
            first_sentence = re.split(r"[.!?]\s+", content)[0].strip()
            if len(first_sentence) > 15:
                cite_suffix = f" (citing pp. {', '.join(cited_pages)})" if cited_pages else ""
                a_point = f"Key takeaway: {first_sentence[:120]}{cite_suffix}"
                if a_point not in seen_points:
                    seen_points.add(a_point)
                    key_points.append(a_point)

    # Filter to fit max_tokens
    final_lines: list[str] = []
    current_tokens = estimate_tokens("Earlier Discussion Summary:\n")
    for point in key_points:
        pt_tokens = estimate_tokens(f"- {point}\n")
        if current_tokens + pt_tokens > max_tokens:
            break
        final_lines.append(f"- {point}")
        current_tokens += pt_tokens

    if not final_lines:
        return existing_summary[:max_tokens * 4]

    return "Earlier Discussion Summary:\n" + "\n".join(final_lines)


def _safe_truncate_message(msg: ChatMessage, token_budget: int) -> ChatMessage:
    """Return a safely truncated shallow copy of an oversized message to fit token_budget.

    The original message in ConversationThread is preserved untouched.
    """
    if msg.token_count <= token_budget:
        return msg

    suffix = " ... [truncated to fit context budget]"
    suffix_tokens = estimate_tokens(suffix)
    available_content_tokens = max(5, token_budget - suffix_tokens)
    target_chars = max(20, int(available_content_tokens * 3.5))

    truncated_content = msg.content[:target_chars].rstrip() + suffix
    new_tokens = estimate_tokens(truncated_content)
    while new_tokens > token_budget and target_chars > 20:
        target_chars -= 20
        truncated_content = msg.content[:target_chars].rstrip() + suffix
        new_tokens = estimate_tokens(truncated_content)

    return ChatMessage(
        role=msg.role,
        content=truncated_content,
        message_id=msg.message_id,
        timestamp=msg.timestamp,
        token_count=new_tokens,
        citations=deepcopy(msg.citations),
        metadata=deepcopy(msg.metadata),
    )


SummarizerCallable = Callable[[list[ChatMessage], str], str]


class SlidingWindowMemoryManager:
    """Manages sliding-window context assembly, token budgeting, and out-of-window summarization."""

    def __init__(
        self,
        config: ConversationMemoryConfig | None = None,
        summarizer: SummarizerCallable | None = None,
    ) -> None:
        self.config = config or ConversationMemoryConfig()
        self._summarizer = summarizer

    def build_prompt_history(
        self,
        thread: ConversationThread,
        available_budget: int | None = None,
    ) -> tuple[str, list[ChatMessage]]:
        """Construct prompt conversation history respecting token budget and turn boundaries.

        Guarantees:
        - Token budget is strictly respected (never exceeded).
        - Recent turns are preserved in chronological order.
        - Pair-awareness: Avoids retaining an orphan assistant answer at the start of
          the prompt history whose preceding user question was excluded by budget limits.
        - Safe oversized handling: If the single most recent message exceeds available
          budget, it is safely truncated without mutating the thread transcript.

        Returns:
            (summary_memory, recent_verbatim_messages)
        """
        budget = (
            available_budget
            if available_budget is not None
            else self.config.max_history_tokens
        )
        if budget <= 0 or not thread.messages:
            return "", []

        # 1. Budget summary memory first if present
        summary_text = (thread.summary_memory or "").strip()
        summary_tokens = estimate_tokens(summary_text) if summary_text else 0

        # If summary alone exceeds 40% of budget, truncate summary to leave room for recent turns
        max_summary_tokens = min(
            self.config.summary_max_tokens, int(budget * 0.40)
        )
        if summary_tokens > max_summary_tokens:
            # Deterministically trim summary
            summary_text = summary_text[: max_summary_tokens * 4].strip() + "..."
            summary_tokens = estimate_tokens(summary_text)

        remaining_budget = max(0, budget - summary_tokens)
        if remaining_budget <= 0:
            return summary_text, []

        # 2. Iterate backward through messages to include recent turns verbatim
        selected_rev: list[ChatMessage] = []
        accumulated_tokens = 0

        for idx, msg in enumerate(reversed(thread.messages)):
            if accumulated_tokens + msg.token_count <= remaining_budget:
                selected_rev.append(msg)
                accumulated_tokens += msg.token_count
            else:
                # If this is the newest message and it exceeds the entire remaining budget,
                # provide safe truncation so the prompt doesn't drop the latest interaction
                if idx == 0 and remaining_budget >= 20:
                    truncated = _safe_truncate_message(msg, remaining_budget)
                    if truncated.token_count <= remaining_budget:
                        selected_rev.append(truncated)
                        accumulated_tokens += truncated.token_count
                break

        # Restore chronological order
        selected_messages = list(reversed(selected_rev))

        # 3. Pair-Awareness: Prevent orphan assistant message at the start of history
        # If the oldest message included is an assistant response whose preceding user query
        # was cut off due to budget limits, drop that orphan assistant answer.
        while selected_messages and selected_messages[0].role == "assistant":
            first_mid = selected_messages[0].message_id
            orig_idx = next(
                (i for i, m in enumerate(thread.messages) if m.message_id == first_mid),
                -1,
            )
            # If there was a preceding user message in thread that couldn't fit, drop the orphan
            if orig_idx > 0 and thread.messages[orig_idx - 1].role == "user":
                dropped = selected_messages.pop(0)
                logger.debug(
                    "Dropped orphan assistant message %s to preserve pair-aware turn boundary",
                    dropped.message_id,
                )
            else:
                break

        return summary_text, selected_messages

    def update_memory(
        self,
        thread: ConversationThread,
        force_summary: bool = False,
    ) -> bool:
        """Compress older turns outside the recent window into summary memory.

        Uses the injected LLM summarizer if provided; gracefully falls back to
        deterministic extractive summarization if unavailable or upon error.
        Never destroys thread history.
        """
        if not thread.messages:
            return False

        # Identify messages outside the recent turn threshold
        # A turn is a user-assistant pair (approx 2 messages)
        recent_msg_count = max(2, self.config.min_recent_turns * 2)
        total_tokens = thread.get_total_tokens()

        needs_summary = force_summary or (
            len(thread.messages) > recent_msg_count
            and total_tokens > self.config.max_history_tokens
        )
        if not needs_summary:
            return False

        # Older messages are everything except the last recent_msg_count
        older_messages = thread.messages[:-recent_msg_count]
        if not older_messages:
            return False

        new_summary = ""
        # Try injected summarizer first
        if self._summarizer is not None:
            try:
                new_summary = self._summarizer(
                    older_messages, thread.summary_memory
                )
            except Exception as exc:
                logger.warning(
                    "Injected chat summarizer failed; falling back to deterministic summary: %s",
                    exc,
                )
                new_summary = ""

        # Fallback to deterministic extractive summary if LLM summarizer produced nothing or failed
        if not new_summary or not new_summary.strip():
            new_summary = deterministic_extractive_summary(
                older_messages,
                existing_summary=thread.summary_memory,
                max_tokens=self.config.summary_max_tokens,
            )

        if new_summary and new_summary.strip():
            thread.summary_memory = new_summary.strip()
            thread.updated_at = datetime.now(timezone.utc).isoformat()
            return True

        return False

    def prune_older_messages(
        self,
        thread: ConversationThread,
        keep_last_n_turns: int | None = None,
    ) -> int:
        """Opt-in destructive transcript trimming for memory-constrained environments.

        IMPORTANT ARCHITECTURAL NOTE:
        This operation is NEVER called automatically by update_memory() or build_prompt_history().
        By default, Studyastra preserves the complete original transcript in thread.messages
        for UI chat rendering, Markdown export, and JSON export.
        This method is an optional utility provided exclusively for callers that explicitly
        request hard in-memory truncation of the thread's underlying transcript nodes.

        Returns:
            Number of messages removed.
        """
        n_turns = keep_last_n_turns or self.config.min_recent_turns
        keep_count = max(2, n_turns * 2)
        if len(thread.messages) <= keep_count:
            return 0

        prune_count = len(thread.messages) - keep_count
        del thread.messages[:prune_count]
        thread.updated_at = datetime.now(timezone.utc).isoformat()
        return prune_count

        prune_count = len(thread.messages) - keep_count
        del thread.messages[:prune_count]
        thread.updated_at = datetime.now(timezone.utc).isoformat()
        return prune_count


class ConversationManager:
    """Manages multiple conversation threads per document with session-level isolation."""

    def __init__(self) -> None:
        # Map: document_id -> dict of (conversation_id -> ConversationThread)
        self._threads: dict[str, dict[str, ConversationThread]] = {}

    @property
    def threads(self) -> dict[str, dict[str, ConversationThread]]:
        """Access internal threads map."""
        return self._threads

    def create_thread(
        self,
        document_id: str,
        title: str | None = None,
    ) -> ConversationThread:
        """Create a fresh conversation thread with a newly generated UUID."""
        return self.get_or_create_thread(document_id=document_id, title=title)

    def get_or_create_thread(
        self,
        document_id: str,
        conversation_id: str | None = None,
        title: str | None = None,
    ) -> ConversationThread:
        """Retrieve existing thread or create a new isolated conversation."""
        doc_id = str(document_id).strip()
        if not doc_id:
            raise ValueError("document_id cannot be empty")

        doc_threads = self._threads.setdefault(doc_id, {})

        if conversation_id and is_valid_uuid(conversation_id):
            if conversation_id in doc_threads:
                return doc_threads[conversation_id]
            # Create with specified UUID
            new_thread = ConversationThread(
                document_id=doc_id,
                conversation_id=conversation_id,
                title=title or "New Conversation",
            )
            doc_threads[conversation_id] = new_thread
            return new_thread

        # Create fresh thread with new UUID
        new_thread = ConversationThread(
            document_id=doc_id,
            title=title or "New Conversation",
        )
        doc_threads[new_thread.conversation_id] = new_thread
        return new_thread

    def get_thread(
        self, document_id: str, conversation_id: str
    ) -> ConversationThread | None:
        """Get thread by document_id and conversation_id."""
        return self._threads.get(str(document_id).strip(), {}).get(
            str(conversation_id)
        )

    def list_threads(self, document_id: str) -> list[ConversationThread]:
        """List all active threads for a specific document in descending update order."""
        threads = list(self._threads.get(str(document_id).strip(), {}).values())
        threads.sort(key=lambda t: t.updated_at, reverse=True)
        return threads

    def delete_thread(self, document_id: str, conversation_id: str) -> bool:
        """Delete a single conversation thread."""
        doc_threads = self._threads.get(str(document_id).strip())
        if doc_threads and str(conversation_id) in doc_threads:
            del doc_threads[str(conversation_id)]
            return True
        return False

    def clear_document_threads(self, document_id: str) -> None:
        """Remove all conversations associated with a document (e.g. on PDF deletion)."""
        doc_id = str(document_id).strip()
        self._threads.pop(doc_id, None)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all document threads to dictionary."""
        serialized: dict[str, Any] = {}
        for doc_id, threads in self._threads.items():
            serialized[doc_id] = {
                cid: thread.to_dict() for cid, thread in threads.items()
            }
        return serialized

    def load_dict(self, data: dict[str, Any]) -> None:
        """Restore all document threads from dictionary with error resilience."""
        if not isinstance(data, dict):
            logger.warning("Invalid data type for ConversationManager.load_dict; resetting.")
            self._threads.clear()
            return

        restored: dict[str, dict[str, ConversationThread]] = {}
        for doc_id, threads_map in data.items():
            if not isinstance(threads_map, dict):
                continue
            doc_threads: dict[str, ConversationThread] = {}
            for cid, thread_dict in threads_map.items():
                if isinstance(thread_dict, dict):
                    try:
                        th = ConversationThread.from_dict(thread_dict)
                        doc_threads[th.conversation_id] = th
                    except Exception as err:
                        logger.warning("Failed to restore thread %s: %s", cid, err)
            if doc_threads:
                restored[doc_id] = doc_threads

        self._threads = restored
