import os
from pathlib import Path
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _safe_int(val: str | None, default: int) -> int:
    """Safely cast string to int, falling back to default if absent or invalid."""
    if val is None or not str(val).strip():
        return default
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return default


def _safe_float(val: str | None, default: float) -> float:
    """Safely cast string to float, falling back to default if absent or invalid."""
    if val is None or not str(val).strip():
        return default
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default


def _safe_bool(val: str | None, default: bool) -> bool:
    """Safely parse truthy string to boolean, falling back to default if absent."""
    if val is None or not str(val).strip():
        return default
    return str(val).strip().lower() in ("true", "1", "yes")


# LLM provider configuration (kept independent of embeddings & vector store)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_TIMEOUT_SECONDS = _safe_float(os.getenv("LLM_TIMEOUT_SECONDS"), 60.0)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key") or ""
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Ollama local LLM configuration (Qwen2.5 7B)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TEMPERATURE = _safe_float(os.getenv("OLLAMA_TEMPERATURE"), 0.2)
OLLAMA_MAX_TOKENS = _safe_int(os.getenv("OLLAMA_MAX_TOKENS"), 2048)
OLLAMA_CONTEXT_WINDOW = _safe_int(os.getenv("OLLAMA_CONTEXT_WINDOW"), 4096)
OLLAMA_TIMEOUT_SECONDS = _safe_float(os.getenv("OLLAMA_TIMEOUT_SECONDS"), 60.0)
OLLAMA_CONNECT_TIMEOUT_SECONDS = _safe_float(
    os.getenv("OLLAMA_CONNECT_TIMEOUT_SECONDS"), 10.0
)
OLLAMA_READ_TIMEOUT_SECONDS = _safe_float(
    os.getenv("OLLAMA_READ_TIMEOUT_SECONDS"), 120.0
)
OLLAMA_MAX_RETRIES = _safe_int(os.getenv("OLLAMA_MAX_RETRIES"), 2)
OLLAMA_RETRY_BACKOFF_FACTOR = _safe_float(
    os.getenv("OLLAMA_RETRY_BACKOFF_FACTOR"), 0.5
)
OLLAMA_RETRY_BACKOFF_MAX_SECONDS = _safe_float(
    os.getenv("OLLAMA_RETRY_BACKOFF_MAX_SECONDS"), 3.0
)
OLLAMA_STREAM_IDLE_TIMEOUT_SECONDS = _safe_float(
    os.getenv("OLLAMA_STREAM_IDLE_TIMEOUT_SECONDS"), 30.0
)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
OLLAMA_HEALTH_CHECK_TTL_SECONDS = _safe_float(
    os.getenv("OLLAMA_HEALTH_CHECK_TTL_SECONDS"), 20.0
)

# Summarization configuration (token budget accounts for prompts, headers, and text)
SUMMARY_MAX_BATCH_TOKENS = _safe_int(os.getenv("SUMMARY_MAX_BATCH_TOKENS"), 2400)
SUMMARY_TOPIC_TOP_K = _safe_int(os.getenv("SUMMARY_TOPIC_TOP_K"), 6)
MAX_REDUCTION_LEVELS = _safe_int(os.getenv("MAX_REDUCTION_LEVELS"), 6)

# PDF text extraction configuration
PDF_LAYOUT_SORT_ENABLED = _safe_bool(os.getenv("PDF_LAYOUT_SORT_ENABLED"), True)

# Study notes configuration
NOTES_MAX_BATCH_TOKENS = _safe_int(os.getenv("NOTES_MAX_BATCH_TOKENS"), 2400)
NOTES_TOPIC_TOP_K = _safe_int(os.getenv("NOTES_TOPIC_TOP_K"), 6)

# Flashcard generation configuration
FLASHCARD_MAX_BATCH_TOKENS = _safe_int(os.getenv("FLASHCARD_MAX_BATCH_TOKENS"), 2400)
FLASHCARD_DEFAULT_COUNT = _safe_int(os.getenv("FLASHCARD_DEFAULT_COUNT"), 6)
FLASHCARD_MAX_COUNT = _safe_int(os.getenv("FLASHCARD_MAX_COUNT"), 20)
FLASHCARD_TOPIC_TOP_K = _safe_int(os.getenv("FLASHCARD_TOPIC_TOP_K"), 6)
FLASHCARD_MAX_RETRIES = _safe_int(os.getenv("FLASHCARD_MAX_RETRIES"), 1)
FLASHCARD_MAX_OUTPUT_TOKENS = _safe_int(os.getenv("FLASHCARD_MAX_OUTPUT_TOKENS"), 1400)

# Follow-up question & concept explanation configuration
FOLLOW_UP_DEFAULT_COUNT = _safe_int(os.getenv("FOLLOW_UP_DEFAULT_COUNT"), 3)
EXPLANATION_MAX_TOKENS = _safe_int(os.getenv("EXPLANATION_MAX_TOKENS"), 800)
EXPLANATION_TOP_K = _safe_int(os.getenv("EXPLANATION_TOP_K"), 4)

# Practice question generation configuration
QUESTION_MAX_BATCH_TOKENS = _safe_int(os.getenv("QUESTION_MAX_BATCH_TOKENS"), 2400)
QUESTION_DEFAULT_COUNT = _safe_int(os.getenv("QUESTION_DEFAULT_COUNT"), 5)
QUESTION_MAX_COUNT = _safe_int(os.getenv("QUESTION_MAX_COUNT"), 20)
QUESTION_TOPIC_TOP_K = _safe_int(os.getenv("QUESTION_TOPIC_TOP_K"), 6)
QUESTION_MAX_RETRIES = _safe_int(os.getenv("QUESTION_MAX_RETRIES"), 1)
QUESTION_MAX_OUTPUT_TOKENS = _safe_int(os.getenv("QUESTION_MAX_OUTPUT_TOKENS"), 1400)
QUESTION_MAX_PREVIOUS_STEMS = _safe_int(os.getenv("QUESTION_MAX_PREVIOUS_STEMS"), 15)

# Interactive quiz evaluation configuration
QUIZ_SHORT_ANSWER_EVALUATOR = os.getenv("QUIZ_SHORT_ANSWER_EVALUATOR", "llm")
QUIZ_SHORT_ANSWER_CORRECT_THRESHOLD = _safe_float(
    os.getenv("QUIZ_SHORT_ANSWER_CORRECT_THRESHOLD"), 0.75
)
QUIZ_SHORT_ANSWER_PARTIAL_THRESHOLD = _safe_float(
    os.getenv("QUIZ_SHORT_ANSWER_PARTIAL_THRESHOLD"), 0.40
)

# Local embedding configuration (independent of LLM provider)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = _safe_int(os.getenv("EMBEDDING_BATCH_SIZE"), 32)

# Token-based chunking configuration
# Default 220/40 fits within all-MiniLM-L6-v2's 256-token max sequence length.
CHUNK_SIZE_TOKENS = _safe_int(os.getenv("CHUNK_SIZE_TOKENS"), 220)
CHUNK_OVERLAP_TOKENS = _safe_int(os.getenv("CHUNK_OVERLAP_TOKENS"), 40)

# Backwards-compatible aliases
CHUNK_SIZE = CHUNK_SIZE_TOKENS
CHUNK_OVERLAP = CHUNK_OVERLAP_TOKENS

# Retrieval & persistent vector storage configuration
TOP_K = _safe_int(os.getenv("TOP_K"), 3)
VECTOR_STORE_DIR = Path(
    os.getenv("VECTOR_STORE_DIR", str(PROJECT_ROOT / "data" / "vector_store"))
)
VECTOR_STORE_CLEANUP_STAGING = _safe_bool(
    os.getenv("VECTOR_STORE_CLEANUP_STAGING"), True
)
VECTOR_STORE_STAGING_MAX_AGE_SECONDS = _safe_float(
    os.getenv("VECTOR_STORE_STAGING_MAX_AGE_SECONDS"), 3600.0
)

# Study timers and statistics configuration
STUDY_STATS_FILE = Path(
    os.getenv("STUDY_STATS_FILE", str(PROJECT_ROOT / "data" / "study_stats.json"))
)
POMODORO_DEFAULT_FOCUS_MINUTES = _safe_int(os.getenv("POMODORO_DEFAULT_FOCUS_MINUTES"), 25)
POMODORO_DEFAULT_SHORT_BREAK_MINUTES = _safe_int(
    os.getenv("POMODORO_DEFAULT_SHORT_BREAK_MINUTES"), 5
)
POMODORO_DEFAULT_LONG_BREAK_MINUTES = _safe_int(
    os.getenv("POMODORO_DEFAULT_LONG_BREAK_MINUTES"), 15
)
POMODORO_DEFAULT_SESSIONS_PER_CYCLE = _safe_int(
    os.getenv("POMODORO_DEFAULT_SESSIONS_PER_CYCLE"), 4
)

# Countdown study challenges configuration
CHALLENGE_PRESET_DURATIONS = [5, 10, 15, 25, 30, 45, 60]
CHALLENGE_DEFAULT_MINUTES = _safe_int(os.getenv("CHALLENGE_DEFAULT_MINUTES"), 25)
CHALLENGE_MIN_MINUTES = _safe_int(os.getenv("CHALLENGE_MIN_MINUTES"), 1)
CHALLENGE_MAX_MINUTES = _safe_int(os.getenv("CHALLENGE_MAX_MINUTES"), 180)

# Timed interactive quiz configuration
TIMED_QUIZ_PRESET_DURATIONS = [5, 10, 15, 20, 30]
TIMED_QUIZ_DEFAULT_MINUTES = _safe_int(os.getenv("TIMED_QUIZ_DEFAULT_MINUTES"), 10)
TIMED_QUIZ_MIN_MINUTES = _safe_int(os.getenv("TIMED_QUIZ_MIN_MINUTES"), 1)
TIMED_QUIZ_MAX_MINUTES = _safe_int(os.getenv("TIMED_QUIZ_MAX_MINUTES"), 60)
TIMED_QUIZ_WARNING_SECONDS = _safe_int(os.getenv("TIMED_QUIZ_WARNING_SECONDS"), 60)

# Multi-turn conversational PDF chatbot configuration (Stage 10)
CHAT_CONTEXT_WINDOW_TOKENS = _safe_int(os.getenv("CHAT_CONTEXT_WINDOW_TOKENS"), 4096)
CHAT_MAX_HISTORY_TOKENS = _safe_int(os.getenv("CHAT_MAX_HISTORY_TOKENS"), 1500)
CHAT_OUTPUT_RESERVE_TOKENS = _safe_int(os.getenv("CHAT_OUTPUT_RESERVE_TOKENS"), 1000)
CHAT_SYSTEM_PROMPT_RESERVE_TOKENS = _safe_int(
    os.getenv("CHAT_SYSTEM_PROMPT_RESERVE_TOKENS"), 400
)
CHAT_DOC_CONTEXT_RESERVE_TOKENS = _safe_int(
    os.getenv("CHAT_DOC_CONTEXT_RESERVE_TOKENS"), 1000
)
CHAT_MIN_RECENT_TURNS = _safe_int(os.getenv("CHAT_MIN_RECENT_TURNS"), 2)
CHAT_SUMMARY_MAX_TOKENS = _safe_int(os.getenv("CHAT_SUMMARY_MAX_TOKENS"), 300)
CHAT_SAFETY_MARGIN_RATIO = _safe_float(os.getenv("CHAT_SAFETY_MARGIN_RATIO"), 0.10)
CHAT_REWRITE_ENABLED = _safe_bool(os.getenv("CHAT_REWRITE_ENABLED"), True)
CHAT_REWRITE_TIMEOUT_SECONDS = _safe_float(
    os.getenv("CHAT_REWRITE_TIMEOUT_SECONDS"), 5.0
)
CHAT_TOP_K = _safe_int(os.getenv("CHAT_TOP_K"), 4)


def validate_production_config() -> tuple[bool, list[str], list[str]]:
    """Validate application configuration against production readiness rules.

    Returns:
        tuple[bool, list[str], list[str]]: (is_valid, errors, warnings)
        - is_valid: True if no critical errors were detected.
        - errors: List of actionable failure messages.
        - warnings: List of advisory non-blocking messages.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Enforce LLM_PROVIDER == "ollama"
    normalized_provider = (LLM_PROVIDER or "").strip().lower()
    if normalized_provider != "ollama":
        if normalized_provider == "groq":
            errors.append(
                "LLM_PROVIDER='groq' is prohibited in production. StudyAstra operates strictly "
                "local-only using Ollama (qwen2.5:7b). Set LLM_PROVIDER=ollama in .env."
            )
        else:
            errors.append(
                f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Production target strictly requires 'ollama'."
            )

    # 2. Validate OLLAMA_BASE_URL
    if not OLLAMA_BASE_URL:
        errors.append("OLLAMA_BASE_URL cannot be empty.")
    else:
        parsed_url = urllib.parse.urlparse(OLLAMA_BASE_URL)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            errors.append(
                f"Invalid OLLAMA_BASE_URL '{OLLAMA_BASE_URL}': must use http or https and specify a valid host."
            )

    # 3. Validate numeric bounds
    # Check raw env values if present to catch out-of-bounds before safe-fallback
    raw_temp_str = os.getenv("OLLAMA_TEMPERATURE")
    if raw_temp_str is not None and str(raw_temp_str).strip():
        try:
            raw_temp = float(str(raw_temp_str).strip())
            if not (0.0 <= raw_temp <= 2.0):
                errors.append(f"OLLAMA_TEMPERATURE must be between 0.0 and 2.0 (got {raw_temp}).")
        except ValueError:
            errors.append(f"OLLAMA_TEMPERATURE '{raw_temp_str}' is not a valid float.")
    elif not (0.0 <= OLLAMA_TEMPERATURE <= 2.0):
        errors.append(f"OLLAMA_TEMPERATURE must be between 0.0 and 2.0 (got {OLLAMA_TEMPERATURE}).")

    raw_tokens_str = os.getenv("OLLAMA_MAX_TOKENS")
    if raw_tokens_str is not None and str(raw_tokens_str).strip():
        try:
            raw_tokens = int(str(raw_tokens_str).strip())
            if raw_tokens < 64:
                errors.append(f"OLLAMA_MAX_TOKENS must be >= 64 (got {raw_tokens}).")
        except ValueError:
            errors.append(f"OLLAMA_MAX_TOKENS '{raw_tokens_str}' is not a valid integer.")
    elif OLLAMA_MAX_TOKENS < 64:
        errors.append(f"OLLAMA_MAX_TOKENS must be >= 64 (got {OLLAMA_MAX_TOKENS}).")

    if CHUNK_SIZE_TOKENS < 50:
        errors.append(f"CHUNK_SIZE_TOKENS must be >= 50 (got {CHUNK_SIZE_TOKENS}).")

    if CHUNK_OVERLAP_TOKENS < 0:
        errors.append(f"CHUNK_OVERLAP_TOKENS must be >= 0 (got {CHUNK_OVERLAP_TOKENS}).")
    elif CHUNK_OVERLAP_TOKENS >= CHUNK_SIZE_TOKENS:
        errors.append(
            f"CHUNK_OVERLAP_TOKENS ({CHUNK_OVERLAP_TOKENS}) must be strictly less than "
            f"CHUNK_SIZE_TOKENS ({CHUNK_SIZE_TOKENS})."
        )

    if EMBEDDING_BATCH_SIZE < 1:
        errors.append(f"EMBEDDING_BATCH_SIZE must be >= 1 (got {EMBEDDING_BATCH_SIZE}).")

    if TOP_K < 1:
        errors.append(f"TOP_K must be >= 1 (got {TOP_K}).")

    # 4. Validate storage paths non-destructively
    # Check parent directory of vector store
    vs_parent = VECTOR_STORE_DIR if VECTOR_STORE_DIR.exists() else VECTOR_STORE_DIR.parent
    if vs_parent.exists():
        if not os.access(vs_parent, os.W_OK):
            errors.append(f"Vector store parent directory '{vs_parent}' is not writable.")
    else:
        # Check first existing ancestor
        curr = vs_parent
        while not curr.exists() and curr != curr.parent:
            curr = curr.parent
        if not os.access(curr, os.W_OK):
            errors.append(f"Storage ancestor path '{curr}' is not writable for vector store.")

    # Check parent directory of study stats file
    stats_parent = STUDY_STATS_FILE.parent
    if stats_parent.exists():
        if not os.access(stats_parent, os.W_OK):
            errors.append(f"Study stats parent directory '{stats_parent}' is not writable.")
    else:
        curr = stats_parent
        while not curr.exists() and curr != curr.parent:
            curr = curr.parent
        if not os.access(curr, os.W_OK):
            errors.append(f"Storage ancestor path '{curr}' is not writable for study stats.")

    # 5. Check for advisory warnings
    if os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key"):
        warnings.append(
            "GROQ_API_KEY is present in environment, but StudyAstra operates local-only; Groq is ignored."
        )

    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        try:
            for item in data_dir.iterdir():
                if item.is_dir() and item.name.startswith("data.quarantine"):
                    warnings.append(f"Leftover quarantine folder found: '{item.name}'. Operator review recommended.")
        except Exception:
            pass

    if (PROJECT_ROOT / "data.old").exists():
        warnings.append("Leftover 'data.old' directory found from a previous restore. Operator review recommended.")

    return (len(errors) == 0, errors, warnings)
