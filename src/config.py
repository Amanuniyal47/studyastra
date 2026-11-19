import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# LLM provider configuration (kept independent of embeddings & vector store)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Summarization configuration (token budget accounts for prompts, headers, and text)
SUMMARY_MAX_BATCH_TOKENS = int(os.getenv("SUMMARY_MAX_BATCH_TOKENS", "2400"))
SUMMARY_TOPIC_TOP_K = int(os.getenv("SUMMARY_TOPIC_TOP_K", "6"))

# Practice question generation configuration
QUESTION_MAX_BATCH_TOKENS = int(os.getenv("QUESTION_MAX_BATCH_TOKENS", "2400"))
QUESTION_DEFAULT_COUNT = int(os.getenv("QUESTION_DEFAULT_COUNT", "5"))
QUESTION_MAX_COUNT = int(os.getenv("QUESTION_MAX_COUNT", "20"))
QUESTION_TOPIC_TOP_K = int(os.getenv("QUESTION_TOPIC_TOP_K", "6"))
QUESTION_MAX_RETRIES = int(os.getenv("QUESTION_MAX_RETRIES", "1"))
QUESTION_MAX_OUTPUT_TOKENS = int(os.getenv("QUESTION_MAX_OUTPUT_TOKENS", "1400"))
QUESTION_MAX_PREVIOUS_STEMS = int(os.getenv("QUESTION_MAX_PREVIOUS_STEMS", "15"))

# Interactive quiz evaluation configuration
QUIZ_SHORT_ANSWER_EVALUATOR = os.getenv("QUIZ_SHORT_ANSWER_EVALUATOR", "llm")
QUIZ_SHORT_ANSWER_CORRECT_THRESHOLD = float(
    os.getenv("QUIZ_SHORT_ANSWER_CORRECT_THRESHOLD", "0.75")
)
QUIZ_SHORT_ANSWER_PARTIAL_THRESHOLD = float(
    os.getenv("QUIZ_SHORT_ANSWER_PARTIAL_THRESHOLD", "0.40")
)

# Local embedding configuration (independent of LLM provider)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Token-based chunking configuration
# Default 220/40 fits within all-MiniLM-L6-v2's 256-token max sequence length.
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "220"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))

# Backwards-compatible aliases
CHUNK_SIZE = CHUNK_SIZE_TOKENS
CHUNK_OVERLAP = CHUNK_OVERLAP_TOKENS

# Retrieval & persistent vector storage configuration
TOP_K = int(os.getenv("TOP_K", "3"))
VECTOR_STORE_DIR = Path(
    os.getenv("VECTOR_STORE_DIR", str(PROJECT_ROOT / "data" / "vector_store"))
)
