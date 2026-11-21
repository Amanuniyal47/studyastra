"""Unit tests for Stage 10.2: Conversational Retrieval, Query Rewriting, and Grounded Prompts."""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

import src.config as config
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
from src.generation.llm_provider import BaseLLMProvider, LLMProviderError


class MockLLMProvider(BaseLLMProvider):
    """Simple mock LLM provider for testing query rewriting."""

    provider_name = "mock"

    def __init__(self, response: str = "", delay: float = 0.0, should_raise: bool = False):
        super().__init__(model_name="mock-model")
        self.response = response
        self.delay = delay
        self.should_raise = should_raise
        self.call_count = 0
        self.last_system_prompt = ""
        self.last_user_prompt = ""
        self.timeout: float | None = None

    def is_configured(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        if self.delay > 0:
            time.sleep(self.delay)
        if self.should_raise:
            raise LLMProviderError("Mock LLM generation failure")
        return self.response


class TestQueryRewriter(unittest.TestCase):
    """Test suite for query rewriting heuristic and execution."""

    def setUp(self):
        reset_rewrite_pool()
        self.thread = ConversationThread(document_id="doc_bio_1")
        self.thread.add_user_message("What is cellular respiration?")
        self.thread.add_assistant_message(
            "Cellular respiration is a metabolic pathway that breaks down glucose to produce ATP."
        )

    def tearDown(self):
        reset_rewrite_pool()

    def test_is_follow_up_candidate_empty_or_no_history(self):
        """Empty query or empty history must never be treated as follow-up candidates."""
        self.assertFalse(is_follow_up_candidate("", self.thread.messages))
        self.assertFalse(is_follow_up_candidate("   ", self.thread.messages))
        self.assertFalse(is_follow_up_candidate("How does it work?", []))

    def test_is_follow_up_candidate_standalone_questions(self):
        """Complete, self-contained questions must NOT trigger rewriting."""
        self.assertFalse(
            is_follow_up_candidate(
                "What is photosynthesis?",
                self.thread.messages,
            )
        )
        self.assertFalse(
            is_follow_up_candidate(
                "Define entropy in the context of thermodynamics.",
                self.thread.messages,
            )
        )
        self.assertFalse(
            is_follow_up_candidate(
                "What are the primary symptoms of diabetes according to the text?",
                self.thread.messages,
            )
        )
        self.assertFalse(
            is_follow_up_candidate(
                "How does the ribosome synthesize polypeptides from amino acids?",
                self.thread.messages,
            )
        )

    def test_is_follow_up_candidate_pronouns(self):
        """Questions with context-dependent pronouns must be identified as follow-up candidates."""
        self.assertTrue(
            is_follow_up_candidate("How does it produce ATP?", self.thread.messages)
        )
        self.assertTrue(
            is_follow_up_candidate("Why is this important?", self.thread.messages)
        )
        self.assertTrue(
            is_follow_up_candidate("What are its main stages?", self.thread.messages)
        )
        self.assertTrue(
            is_follow_up_candidate("Can they function without oxygen?", self.thread.messages)
        )
        self.assertTrue(
            is_follow_up_candidate("Tell me more about it.", self.thread.messages)
        )

    def test_is_follow_up_candidate_anaphoric_phrases(self):
        """Questions with explicit anaphoric references must be identified as follow-up candidates."""
        self.assertTrue(
            is_follow_up_candidate("What about the second method?", self.thread.messages)
        )
        self.assertTrue(
            is_follow_up_candidate(
                "Compare the former approach with the latter.", self.thread.messages
            )
        )
        self.assertTrue(
            is_follow_up_candidate(
                "Can you elaborate on that topic?", self.thread.messages
            )
        )
        self.assertTrue(
            is_follow_up_candidate(
                "Explain the previous concept in more detail.", self.thread.messages
            )
        )

    def test_is_follow_up_candidate_connectives_and_short_queries(self):
        """Short ellipsis queries or continuation connectives must be identified as follow-up candidates."""
        self.assertTrue(is_follow_up_candidate("Why?", self.thread.messages))
        self.assertTrue(is_follow_up_candidate("How so?", self.thread.messages))
        self.assertTrue(is_follow_up_candidate("And the cost?", self.thread.messages))
        self.assertTrue(is_follow_up_candidate("Also, why?", self.thread.messages))
        self.assertTrue(is_follow_up_candidate("Give an example.", self.thread.messages))

    def test_clean_rewritten_query(self):
        """Preamble, label prefixes, quotes, and backticks must be cleanly stripped."""
        self.assertEqual(
            clean_rewritten_query('Rewritten Query: "What is cellular respiration?"', "orig"),
            "What is cellular respiration?",
        )
        self.assertEqual(
            clean_rewritten_query("Standalone question: How is ATP synthesized?", "orig"),
            "How is ATP synthesized?",
        )
        self.assertEqual(
            clean_rewritten_query("`What are the stages of glycolysis?`", "orig"),
            "What are the stages of glycolysis?",
        )
        self.assertEqual(clean_rewritten_query("", "orig question"), "orig question")
        self.assertEqual(clean_rewritten_query("   ", "orig question"), "orig question")

    def test_build_rewrite_prompt(self):
        """Prompt construction includes recent history and follow-up query."""
        prompt = build_rewrite_prompt("Why does it happen?", self.thread.messages)
        self.assertIn("User: What is cellular respiration?", prompt)
        self.assertIn("Assistant: Cellular respiration is a metabolic pathway", prompt)
        self.assertIn("Follow-up Question: Why does it happen?", prompt)

    def test_rewrite_query_skipped_when_disabled(self):
        """When CHAT_REWRITE_ENABLED is False, provider is never called."""
        mock_provider = MockLLMProvider("How does cellular respiration produce ATP?")
        res, rewritten = rewrite_query(
            "How does it produce ATP?",
            self.thread.messages,
            provider=mock_provider,
            enabled=False,
        )
        self.assertEqual(res, "How does it produce ATP?")
        self.assertFalse(rewritten)
        self.assertEqual(mock_provider.call_count, 0)

    def test_rewrite_query_skipped_for_standalone_query(self):
        """Standalone questions bypass query rewriting to save latency and token usage."""
        mock_provider = MockLLMProvider("Whatever")
        res, rewritten = rewrite_query(
            "What are the primary symptoms of diabetes?",
            self.thread.messages,
            provider=mock_provider,
            enabled=True,
        )
        self.assertEqual(res, "What are the primary symptoms of diabetes?")
        self.assertFalse(rewritten)
        self.assertEqual(mock_provider.call_count, 0)

    def test_rewrite_query_executes_when_candidate(self):
        """Follow-up questions invoke provider and return rewritten standalone query."""
        mock_provider = MockLLMProvider(
            "How does cellular respiration produce adenosine triphosphate (ATP)?"
        )
        res, rewritten = rewrite_query(
            "How does it produce ATP?",
            self.thread.messages,
            provider=mock_provider,
            enabled=True,
        )
        self.assertEqual(
            res, "How does cellular respiration produce adenosine triphosphate (ATP)?"
        )
        self.assertTrue(rewritten)
        self.assertEqual(mock_provider.call_count, 1)

    def test_rewrite_query_timeout_fallback(self):
        """When provider exceeds timeout, gracefully fall back to original query without blocking."""
        mock_provider = MockLLMProvider(
            "Slow rewrite",
            delay=0.4,
        )
        start_time = time.time()
        res, rewritten = rewrite_query(
            "Why does it happen?",
            self.thread.messages,
            provider=mock_provider,
            timeout=0.1,
            enabled=True,
        )
        elapsed = time.time() - start_time
        self.assertEqual(res, "Why does it happen?")
        self.assertFalse(rewritten)
        self.assertLess(elapsed, 0.35)

    def test_rewrite_query_timeout_substantially_longer_delay(self):
        """Area 1 Audit: A provider taking 2.5s must abort within 0.2s without blocking caller."""
        mock_provider = MockLLMProvider(
            "Very slow rewrite output",
            delay=2.5,
        )
        start_time = time.time()
        res, rewritten = rewrite_query(
            "Why does it happen?",
            self.thread.messages,
            provider=mock_provider,
            timeout=0.1,
            enabled=True,
        )
        elapsed = time.time() - start_time
        self.assertEqual(res, "Why does it happen?")
        self.assertFalse(rewritten)
        self.assertLess(elapsed, 0.4)

    def test_rewrite_pipeline_usable_after_timeout(self):
        """Area 1 Audit: Pipeline must remain fully functional immediately after a timed-out call."""
        slow_provider = MockLLMProvider("Slow response", delay=1.5)
        res1, rewritten1 = rewrite_query(
            "Why does it happen?",
            self.thread.messages,
            provider=slow_provider,
            timeout=0.08,
            enabled=True,
        )
        self.assertEqual(res1, "Why does it happen?")
        self.assertFalse(rewritten1)

        fast_provider = MockLLMProvider("How does cellular respiration function?")
        res2, rewritten2 = rewrite_query(
            "How does it work?",
            self.thread.messages,
            provider=fast_provider,
            timeout=1.0,
            enabled=True,
        )
        self.assertEqual(res2, "How does cellular respiration function?")
        self.assertTrue(rewritten2)
        self.assertEqual(fast_provider.call_count, 1)

    def test_rewrite_repeated_timeouts_do_not_accumulate_blocking(self):
        """Area 1 Audit: Repeated timeouts must never accumulate blocking delays or freeze."""
        slow_provider = MockLLMProvider("Slow response", delay=1.0)
        start_time = time.time()
        for _ in range(4):
            res, rewritten = rewrite_query(
                "Why does it happen?",
                self.thread.messages,
                provider=slow_provider,
                timeout=0.05,
                enabled=True,
            )
            self.assertEqual(res, "Why does it happen?")
            self.assertFalse(rewritten)
        total_elapsed = time.time() - start_time
        self.assertLess(total_elapsed, 0.6)

    def test_rewrite_query_error_fallback(self):
        """When provider raises an exception, gracefully fall back to original query."""
        mock_provider = MockLLMProvider(should_raise=True)
        res, rewritten = rewrite_query(
            "Why does it happen?",
            self.thread.messages,
            provider=mock_provider,
            enabled=True,
        )
        self.assertEqual(res, "Why does it happen?")
        self.assertFalse(rewritten)

    def test_provider_receives_configured_timeout(self):
        """Area 1 Audit: The provider receives the configured timeout setting."""
        mock_provider = MockLLMProvider("Rewritten query")
        mock_provider.timeout = None
        res, rewritten = rewrite_query(
            "Why does it happen?",
            self.thread.messages,
            provider=mock_provider,
            timeout=1.85,
            enabled=True,
        )
        self.assertEqual(mock_provider.timeout, 1.85)

    def test_bounded_worker_pool_prevents_unbounded_thread_creation(self):
        """Area 1 Audit: Permanently blocked tasks never cause unbounded worker thread creation."""
        blocking_provider = MockLLMProvider("Never", delay=3.0)

        # Call 1 and 2 occupy the 2 workers in MAX_REWRITE_WORKERS
        res1, rew1 = rewrite_query(
            "Why does it happen?",
            self.thread.messages,
            provider=blocking_provider,
            timeout=0.08,
            enabled=True,
        )
        self.assertEqual(res1, "Why does it happen?")
        self.assertFalse(rew1)

        res2, rew2 = rewrite_query(
            "Why does it happen?",
            self.thread.messages,
            provider=blocking_provider,
            timeout=0.08,
            enabled=True,
        )
        self.assertEqual(res2, "Why does it happen?")
        self.assertFalse(rew2)

        # Active workers are bounded by MAX_REWRITE_WORKERS
        active_count = get_active_rewrite_worker_count()
        self.assertLessEqual(active_count, MAX_REWRITE_WORKERS)

        # Call 3: Pool is saturated with 2 busy workers; must return IMMEDIATELY (< 0.05s)
        # without queuing or spawning a 3rd thread
        sat_start = time.time()
        res3, rew3 = rewrite_query(
            "Why does it happen?",
            self.thread.messages,
            provider=blocking_provider,
            timeout=0.08,
            enabled=True,
        )
        sat_elapsed = time.time() - sat_start
        self.assertEqual(res3, "Why does it happen?")
        self.assertFalse(rew3)
        self.assertLess(sat_elapsed, 0.05)
        self.assertLessEqual(get_active_rewrite_worker_count(), MAX_REWRITE_WORKERS)


class TestConversationalRetriever(unittest.TestCase):
    """Test suite for document-scoped chat retrieval."""

    def test_retrieve_chat_context_requires_document_id(self):
        """Calling retrieval with empty or whitespace document_id must raise ValueError."""
        with self.assertRaises(ValueError):
            retrieve_chat_context("photosynthesis", "")
        with self.assertRaises(ValueError):
            retrieve_chat_context("photosynthesis", "   ")

    def test_retrieve_chat_context_empty_query(self):
        """Empty query returns empty list without calling store."""
        chunks = retrieve_chat_context("", "doc_1")
        self.assertEqual(chunks, [])

    def test_retrieve_chat_context_zero_or_negative_top_k(self):
        """top_k <= 0 returns empty list immediately."""
        chunks = retrieve_chat_context("query", "doc_1", top_k=0)
        self.assertEqual(chunks, [])

    @patch("src.chat.conversational_retriever.retrieve_chunks")
    def test_retrieve_chat_context_scoping_and_filtering(self, mock_retrieve_chunks):
        """Retrieval strictly enforces document_id scoping and filters out leaky chunks."""
        target_doc_id = "doc_target_123"

        mock_retrieve_chunks.return_value = [
            {
                "text": "Correct doc chunk 1",
                "metadata": {"document_id": target_doc_id, "page": 1, "filename": "t.pdf"},
            },
            {
                "text": "Leaked chunk from wrong doc",
                "metadata": {"document_id": "doc_OTHER_456", "page": 2, "filename": "o.pdf"},
            },
            {
                "text": "Correct doc chunk 2",
                "metadata": {"document_id": target_doc_id, "page": 3, "filename": "t.pdf"},
            },
        ]

        results = retrieve_chat_context(
            "test query",
            document_id=target_doc_id,
            top_k=3,
        )

        mock_retrieve_chunks.assert_called_once()
        _, kwargs = mock_retrieve_chunks.call_args
        self.assertEqual(kwargs.get("document_id"), target_doc_id)
        self.assertEqual(kwargs.get("top_k"), 3)

        self.assertEqual(len(results), 2)
        for c in results:
            self.assertEqual(c["metadata"]["document_id"], target_doc_id)


class TestChatPromptsAndCitations(unittest.TestCase):
    """Test suite for grounded chat prompt construction and citation validation."""

    def setUp(self):
        self.sample_chunks = [
            {
                "text": "Mitochondria generate ATP through oxidative phosphorylation.",
                "metadata": {
                    "page": 3,
                    "filename": "biology.pdf",
                    "chunk_id": "chk_1",
                    "document_id": "doc_bio",
                },
                "score": 0.92,
            },
            {
                "text": "Glycolysis breaks down glucose in the cytoplasm without oxygen.",
                "metadata": {
                    "page": 5,
                    "filename": "biology.pdf",
                    "chunk_id": "chk_2",
                    "document_id": "doc_bio",
                },
                "score": 0.85,
            },
        ]

    def test_build_document_evidence_block_empty(self):
        """Empty chunks list returns standard placeholder and empty source_map."""
        block, tokens, source_map = build_document_evidence_block([])
        self.assertIn("No relevant document excerpts were retrieved", block)
        self.assertEqual(source_map, {})
        self.assertGreater(tokens, 0)

    def test_build_document_evidence_block_formatting(self):
        """Retrieved chunks are formatted with deterministic [Source X, Page Y] markers."""
        block, tokens, source_map = build_document_evidence_block(self.sample_chunks)
        self.assertIn("=== DOCUMENT EVIDENCE (Authoritative) ===", block)
        self.assertIn("[Source 1, Page 3] (File: biology.pdf)", block)
        self.assertIn("Mitochondria generate ATP through oxidative phosphorylation.", block)
        self.assertIn("[Source 2, Page 5] (File: biology.pdf)", block)
        self.assertIn("=== END DOCUMENT EVIDENCE ===", block)

        self.assertEqual(len(source_map), 2)
        self.assertEqual(source_map[1]["page"], 3)
        self.assertEqual(source_map[1]["chunk_id"], "chk_1")
        self.assertEqual(source_map[2]["page"], 5)
        self.assertEqual(source_map[2]["chunk_id"], "chk_2")

    def test_build_grounded_chat_prompt_prioritizes_evidence(self):
        """Document evidence is budgeted first; history budget shrinks dynamically."""
        cfg = ConversationMemoryConfig(
            context_window_tokens=2000,
            output_reserve_tokens=400,
            safety_margin_ratio=0.10,
        )
        thread = ConversationThread(document_id="doc_bio")
        for i in range(10):
            thread.add_user_message(f"User turn {i} about biology concepts and cellular pathways.")
            thread.add_assistant_message(f"Assistant turn {i} explaining energy production in detail.")

        prompt = build_grounded_chat_prompt(
            "How does mitochondria work?",
            self.sample_chunks,
            thread,
            config_obj=cfg,
        )

        self.assertIsInstance(prompt, GroundedChatPrompt)
        self.assertGreater(prompt.evidence_tokens, 0)
        self.assertIn("=== DOCUMENT EVIDENCE (Authoritative) ===", prompt.user_prompt)
        self.assertIn("How does mitochondria work?", prompt.user_prompt)
        self.assertGreaterEqual(prompt.history_budget, 0)
        self.assertLessEqual(prompt.history_tokens, prompt.history_budget)

    def test_grounded_chat_prompt_unpacking(self):
        """GroundedChatPrompt supports direct tuple unpacking."""
        prompt = build_grounded_chat_prompt("Query", self.sample_chunks)
        sys_p, user_p, s_map = prompt
        self.assertEqual(sys_p, prompt.system_prompt)
        self.assertEqual(user_p, prompt.user_prompt)
        self.assertEqual(s_map, prompt.source_map)

    def test_build_grounded_chat_prompt_without_history(self):
        """Constructs valid grounded prompt when thread is None."""
        prompt = build_grounded_chat_prompt(
            "What is glycolysis?",
            self.sample_chunks,
            thread=None,
        )
        self.assertIn("=== DOCUMENT EVIDENCE (Authoritative) ===", prompt.user_prompt)
        self.assertNotIn("=== RECENT CONVERSATION HISTORY", prompt.user_prompt)
        self.assertIn("What is glycolysis?", prompt.user_prompt)

    def test_evidence_fits_within_budget(self):
        """Area 2 Audit: Evidence fitting within budget retains all chunks without exclusion."""
        cfg = ConversationMemoryConfig(
            context_window_tokens=2048,
            output_reserve_tokens=400,
            safety_margin_ratio=0.10,
        )
        prompt = build_grounded_chat_prompt(
            "What is glycolysis?",
            self.sample_chunks,
            config_obj=cfg,
        )
        self.assertEqual(prompt.excluded_chunks, 0)
        self.assertFalse(prompt.evidence_truncated)
        self.assertEqual(len(prompt.source_map), 2)
        self.assertLessEqual(
            prompt.total_prompt_tokens + prompt.output_reserve_tokens,
            prompt.context_window_tokens,
        )

    def test_evidence_exceeds_budget_excludes_lower_ranked_chunks(self):
        """Area 2 Audit: When evidence exceeds budget, lower-ranked chunks are explicitly excluded."""
        five_chunks = [
            {
                "text": f"Chunk {i} describes specialized biochemical reactions in organelle {i}.",
                "metadata": {"page": i, "filename": "bio.pdf", "chunk_id": f"chk_{i}"},
            }
            for i in range(1, 6)
        ]
        cfg = ConversationMemoryConfig(
            context_window_tokens=550,
            output_reserve_tokens=100,
            safety_margin_ratio=0.05,
        )
        prompt = build_grounded_chat_prompt(
            "Explain all organelles",
            five_chunks,
            config_obj=cfg,
        )

        self.assertGreater(prompt.excluded_chunks, 0)
        self.assertEqual(len(prompt.source_map), len(five_chunks) - prompt.excluded_chunks)
        self.assertIn("omitted to strictly adhere to model context limits", prompt.user_prompt)
        self.assertLessEqual(
            prompt.total_prompt_tokens + prompt.output_reserve_tokens,
            prompt.context_window_tokens,
        )

    def test_history_exceeds_remaining_budget_reduces_history_first(self):
        """Area 2 Audit: Conversation history is reduced before authoritative evidence is touched."""
        cfg = ConversationMemoryConfig(
            context_window_tokens=1200,
            output_reserve_tokens=200,
            safety_margin_ratio=0.10,
        )
        thread = ConversationThread(document_id="doc_bio")
        for i in range(12):
            thread.add_user_message(f"User detailed inquiry {i} about enzyme catalysis mechanisms.")
            thread.add_assistant_message(f"Assistant comprehensive explanation {i} of activation energies.")

        prompt = build_grounded_chat_prompt(
            "Explain glycolysis",
            self.sample_chunks,
            thread=thread,
            config_obj=cfg,
        )

        self.assertEqual(prompt.excluded_chunks, 0)
        self.assertFalse(prompt.evidence_truncated)
        self.assertEqual(len(prompt.source_map), 2)
        self.assertLessEqual(prompt.history_tokens, prompt.history_budget)
        self.assertLessEqual(
            prompt.total_prompt_tokens + prompt.output_reserve_tokens,
            prompt.context_window_tokens,
        )

    def test_zero_available_history_budget_omits_history_preserves_evidence(self):
        """Area 2 Audit: When evidence consumes available capacity, history budget is 0 and history is omitted."""
        large_chunks = [
            {
                "text": "Detailed excerpt describing mitochondrial membrane complexes and electron transport. " * 8,
                "metadata": {"page": 1, "filename": "bio.pdf", "chunk_id": "large_chk_1"},
            },
            {
                "text": "Second detailed excerpt describing cytochrome c oxidase and ATP synthase function. " * 8,
                "metadata": {"page": 2, "filename": "bio.pdf", "chunk_id": "large_chk_2"},
            },
        ]
        cfg = ConversationMemoryConfig(
            context_window_tokens=600,
            output_reserve_tokens=100,
            safety_margin_ratio=0.05,
        )
        thread = ConversationThread(document_id="doc_bio")
        thread.add_user_message("Prior question about respiration.")
        thread.add_assistant_message("Prior answer about respiration.")

        prompt = build_grounded_chat_prompt(
            "How does electron transport work?",
            large_chunks,
            thread=thread,
            config_obj=cfg,
        )

        self.assertEqual(prompt.history_budget, 0)
        self.assertEqual(prompt.history_tokens, 0)
        self.assertNotIn("=== RECENT CONVERSATION HISTORY", prompt.user_prompt)
        self.assertIn("=== DOCUMENT EVIDENCE (Authoritative) ===", prompt.user_prompt)
        self.assertLessEqual(
            prompt.total_prompt_tokens + prompt.output_reserve_tokens,
            prompt.context_window_tokens,
        )

    def test_prompt_size_at_exact_context_limit(self):
        """Area 2 Audit: Prompt size right at the limit never exceeds context window."""
        cfg = ConversationMemoryConfig(
            context_window_tokens=500,
            output_reserve_tokens=100,
            safety_margin_ratio=0.0,
        )
        prompt = build_grounded_chat_prompt(
            "What is glycolysis?",
            self.sample_chunks,
            config_obj=cfg,
        )
        self.assertLessEqual(
            prompt.total_prompt_tokens + prompt.output_reserve_tokens,
            500,
        )

    def test_prompt_size_one_token_above_limit_clamped_strictly(self):
        """Area 2 Audit: Prompt that would otherwise be 1 token above limit is strictly clamped."""
        cfg = ConversationMemoryConfig(
            context_window_tokens=460,
            output_reserve_tokens=100,
            safety_margin_ratio=0.0,
        )
        prompt = build_grounded_chat_prompt(
            "What is glycolysis and how does it function?",
            self.sample_chunks,
            config_obj=cfg,
        )
        self.assertLessEqual(
            prompt.total_prompt_tokens + prompt.output_reserve_tokens,
            460,
        )

    def test_single_oversized_chunk_safely_truncated(self):
        """Area 2 Audit: Single chunk exceeding the entire evidence budget is safely truncated with explicit marker."""
        huge_chunk = [
            {
                "text": "This is a massive continuous block of biological text that goes on for pages. " * 30,
                "metadata": {"page": 12, "filename": "huge.pdf", "chunk_id": "huge_1"},
            }
        ]
        cfg = ConversationMemoryConfig(
            context_window_tokens=500,
            output_reserve_tokens=100,
            safety_margin_ratio=0.10,
        )
        prompt = build_grounded_chat_prompt(
            "Summarize this",
            huge_chunk,
            config_obj=cfg,
        )
        self.assertTrue(prompt.evidence_truncated)
        self.assertIn("Source 1 excerpt truncated to fit context budget", prompt.user_prompt)
        self.assertEqual(len(prompt.source_map), 1)
        self.assertTrue(prompt.source_map[1].get("truncated"))
        self.assertLessEqual(
            prompt.total_prompt_tokens + prompt.output_reserve_tokens,
            prompt.context_window_tokens,
        )

    def test_long_question_fits_after_history_removal(self):
        """Area 2 Audit: Complete long question is preserved byte-for-byte when history is removed."""
        cfg = ConversationMemoryConfig(
            context_window_tokens=1000,
            output_reserve_tokens=200,
            safety_margin_ratio=0.10,
        )
        thread = ConversationThread(document_id="doc_bio")
        for i in range(10):
            thread.add_user_message(f"Detailed user discussion {i} on cellular metabolic pathways.")
            thread.add_assistant_message(f"Detailed assistant explanation {i} of thermodynamics.")

        long_question = (
            "What are the exact regulatory mechanisms that control phosphofructokinase-1 "
            "during glycolysis in mammalian liver tissue?"
        )
        prompt = build_grounded_chat_prompt(
            long_question,
            self.sample_chunks,
            thread=thread,
            config_obj=cfg,
        )

        self.assertIn(f"User Question: {long_question}", prompt.user_prompt)
        self.assertLessEqual(
            prompt.total_prompt_tokens + prompt.output_reserve_tokens,
            prompt.context_window_tokens,
        )

    def test_long_question_fits_after_evidence_reduction(self):
        """Area 2 Audit: Complete long question is preserved byte-for-byte when evidence is reduced."""
        five_chunks = [
            {
                "text": f"Substantial evidence excerpt {i} describing organelle {i} mechanics. " * 3,
                "metadata": {"page": i, "filename": "bio.pdf", "chunk_id": f"chk_{i}"},
            }
            for i in range(1, 6)
        ]
        cfg = ConversationMemoryConfig(
            context_window_tokens=650,
            output_reserve_tokens=100,
            safety_margin_ratio=0.05,
        )
        long_question = (
            "Explain how chemiosmotic coupling across the inner mitochondrial membrane "
            "creates the electrochemical gradient required for ATP synthesis."
        )
        prompt = build_grounded_chat_prompt(
            long_question,
            five_chunks,
            config_obj=cfg,
        )

        self.assertGreater(prompt.excluded_chunks, 0)
        self.assertIn(f"User Question: {long_question}", prompt.user_prompt)
        self.assertLessEqual(
            prompt.total_prompt_tokens + prompt.output_reserve_tokens,
            prompt.context_window_tokens,
        )

    def test_question_cannot_fit_raises_prompt_budget_exceeded_error(self):
        """Area 2 Audit: Oversized question that cannot fit raises PromptBudgetExceededError instead of truncation."""
        cfg = ConversationMemoryConfig(
            context_window_tokens=500,
            output_reserve_tokens=100,
        )
        oversized_question = (
            "Can you explain in comprehensive, exhaustive, rigorous academic detail every single biochemical "
            "step, enzyme, cofactor, intermediate, energetic yield, delta-G value, and regulatory feedback loop "
            "of the citric acid cycle from citrate synthase through malate dehydrogenase in eukaryotes? " * 15
        )

        with self.assertRaises(PromptBudgetExceededError) as ctx:
            build_grounded_chat_prompt(
                oversized_question,
                self.sample_chunks,
                config_obj=cfg,
            )

        self.assertIn("exceeds the maximum allowed prompt capacity", str(ctx.exception))
        self.assertIn("The complete question cannot fit without truncation", str(ctx.exception))

    def test_original_question_remains_byte_for_byte_identical(self):
        """Area 2 Audit: Verify user question containing punctuation and math is byte-for-byte identical."""
        special_question = (
            "How does ATP (adenosine 5'-triphosphate) hydrolysis (ATP + H2O -> ADP + Pi) "
            "release -30.5 kJ/mol of free energy (ΔG°') under standard conditions?"
        )
        prompt = build_grounded_chat_prompt(special_question, self.sample_chunks)
        self.assertIn(f"User Question: {special_question}", prompt.user_prompt)

    def test_no_prompt_exceeding_estimated_token_ceiling(self):
        """Area 2 Audit: Exhaustive check ensuring total prompt + output reserve <= context_window_tokens."""
        test_windows = [600, 1000, 2048, 4096]
        test_reserves = [100, 250, 500, 1000]

        for win, res in zip(test_windows, test_reserves):
            cfg = ConversationMemoryConfig(
                context_window_tokens=win,
                output_reserve_tokens=res,
                safety_margin_ratio=0.10,
            )
            prompt = build_grounded_chat_prompt(
                "What is cellular energy?",
                self.sample_chunks,
                config_obj=cfg,
            )
            self.assertLessEqual(
                prompt.total_prompt_tokens + prompt.output_reserve_tokens,
                win,
                f"Prompt total {prompt.total_prompt_tokens} + reserve {res} exceeded window {win}",
            )

    def test_validate_chat_citations_valid(self):
        """Valid source citations with correct page numbers are properly recognized."""
        _, _, source_map = build_document_evidence_block(self.sample_chunks)
        answer = (
            "Mitochondria are the powerhouses [Source 1, Page 3]. "
            "In contrast, glycolysis occurs in cytoplasm [Source 2, Page 5]."
        )
        citations = validate_chat_citations(answer, source_map)
        self.assertEqual(len(citations), 2)

        self.assertEqual(citations[0]["source_id"], 1)
        self.assertTrue(citations[0]["is_valid"])
        self.assertTrue(citations[0]["page_valid"])
        self.assertEqual(citations[0]["page"], 3)
        self.assertEqual(citations[0]["filename"], "biology.pdf")

        self.assertEqual(citations[1]["source_id"], 2)
        self.assertTrue(citations[1]["is_valid"])
        self.assertTrue(citations[1]["page_valid"])
        self.assertEqual(citations[1]["page"], 5)

    def test_validate_chat_citations_mismatched_page(self):
        """Valid source identifier but incorrect cited page number flags page_valid=False."""
        _, _, source_map = build_document_evidence_block(self.sample_chunks)
        answer = "Mitochondria produce ATP [Source 1, Page 99]."
        citations = validate_chat_citations(answer, source_map)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["source_id"], 1)
        self.assertTrue(citations[0]["is_valid"])
        self.assertFalse(citations[0]["page_valid"])
        self.assertEqual(citations[0]["page"], 3)
        self.assertEqual(citations[0]["cited_page"], "99")

    def test_validate_chat_citations_hallucinated_source(self):
        """Citations referencing non-existent sources (e.g. Source 9) are marked invalid."""
        _, _, source_map = build_document_evidence_block(self.sample_chunks)
        answer = "Nuclear pores control RNA transport [Source 9, Page 1]."
        citations = validate_chat_citations(answer, source_map)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["source_id"], 9)
        self.assertFalse(citations[0]["is_valid"])
        self.assertFalse(citations[0]["page_valid"])

    def test_validate_chat_citations_none_found(self):
        """Answers without citations return empty list cleanly."""
        _, _, source_map = build_document_evidence_block(self.sample_chunks)
        citations = validate_chat_citations("I cannot answer this question.", source_map)
        self.assertEqual(citations, [])

    def test_sanitize_and_repair_citations(self):
        """Sanitizer strips hallucinated sources and corrects mismatched page numbers."""
        _, _, source_map = build_document_evidence_block(self.sample_chunks)

        raw_answer = (
            "ATP is synthesized here [Source 1, Page 99]. "
            "Hallucinated claim [Source 8, Page 12]. "
            "Glycolysis step [Source 2, Page 5]."
        )
        sanitized = sanitize_and_repair_citations(raw_answer, source_map)

        # Mismatched page 99 should be repaired to page 3
        self.assertIn("[Source 1, Page 3]", sanitized)
        self.assertNotIn("[Source 1, Page 99]", sanitized)

        # Hallucinated Source 8 must be removed
        self.assertNotIn("Source 8", sanitized)

        # Valid citation should be preserved
        self.assertIn("[Source 2, Page 5]", sanitized)


if __name__ == "__main__":
    unittest.main()
