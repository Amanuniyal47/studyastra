"""Unit tests for Multi-Turn RAG Pipeline Orchestrator (Stage 10.3).

Comprehensive test coverage verifying:
1. Complete successful single-turn conversation.
2. Successful multi-turn conversation.
3. Pronoun-dependent follow-up retrieval.
4. Standalone question without rewriting.
5. Empty retrieval results.
6. Document isolation.
7. Citation validation and repair.
8. Prompt-budget overflow (PromptBudgetExceededError).
9. Query rewrite timeout fallback.
10. Worker saturation fallback.
11. Ollama unavailable.
12. Generation timeout.
13. Empty model response.
14. Failed generation without incomplete assistant messages.
15. Retry behaviour without duplicate messages.
16. Transcript preservation.
17. Correct source metadata persistence.
18. Conversation thread isolation.
19. Request validation (missing doc, empty query, store empty).
20. Streaming turn support and on_token callback.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Iterator
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.chat.chat_orchestrator import (
    ChatOrchestrator,
    ChatOrchestratorResult,
    orchestrate_chat,
)
from src.chat.conversation_memory import (
    ChatMessage,
    ConversationManager,
    ConversationMemoryConfig,
    ConversationThread,
    SlidingWindowMemoryManager,
)
from src.chat.chat_prompts import build_grounded_chat_prompt
from src.chat.query_rewriter import reset_rewrite_pool
from src.generation.llm_provider import BaseLLMProvider, LLMProviderError


class MockLLMProvider(BaseLLMProvider):
    """Configurable mock LLM provider for deterministic orchestrator tests."""

    provider_name = "mock"

    def __init__(
        self,
        response: str = "Default mock answer [Source 1, Page 3].",
        delay: float = 0.0,
        should_raise: bool = False,
        raise_exc: Exception | None = None,
        fail_after_tokens: int | None = None,
        model_name: str = "qwen2.5:7b",
    ) -> None:
        super().__init__(model_name=model_name)
        self.response = response
        self.delay = delay
        self.should_raise = should_raise
        self.raise_exc = raise_exc
        self.fail_after_tokens = fail_after_tokens
        self.call_count = 0
        self.history: list[dict[str, Any]] = []

    def is_configured(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        self.call_count += 1
        self.history.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if self.delay > 0:
            time.sleep(self.delay)
        if self.should_raise:
            if self.raise_exc:
                raise self.raise_exc
            raise LLMProviderError("Mock LLM generation failure")
        return self.response

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        self.call_count += 1
        if self.delay > 0:
            time.sleep(self.delay)
        if self.should_raise:
            if self.raise_exc:
                raise self.raise_exc
            raise LLMProviderError("Mock LLM streaming generation failure")
        words = self.response.split(" ")
        for i, word in enumerate(words):
            if self.fail_after_tokens is not None and i >= self.fail_after_tokens:
                if self.raise_exc:
                    raise self.raise_exc
                raise LLMProviderError("Mock LLM mid-stream network dropped")
            yield word + (" " if i < len(words) - 1 else "")


class MockEmbeddingService:
    """Mock embedding service returning 384-dimensional zero vectors."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    def embed_query(self, text: str, normalize: bool = True) -> np.ndarray:
        return np.zeros(384, dtype=np.float32)

    def get_dimension(self) -> int:
        return 384


class MockVectorStore:
    """Mock FAISSVectorStore with configurable indexed documents and chunks."""

    def __init__(
        self,
        indexed_docs: dict[str, list[dict[str, Any]]] | None = None,
        total_vectors: int = 10,
    ) -> None:
        self._indexed_docs = indexed_docs or {}
        self._total_vectors = total_vectors

    @property
    def total_vectors(self) -> int:
        return self._total_vectors

    def is_document_indexed(self, document_id: str) -> bool:
        return document_id in self._indexed_docs

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if document_id and document_id in self._indexed_docs:
            return self._indexed_docs[document_id][:top_k]
        return []


class TestChatOrchestrator(unittest.TestCase):
    """Comprehensive test suite for ChatOrchestrator."""

    def setUp(self) -> None:
        reset_rewrite_pool()
        ChatOrchestrator.clear_all_thread_locks()
        self.doc_id = "doc_bio_101"
        self.sample_chunks = [
            {
                "chunk_id": "chk_1",
                "text": "Mitochondria produce cellular ATP through oxidative phosphorylation across the inner membrane.",
                "page": 3,
                "score": 0.92,
                "metadata": {
                    "document_id": self.doc_id,
                    "filename": "biology.pdf",
                    "page": 3,
                    "chunk_id": "chk_1",
                },
            },
            {
                "chunk_id": "chk_2",
                "text": "Glycolysis breaks down glucose into pyruvate in the cytoplasm, yielding 2 ATP net.",
                "page": 5,
                "score": 0.85,
                "metadata": {
                    "document_id": self.doc_id,
                    "filename": "biology.pdf",
                    "page": 5,
                    "chunk_id": "chk_2",
                },
            },
        ]
        self.mock_store = MockVectorStore(
            indexed_docs={self.doc_id: self.sample_chunks},
            total_vectors=len(self.sample_chunks),
        )
        self.mock_embeddings = MockEmbeddingService()
        self.mock_provider = MockLLMProvider(
            response="Mitochondria generate cellular energy [Source 1, Page 3]."
        )
        self.memory_config = ConversationMemoryConfig(
            context_window_tokens=4096,
            max_history_tokens=1500,
            output_reserve_tokens=1000,
        )

    def tearDown(self) -> None:
        reset_rewrite_pool()
        ChatOrchestrator.clear_all_thread_locks()

    # -----------------------------------------------------------------
    # Test 1: Complete Successful Single-Turn Conversation
    # -----------------------------------------------------------------
    def test_complete_successful_single_turn(self) -> None:
        """Requirement 1: Complete single-turn conversation with evidence retrieval and citation."""
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
            memory_config=self.memory_config,
        )
        thread = ConversationThread(document_id=self.doc_id)

        result = orchestrator.orchestrate_turn(
            "What do mitochondria do?",
            self.doc_id,
            thread=thread,
        )

        self.assertTrue(result.success)
        self.assertIn("Mitochondria generate cellular energy", result.answer)
        self.assertIn("[Source 1, Page 3]", result.answer)
        self.assertEqual(result.original_query, "What do mitochondria do?")
        self.assertEqual(result.retrieval_query, "What do mitochondria do?")
        self.assertFalse(result.is_rewritten)
        self.assertEqual(result.document_id, self.doc_id)
        self.assertEqual(len(result.raw_chunks), 2)
        self.assertEqual(len(result.source_map), 2)

        # Citations check
        self.assertEqual(len(result.citations), 1)
        self.assertTrue(result.citations[0]["is_valid"])
        self.assertEqual(result.citations[0]["page"], 3)

        # Thread messages check (1 user + 1 assistant)
        self.assertEqual(len(thread.messages), 2)
        self.assertEqual(thread.messages[0].role, "user")
        self.assertEqual(thread.messages[0].content, "What do mitochondria do?")
        self.assertEqual(thread.messages[1].role, "assistant")
        self.assertIn("Mitochondria generate cellular energy", thread.messages[1].content)
        self.assertEqual(len(thread.messages[1].citations), 1)

        # Timing check
        self.assertIn("total_ms", result.timing_ms)
        self.assertGreater(result.timing_ms["total_ms"], 0)

    # -----------------------------------------------------------------
    # Test 2: Successful Multi-Turn Conversation
    # -----------------------------------------------------------------
    def test_successful_multi_turn_conversation(self) -> None:
        """Requirement 2: Multi-turn conversation retains prior history across turns."""
        provider = MockLLMProvider(
            response="Glycolysis occurs in the cytoplasm [Source 2, Page 5]."
        )
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=provider,
            memory_config=self.memory_config,
        )
        thread = ConversationThread(document_id=self.doc_id)

        # Turn 1
        res1 = orchestrator.orchestrate_turn("What is cellular respiration?", self.doc_id, thread=thread)
        self.assertTrue(res1.success)
        self.assertEqual(len(thread.messages), 2)

        # Turn 2
        res2 = orchestrator.orchestrate_turn("Where does glycolysis happen?", self.doc_id, thread=thread)
        self.assertTrue(res2.success)
        self.assertEqual(len(thread.messages), 4)

        # Verify prior turn history was presented in the prompt of Turn 2
        last_call = provider.history[-1]
        self.assertIn("What is cellular respiration?", last_call["user_prompt"])
        self.assertIn("Where does glycolysis happen?", last_call["user_prompt"])

    # -----------------------------------------------------------------
    # Test 3: Pronoun-Dependent Follow-Up Retrieval
    # -----------------------------------------------------------------
    def test_pronoun_dependent_follow_up_retrieval(self) -> None:
        """Requirement 3: Follow-up question with pronoun triggers rewriting for retrieval."""
        class DualResponseProvider(BaseLLMProvider):
            def __init__(self):
                super().__init__(model_name="qwen2.5:7b")
                self.calls = 0

            def is_configured(self):
                return True

            def generate(self, sys_prompt, user_prompt, **kwargs):
                self.calls += 1
                # If prompt is rewrite prompt, return rewritten query
                if "rewriting assistant" in sys_prompt.lower():
                    return "Where does glycolysis occur in the cell?"
                # Main answer generation
                return "Glycolysis occurs in the cytoplasm [Source 2, Page 5]."

        provider = DualResponseProvider()
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=provider,
            memory_config=self.memory_config,
        )
        thread = ConversationThread(document_id=self.doc_id)
        thread.add_user_message("Explain glycolysis.")
        thread.add_assistant_message("Glycolysis is the metabolic pathway converting glucose into pyruvate.")

        result = orchestrator.orchestrate_turn(
            "Where does it happen?",
            self.doc_id,
            thread=thread,
        )

        self.assertTrue(result.success)
        self.assertTrue(result.is_rewritten)
        self.assertEqual(result.original_query, "Where does it happen?")
        self.assertEqual(result.retrieval_query, "Where does glycolysis occur in the cell?")
        # Original user query must be preserved in transcript
        self.assertEqual(thread.messages[-2].content, "Where does it happen?")

    # -----------------------------------------------------------------
    # Test 4: Standalone Question Without Rewriting
    # -----------------------------------------------------------------
    def test_standalone_question_without_rewriting(self) -> None:
        """Requirement 4: Standalone question skips query rewriter to avoid unnecessary LLM calls."""
        provider = MockLLMProvider(
            response="Photosynthesis creates sugars [Source 1, Page 3]."
        )
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=provider,
            memory_config=self.memory_config,
        )
        thread = ConversationThread(document_id=self.doc_id)
        thread.add_user_message("What is cellular respiration?")
        thread.add_assistant_message("Respiration produces ATP.")

        # Standalone question with no pronouns or anaphoric references
        result = orchestrator.orchestrate_turn(
            "Describe the overall chemical equation of photosynthesis in plants.",
            self.doc_id,
            thread=thread,
        )

        self.assertTrue(result.success)
        self.assertFalse(result.is_rewritten)
        self.assertEqual(result.retrieval_query, result.original_query)
        # Exactly 1 LLM call was made (for generation only; rewrite was skipped)
        self.assertEqual(provider.call_count, 1)

    # -----------------------------------------------------------------
    # Test 5: Empty Retrieval Results Handled Gracefully
    # -----------------------------------------------------------------
    def test_empty_retrieval_results_handled_gracefully(self) -> None:
        """Requirement 5: Empty retrieval results are handled gracefully without raising exceptions."""
        empty_store = MockVectorStore(
            indexed_docs={self.doc_id: []},
            total_vectors=5,
        )
        provider = MockLLMProvider(
            response="I cannot answer this question based on the provided document context."
        )
        orchestrator = ChatOrchestrator(
            store=empty_store,
            embedding_service=self.mock_embeddings,
            provider=provider,
            memory_config=self.memory_config,
        )
        thread = ConversationThread(document_id=self.doc_id)

        result = orchestrator.orchestrate_turn(
            "What is quantum entanglement?",
            self.doc_id,
            thread=thread,
        )

        self.assertTrue(result.success)
        self.assertEqual(len(result.raw_chunks), 0)
        self.assertIn("cannot answer this question based on the provided document context", result.answer)
        self.assertEqual(len(thread.messages), 2)

    # -----------------------------------------------------------------
    # Test 6: Document Isolation & Leakage Prevention
    # -----------------------------------------------------------------
    def test_document_isolation_and_leakage_prevention(self) -> None:
        """Requirement 6: Retrieval and conversation are strictly isolated by document_id."""
        doc_a = "doc_alpha"
        doc_b = "doc_beta"

        chunks_a = [{"chunk_id": "c1", "text": "Alpha content", "page": 1, "metadata": {"document_id": doc_a, "page": 1}}]
        chunks_b = [{"chunk_id": "c2", "text": "Beta content", "page": 2, "metadata": {"document_id": doc_b, "page": 2}}]

        store = MockVectorStore(
            indexed_docs={doc_a: chunks_a, doc_b: chunks_b},
            total_vectors=2,
        )
        orchestrator = ChatOrchestrator(
            store=store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
        )

        # Thread for Doc A
        thread_a = ConversationThread(document_id=doc_a)
        res_a = orchestrator.orchestrate_turn("Tell me about Alpha", doc_a, thread=thread_a)
        self.assertTrue(res_a.success)
        self.assertEqual(res_a.raw_chunks[0]["metadata"]["document_id"], doc_a)

        # Passing Thread A to query Doc B must be rejected due to document ID mismatch
        res_mismatch = orchestrator.orchestrate_turn("Tell me about Beta", doc_b, thread=thread_a)
        self.assertFalse(res_mismatch.success)
        self.assertIn("does not match requested document_id", res_mismatch.error_message)

    # -----------------------------------------------------------------
    # Test 7: Citation Validation and Repair
    # -----------------------------------------------------------------
    def test_citation_validation_and_repair(self) -> None:
        """Requirement 7: Hallucinated citations are stripped and page numbers are corrected."""
        # Model returns Source 1 with wrong page (page 99 instead of 3), and hallucinated Source 99
        hallucinated_answer = (
            "ATP is created by mitochondria [Source 1, Page 99], "
            "and ribosomes translate proteins [Source 99, Page 1]."
        )
        provider = MockLLMProvider(response=hallucinated_answer)
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=provider,
            memory_config=self.memory_config,
        )
        thread = ConversationThread(document_id=self.doc_id)

        result = orchestrator.orchestrate_turn("How is ATP made?", self.doc_id, thread=thread)

        self.assertTrue(result.success)
        # Source 1 cited page 99 is repaired to actual page 3
        self.assertIn("[Source 1, Page 3]", result.answer)
        # Hallucinated Source 99 is completely removed from the repaired answer
        self.assertNotIn("[Source 99", result.answer)
        # Assistant message in thread stores the repaired answer
        self.assertEqual(thread.messages[-1].content, result.answer)

    # -----------------------------------------------------------------
    # Test 8: Prompt Budget Overflow Handled Safely
    # -----------------------------------------------------------------
    def test_prompt_budget_overflow_handled_safely(self) -> None:
        """Requirement 8: PromptBudgetExceededError is caught and handled safely without corrupted LLM calls."""
        # Very tiny context window where an oversized user question cannot fit
        tiny_config = ConversationMemoryConfig(
            context_window_tokens=300,
            output_reserve_tokens=100,
        )
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
            memory_config=tiny_config,
        )
        thread = ConversationThread(document_id=self.doc_id)
        huge_question = "Explain every single biological detail of life on Earth. " * 50

        result = orchestrator.orchestrate_turn(huge_question, self.doc_id, thread=thread)

        self.assertFalse(result.success)
        self.assertIn("exceeds the maximum allowed prompt capacity", result.error_message)
        # No assistant message was added
        self.assertEqual(len([m for m in thread.messages if m.role == "assistant"]), 0)
        # LLM was never called
        self.assertEqual(self.mock_provider.call_count, 0)

    # -----------------------------------------------------------------
    # Test 9: Query Rewrite Timeout Fallback
    # -----------------------------------------------------------------
    def test_query_rewrite_timeout_fallback(self) -> None:
        """Requirement 9: Timed-out query rewriting gracefully falls back to original query."""
        class SlowRewriteProvider(BaseLLMProvider):
            def __init__(self):
                super().__init__(model_name="qwen2.5:7b")

            def is_configured(self):
                return True

            def generate(self, sys_prompt, user_prompt, **kwargs):
                if "rewriting assistant" in sys_prompt.lower():
                    time.sleep(0.3)  # Exceeds 0.05s timeout
                    return "Rewritten query"
                return "Answer based on original query [Source 1, Page 3]."

        provider = SlowRewriteProvider()
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=provider,
            rewrite_timeout_seconds=0.05,
        )
        thread = ConversationThread(document_id=self.doc_id)
        thread.add_user_message("What is respiration?")
        thread.add_assistant_message("It makes ATP.")

        result = orchestrator.orchestrate_turn("Why does it happen?", self.doc_id, thread=thread)

        self.assertTrue(result.success)
        # Timed out rewrite falls back to original query
        self.assertEqual(result.retrieval_query, "Why does it happen?")
        self.assertFalse(result.is_rewritten)
        self.assertIn("Answer based on original query", result.answer)

    # -----------------------------------------------------------------
    # Test 10: Worker Saturation Fallback
    # -----------------------------------------------------------------
    def test_worker_saturation_fallback(self) -> None:
        """Requirement 10: Saturated rewrite pool falls back immediately to original query."""
        # Fill both rewrite workers with permanently blocked tasks
        from src.chat.query_rewriter import get_rewrite_pool, rewrite_query

        _, sem = get_rewrite_pool()
        sem.acquire()
        sem.acquire()

        try:
            orchestrator = ChatOrchestrator(
                store=self.mock_store,
                embedding_service=self.mock_embeddings,
                provider=self.mock_provider,
            )
            thread = ConversationThread(document_id=self.doc_id)
            thread.add_user_message("What is respiration?")
            thread.add_assistant_message("It makes ATP.")

            t0 = time.perf_counter()
            result = orchestrator.orchestrate_turn("Why does it happen?", self.doc_id, thread=thread)
            elapsed = time.perf_counter() - t0

            # Saturated pool returns immediately (under 0.2s)
            self.assertLess(elapsed, 0.2)
            self.assertTrue(result.success)
            self.assertEqual(result.retrieval_query, "Why does it happen?")
            self.assertFalse(result.is_rewritten)
        finally:
            sem.release()
            sem.release()

    # -----------------------------------------------------------------
    # Test 11: Ollama Unavailable Handled Safely
    # -----------------------------------------------------------------
    def test_ollama_unavailable_handled_safely(self) -> None:
        """Requirement 11: Ollama connection error is handled safely without orphan assistant message."""
        provider = MockLLMProvider(
            should_raise=True,
            raise_exc=LLMProviderError(
                "Cannot connect to Ollama at http://localhost:11434. Please ensure Ollama is installed."
            ),
        )
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        result = orchestrator.orchestrate_turn("What is ATP?", self.doc_id, thread=thread)

        self.assertFalse(result.success)
        self.assertIn("Cannot connect to Ollama", result.error_message)
        # Assistant message was NOT added
        self.assertEqual(len(thread.messages), 1)
        self.assertEqual(thread.messages[0].role, "user")

    # -----------------------------------------------------------------
    # Test 12: Generation Timeout Handled Safely
    # -----------------------------------------------------------------
    def test_generation_timeout_handled_safely(self) -> None:
        """Requirement 12: LLM generation timeout produces clean error result without corrupted messages."""
        provider = MockLLMProvider(
            should_raise=True,
            raise_exc=LLMProviderError("Ollama request timed out after 30.0s for model 'qwen2.5:7b'."),
        )
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        result = orchestrator.orchestrate_turn("What is ATP?", self.doc_id, thread=thread)

        self.assertFalse(result.success)
        self.assertIn("timed out", result.error_message)
        self.assertEqual(len([m for m in thread.messages if m.role == "assistant"]), 0)

    # -----------------------------------------------------------------
    # Test 13: Empty Model Response Handled Safely
    # -----------------------------------------------------------------
    def test_empty_model_response_handled_safely(self) -> None:
        """Requirement 13: Empty completion text from provider returns clean failure result."""
        provider = MockLLMProvider(response="   ")
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        result = orchestrator.orchestrate_turn("What is ATP?", self.doc_id, thread=thread)

        self.assertFalse(result.success)
        self.assertIn("empty completion", result.error_message)
        self.assertEqual(len([m for m in thread.messages if m.role == "assistant"]), 0)

    # -----------------------------------------------------------------
    # Test 14: Failed Generation Leaves No Incomplete Assistant Message
    # -----------------------------------------------------------------
    def test_failed_generation_leaves_no_incomplete_assistant_message(self) -> None:
        """Requirement 14: Failed turn leaves zero assistant messages in the conversation thread."""
        provider = MockLLMProvider(should_raise=True)
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=provider,
        )
        thread = ConversationThread(document_id=self.doc_id)
        thread.add_user_message("Prior user message")
        thread.add_assistant_message("Prior assistant message")

        result = orchestrator.orchestrate_turn("New failing query", self.doc_id, thread=thread)

        self.assertFalse(result.success)
        # Thread must have exactly 1 prior assistant message, not 2
        assistant_msgs = [m for m in thread.messages if m.role == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(assistant_msgs[0].content, "Prior assistant message")

    # -----------------------------------------------------------------
    # Test 15: Retry Behaviour Without Duplicate Messages
    # -----------------------------------------------------------------
    def test_retry_behaviour_without_duplicate_messages(self) -> None:
        """Requirement 15: Retrying a failed turn reuses the existing user message without duplicating it."""
        failing_provider = MockLLMProvider(should_raise=True)
        working_provider = MockLLMProvider(response="Recovered answer [Source 1, Page 3].")

        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=failing_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        # Attempt 1: Fails
        res1 = orchestrator.orchestrate_turn("Explain mitochondria", self.doc_id, thread=thread)
        self.assertFalse(res1.success)
        self.assertEqual(len(thread.messages), 1)
        self.assertEqual(thread.messages[0].content, "Explain mitochondria")

        # Attempt 2: Retries same query with working provider
        orchestrator.provider = working_provider
        res2 = orchestrator.orchestrate_turn("Explain mitochondria", self.doc_id, thread=thread)
        self.assertTrue(res2.success)

        # Thread must now have exactly 2 messages (1 user, 1 assistant), NOT 3
        self.assertEqual(len(thread.messages), 2)
        self.assertEqual(thread.messages[0].role, "user")
        self.assertEqual(thread.messages[0].content, "Explain mitochondria")
        self.assertEqual(thread.messages[1].role, "assistant")
        self.assertEqual(thread.messages[1].content, "Recovered answer [Source 1, Page 3].")

    # -----------------------------------------------------------------
    # Test 16: Transcript Preservation Verbatim
    # -----------------------------------------------------------------
    def test_transcript_preservation_verbatim(self) -> None:
        """Requirement 16: Messages with special characters and math remain byte-for-byte identical."""
        special_query = "What is the ΔG°' of ATP hydrolysis (ATP + H2O -> ADP + Pi) at pH 7.0?"
        special_answer = "Under standard biochemical conditions, ΔG°' is approximately -30.5 kJ/mol."

        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)
        thread.add_user_message(special_query)
        thread.add_assistant_message(special_answer)

        # Run next turn
        result = orchestrator.orchestrate_turn("What about in living cells?", self.doc_id, thread=thread)
        self.assertTrue(result.success)

        # Verify prior messages were not mutated
        self.assertEqual(thread.messages[0].content, special_query)
        self.assertEqual(thread.messages[1].content, special_answer)

    # -----------------------------------------------------------------
    # Test 17: Correct Source Metadata Persistence
    # -----------------------------------------------------------------
    def test_correct_source_metadata_persistence(self) -> None:
        """Requirement 17: Assistant message stores full citation and retrieval metadata."""
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        result = orchestrator.orchestrate_turn("What do mitochondria do?", self.doc_id, thread=thread)
        self.assertTrue(result.success)

        assistant_msg = thread.messages[-1]
        self.assertEqual(len(assistant_msg.citations), 1)
        cite = assistant_msg.citations[0]
        self.assertEqual(cite["source_id"], 1)
        self.assertEqual(cite["page"], 3)
        self.assertEqual(cite["filename"], "biology.pdf")
        self.assertEqual(cite["chunk_id"], "chk_1")

        meta = assistant_msg.metadata
        self.assertEqual(meta["evidence_count"], 2)
        self.assertIn("prompt_tokens", meta)
        self.assertEqual(meta["model"], "qwen2.5:7b")

    # -----------------------------------------------------------------
    # Test 18: Conversation Thread Isolation
    # -----------------------------------------------------------------
    def test_conversation_thread_isolation(self) -> None:
        """Requirement 18: Multiple conversation threads maintain complete data isolation."""
        manager = ConversationManager()
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
            conversation_manager=manager,
        )

        thread_1 = manager.get_or_create_thread(self.doc_id, conversation_id="11111111-1111-4111-8111-111111111111")
        thread_2 = manager.get_or_create_thread(self.doc_id, conversation_id="22222222-2222-4222-8222-222222222222")

        orchestrator.orchestrate_turn("Question for thread 1", self.doc_id, thread=thread_1)
        orchestrator.orchestrate_turn("Question for thread 2", self.doc_id, thread=thread_2)

        self.assertEqual(len(thread_1.messages), 2)
        self.assertEqual(len(thread_2.messages), 2)
        self.assertEqual(thread_1.messages[0].content, "Question for thread 1")
        self.assertEqual(thread_2.messages[0].content, "Question for thread 2")

    # -----------------------------------------------------------------
    # Test 19: Request Validation Edge Cases
    # -----------------------------------------------------------------
    def test_request_validation_edge_cases(self) -> None:
        """Requirement 19: Missing doc_id, empty query, and unindexed document are cleanly rejected."""
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
        )

        # Empty document_id
        res1 = orchestrator.orchestrate_turn("Query", "")
        self.assertFalse(res1.success)
        self.assertIn("document_id must be a non-empty string", res1.error_message)

        # Empty query
        res2 = orchestrator.orchestrate_turn("   ", self.doc_id)
        self.assertFalse(res2.success)
        self.assertIn("User query cannot be empty", res2.error_message)

        # Unindexed document
        res3 = orchestrator.orchestrate_turn("Valid query", "doc_not_indexed")
        self.assertFalse(res3.success)
        self.assertIn("not found in document registry", res3.error_message)

    # -----------------------------------------------------------------
    # Test 20: Streaming Turn Support and on_token Callback
    # -----------------------------------------------------------------
    def test_streaming_turn_support_and_callback(self) -> None:
        """Requirement 20: stream_turn yields text chunks and populates last_result."""
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        stream = orchestrator.stream_turn("What do mitochondria do?", self.doc_id, thread=thread)
        chunks = list(stream)

        # Verify streamed chunks match the answer
        assembled = "".join(chunks)
        self.assertIn("Mitochondria generate cellular energy", assembled)

        # Verify last_result is populated
        self.assertIsNotNone(orchestrator.last_result)
        self.assertTrue(orchestrator.last_result.success)
        self.assertEqual(len(thread.messages), 2)

    def test_orchestrate_chat_functional_interface(self) -> None:
        """Convenience function orchestrate_chat behaves identically to orchestrator instance."""
        res = orchestrate_chat(
            "What do mitochondria do?",
            self.doc_id,
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
            memory_config=self.memory_config,
        )
        self.assertTrue(res.success)
        self.assertIn("Mitochondria generate cellular energy", res.answer)

    # =================================================================
    # Area 1 Audit: Streaming Failure and Transcript Consistency
    # =================================================================
    def test_mid_stream_failure_never_commits_assistant_message(self) -> None:
        """Area 1 Audit: Mid-stream generation failure stops cleanly and never commits an assistant message."""
        failing_stream_provider = MockLLMProvider(
            response="Mitochondria produce ATP through respiratory chains across the cristae.",
            fail_after_tokens=2,
        )
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=failing_stream_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        stream = orchestrator.stream_turn("How do mitochondria produce ATP?", self.doc_id, thread=thread)
        yielded_tokens = list(stream)

        # Tokens were yielded before the failure
        self.assertEqual(len(yielded_tokens), 2)

        # Result is marked as failure
        self.assertIsNotNone(orchestrator.last_result)
        self.assertFalse(orchestrator.last_result.success)
        self.assertIn("Streaming failed", orchestrator.last_result.error_message)

        # Transcript MUST NOT contain any assistant message
        assistant_msgs = [m for m in thread.messages if m.role == "assistant"]
        self.assertEqual(len(assistant_msgs), 0)

        # User message remains in thread with failed turn status
        self.assertEqual(len(thread.messages), 1)
        self.assertEqual(thread.messages[0].role, "user")
        self.assertEqual(thread.messages[0].metadata.get("turn_status"), "failed")

    def test_streaming_callback_failure_does_not_corrupt_transcript(self) -> None:
        """Area 1 Audit: An on_token callback exception is logged and does not fail turn execution or corrupt state."""
        def broken_callback(tok: str) -> None:
            raise RuntimeError("UI rendering crash inside callback")

        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        result = orchestrator.orchestrate_turn(
            "What do mitochondria do?",
            self.doc_id,
            thread=thread,
            on_token=broken_callback,
        )

        self.assertTrue(result.success)
        self.assertIn("Mitochondria generate cellular energy", result.answer)
        # Assistant message was still committed cleanly
        self.assertEqual(len(thread.messages), 2)
        self.assertEqual(thread.messages[1].role, "assistant")

    def test_retry_after_partial_streaming_failure(self) -> None:
        """Area 1 Audit: Retrying after a mid-stream failure reuses the user message without duplication."""
        failing_provider = MockLLMProvider(
            response="Partial stream text",
            fail_after_tokens=1,
        )
        working_provider = MockLLMProvider(
            response="Fully recovered answer [Source 1, Page 3]."
        )
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=failing_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        # Attempt 1: Fails mid-stream
        list(orchestrator.stream_turn("How is ATP made?", self.doc_id, thread=thread))
        self.assertFalse(orchestrator.last_result.success)
        self.assertEqual(len(thread.messages), 1)

        # Attempt 2: Retry with working provider
        orchestrator.provider = working_provider
        res2 = orchestrator.orchestrate_turn("How is ATP made?", self.doc_id, thread=thread)
        self.assertTrue(res2.success)

        # Thread must have exactly 1 user message and 1 assistant message (no duplicates)
        self.assertEqual(len(thread.messages), 2)
        self.assertEqual(thread.messages[0].role, "user")
        self.assertEqual(thread.messages[1].role, "assistant")
        self.assertIn("Fully recovered answer", thread.messages[1].content)

    # =================================================================
    # Area 2 Audit: Retry Idempotency and Concurrency Control
    # =================================================================
    def test_consecutive_identical_questions_create_separate_legitimate_turns(self) -> None:
        """Area 2 Audit: Identical consecutive queries on completed turns are treated as legitimate separate turns."""
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        # Turn 1: Initial question
        res1 = orchestrator.orchestrate_turn("Explain mitochondria", self.doc_id, thread=thread)
        self.assertTrue(res1.success)
        self.assertEqual(len(thread.messages), 2)

        # Turn 2: Exact same question asked again legitimately
        res2 = orchestrator.orchestrate_turn("Explain mitochondria", self.doc_id, thread=thread)
        self.assertTrue(res2.success)

        # Must have 4 messages: [User, Assistant, User, Assistant]
        self.assertEqual(len(thread.messages), 4)
        roles = [m.role for m in thread.messages]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
        self.assertEqual(thread.messages[0].content, "Explain mitochondria")
        self.assertEqual(thread.messages[2].content, "Explain mitochondria")

    def test_failed_question_followed_by_different_question_preserves_both(self) -> None:
        """Area 1 Audit: A failed question is never deleted when the user asks a different question."""
        failing_provider = MockLLMProvider(should_raise=True)
        working_provider = MockLLMProvider(response="Ribosomes synthesize proteins [Source 1, Page 3].")

        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=failing_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        # Turn 1 fails
        res1 = orchestrator.orchestrate_turn("What is mitochondria?", self.doc_id, thread=thread)
        self.assertFalse(res1.success)
        self.assertEqual(len(thread.messages), 1)
        q1_msg = thread.messages[0]
        self.assertEqual(q1_msg.role, "user")
        self.assertEqual(q1_msg.content, "What is mitochondria?")
        self.assertEqual(q1_msg.metadata["turn_status"], "failed")
        q1_id = q1_msg.message_id
        q1_time = q1_msg.timestamp
        q1_req = q1_msg.metadata["request_id"]

        # Turn 2: User asks different question instead of retrying
        orchestrator.provider = working_provider
        res2 = orchestrator.orchestrate_turn("What is ribosome?", self.doc_id, thread=thread)
        self.assertTrue(res2.success)

        # Thread must have 3 messages: failed Q1, new Q2, and Assistant Answer 2
        self.assertEqual(len(thread.messages), 3)
        self.assertEqual(thread.messages[0].message_id, q1_id)
        self.assertEqual(thread.messages[0].timestamp, q1_time)
        self.assertEqual(thread.messages[0].content, "What is mitochondria?")
        self.assertEqual(thread.messages[0].metadata["turn_status"], "failed")
        self.assertEqual(thread.messages[0].metadata["request_id"], q1_req)

        self.assertEqual(thread.messages[1].role, "user")
        self.assertEqual(thread.messages[1].content, "What is ribosome?")
        self.assertEqual(thread.messages[1].metadata["turn_status"], "completed")

        self.assertEqual(thread.messages[2].role, "assistant")
        self.assertIn("Ribosomes synthesize proteins", thread.messages[2].content)
        self.assertEqual(thread.messages[2].metadata["turn_status"], "completed")

    def test_failed_question_followed_by_retry_preserves_identity_and_metadata(self) -> None:
        """Area 1 Audit: Retrying a failed turn associates with original message, preserving id and timestamp."""
        failing_provider = MockLLMProvider(should_raise=True)
        working_provider = MockLLMProvider(response="Mitochondria produce ATP [Source 1, Page 3].")

        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=failing_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        # Turn 1: Fails
        res1 = orchestrator.orchestrate_turn("Explain mitochondria", self.doc_id, thread=thread)
        self.assertFalse(res1.success)
        orig_msg = thread.messages[0]
        orig_id = orig_msg.message_id
        orig_ts = orig_msg.timestamp
        orig_req = orig_msg.metadata["request_id"]
        self.assertEqual(orig_msg.metadata["turn_status"], "failed")

        # Turn 2: Retries same query with working provider
        orchestrator.provider = working_provider
        res2 = orchestrator.orchestrate_turn("Explain mitochondria", self.doc_id, thread=thread, is_retry=True)
        self.assertTrue(res2.success)

        # Exactly 2 messages in thread
        self.assertEqual(len(thread.messages), 2)
        user_node = thread.messages[0]
        self.assertEqual(user_node.message_id, orig_id)
        self.assertEqual(user_node.timestamp, orig_ts)
        self.assertEqual(user_node.content, "Explain mitochondria")
        self.assertEqual(user_node.metadata["initial_request_id"], orig_req)
        self.assertEqual(user_node.metadata["retry_count"], 1)
        self.assertEqual(user_node.metadata["turn_status"], "completed")

        assistant_node = thread.messages[1]
        self.assertEqual(assistant_node.role, "assistant")
        self.assertIn("Mitochondria produce ATP", assistant_node.content)

    def test_two_consecutive_identical_questions_create_independent_turns(self) -> None:
        """Area 1 Audit: Consecutive identical questions after a completed turn create independent turns."""
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        res1 = orchestrator.orchestrate_turn("Tell me about mitochondria", self.doc_id, thread=thread)
        self.assertTrue(res1.success)

        res2 = orchestrator.orchestrate_turn("Tell me about mitochondria", self.doc_id, thread=thread)
        self.assertTrue(res2.success)

        self.assertEqual(len(thread.messages), 4)
        self.assertEqual([m.role for m in thread.messages], ["user", "assistant", "user", "assistant"])
        self.assertNotEqual(thread.messages[0].message_id, thread.messages[2].message_id)
        self.assertEqual(thread.messages[0].content, "Tell me about mitochondria")
        self.assertEqual(thread.messages[2].content, "Tell me about mitochondria")

    def test_transcript_export_after_failed_turns(self) -> None:
        """Area 1 Audit: Markdown, dictionary, and JSON export handle failed turns cleanly."""
        thread = ConversationThread(document_id=self.doc_id, title="Biology Discussion")
        # Add failed turn
        thread.add_user_message("What is an organelle?")
        thread.messages[0].metadata["turn_status"] = "failed"
        thread.messages[0].metadata["last_error"] = "Ollama connection timeout"

        # Add completed turn
        thread.add_user_message("What is mitochondria?")
        thread.messages[1].metadata["turn_status"] = "completed"
        thread.add_assistant_message(
            "Mitochondria are membrane-bound cell organelles.",
            citations=[{"source_id": 1, "page": 3, "filename": "bio.pdf", "chunk_id": "c1"}],
            metadata={"turn_status": "completed"},
        )

        # Markdown export
        md = thread.to_markdown()
        self.assertIn("# Study Chat Transcript — Biology Discussion", md)
        self.assertIn("### Turn 1 — 👤 **User**", md)
        self.assertIn("What is an organelle?", md)
        self.assertIn("### Turn 2 — 👤 **User**", md)
        self.assertIn("What is mitochondria?", md)
        self.assertIn("### Turn 3 — 🤖 **Assistant**", md)
        self.assertIn("Mitochondria are membrane-bound", md)
        self.assertIn("Page 3", md)

        # Dict and JSON export
        d = thread.to_dict()
        self.assertEqual(len(d["messages"]), 3)
        self.assertEqual(d["messages"][0]["metadata"]["turn_status"], "failed")
        self.assertEqual(d["messages"][1]["metadata"]["turn_status"], "completed")

        j = thread.to_json()
        self.assertIn("What is an organelle?", j)
        self.assertIn("failed", j)

    def test_sliding_window_history_containing_failed_turns(self) -> None:
        """Area 1 Audit: Sliding window memory manager handles failed and unanswered turns safely."""
        mem_mgr = SlidingWindowMemoryManager(
            config=ConversationMemoryConfig(max_history_tokens=200, min_recent_turns=2)
        )
        thread = ConversationThread(document_id=self.doc_id)
        # Turn 1: failed (no assistant answer)
        thread.add_user_message("Failed inquiry 1")
        thread.messages[-1].metadata["turn_status"] = "failed"

        # Turn 2: succeeded
        thread.add_user_message("Successful inquiry 2")
        thread.add_assistant_message("Successful response 2")

        # Turn 3: failed (no assistant answer)
        thread.add_user_message("Failed inquiry 3")
        thread.messages[-1].metadata["turn_status"] = "failed"

        # Turn 4: succeeded
        thread.add_user_message("Successful inquiry 4")
        thread.add_assistant_message("Successful response 4")

        summary, recent = mem_mgr.build_prompt_history(thread, available_budget=300)
        self.assertTrue(len(recent) >= 2)
        # Verify no orphan assistant message at the head
        if recent and recent[0].role == "assistant":
            self.fail("Orphan assistant message appeared at head of history")

        # Verify prompt construction with this thread works without error
        prompt = build_grounded_chat_prompt(
            "New follow up question",
            [{"text": "Sample evidence", "metadata": {"page": 1, "filename": "doc.pdf"}}],
            thread=thread,
            memory_manager=mem_mgr,
        )
        self.assertIn("New follow up question", prompt.user_prompt)

    def test_original_message_ids_and_content_remain_unchanged(self) -> None:
        """Area 1 Audit: Byte-for-byte fidelity of content, message IDs, and timestamps is preserved."""
        failing_provider = MockLLMProvider(should_raise=True)
        working_provider = MockLLMProvider(response="Answer for second question [Source 1, Page 3].")

        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=failing_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)
        special_query = "What is ΔG°' of glucose-6-phosphate + H2O -> glucose + Pi?"

        res1 = orchestrator.orchestrate_turn(special_query, self.doc_id, thread=thread)
        self.assertFalse(res1.success)

        m0 = thread.messages[0]
        m0_id = m0.message_id
        m0_ts = m0.timestamp
        m0_content = m0.content

        # Subsequent turns
        orchestrator.provider = working_provider
        res2 = orchestrator.orchestrate_turn("Different follow up question", self.doc_id, thread=thread)
        self.assertTrue(res2.success)

        res3 = orchestrator.orchestrate_turn("Yet another question", self.doc_id, thread=thread)
        self.assertTrue(res3.success)

        # Message 0 MUST be completely unchanged
        self.assertEqual(thread.messages[0].message_id, m0_id)
        self.assertEqual(thread.messages[0].timestamp, m0_ts)
        self.assertEqual(thread.messages[0].content, m0_content)
        self.assertEqual(thread.messages[0].content, special_query)

    def test_lock_reference_held_without_acquisition_blocks_eviction(self) -> None:
        """Area 1 & 3 Audit: Obtaining a lock reference (lease) without acquiring it blocks eviction."""
        cid = "11111111-1111-4111-8111-111111111111"

        # Caller obtains lock reference/lease but DOES NOT acquire underlying RLock yet
        lease = ChatOrchestrator.acquire_conversation_lock(cid)
        self.assertEqual(ChatOrchestrator.get_lock_ref_count(cid), 1)

        # Cleanup/eviction attempted while reference is held without acquisition
        evicted = ChatOrchestrator.release_thread_lock(cid)
        # Eviction MUST be refused/deferred
        self.assertFalse(evicted)
        self.assertTrue(ChatOrchestrator.is_lock_pending_eviction(cid))
        with ChatOrchestrator._thread_locks_guard:
            self.assertIn(cid, ChatOrchestrator._thread_locks)

        # Once caller finishes and releases lease, deferred eviction removes entry automatically
        lease.release()
        self.assertEqual(ChatOrchestrator.get_lock_ref_count(cid), 0)
        with ChatOrchestrator._thread_locks_guard:
            self.assertNotIn(cid, ChatOrchestrator._thread_locks)

    def test_two_concurrent_requests_for_same_conversation_share_same_lock_instance(self) -> None:
        """Area 1 & 3 Audit: Two concurrent requests for the same conversation always receive the same lock."""
        cid = "22222222-2222-4222-8222-222222222222"

        lease_a = ChatOrchestrator.acquire_conversation_lock(cid)
        lease_b = ChatOrchestrator.acquire_conversation_lock(cid)

        # Both leases MUST reference the exact same underlying RLock instance
        self.assertIs(lease_a.lock, lease_b.lock)
        self.assertEqual(ChatOrchestrator.get_lock_ref_count(cid), 2)

        # Test mutual exclusion deterministically with Events
        worker_entered_event = threading.Event()
        worker_finished_event = threading.Event()
        caller_released_event = threading.Event()

        def _worker_b():
            # Tries to acquire lease_b; must block until lease_a releases
            with lease_b:
                worker_entered_event.set()
            worker_finished_event.set()

        with lease_a:
            t = threading.Thread(target=_worker_b, daemon=True)
            t.start()

            # Worker B cannot enter while lease_a is held
            entered_early = worker_entered_event.wait(timeout=0.05)
            self.assertFalse(entered_early)

        # Lease A is now exited/released
        # Worker B should now acquire and finish
        worker_entered = worker_entered_event.wait(timeout=1.0)
        self.assertTrue(worker_entered)
        t.join(timeout=1.0)

        lease_a.release()
        self.assertEqual(ChatOrchestrator.get_lock_ref_count(cid), 0)

    def test_thread_deleted_during_active_request_deferred_eviction_on_completion(self) -> None:
        """Area 2 & 3 Audit: A thread deleted during an active request is safely evicted upon request completion."""
        manager = ConversationManager()
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
            conversation_manager=manager,
        )
        t = orchestrator.get_or_create_thread(self.doc_id, conversation_id="33333333-3333-4333-8333-333333333333")
        cid = t.conversation_id

        # Active request turn begins
        with orchestrator.acquire_conversation_lock(cid):
            self.assertEqual(ChatOrchestrator.get_lock_ref_count(cid), 1)

            # Thread deletion occurs while request is in-flight
            deleted = orchestrator.delete_thread(self.doc_id, cid)
            self.assertTrue(deleted)
            # Memory manager deleted the thread
            self.assertIsNone(manager.get_thread(self.doc_id, cid))

            # Eviction was deferred because ref_count > 0
            self.assertTrue(ChatOrchestrator.is_lock_pending_eviction(cid))
            with ChatOrchestrator._thread_locks_guard:
                self.assertIn(cid, ChatOrchestrator._thread_locks)

        # Upon exiting request context, deferred eviction triggers automatically on final release
        self.assertEqual(ChatOrchestrator.get_lock_ref_count(cid), 0)
        with ChatOrchestrator._thread_locks_guard:
            self.assertNotIn(cid, ChatOrchestrator._thread_locks)

    def test_independent_conversations_execute_concurrently_with_barrier(self) -> None:
        """Area 1 & 3 Audit: Independent conversations acquire distinct locks and execute concurrently."""
        cid1 = "44444444-4444-4444-8444-444444444444"
        cid2 = "55555555-5555-4555-8555-555555555555"

        barrier = threading.Barrier(2, timeout=2.0)
        thread1_in_critical = False
        thread2_in_critical = False
        both_active_concurrently = False

        def _worker_1():
            nonlocal thread1_in_critical, both_active_concurrently
            with ChatOrchestrator.acquire_conversation_lock(cid1):
                thread1_in_critical = True
                barrier.wait()
                if thread2_in_critical:
                    both_active_concurrently = True
                thread1_in_critical = False

        def _worker_2():
            nonlocal thread2_in_critical, both_active_concurrently
            with ChatOrchestrator.acquire_conversation_lock(cid2):
                thread2_in_critical = True
                barrier.wait()
                if thread1_in_critical:
                    both_active_concurrently = True
                thread2_in_critical = False

        t1 = threading.Thread(target=_worker_1, daemon=True)
        t2 = threading.Thread(target=_worker_2, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        self.assertTrue(both_active_concurrently)

    def test_repeated_thread_creation_and_deletion_zero_registry_growth(self) -> None:
        """Area 2 & 3 Audit: Repeated cycles of thread use and deletion leave registry at zero growth."""
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
        )

        for i in range(25):
            cid = f"transient-thread-{i}"
            with orchestrator.acquire_conversation_lock(cid):
                pass
            orchestrator.release_thread_lock(cid)

        self.assertEqual(ChatOrchestrator.get_active_lock_count(), 0)

    def test_exceptions_during_request_execution_release_references_and_permit_cleanup(self) -> None:
        """Area 3 Audit: Exceptions inside turn execution reliably decrement ref_count and execute deferred eviction."""
        cid = "66666666-6666-4666-8666-666666666666"

        try:
            with ChatOrchestrator.acquire_conversation_lock(cid):
                # Trigger deferred eviction request during turn
                ChatOrchestrator.release_thread_lock(cid)
                raise RuntimeError("Simulated mid-turn crash")
        except RuntimeError:
            pass

        # Reference count MUST be decremented to 0 despite exception
        self.assertEqual(ChatOrchestrator.get_lock_ref_count(cid), 0)
        # Deferred eviction MUST have executed
        with ChatOrchestrator._thread_locks_guard:
            self.assertNotIn(cid, ChatOrchestrator._thread_locks)

    def test_concurrent_requests_for_same_thread_are_safely_serialized(self) -> None:
        """Area 2 Audit: Concurrent requests targeting the same conversation thread are serialized safely."""
        import concurrent.futures

        class SlowMockProvider(BaseLLMProvider):
            def __init__(self):
                super().__init__(model_name="qwen2.5:7b")
                self.calls = 0

            def is_configured(self):
                return True

            def generate(self, sys, usr, **kwargs):
                self.calls += 1
                time.sleep(0.04)
                return f"Grounded response {self.calls} [Source 1, Page 3]."

        slow_provider = SlowMockProvider()
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=slow_provider,
        )
        thread = ConversationThread(document_id=self.doc_id)

        def _run_turn(query: str):
            return orchestrator.orchestrate_turn(query, self.doc_id, thread=thread)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(_run_turn, "Concurrent inquiry A")
            f2 = executor.submit(_run_turn, "Concurrent inquiry B")
            r1 = f1.result()
            r2 = f2.result()

        self.assertTrue(r1.success)
        self.assertTrue(r2.success)
        # 4 messages, strictly alternating user and assistant
        self.assertEqual(len(thread.messages), 4)
        roles = [m.role for m in thread.messages]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])

    def test_concurrent_requests_for_different_threads_are_independent(self) -> None:
        """Area 2 Audit: Concurrent requests targeting different threads execute independently."""
        import concurrent.futures

        manager = ConversationManager()
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
            conversation_manager=manager,
        )

        thread_a = manager.get_or_create_thread(self.doc_id, conversation_id="aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa")
        thread_b = manager.get_or_create_thread(self.doc_id, conversation_id="bbbbbbbb-bbbb-4bbb-bbbb-bbbbbbbbbbbb")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fa = executor.submit(orchestrator.orchestrate_turn, "Question for A", self.doc_id, thread=thread_a)
            fb = executor.submit(orchestrator.orchestrate_turn, "Question for B", self.doc_id, thread=thread_b)
            ra = fa.result()
            rb = fb.result()

        self.assertTrue(ra.success)
        self.assertTrue(rb.success)
        self.assertEqual(len(thread_a.messages), 2)
        self.assertEqual(len(thread_b.messages), 2)
        self.assertEqual(thread_a.messages[0].content, "Question for A")
        self.assertEqual(thread_b.messages[0].content, "Question for B")

    # =================================================================
    # Area 3 Audit: Summary-Memory Failure Isolation
    # =================================================================
    def test_summary_generation_failure_does_not_invalidate_answer(self) -> None:
        """Area 3 Audit: Secondary summarisation failure leaves answer valid and records summary_error."""
        mem_mgr = SlidingWindowMemoryManager(
            config=ConversationMemoryConfig(max_history_tokens=50),
        )
        mem_mgr.update_memory = MagicMock(
            side_effect=RuntimeError("Summarization engine out of memory")
        )
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
            memory_manager=mem_mgr,
        )
        thread = ConversationThread(document_id=self.doc_id)
        # Populate history to force summarization trigger
        for i in range(5):
            thread.add_user_message(f"Historical user inquiry {i}")
            thread.add_assistant_message(f"Historical assistant response {i}")

        result = orchestrator.orchestrate_turn(
            "New question about mitochondria", self.doc_id, thread=thread
        )

        self.assertTrue(result.success)
        self.assertIn("Mitochondria generate cellular energy", result.answer)
        self.assertIsNotNone(result.summary_error)
        self.assertIn("Summarization engine out of memory", result.summary_error)
        self.assertIsNone(result.error_message)
        # Assistant message was still committed to thread
        self.assertEqual(thread.messages[-1].role, "assistant")

    def test_failed_summary_preserves_existing_summary_memory(self) -> None:
        """Area 3 Audit: Failed summary generation preserves existing summary memory without corrupting it."""
        mem_mgr = SlidingWindowMemoryManager(
            config=ConversationMemoryConfig(max_history_tokens=50),
        )
        mem_mgr.update_memory = MagicMock(
            side_effect=RuntimeError("Summarizer connection dropped")
        )
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
            memory_manager=mem_mgr,
        )
        thread = ConversationThread(document_id=self.doc_id)
        thread.summary_memory = "Established historical academic summary of cell biology."
        for i in range(5):
            thread.add_user_message(f"Historical user inquiry {i}")
            thread.add_assistant_message(f"Historical assistant response {i}")

        result = orchestrator.orchestrate_turn("How is ATP made?", self.doc_id, thread=thread)

        self.assertTrue(result.success)
        # Existing summary memory MUST be preserved untouched
        self.assertEqual(
            thread.summary_memory,
            "Established historical academic summary of cell biology.",
        )
        self.assertIsNotNone(result.summary_error)

    def test_repeated_summary_updates_preserve_transcript_integrity(self) -> None:
        """Area 3 Audit: Successive turns with memory updates keep all thread.messages nodes intact."""
        orchestrator = ChatOrchestrator(
            store=self.mock_store,
            embedding_service=self.mock_embeddings,
            provider=self.mock_provider,
            memory_config=ConversationMemoryConfig(max_history_tokens=100),
        )
        thread = ConversationThread(document_id=self.doc_id)

        for i in range(4):
            res = orchestrator.orchestrate_turn(f"Question turn {i}", self.doc_id, thread=thread)
            self.assertTrue(res.success)

        # 4 turns = exactly 8 messages in transcript
        self.assertEqual(len(thread.messages), 8)
        # Verify messages are intact and chronological
        for i in range(4):
            self.assertEqual(thread.messages[i * 2].role, "user")
            self.assertEqual(thread.messages[i * 2].content, f"Question turn {i}")
            self.assertEqual(thread.messages[i * 2 + 1].role, "assistant")


if __name__ == "__main__":
    unittest.main()
