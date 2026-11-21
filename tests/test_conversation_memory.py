"""Unit tests for Stage 10.1 conversation memory and sliding-window engine.

Covers all 14 required verification aspects:
1. Message creation and serialization
2. Conversation creation and serialization
3. Full 36-character UUID identifiers
4. Token estimation
5. Sliding-window behavior
6. Preservation of recent turns
7. Older-turn summarization (injectable)
8. Deterministic fallback when summarization fails
9. Context-budget enforcement and safety margin
10. Empty conversations and clear functionality
11. Multiple conversation isolation per document
12. Corrupted and incomplete serialized data resilience
13. Markdown and JSON exports
14. Backward-compatible schema handling
"""

import json
import unittest
import uuid

from src.chat.conversation_memory import (
    ChatMessage,
    ConversationMemoryConfig,
    ConversationThread,
    ConversationManager,
    SlidingWindowMemoryManager,
    deterministic_extractive_summary,
    is_valid_uuid,
    new_uuid,
)


class TestChatMessage(unittest.TestCase):
    """Test ChatMessage creation, validation, tokens, and serialization."""

    def test_01_message_creation_and_full_uuid(self):
        msg = ChatMessage(role="user", content="Explain ATP synthesis.")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Explain ATP synthesis.")
        self.assertTrue(is_valid_uuid(msg.message_id))
        self.assertEqual(len(msg.message_id), 36)
        self.assertGreater(msg.token_count, 0)
        self.assertIsInstance(msg.timestamp, str)

    def test_02_invalid_role_rejected(self):
        with self.assertRaises(ValueError):
            ChatMessage(role="moderator", content="Hello")

    def test_03_serialization_round_trip(self):
        citations = [
            {
                "source_index": 1,
                "chunk_id": "c1",
                "filename": "bio.pdf",
                "page": 4,
                "page_range": "4",
                "text": "Mitochondria produce ATP.",
                "score": 0.88,
            }
        ]
        msg = ChatMessage(
            role="assistant",
            content="ATP is generated in the mitochondria.",
            citations=citations,
            metadata={"rewritten_query": "ATP synthesis mechanism"},
        )
        data = msg.to_dict()
        restored = ChatMessage.from_dict(data)

        self.assertEqual(restored.message_id, msg.message_id)
        self.assertEqual(restored.role, "assistant")
        self.assertEqual(restored.content, msg.content)
        self.assertEqual(restored.token_count, msg.token_count)
        self.assertEqual(len(restored.citations), 1)
        self.assertEqual(restored.citations[0]["chunk_id"], "c1")
        self.assertEqual(restored.metadata["rewritten_query"], "ATP synthesis mechanism")

    def test_04_from_dict_with_invalid_uuid_replaces_with_valid_uuid(self):
        data = {
            "role": "user",
            "content": "Short ID test",
            "message_id": "short-id-123",  # Not a valid 36-char UUID
        }
        msg = ChatMessage.from_dict(data)
        self.assertTrue(is_valid_uuid(msg.message_id))
        self.assertEqual(len(msg.message_id), 36)


class TestConversationThread(unittest.TestCase):
    """Test ConversationThread lifecycle, turns, title auto-generation, and exports."""

    def test_01_thread_creation_and_full_uuid(self):
        thread = ConversationThread(document_id="doc_bio_123")
        self.assertEqual(thread.document_id, "doc_bio_123")
        self.assertTrue(is_valid_uuid(thread.conversation_id))
        self.assertEqual(len(thread.conversation_id), 36)
        self.assertEqual(thread.title, "New Conversation")
        self.assertEqual(len(thread.messages), 0)
        self.assertEqual(thread.get_turn_count(), 0)

    def test_02_add_messages_and_auto_title(self):
        thread = ConversationThread(document_id="doc_bio_123")
        user_msg = thread.add_user_message("What is the function of chloroplasts?")
        self.assertEqual(user_msg.role, "user")
        self.assertEqual(thread.get_turn_count(), 1)
        # Title should automatically derive from first user question
        self.assertEqual(thread.title, "What is the function of chloroplasts?")

        asst_msg = thread.add_assistant_message(
            "Chloroplasts conduct photosynthesis.",
            citations=[{"page": 2, "filename": "bio.pdf", "chunk_id": "c2"}],
        )
        self.assertEqual(asst_msg.role, "assistant")
        self.assertEqual(len(thread.messages), 2)
        self.assertGreater(thread.get_total_tokens(), 0)

    def test_03_serialization_round_trip_and_json(self):
        thread = ConversationThread(document_id="doc_bio_123", title="Cell Biology")
        thread.add_user_message("Q1")
        thread.add_assistant_message("A1")
        thread.summary_memory = "Summary of prior turns"

        serialized = thread.to_dict()
        self.assertIn("version", serialized)
        self.assertEqual(serialized["version"], 1)

        restored = ConversationThread.from_dict(serialized)
        self.assertEqual(restored.conversation_id, thread.conversation_id)
        self.assertEqual(restored.document_id, "doc_bio_123")
        self.assertEqual(restored.title, "Cell Biology")
        self.assertEqual(len(restored.messages), 2)
        self.assertEqual(restored.summary_memory, "Summary of prior turns")

        # JSON output check
        json_str = thread.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["conversation_id"], thread.conversation_id)

    def test_04_markdown_export(self):
        thread = ConversationThread(document_id="doc_bio_123", title="Plant Biology")
        thread.add_user_message("Where does the Calvin cycle occur?")
        thread.add_assistant_message(
            "The Calvin cycle occurs in the stroma of chloroplasts.",
            citations=[
                {
                    "page": 12,
                    "filename": "biology.pdf",
                    "chunk_id": "chk_12",
                    "text": "The Calvin cycle takes place in the stroma.",
                    "score": 0.92,
                }
            ],
        )
        md = thread.to_markdown()
        self.assertIn("# Study Chat Transcript — Plant Biology", md)
        self.assertIn("doc_bio_123", md)
        self.assertIn("Where does the Calvin cycle occur?", md)
        self.assertIn("The Calvin cycle occurs in the stroma", md)
        self.assertIn("biology.pdf", md)
        self.assertIn("chk_12", md)

    def test_05_empty_thread_and_clear(self):
        thread = ConversationThread(document_id="doc_bio_123")
        self.assertEqual(thread.get_total_tokens(), 0)
        md_empty = thread.to_markdown()
        self.assertIn("*No messages in this conversation yet.*", md_empty)

        thread.add_user_message("Hello")
        thread.add_assistant_message("Hi")
        thread.summary_memory = "Some summary"
        self.assertEqual(len(thread.messages), 2)

        thread.clear()
        self.assertEqual(len(thread.messages), 0)
        self.assertEqual(thread.summary_memory, "")
        self.assertEqual(thread.get_total_tokens(), 0)


class TestContextBudgetAndSlidingWindow(unittest.TestCase):
    """Test dynamic context-budget calculation, sliding-window selection, and truncation."""

    def test_01_configurable_context_budget_calculation(self):
        cfg = ConversationMemoryConfig(
            context_window_tokens=4096,
            max_history_tokens=1500,
            output_reserve_tokens=1000,
            system_prompt_reserve_tokens=400,
            doc_context_reserve_tokens=1000,
            safety_margin_ratio=0.10,
        )
        # When doc context is 800 tokens, sys prompt is 300 tokens, user query is 50:
        # non_history = 300 + 800 + 50 + 1000 = 2150
        # raw_remaining = 4096 - 2150 = 1946
        # safe_remaining = 1946 * 0.90 = 1751
        # clamped to max_history_tokens (1500) -> 1500
        budget = cfg.compute_available_history_budget(
            doc_context_tokens=800,
            system_prompt_tokens=300,
            user_query_tokens=50,
        )
        self.assertEqual(budget, 1500)

        # When context window is smaller (e.g. 2048):
        cfg_small = ConversationMemoryConfig(
            context_window_tokens=2048,
            max_history_tokens=1500,
            output_reserve_tokens=600,
            safety_margin_ratio=0.10,
        )
        # doc_context: 800, sys: 300, user: 50 -> non_history: 1750
        # raw_remaining = 2048 - 1750 = 298
        # safe_remaining = 298 * 0.90 = 268
        budget_small = cfg_small.compute_available_history_budget(
            doc_context_tokens=800,
            system_prompt_tokens=300,
            user_query_tokens=50,
        )
        self.assertEqual(budget_small, 268)
        self.assertLess(budget_small, 1500)

    def test_02_sliding_window_under_budget_includes_all_messages(self):
        manager = SlidingWindowMemoryManager()
        thread = ConversationThread(document_id="doc_1")
        thread.add_user_message("What is DNA?")
        thread.add_assistant_message("DNA is deoxyribonucleic acid.")

        summary, msgs = manager.build_prompt_history(thread, available_budget=1000)
        self.assertEqual(summary, "")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].content, "What is DNA?")
        self.assertEqual(msgs[1].content, "DNA is deoxyribonucleic acid.")

    def test_03_sliding_window_exceeding_budget_preserves_most_recent_turns(self):
        manager = SlidingWindowMemoryManager()
        thread = ConversationThread(document_id="doc_1")

        # Add 5 conversational turns with known token footprints
        for i in range(1, 6):
            thread.add_user_message(f"User Question {i} " * 15)  # ~30 tokens
            thread.add_assistant_message(f"Assistant Answer {i} " * 25)  # ~50 tokens

        # Request a strict budget that only fits the last 2 turns (~160 tokens)
        budget = 180
        summary, msgs = manager.build_prompt_history(thread, available_budget=budget)

        total_selected_tokens = sum(m.token_count for m in msgs)
        self.assertLessEqual(total_selected_tokens, budget)
        # Must preserve the most recent message (Answer 5 and Question 5)
        self.assertEqual(msgs[-1].content.strip(), f"{('Assistant Answer 5 ' * 25).strip()}")
        self.assertEqual(msgs[-2].content.strip(), f"{('User Question 5 ' * 15).strip()}")
        # Older turns (1, 2, 3) must be excluded to strictly respect budget
        self.assertFalse(any("Question 1" in m.content for m in msgs))

    def test_04_explicit_zero_tokens_budget_not_overwritten_by_defaults(self):
        cfg = ConversationMemoryConfig(
            context_window_tokens=4096,
            max_history_tokens=3000,
            output_reserve_tokens=1000,
            system_prompt_reserve_tokens=400,
            doc_context_reserve_tokens=1000,
            safety_margin_ratio=0.10,
        )
        # When explicitly passing 0 for system and doc tokens:
        # non_history = 0 + 0 + 0 + 1000 = 1000
        # raw_remaining = 4096 - 1000 = 3096
        # safe_remaining = int(3096 * 0.90) = 2786
        budget_zeros = cfg.compute_available_history_budget(
            doc_context_tokens=0,
            system_prompt_tokens=0,
            user_query_tokens=0,
        )
        self.assertEqual(budget_zeros, 2786)

        # When passing None (unsupplied), it must use the configured default reserves:
        # non_history = 400 + 1000 + 0 + 1000 = 2400
        # raw_remaining = 4096 - 2400 = 1696
        # safe_remaining = int(1696 * 0.90) = 1526
        budget_defaults = cfg.compute_available_history_budget(
            doc_context_tokens=None,
            system_prompt_tokens=None,
            user_query_tokens=None,
        )
        self.assertEqual(budget_defaults, 1526)
        self.assertNotEqual(budget_zeros, budget_defaults)

        # Non-negative guard when reserves exceed context window
        budget_overflow = cfg.compute_available_history_budget(
            doc_context_tokens=3500,
            system_prompt_tokens=1000,
        )
        self.assertEqual(budget_overflow, 0)

    def test_05_pair_aware_turn_selection_drops_orphan_assistant_message(self):
        manager = SlidingWindowMemoryManager()
        thread = ConversationThread(document_id="doc_1")

        # Turn 1: User 1 (~60 tokens), Assistant 1 (~60 tokens)
        thread.add_user_message("What is cell respiration? " * 15)
        thread.add_assistant_message("Cellular respiration converts glucose into ATP. " * 10)

        # Turn 2: User 2 (~20 tokens), Assistant 2 (~25 tokens)
        thread.add_user_message("Where does it occur?")
        thread.add_assistant_message("It occurs in the mitochondria and cytoplasm.")

        turn2_tokens = thread.messages[2].token_count + thread.messages[3].token_count
        asst1_tokens = thread.messages[1].token_count

        # Set budget to fit Turn 2 + Assistant 1, but NOT User 1
        budget = turn2_tokens + asst1_tokens + 5
        summary, msgs = manager.build_prompt_history(thread, available_budget=budget)

        # Pair-awareness: The orphan Assistant 1 message must be dropped
        # The prompt history must begin with User 2, not Assistant 1
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].role, "user")
        self.assertEqual(msgs[0].content, "Where does it occur?")
        self.assertEqual(msgs[1].role, "assistant")
        self.assertIn("mitochondria", msgs[1].content)

    def test_06_oversized_single_message_safe_truncation(self):
        manager = SlidingWindowMemoryManager()
        thread = ConversationThread(document_id="doc_1")
        long_content = "Photosynthesis electron transport chain " * 80  # ~320 tokens
        thread.add_user_message(long_content)

        self.assertGreater(thread.messages[0].token_count, 200)

        # Strict budget of 80 tokens
        budget = 80
        summary, msgs = manager.build_prompt_history(thread, available_budget=budget)

        self.assertEqual(len(msgs), 1)
        self.assertLessEqual(msgs[0].token_count, budget)
        self.assertIn("[truncated to fit context budget]", msgs[0].content)

        # Verify the original message in thread.messages is unmutated!
        self.assertEqual(thread.messages[0].content, long_content)
        self.assertNotIn("[truncated", thread.messages[0].content)


class TestMemorySummarization(unittest.TestCase):
    """Test injectable summarizer, deterministic fallback, and memory preservation."""

    def test_01_injectable_llm_summarizer_called(self):
        called_with = []

        def mock_llm_summarizer(messages, existing_summary):
            called_with.append((messages, existing_summary))
            return "Synthesized summary from Qwen LLM"

        manager = SlidingWindowMemoryManager(
            config=ConversationMemoryConfig(max_history_tokens=100, min_recent_turns=1),
            summarizer=mock_llm_summarizer,
        )
        thread = ConversationThread(document_id="doc_1")
        # Add 4 turns (8 messages) to trigger threshold
        for i in range(1, 5):
            thread.add_user_message(f"What is concept {i} in genetics?")
            thread.add_assistant_message(f"Concept {i} relates to allele dominance and inheritance.")

        updated = manager.update_memory(thread, force_summary=True)
        self.assertTrue(updated)
        self.assertEqual(len(called_with), 1)
        self.assertEqual(thread.summary_memory, "Synthesized summary from Qwen LLM")

    def test_02_deterministic_fallback_when_summarizer_raises(self):
        def failing_summarizer(messages, existing_summary):
            raise ConnectionError("Ollama daemon unreachable at http://localhost:11434")

        manager = SlidingWindowMemoryManager(
            config=ConversationMemoryConfig(max_history_tokens=100, min_recent_turns=1),
            summarizer=failing_summarizer,
        )
        thread = ConversationThread(document_id="doc_1")
        thread.add_user_message("Explain the Krebs cycle.")
        thread.add_assistant_message("The citric acid cycle generates NADH and FADH2 in the mitochondrial matrix.")
        thread.add_user_message("What about glycolysis?")
        thread.add_assistant_message("Glycolysis takes place in the cytosol.")

        # update_memory must not raise; it must fallback to deterministic extractive summary
        updated = manager.update_memory(thread, force_summary=True)
        self.assertTrue(updated)
        self.assertIn("Earlier Discussion Summary", thread.summary_memory)
        self.assertIn("Krebs cycle", thread.summary_memory)
        # Existing messages in thread must remain completely intact
        self.assertEqual(len(thread.messages), 4)

    def test_03_deterministic_extractive_summary_content(self):
        msgs = [
            ChatMessage(role="user", content="Can you explain photolysis in photosystem II?"),
            ChatMessage(
                role="assistant",
                content="Photolysis splits water molecules into oxygen, protons, and electrons.",
                citations=[{"page": 8, "filename": "plants.pdf"}],
            ),
        ]
        summary = deterministic_extractive_summary(msgs, existing_summary="", max_tokens=200)
        self.assertIn("User inquired about:", summary)
        self.assertIn("photolysis in photosystem II", summary)
        self.assertIn("citing pp. 8", summary)

    def test_04_prune_older_messages_once_folded(self):
        manager = SlidingWindowMemoryManager(
            config=ConversationMemoryConfig(min_recent_turns=2)
        )
        thread = ConversationThread(document_id="doc_1")
        for i in range(1, 6):
            thread.add_user_message(f"Question {i}")
            thread.add_assistant_message(f"Answer {i}")

        self.assertEqual(len(thread.messages), 10)
        # Keep last 2 turns = 4 messages; 6 older messages should be pruned
        pruned_count = manager.prune_older_messages(thread, keep_last_n_turns=2)
        self.assertEqual(pruned_count, 6)
        self.assertEqual(len(thread.messages), 4)
        self.assertEqual(thread.messages[0].content, "Question 4")
        self.assertEqual(thread.messages[-1].content, "Answer 5")

    def test_05_summarization_preserves_full_original_transcript(self):
        """Verify that updating summary memory NEVER mutates or deletes original transcript messages."""
        manager = SlidingWindowMemoryManager(
            config=ConversationMemoryConfig(max_history_tokens=100, min_recent_turns=1)
        )
        thread = ConversationThread(document_id="doc_bio_preserve")
        for i in range(1, 6):
            thread.add_user_message(f"Explain biological mechanism {i} in detail.")
            thread.add_assistant_message(
                f"Mechanism {i} involves cellular signaling and enzymes.",
                citations=[{"page": i, "chunk_id": f"chk_{i}", "filename": "bio.pdf"}],
            )

        self.assertEqual(len(thread.messages), 10)
        original_ids = [m.message_id for m in thread.messages]
        original_contents = [m.content for m in thread.messages]
        original_timestamps = [m.timestamp for m in thread.messages]

        # Trigger summarization
        updated = manager.update_memory(thread, force_summary=True)
        self.assertTrue(updated)
        self.assertTrue(len(thread.summary_memory) > 0)

        # Original messages MUST be 100% intact
        self.assertEqual(len(thread.messages), 10)
        self.assertEqual([m.message_id for m in thread.messages], original_ids)
        self.assertEqual([m.content for m in thread.messages], original_contents)
        self.assertEqual([m.timestamp for m in thread.messages], original_timestamps)

        # Markdown export and JSON export must preserve all 10 messages
        md = thread.to_markdown()
        self.assertEqual(md.count("Turn "), 10)
        self.assertIn("Summary of Earlier Discussion", md)

        data = thread.to_dict()
        self.assertEqual(len(data["messages"]), 10)


class TestConversationManager(unittest.TestCase):
    """Test multi-thread isolation per document and persistence management."""

    def test_01_create_and_retrieve_multiple_threads_per_document(self):
        mgr = ConversationManager()
        bio_t1 = mgr.get_or_create_thread(document_id="bio_pdf", title="Genetics Chat")
        bio_t2 = mgr.get_or_create_thread(document_id="bio_pdf", title="Ecology Chat")

        self.assertNotEqual(bio_t1.conversation_id, bio_t2.conversation_id)
        self.assertEqual(len(mgr.list_threads("bio_pdf")), 2)

        # Retrieve specific thread
        fetched = mgr.get_thread("bio_pdf", bio_t1.conversation_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.title, "Genetics Chat")

    def test_02_document_isolation(self):
        mgr = ConversationManager()
        bio_thread = mgr.get_or_create_thread(document_id="bio_pdf")
        bio_thread.add_user_message("Biology question")

        hist_thread = mgr.get_or_create_thread(document_id="history_pdf")
        hist_thread.add_user_message("History question")

        # Verify threads are strictly partitioned by document
        self.assertEqual(len(mgr.list_threads("bio_pdf")), 1)
        self.assertEqual(len(mgr.list_threads("history_pdf")), 1)
        self.assertIsNone(mgr.get_thread("history_pdf", bio_thread.conversation_id))

    def test_03_thread_deletion_and_clear_document(self):
        mgr = ConversationManager()
        t1 = mgr.get_or_create_thread("chem_pdf")
        t2 = mgr.get_or_create_thread("chem_pdf")

        # Delete single thread
        deleted = mgr.delete_thread("chem_pdf", t1.conversation_id)
        self.assertTrue(deleted)
        self.assertEqual(len(mgr.list_threads("chem_pdf")), 1)

        # Clear document threads completely (e.g. upon document deletion)
        mgr.clear_document_threads("chem_pdf")
        self.assertEqual(len(mgr.list_threads("chem_pdf")), 0)

    def test_04_serialization_and_corrupted_data_recovery(self):
        mgr = ConversationManager()
        t = mgr.get_or_create_thread("physics_pdf", title="Quantum Mechanics")
        t.add_user_message("What is entanglement?")
        t.add_assistant_message("Quantum entanglement is...")

        data = mgr.to_dict()
        restored_mgr = ConversationManager()
        restored_mgr.load_dict(data)

        threads = restored_mgr.list_threads("physics_pdf")
        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0].title, "Quantum Mechanics")
        self.assertEqual(len(threads[0].messages), 2)

        # Test recovery from invalid / corrupted data
        corrupted = {"physics_pdf": {"bad_thread": "not a dictionary"}}
        robust_mgr = ConversationManager()
        robust_mgr.load_dict(corrupted)
        self.assertEqual(len(robust_mgr.list_threads("physics_pdf")), 0)

        none_mgr = ConversationManager()
        none_mgr.load_dict("not a dictionary")
        self.assertEqual(none_mgr.to_dict(), {})


if __name__ == "__main__":
    unittest.main()
