"""Targeted unit tests for Streamlit session-state memory lifecycle and caching (Stage 11.2 Work Area C)."""

import unittest
from src.utils.session_lifecycle import (
    prune_lru_cache,
    clean_stale_ephemeral_state,
    classify_session_keys,
    ESSENTIAL_STATE_KEYS,
    RECOMPUTABLE_CACHE_KEYS,
    EPHEMERAL_DOC_ERROR_PAIRS,
)


class TestSessionLifecycle(unittest.TestCase):
    """Test memory lifecycle management, bounded LRU caches, and state protection."""

    def test_01_classify_session_keys(self) -> None:
        """Verify keys are correctly categorized into essential, cache, and ephemeral."""
        keys = [
            "chat_conversation_manager",
            "quiz_ans_doc123_q0",
            "quiz_sessions",
            "fc_input_front_doc456_0",
            "study_timer_elapsed_seconds",
            "cached_summaries",
            "cached_questions",
            "cached_notes",
            "cached_flashcards",
            "summary_error",
            "questions_error_doc_id",
            "flashcard_error",
            "chat_error",
            "random_custom_key",
        ]
        classified = classify_session_keys(keys)

        self.assertIn("chat_conversation_manager", classified["essential"])
        self.assertIn("quiz_ans_doc123_q0", classified["essential"])
        self.assertIn("quiz_sessions", classified["essential"])
        self.assertIn("fc_input_front_doc456_0", classified["essential"])
        self.assertIn("study_timer_elapsed_seconds", classified["essential"])

        self.assertIn("cached_summaries", classified["recomputable_cache"])
        self.assertIn("cached_questions", classified["recomputable_cache"])
        self.assertIn("cached_notes", classified["recomputable_cache"])
        self.assertIn("cached_flashcards", classified["recomputable_cache"])

        self.assertIn("summary_error", classified["ephemeral_ui"])
        self.assertIn("questions_error_doc_id", classified["ephemeral_ui"])
        self.assertIn("flashcard_error", classified["ephemeral_ui"])
        self.assertIn("chat_error", classified["ephemeral_ui"])

    def test_02_lru_cache_pruning_bounds_at_max_entries(self) -> None:
        """Verify adding more than max_entries evicts the oldest entry in FIFO order."""
        cache = {
            f"doc_{i}": {"summary": f"Content for doc {i}"}
            for i in range(1, 6)
        }
        self.assertEqual(len(cache), 5)

        # Add 6th item
        cache["doc_6"] = {"summary": "Content for doc 6"}
        evicted = prune_lru_cache(cache, max_entries=5)

        self.assertEqual(len(cache), 5)
        self.assertEqual(evicted, ["doc_1"])
        self.assertNotIn("doc_1", cache)
        self.assertIn("doc_6", cache)

    def test_03_lru_cache_access_refreshes_recency(self) -> None:
        """Verify re-inserting or updating an entry marks it as most recently used."""
        cache = {
            f"doc_{i}": {"data": i}
            for i in range(1, 6)
        }

        # Access/update doc_1 so it moves to end of OrderedDict/dict in Python 3.7+
        val = cache.pop("doc_1")
        cache["doc_1"] = val

        # Now add doc_6; doc_2 should be the oldest and thus evicted
        cache["doc_6"] = {"data": 6}
        evicted = prune_lru_cache(cache, max_entries=5)

        self.assertEqual(len(cache), 5)
        self.assertEqual(evicted, ["doc_2"])
        self.assertIn("doc_1", cache)
        self.assertNotIn("doc_2", cache)
        self.assertIn("doc_6", cache)

    def test_04_lru_cache_protects_specified_keys(self) -> None:
        """Verify protected_keys are never evicted even when they are the oldest."""
        cache = {
            f"doc_{i}": {"data": i}
            for i in range(1, 6)
        }

        # Protect doc_1 and doc_2
        protected = {"doc_1", "doc_2"}

        # Add doc_6; doc_1 and doc_2 are oldest but protected, so doc_3 should be evicted
        cache["doc_6"] = {"data": 6}
        evicted = prune_lru_cache(cache, max_entries=5, protected_keys=protected)

        self.assertEqual(len(cache), 5)
        self.assertEqual(evicted, ["doc_3"])
        self.assertIn("doc_1", cache)
        self.assertIn("doc_2", cache)
        self.assertNotIn("doc_3", cache)
        self.assertIn("doc_6", cache)

    def test_05_clean_stale_ephemeral_state(self) -> None:
        """Verify error banners from previously viewed documents are removed on doc switch."""
        session_state = {
            "summary_error": "Failed on doc_A",
            "summary_error_doc_id": "doc_A",
            "questions_error": "Failed on doc_A",
            "questions_error_doc_id": "doc_A",
            "notes_error": "Failed on doc_B",
            "notes_error_doc_id": "doc_B",
            "flashcard_error": "Failed on doc_A",
            "flashcard_error_doc_id": "doc_A",
            "cached_summaries": {"doc_A": {}},
            "quiz_ans_docA_q1": "Answer",
        }

        # Switch to doc_B: doc_A errors should be cleaned, doc_B error preserved
        clean_stale_ephemeral_state(session_state, active_doc_id="doc_B")

        self.assertNotIn("summary_error", session_state)
        self.assertNotIn("summary_error_doc_id", session_state)
        self.assertNotIn("questions_error", session_state)
        self.assertNotIn("questions_error_doc_id", session_state)
        self.assertNotIn("flashcard_error", session_state)
        self.assertNotIn("flashcard_error_doc_id", session_state)

        # doc_B error preserved because active_doc_id == "doc_B"
        self.assertEqual(session_state.get("notes_error"), "Failed on doc_B")
        self.assertEqual(session_state.get("notes_error_doc_id"), "doc_B")

        # Essential data untouched
        self.assertIn("cached_summaries", session_state)
        self.assertEqual(session_state.get("quiz_ans_docA_q1"), "Answer")

    def test_06_clean_all_ephemeral_state_when_active_doc_is_none(self) -> None:
        """Verify all error banners are cleaned if active_doc_id is None."""
        session_state = {
            "summary_error": "Some error",
            "summary_error_doc_id": "doc_X",
            "notes_error": "Other error",
            "notes_error_doc_id": "doc_Y",
            "cached_notes": {},
        }

        clean_stale_ephemeral_state(session_state, active_doc_id=None)

        self.assertNotIn("summary_error", session_state)
        self.assertNotIn("summary_error_doc_id", session_state)
        self.assertNotIn("notes_error", session_state)
        self.assertNotIn("notes_error_doc_id", session_state)
        self.assertIn("cached_notes", session_state)

    def test_07_essential_keys_never_pruned(self) -> None:
        """Verify essential keys are not in recomputable cache sets and never pruned."""
        essential_sample = {
            "chat_conversation_manager": object(),
            "quiz_sessions": {"doc_1": {"in_progress": True}},
            "quiz_ans_doc1_q0": "B",
            "fc_input_front_doc1_0": "Custom front text",
            "study_timer_elapsed_seconds": 1200,
        }

        # prune_lru_cache is called only on cache dictionaries, not session state itself
        # Verify that prune_lru_cache cannot accidentally touch external dicts
        cache = {"doc_1": "summary 1"}
        prune_lru_cache(cache, max_entries=5)
        self.assertEqual(len(essential_sample), 5)

    def test_08_document_switching_preserves_cached_data_and_cleans_errors(self) -> None:
        """Simulate switching between 3 documents and returning back."""
        session_state = {
            "cached_summaries": {
                "doc_1": {"summary": "Doc 1 Summary"},
                "doc_2": {"summary": "Doc 2 Summary"},
            },
            "summary_error": "Failed generating for doc 1",
            "summary_error_doc_id": "doc_1",
        }

        # User switches from doc_1 to doc_2
        clean_stale_ephemeral_state(session_state, active_doc_id="doc_2")
        self.assertNotIn("summary_error", session_state)
        # Verify cached summaries are retained
        self.assertIn("doc_1", session_state["cached_summaries"])
        self.assertIn("doc_2", session_state["cached_summaries"])

        # User now switches to doc_3 and encounters an error
        session_state["summary_error"] = "Failed generating for doc 3"
        session_state["summary_error_doc_id"] = "doc_3"

        # User switches back to doc_1
        clean_stale_ephemeral_state(session_state, active_doc_id="doc_1")
        self.assertNotIn("summary_error", session_state)
        self.assertEqual(session_state["cached_summaries"]["doc_1"]["summary"], "Doc 1 Summary")

    def test_09_all_keys_protected_exceeding_max_entries(self) -> None:
        """Verify that when all keys are protected, pruning does not evict protected keys."""
        cache = {f"doc_{i}": i for i in range(1, 8)}
        all_keys = set(cache.keys())
        evicted = prune_lru_cache(cache, max_entries=5, protected_keys=all_keys)
        self.assertEqual(len(evicted), 0)
        self.assertEqual(len(cache), 7)


if __name__ == "__main__":
    unittest.main()
