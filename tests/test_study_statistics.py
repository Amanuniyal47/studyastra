from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import unittest

from src.utils.study_statistics import StudyStatisticsManager


class TestStudyStatisticsManager(unittest.TestCase):
    def test_01_init_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            mgr = StudyStatisticsManager(stats_path)
            data = mgr.to_dict()
            self.assertEqual(data["completed_focus_sessions"], 0)
            self.assertEqual(data["total_focus_minutes"], 0.0)
            self.assertEqual(data["total_break_minutes"], 0.0)
            self.assertEqual(data["current_streak"], 0)
            self.assertEqual(len(data["session_history"]), 0)
            self.assertEqual(data["completed_challenges"], 0)
            self.assertEqual(data["total_challenge_minutes"], 0.0)
            self.assertEqual(len(data["challenge_history"]), 0)
            self.assertEqual(data["completed_timed_quizzes"], 0)
            self.assertEqual(data["total_timed_quiz_attempts"], 0)
            self.assertEqual(data["total_timed_quiz_minutes"], 0.0)
            self.assertEqual(data["average_quiz_score_pct"], 0.0)
            self.assertEqual(data["best_quiz_score_pct"], 0.0)
            self.assertEqual(len(data["timed_quiz_history"]), 0)

    def test_02_record_focus_and_break_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            mgr = StudyStatisticsManager(stats_path)

            # Record 25 min focus
            mgr.record_session("focus", 1500)
            data = mgr.to_dict()
            self.assertEqual(data["completed_focus_sessions"], 1)
            self.assertEqual(data["total_focus_minutes"], 25.0)
            self.assertEqual(data["current_streak"], 1)
            self.assertEqual(len(data["session_history"]), 1)

            # Record 5 min short break
            mgr.record_session("short_break", 300)
            data = mgr.to_dict()
            self.assertEqual(data["completed_focus_sessions"], 1)
            self.assertEqual(data["total_focus_minutes"], 25.0)
            self.assertEqual(data["total_break_minutes"], 5.0)
            self.assertEqual(len(data["session_history"]), 2)

    def test_03_streak_calculation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            mgr = StudyStatisticsManager(stats_path)

            day1 = datetime(2026, 11, 15, 10, 0, tzinfo=timezone.utc)
            mgr.record_session("focus", 1500, timestamp=day1)
            self.assertEqual(mgr.to_dict()["current_streak"], 1)

            # Same day session -> streak stays 1
            day1_later = datetime(2026, 11, 15, 14, 0, tzinfo=timezone.utc)
            mgr.record_session("focus", 1500, timestamp=day1_later)
            self.assertEqual(mgr.to_dict()["current_streak"], 1)

            # Next day session (consecutive) -> streak = 2
            day2 = datetime(2026, 11, 16, 9, 30, tzinfo=timezone.utc)
            mgr.record_session("focus", 1500, timestamp=day2)
            self.assertEqual(mgr.to_dict()["current_streak"], 2)

            # Skip 2 days (gap) -> streak resets to 1
            day4 = datetime(2026, 11, 19, 11, 0, tzinfo=timezone.utc)
            mgr.record_session("focus", 1500, timestamp=day4)
            self.assertEqual(mgr.to_dict()["current_streak"], 1)

    def test_04_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            mgr1 = StudyStatisticsManager(stats_path)
            mgr1.record_session("focus", 1800)

            # Reload with a new instance
            mgr2 = StudyStatisticsManager(stats_path)
            data = mgr2.to_dict()
            self.assertEqual(data["completed_focus_sessions"], 1)
            self.assertEqual(data["total_focus_minutes"], 30.0)

    def test_05_corrupted_json_recovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                f.write("{ invalid json corrupted ...")

            mgr = StudyStatisticsManager(stats_path)
            data = mgr.to_dict()
            self.assertEqual(data["completed_focus_sessions"], 0)

    def test_06_reset_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            mgr = StudyStatisticsManager(stats_path)
            mgr.record_session("focus", 1500)
            mgr.record_session("short_break", 300)
            self.assertEqual(mgr.to_dict()["completed_focus_sessions"], 1)

            mgr.reset_statistics()
            data = mgr.to_dict()
            self.assertEqual(data["completed_focus_sessions"], 0)
            self.assertEqual(data["total_focus_minutes"], 0.0)
            self.assertEqual(len(data["session_history"]), 0)
            self.assertEqual(data["completed_challenges"], 0)
            self.assertEqual(data["total_challenge_minutes"], 0.0)
            self.assertEqual(len(data["challenge_history"]), 0)
            self.assertEqual(data["completed_timed_quizzes"], 0)
            self.assertEqual(data["total_timed_quiz_attempts"], 0)
            self.assertEqual(data["total_timed_quiz_minutes"], 0.0)
            self.assertEqual(data["average_quiz_score_pct"], 0.0)
            self.assertEqual(data["best_quiz_score_pct"], 0.0)
            self.assertEqual(len(data["timed_quiz_history"]), 0)

    def test_07_record_challenge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            mgr = StudyStatisticsManager(stats_path)

            # Record 15-min challenge
            mgr.record_challenge("Cell Division Review", 900, 900)
            data = mgr.to_dict()
            self.assertEqual(data["completed_challenges"], 1)
            self.assertEqual(data["total_challenge_minutes"], 15.0)
            self.assertEqual(len(data["challenge_history"]), 1)

            entry = data["challenge_history"][0]
            self.assertEqual(entry["title"], "Cell Division Review")
            self.assertEqual(entry["planned_duration_seconds"], 900)
            self.assertEqual(entry["actual_duration_seconds"], 900)
            self.assertEqual(entry["duration_minutes"], 15.0)
            self.assertTrue(entry.get("challenge_id"))
            self.assertTrue(entry.get("completed_at"))

            # Record second challenge with partial time
            mgr.record_challenge("Formula Cramming", 1800, 1200)
            data2 = mgr.to_dict()
            self.assertEqual(data2["completed_challenges"], 2)
            self.assertEqual(data2["total_challenge_minutes"], 35.0)
            self.assertEqual(len(data2["challenge_history"]), 2)

    def test_08_challenge_pomodoro_independence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            mgr = StudyStatisticsManager(stats_path)

            # Record a focus session
            mgr.record_session("focus", 1500)
            d1 = mgr.to_dict()
            self.assertEqual(d1["completed_focus_sessions"], 1)
            self.assertEqual(d1["total_focus_minutes"], 25.0)
            self.assertEqual(d1["completed_challenges"], 0)
            self.assertEqual(d1["total_challenge_minutes"], 0.0)
            self.assertEqual(len(d1["challenge_history"]), 0)

            # Record a challenge
            mgr.record_challenge("Organic Chemistry Drill", 600, 600)
            d2 = mgr.to_dict()
            # Pomodoro stats remain exactly what they were
            self.assertEqual(d2["completed_focus_sessions"], 1)
            self.assertEqual(d2["total_focus_minutes"], 25.0)
            # Challenge stats are tracked independently
            self.assertEqual(d2["completed_challenges"], 1)
            self.assertEqual(d2["total_challenge_minutes"], 10.0)
            self.assertEqual(len(d2["challenge_history"]), 1)

    def test_09_backward_compatibility_with_stage_9_1_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import json
            stats_path = Path(tmpdir) / "stats.json"
            stage_9_1_data = {
                "completed_focus_sessions": 3,
                "total_focus_minutes": 75.0,
                "total_break_minutes": 15.0,
                "current_streak": 2,
                "last_completed_date": "2026-11-19",
                "session_history": [{"session_id": "abc12345", "session_type": "focus"}],
            }
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stage_9_1_data, f)

            mgr = StudyStatisticsManager(stats_path)
            d = mgr.to_dict()
            # Preserved Stage 9.1 fields
            self.assertEqual(d["completed_focus_sessions"], 3)
            self.assertEqual(d["total_focus_minutes"], 75.0)
            self.assertEqual(d["current_streak"], 2)
            # Initialized Stage 9.2 fields
            self.assertEqual(d["completed_challenges"], 0)
            self.assertEqual(d["total_challenge_minutes"], 0.0)
            self.assertEqual(d["challenge_history"], [])

    def test_10_record_timed_quiz_manual_and_auto(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            mgr = StudyStatisticsManager(stats_path)

            # Record manual submission
            mgr.record_timed_quiz(
                document_id="doc_bio",
                question_set_id="qs_01",
                attempt_number=1,
                configured_duration_seconds=600,
                actual_elapsed_seconds=300,
                earned_points=4.0,
                total_points=5.0,
                percentage=80.0,
                answered_count=4,
                unanswered_count=1,
                auto_submitted=False,
            )
            data = mgr.to_dict()
            self.assertEqual(data["completed_timed_quizzes"], 1)
            self.assertEqual(data["total_timed_quiz_attempts"], 1)
            self.assertEqual(data["total_timed_quiz_minutes"], 5.0)
            self.assertEqual(data["best_quiz_score_pct"], 80.0)
            self.assertEqual(data["average_quiz_score_pct"], 80.0)
            self.assertEqual(len(data["timed_quiz_history"]), 1)

            entry1 = data["timed_quiz_history"][0]
            self.assertEqual(entry1["attempt_id"], "doc_bio_qs_01_a1")
            self.assertEqual(entry1["duration_minutes"], 5.0)
            self.assertFalse(entry1["auto_submitted"])

            # Record automatic submission (timeout)
            mgr.record_timed_quiz(
                document_id="doc_chem",
                question_set_id="qs_02",
                attempt_number=1,
                configured_duration_seconds=900,
                actual_elapsed_seconds=900,
                earned_points=2.0,
                total_points=4.0,
                percentage=50.0,
                answered_count=2,
                unanswered_count=2,
                auto_submitted=True,
            )
            data2 = mgr.to_dict()
            self.assertEqual(data2["completed_timed_quizzes"], 2)
            self.assertEqual(data2["total_timed_quiz_attempts"], 2)
            self.assertEqual(data2["total_timed_quiz_minutes"], 20.0)  # 5.0 + 15.0
            self.assertEqual(data2["best_quiz_score_pct"], 80.0)
            self.assertEqual(data2["average_quiz_score_pct"], 65.0)  # (80 + 50) / 2
            self.assertEqual(len(data2["timed_quiz_history"]), 2)

            entry2 = data2["timed_quiz_history"][1]
            self.assertTrue(entry2["auto_submitted"])

    def test_11_timed_quiz_deduplication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            mgr = StudyStatisticsManager(stats_path)

            # Record attempt 1
            mgr.record_timed_quiz(
                document_id="doc_bio",
                question_set_id="qs_01",
                attempt_number=1,
                configured_duration_seconds=600,
                actual_elapsed_seconds=300,
                earned_points=3.0,
                total_points=3.0,
                percentage=100.0,
                answered_count=3,
                unanswered_count=0,
                auto_submitted=False,
            )
            self.assertEqual(mgr.to_dict()["completed_timed_quizzes"], 1)

            # Record the EXACT same attempt again (simulate double submit trigger)
            mgr.record_timed_quiz(
                document_id="doc_bio",
                question_set_id="qs_01",
                attempt_number=1,
                configured_duration_seconds=600,
                actual_elapsed_seconds=300,
                earned_points=3.0,
                total_points=3.0,
                percentage=100.0,
                answered_count=3,
                unanswered_count=0,
                auto_submitted=False,
            )
            # Counters and history must remain unchanged!
            self.assertEqual(mgr.to_dict()["completed_timed_quizzes"], 1)
            self.assertEqual(mgr.to_dict()["total_timed_quiz_attempts"], 1)
            self.assertEqual(len(mgr.to_dict()["timed_quiz_history"]), 1)

    def test_12_timed_quiz_independence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "stats.json"
            mgr = StudyStatisticsManager(stats_path)

            # 1. Focus session
            mgr.record_session("focus", 1500)
            # 2. Study challenge
            mgr.record_challenge("Speed Challenge", 600, 600)
            # 3. Timed quiz
            mgr.record_timed_quiz(
                document_id="doc_math",
                question_set_id="qs_01",
                attempt_number=1,
                configured_duration_seconds=600,
                actual_elapsed_seconds=420,
                earned_points=4.0,
                total_points=5.0,
                percentage=80.0,
                answered_count=4,
                unanswered_count=1,
                auto_submitted=False,
            )

            d = mgr.to_dict()
            # Focus
            self.assertEqual(d["completed_focus_sessions"], 1)
            self.assertEqual(d["total_focus_minutes"], 25.0)
            # Challenge
            self.assertEqual(d["completed_challenges"], 1)
            self.assertEqual(d["total_challenge_minutes"], 10.0)
            # Quiz
            self.assertEqual(d["completed_timed_quizzes"], 1)
            self.assertEqual(d["total_timed_quiz_minutes"], 7.0)

    def test_13_backward_compatibility_with_stage_9_2_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import json
            stats_path = Path(tmpdir) / "stats.json"
            stage_9_2_data = {
                "completed_focus_sessions": 2,
                "total_focus_minutes": 50.0,
                "total_break_minutes": 10.0,
                "current_streak": 1,
                "last_completed_date": "2026-11-20",
                "session_history": [],
                "completed_challenges": 1,
                "total_challenge_minutes": 15.0,
                "challenge_history": [],
            }
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stage_9_2_data, f)

            mgr = StudyStatisticsManager(stats_path)
            d = mgr.to_dict()
            # Preserved
            self.assertEqual(d["completed_focus_sessions"], 2)
            self.assertEqual(d["completed_challenges"], 1)
            # Initialized Stage 9.3 defaults
            self.assertEqual(d["completed_timed_quizzes"], 0)
            self.assertEqual(d["total_timed_quiz_attempts"], 0)
            self.assertEqual(d["total_timed_quiz_minutes"], 0.0)
            self.assertEqual(d["average_quiz_score_pct"], 0.0)
            self.assertEqual(d["best_quiz_score_pct"], 0.0)
            self.assertEqual(d["timed_quiz_history"], [])


if __name__ == "__main__":
    unittest.main()
