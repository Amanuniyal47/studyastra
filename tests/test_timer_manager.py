import unittest
from unittest.mock import patch
from src.utils.timer_manager import PomodoroTimer, ChallengeTimer, QuizTimer


class TestPomodoroTimer(unittest.TestCase):
    def test_01_initial_state_and_defaults(self):
        timer = PomodoroTimer()
        self.assertEqual(timer.state, PomodoroTimer.STATE_IDLE)
        self.assertEqual(timer.current_session_type, PomodoroTimer.FOCUS)
        self.assertEqual(timer.focus_seconds, 25 * 60)
        self.assertEqual(timer.short_break_seconds, 5 * 60)
        self.assertEqual(timer.long_break_seconds, 15 * 60)
        self.assertEqual(timer.sessions_per_cycle, 4)
        self.assertEqual(timer.cycle_number, 1)
        self.assertEqual(timer.session_in_cycle, 1)
        self.assertEqual(timer.completed_focus_sessions, 0)
        self.assertEqual(timer.remaining_seconds, 25 * 60)
        self.assertEqual(timer.format_time(), "25:00")
        self.assertEqual(timer.get_progress(), 0.0)

    def test_02_custom_configuration(self):
        timer = PomodoroTimer(
            focus_minutes=50,
            short_break_minutes=10,
            long_break_minutes=30,
            sessions_per_cycle=2,
        )
        self.assertEqual(timer.focus_seconds, 50 * 60)
        self.assertEqual(timer.short_break_seconds, 10 * 60)
        self.assertEqual(timer.long_break_seconds, 30 * 60)
        self.assertEqual(timer.sessions_per_cycle, 2)
        self.assertEqual(timer.remaining_seconds, 50 * 60)
        self.assertEqual(timer.format_time(), "50:00")

    def test_03_start_pause_resume(self):
        timer = PomodoroTimer(focus_minutes=10)
        t0 = 1000.0

        # Start at t0
        timer.start(now_monotonic=t0)
        self.assertEqual(timer.state, PomodoroTimer.STATE_RUNNING)
        self.assertEqual(timer.deadline_monotonic, 1600.0)

        # Update at t0 + 100s
        just_completed, stype = timer.update(now_monotonic=1100.0)
        self.assertFalse(just_completed)
        self.assertEqual(timer.remaining_seconds, 500.0)
        self.assertEqual(timer.format_time(), "08:20")
        self.assertAlmostEqual(timer.get_progress(), 100.0 / 600.0)

        # Pause at t0 + 200s
        timer.pause(now_monotonic=1200.0)
        self.assertEqual(timer.state, PomodoroTimer.STATE_PAUSED)
        self.assertEqual(timer.remaining_seconds, 400.0)
        self.assertIsNone(timer.deadline_monotonic)

        # Time passes while paused (t = 2000.0)
        just_completed, _ = timer.update(now_monotonic=2000.0)
        self.assertFalse(just_completed)
        self.assertEqual(timer.remaining_seconds, 400.0)

        # Resume at t = 2000.0
        timer.resume(now_monotonic=2000.0)
        self.assertEqual(timer.state, PomodoroTimer.STATE_RUNNING)
        self.assertEqual(timer.deadline_monotonic, 2400.0)

        # Update at t = 2200.0
        timer.update(now_monotonic=2200.0)
        self.assertEqual(timer.remaining_seconds, 200.0)

    def test_04_reset_and_skip(self):
        timer = PomodoroTimer(focus_minutes=25, short_break_minutes=5)
        timer.start(now_monotonic=100.0)
        timer.update(now_monotonic=200.0)
        self.assertLess(timer.remaining_seconds, 25 * 60)

        # Reset returns to idle and full duration
        timer.reset()
        self.assertEqual(timer.state, PomodoroTimer.STATE_IDLE)
        self.assertEqual(timer.remaining_seconds, 25 * 60)
        self.assertIsNone(timer.deadline_monotonic)

        # Skip from focus advances to short break without counting focus completion
        skipped = timer.skip()
        self.assertEqual(skipped, PomodoroTimer.FOCUS)
        self.assertEqual(timer.current_session_type, PomodoroTimer.SHORT_BREAK)
        self.assertEqual(timer.completed_focus_sessions, 0)
        self.assertEqual(timer.remaining_seconds, 5 * 60)
        self.assertEqual(timer.format_time(), "05:00")

    def test_05_exact_once_completion(self):
        timer = PomodoroTimer(focus_minutes=1)  # 60s
        timer.start(now_monotonic=100.0)

        # Running before deadline
        completed, _ = timer.update(now_monotonic=159.0)
        self.assertFalse(completed)
        self.assertEqual(timer.state, PomodoroTimer.STATE_RUNNING)

        # At or past deadline
        completed, stype = timer.update(now_monotonic=160.0)
        self.assertTrue(completed)
        self.assertEqual(stype, PomodoroTimer.FOCUS)
        self.assertEqual(timer.state, PomodoroTimer.STATE_COMPLETED)
        self.assertEqual(timer.remaining_seconds, 0.0)
        self.assertEqual(timer.get_progress(), 1.0)

        # Subsequent updates must NOT trigger completion again (exact-once guarantee)
        completed_again, _ = timer.update(now_monotonic=165.0)
        self.assertFalse(completed_again)
        self.assertEqual(timer.state, PomodoroTimer.STATE_COMPLETED)

    def test_06_full_cycle_transitions(self):
        timer = PomodoroTimer(
            focus_minutes=1,
            short_break_minutes=1,
            long_break_minutes=2,
            sessions_per_cycle=4,
        )

        # 1. Session 1: Focus -> Short Break
        timer.start(now_monotonic=10.0)
        comp, _ = timer.update(now_monotonic=70.0)
        self.assertTrue(comp)
        nxt, cyc, s_in_c = timer.advance_to_next_session()
        self.assertEqual(nxt, PomodoroTimer.SHORT_BREAK)
        self.assertEqual(timer.completed_focus_sessions, 1)
        self.assertEqual(cyc, 1)
        self.assertEqual(s_in_c, 1)

        # Short break completes -> Focus (Session 2)
        timer.start(now_monotonic=100.0)
        comp, _ = timer.update(now_monotonic=160.0)
        self.assertTrue(comp)
        nxt, cyc, s_in_c = timer.advance_to_next_session()
        self.assertEqual(nxt, PomodoroTimer.FOCUS)
        self.assertEqual(s_in_c, 2)

        # 2. Session 2: Focus completes -> Short Break
        timer.start(now_monotonic=200.0)
        comp, _ = timer.update(now_monotonic=260.0)
        self.assertTrue(comp)
        nxt, cyc, s_in_c = timer.advance_to_next_session()
        self.assertEqual(nxt, PomodoroTimer.SHORT_BREAK)
        self.assertEqual(timer.completed_focus_sessions, 2)
        self.assertEqual(s_in_c, 2)

        # Short break completes -> Focus (Session 3)
        timer.start(now_monotonic=300.0)
        timer.update(now_monotonic=360.0)
        nxt, cyc, s_in_c = timer.advance_to_next_session()
        self.assertEqual(nxt, PomodoroTimer.FOCUS)
        self.assertEqual(s_in_c, 3)

        # 3. Session 3: Focus completes -> Short Break
        timer.start(now_monotonic=400.0)
        timer.update(now_monotonic=460.0)
        timer.advance_to_next_session()
        # Short break completes -> Focus (Session 4)
        timer.start(now_monotonic=500.0)
        timer.update(now_monotonic=560.0)
        nxt, cyc, s_in_c = timer.advance_to_next_session()
        self.assertEqual(nxt, PomodoroTimer.FOCUS)
        self.assertEqual(s_in_c, 4)

        # 4. Session 4: Focus completes -> LONG BREAK!
        timer.start(now_monotonic=600.0)
        timer.update(now_monotonic=660.0)
        nxt, cyc, s_in_c = timer.advance_to_next_session()
        self.assertEqual(nxt, PomodoroTimer.LONG_BREAK)
        self.assertEqual(timer.completed_focus_sessions, 4)
        self.assertEqual(timer.remaining_seconds, 2 * 60)

        # Long break completes -> Cycle 2, Session 1!
        timer.start(now_monotonic=700.0)
        timer.update(now_monotonic=820.0)
        nxt, cyc, s_in_c = timer.advance_to_next_session()
        self.assertEqual(nxt, PomodoroTimer.FOCUS)
        self.assertEqual(cyc, 2)
        self.assertEqual(s_in_c, 1)

    def test_07_serialization_to_and_from_dict(self):
        timer = PomodoroTimer(focus_minutes=20, short_break_minutes=4)
        timer.start(now_monotonic=500.0)
        timer.update(now_monotonic=600.0)

        d = timer.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["state"], PomodoroTimer.STATE_RUNNING)
        self.assertEqual(d["remaining_seconds"], 1100.0)

        new_timer = PomodoroTimer()
        new_timer.load_dict(d)
        self.assertEqual(new_timer.state, PomodoroTimer.STATE_RUNNING)
        self.assertEqual(new_timer.remaining_seconds, 1100.0)
        self.assertEqual(new_timer.focus_seconds, 20 * 60)


class TestChallengeTimer(unittest.TestCase):
    def test_01_initial_state_and_defaults(self):
        timer = ChallengeTimer()
        self.assertEqual(timer.state, ChallengeTimer.STATE_IDLE)
        self.assertEqual(timer.title, "Study Challenge")
        self.assertEqual(timer.duration_seconds, 25 * 60)
        self.assertEqual(timer.remaining_seconds, 25 * 60)
        self.assertEqual(timer.format_time(), "25:00")
        self.assertEqual(timer.get_progress(), 0.0)
        self.assertEqual(timer.actual_elapsed_seconds, 0.0)

    def test_02_custom_challenge_configuration(self):
        timer = ChallengeTimer(title="Math Problem Set", duration_minutes=45)
        self.assertEqual(timer.title, "Math Problem Set")
        self.assertEqual(timer.duration_seconds, 45 * 60)
        self.assertEqual(timer.remaining_seconds, 45 * 60)
        self.assertEqual(timer.format_time(), "45:00")

        # Update via set_challenge
        timer.set_challenge("Physics Revision", 60)
        self.assertEqual(timer.title, "Physics Revision")
        self.assertEqual(timer.duration_seconds, 60 * 60)
        self.assertEqual(timer.remaining_seconds, 60 * 60)
        self.assertEqual(timer.format_time(), "60:00")

    def test_03_start_pause_resume(self):
        timer = ChallengeTimer(title="Deep Reading", duration_minutes=10)
        t0 = 1000.0

        # Start at t0
        timer.start(now_monotonic=t0)
        self.assertEqual(timer.state, ChallengeTimer.STATE_RUNNING)
        self.assertEqual(timer.deadline_monotonic, 1600.0)

        # Update at t0 + 150s
        just_comp, title, elapsed = timer.update(now_monotonic=1150.0)
        self.assertFalse(just_comp)
        self.assertEqual(title, "Deep Reading")
        self.assertEqual(timer.remaining_seconds, 450.0)
        self.assertEqual(timer.format_time(), "07:30")
        self.assertAlmostEqual(timer.get_progress(), 150.0 / 600.0)
        self.assertAlmostEqual(elapsed, 150.0)

        # Pause at t0 + 200s
        timer.pause(now_monotonic=1200.0)
        self.assertEqual(timer.state, ChallengeTimer.STATE_PAUSED)
        self.assertEqual(timer.remaining_seconds, 400.0)
        self.assertIsNone(timer.deadline_monotonic)
        self.assertAlmostEqual(timer.actual_elapsed_seconds, 200.0)

        # Time passes while paused (monotonic time 2500.0)
        just_comp, _, _ = timer.update(now_monotonic=2500.0)
        self.assertFalse(just_comp)
        self.assertEqual(timer.remaining_seconds, 400.0)

        # Resume at t = 2500.0
        timer.resume(now_monotonic=2500.0)
        self.assertEqual(timer.state, ChallengeTimer.STATE_RUNNING)
        self.assertEqual(timer.deadline_monotonic, 2900.0)

        # Update at t = 2600.0
        timer.update(now_monotonic=2600.0)
        self.assertEqual(timer.remaining_seconds, 300.0)
        self.assertAlmostEqual(timer.actual_elapsed_seconds, 300.0)

    def test_04_reset_behavior(self):
        timer = ChallengeTimer(title="Essay Draft", duration_minutes=30)
        timer.start(now_monotonic=500.0)
        timer.update(now_monotonic=700.0)
        self.assertLess(timer.remaining_seconds, 30 * 60)

        # Reset returns to idle and restores full duration
        timer.reset()
        self.assertEqual(timer.state, ChallengeTimer.STATE_IDLE)
        self.assertEqual(timer.remaining_seconds, 30 * 60)
        self.assertEqual(timer.actual_elapsed_seconds, 0.0)
        self.assertIsNone(timer.deadline_monotonic)

        # Update on reset timer must never trigger completion
        just_comp, _, _ = timer.update(now_monotonic=2000.0)
        self.assertFalse(just_comp)

    def test_05_cancel_behavior(self):
        timer = ChallengeTimer(title="Speed Quiz Practice", duration_minutes=15)
        timer.start(now_monotonic=100.0)
        timer.update(now_monotonic=200.0)

        # Cancel challenge halfway
        timer.cancel(now_monotonic=250.0)
        self.assertEqual(timer.state, ChallengeTimer.STATE_CANCELLED)
        self.assertIsNone(timer.deadline_monotonic)
        self.assertAlmostEqual(timer.actual_elapsed_seconds, 150.0)

        # Update on cancelled timer must NEVER trigger completion
        just_comp, _, _ = timer.update(now_monotonic=3000.0)
        self.assertFalse(just_comp)
        self.assertEqual(timer.state, ChallengeTimer.STATE_CANCELLED)

    def test_06_exact_once_completion(self):
        timer = ChallengeTimer(title="Rapid Review", duration_minutes=1)  # 60s
        timer.start(now_monotonic=100.0)

        # Running before deadline
        completed, _, _ = timer.update(now_monotonic=159.0)
        self.assertFalse(completed)
        self.assertEqual(timer.state, ChallengeTimer.STATE_RUNNING)

        # At or past deadline
        completed, title, elapsed = timer.update(now_monotonic=160.0)
        self.assertTrue(completed)
        self.assertEqual(title, "Rapid Review")
        self.assertEqual(elapsed, 60.0)
        self.assertEqual(timer.state, ChallengeTimer.STATE_COMPLETED)
        self.assertEqual(timer.remaining_seconds, 0.0)
        self.assertEqual(timer.get_progress(), 1.0)

        # Subsequent updates must NOT trigger completion again (exact-once guarantee)
        completed_again, _, _ = timer.update(now_monotonic=170.0)
        self.assertFalse(completed_again)
        self.assertEqual(timer.state, ChallengeTimer.STATE_COMPLETED)

    def test_07_serialization_to_and_from_dict(self):
        timer = ChallengeTimer(title="Vocabulary Drill", duration_minutes=20)
        timer.start(now_monotonic=500.0)
        timer.update(now_monotonic=650.0)

        d = timer.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["title"], "Vocabulary Drill")
        self.assertEqual(d["duration_seconds"], 20 * 60)
        self.assertEqual(d["state"], ChallengeTimer.STATE_RUNNING)
        self.assertEqual(d["remaining_seconds"], 1050.0)
        self.assertEqual(d["actual_elapsed_seconds"], 150.0)

        restored = ChallengeTimer()
        restored.load_dict(d)
        self.assertEqual(restored.title, "Vocabulary Drill")
        self.assertEqual(restored.state, ChallengeTimer.STATE_RUNNING)
        self.assertEqual(restored.remaining_seconds, 1050.0)
        self.assertEqual(restored.actual_elapsed_seconds, 150.0)

    def test_08_independence_from_pomodoro(self):
        pomo = PomodoroTimer(focus_minutes=25)
        chal = ChallengeTimer(title="Coding Challenge", duration_minutes=15)

        pomo.start(now_monotonic=100.0)
        chal.start(now_monotonic=100.0)

        self.assertEqual(pomo.state, PomodoroTimer.STATE_RUNNING)
        self.assertEqual(chal.state, ChallengeTimer.STATE_RUNNING)

        # Advance 15 mins (900s) -> chal completes, pomo has 10 mins remaining
        p_comp, _ = pomo.update(now_monotonic=1000.0)
        c_comp, title, elap = chal.update(now_monotonic=1000.0)

        self.assertFalse(p_comp)
        self.assertTrue(c_comp)
        self.assertEqual(title, "Coding Challenge")
        self.assertEqual(chal.state, ChallengeTimer.STATE_COMPLETED)
        self.assertEqual(pomo.state, PomodoroTimer.STATE_RUNNING)
        self.assertEqual(pomo.remaining_seconds, 600.0)


class TestQuizTimer(unittest.TestCase):
    def test_01_initial_state_and_defaults(self):
        timer = QuizTimer(
            document_id="doc_bio",
            question_set_id="qs_01",
            attempt_number=1,
            duration_minutes=10,
        )
        self.assertEqual(timer.state, QuizTimer.STATE_IDLE)
        self.assertEqual(timer.document_id, "doc_bio")
        self.assertEqual(timer.question_set_id, "qs_01")
        self.assertEqual(timer.attempt_number, 1)
        self.assertEqual(timer.duration_seconds, 600)
        self.assertEqual(timer.remaining_seconds, 600.0)
        self.assertEqual(timer.format_time(), "10:00")
        self.assertEqual(timer.attempt_key, "doc_bio_qs_01_a1")
        self.assertFalse(timer.is_warning())
        self.assertEqual(timer.get_progress(), 0.0)

    def test_02_start_and_countdown_accuracy(self):
        timer = QuizTimer(
            document_id="doc_chem",
            question_set_id="qs_02",
            attempt_number=1,
            duration_minutes=10,
        )
        t0 = 1000.0
        timer.start(now_monotonic=t0)
        self.assertEqual(timer.state, QuizTimer.STATE_RUNNING)
        self.assertEqual(timer.deadline_monotonic, 1600.0)

        # Update at t0 + 200s
        expired, elapsed = timer.update(now_monotonic=1200.0)
        self.assertFalse(expired)
        self.assertEqual(timer.remaining_seconds, 400.0)
        self.assertEqual(timer.format_time(), "06:40")
        self.assertAlmostEqual(elapsed, 200.0)
        self.assertAlmostEqual(timer.get_progress(), 200.0 / 600.0)

    def test_03_warning_threshold(self):
        timer = QuizTimer(
            document_id="doc_math",
            question_set_id="qs_03",
            attempt_number=1,
            duration_minutes=5,
            warning_threshold_seconds=60,
        )
        timer.start(now_monotonic=100.0)  # deadline 400.0

        # At t = 330.0 (70s left) -> no warning
        timer.update(now_monotonic=330.0)
        self.assertFalse(timer.is_warning())

        # At t = 350.0 (50s left <= 60s) -> warning active
        timer.update(now_monotonic=350.0)
        self.assertTrue(timer.is_warning())
        self.assertEqual(timer.format_time(), "00:50")

    def test_04_manual_stop_before_expiration(self):
        timer = QuizTimer(
            document_id="doc_phys",
            question_set_id="qs_04",
            attempt_number=1,
            duration_minutes=15,  # 900s
        )
        timer.start(now_monotonic=1000.0)  # deadline 1900.0
        timer.update(now_monotonic=1500.0)

        # Learner clicks Submit Answers at 1600.0 (elapsed 600s, 300s left)
        actual_elapsed = timer.stop(now_monotonic=1600.0)
        self.assertAlmostEqual(actual_elapsed, 600.0)
        self.assertEqual(timer.state, QuizTimer.STATE_SUBMITTED)
        self.assertIsNone(timer.deadline_monotonic)

        # Future updates cannot expire a submitted timer
        expired, _ = timer.update(now_monotonic=2500.0)
        self.assertFalse(expired)
        self.assertEqual(timer.state, QuizTimer.STATE_SUBMITTED)

    def test_05_expiration_and_exact_once_guarantee(self):
        timer = QuizTimer(
            document_id="doc_lit",
            question_set_id="qs_05",
            attempt_number=1,
            duration_minutes=1,  # 60s
        )
        timer.start(now_monotonic=100.0)

        # Before expiration
        expired, _ = timer.update(now_monotonic=159.0)
        self.assertFalse(expired)
        self.assertEqual(timer.state, QuizTimer.STATE_RUNNING)

        # Exactly at / past expiration
        expired, elapsed = timer.update(now_monotonic=160.0)
        self.assertTrue(expired)
        self.assertEqual(elapsed, 60.0)
        self.assertEqual(timer.state, QuizTimer.STATE_EXPIRED)
        self.assertEqual(timer.remaining_seconds, 0.0)
        self.assertEqual(timer.get_progress(), 1.0)

        # Subsequent updates must NEVER trigger expiration again
        expired_again, _ = timer.update(now_monotonic=170.0)
        self.assertFalse(expired_again)
        self.assertEqual(timer.state, QuizTimer.STATE_EXPIRED)

    def test_06_rerun_does_not_extend_deadline(self):
        timer = QuizTimer(
            document_id="doc_hist",
            question_set_id="qs_06",
            attempt_number=1,
            duration_minutes=10,
        )
        timer.start(now_monotonic=1000.0)
        initial_deadline = timer.deadline_monotonic
        self.assertEqual(initial_deadline, 1600.0)

        # Simulate Streamlit reruns
        timer.update(now_monotonic=1100.0)
        self.assertEqual(timer.deadline_monotonic, initial_deadline)
        self.assertEqual(timer.remaining_seconds, 500.0)

        timer.update(now_monotonic=1250.0)
        self.assertEqual(timer.deadline_monotonic, initial_deadline)
        self.assertEqual(timer.remaining_seconds, 350.0)

    def test_07_attempt_isolation_and_retry(self):
        # Attempt 1
        timer1 = QuizTimer(
            document_id="doc_cs",
            question_set_id="qs_cs_01",
            attempt_number=1,
            duration_minutes=10,
        )
        timer1.start(now_monotonic=100.0)

        # Attempt 2 (Retry)
        timer2 = QuizTimer(
            document_id="doc_cs",
            question_set_id="qs_cs_01",
            attempt_number=2,
            duration_minutes=10,
        )
        self.assertEqual(timer2.state, QuizTimer.STATE_IDLE)
        self.assertIsNone(timer2.deadline_monotonic)
        self.assertEqual(timer2.attempt_key, "doc_cs_qs_cs_01_a2")
        self.assertNotEqual(timer1.attempt_key, timer2.attempt_key)

    def test_08_serialization_to_and_from_dict(self):
        timer = QuizTimer(
            document_id="doc_bio",
            question_set_id="qs_01",
            attempt_number=3,
            duration_minutes=20,
        )
        timer.start(now_monotonic=500.0)
        timer.update(now_monotonic=700.0)

        d = timer.to_dict()
        self.assertEqual(d["document_id"], "doc_bio")
        self.assertEqual(d["question_set_id"], "qs_01")
        self.assertEqual(d["attempt_number"], 3)
        self.assertEqual(d["state"], QuizTimer.STATE_RUNNING)
        self.assertEqual(d["remaining_seconds"], 1000.0)

        restored = QuizTimer(
            document_id="dummy",
            question_set_id="dummy",
        )
        restored.load_dict(d)
        self.assertEqual(restored.document_id, "doc_bio")
        self.assertEqual(restored.attempt_number, 3)
        self.assertEqual(restored.state, QuizTimer.STATE_RUNNING)
        self.assertEqual(restored.remaining_seconds, 1000.0)


if __name__ == "__main__":
    unittest.main()
