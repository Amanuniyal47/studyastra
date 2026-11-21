import json
import logging
import os
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import uuid

import src.config as config

logger = logging.getLogger(__name__)


def _default_stats() -> dict[str, Any]:
    return {
        "completed_focus_sessions": 0,
        "total_focus_minutes": 0.0,
        "total_break_minutes": 0.0,
        "current_streak": 0,
        "last_completed_date": None,
        "session_history": [],
        "completed_challenges": 0,
        "total_challenge_minutes": 0.0,
        "challenge_history": [],
        "completed_timed_quizzes": 0,
        "total_timed_quiz_attempts": 0,
        "total_timed_quiz_minutes": 0.0,
        "average_quiz_score_pct": 0.0,
        "best_quiz_score_pct": 0.0,
        "timed_quiz_history": [],
    }


class StudyStatisticsManager:
    """Manages persistent study session metrics and streaks using atomic JSON storage."""

    def __init__(self, stats_file: Path | str | None = None) -> None:
        self.stats_file = (
            Path(stats_file)
            if stats_file is not None
            else Path(config.STUDY_STATS_FILE)
        )
        self._stats: dict[str, Any] = _default_stats()
        self.load()

    def load(self) -> dict[str, Any]:
        """Load statistics from disk or initialize defaults if missing/corrupt."""
        if not self.stats_file.exists():
            self._stats = _default_stats()
            return deepcopy(self._stats)

        try:
            with open(self.stats_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Stats file does not contain a JSON object.")

            defaults = _default_stats()
            # Merge loaded data with defaults to ensure all keys exist
            self._stats = {
                "completed_focus_sessions": int(
                    data.get("completed_focus_sessions", defaults["completed_focus_sessions"])
                ),
                "total_focus_minutes": round(
                    float(data.get("total_focus_minutes", defaults["total_focus_minutes"])),
                    2,
                ),
                "total_break_minutes": round(
                    float(data.get("total_break_minutes", defaults["total_break_minutes"])),
                    2,
                ),
                "current_streak": int(
                    data.get("current_streak", defaults["current_streak"])
                ),
                "last_completed_date": data.get("last_completed_date"),
                "session_history": list(data.get("session_history", defaults["session_history"])),
                "completed_challenges": int(
                    data.get("completed_challenges", defaults["completed_challenges"])
                ),
                "total_challenge_minutes": round(
                    float(data.get("total_challenge_minutes", defaults["total_challenge_minutes"])),
                    2,
                ),
                "challenge_history": list(
                    data.get("challenge_history", defaults["challenge_history"])
                ),
                "completed_timed_quizzes": int(
                    data.get("completed_timed_quizzes", defaults["completed_timed_quizzes"])
                ),
                "total_timed_quiz_attempts": int(
                    data.get("total_timed_quiz_attempts", defaults["total_timed_quiz_attempts"])
                ),
                "total_timed_quiz_minutes": round(
                    float(data.get("total_timed_quiz_minutes", defaults["total_timed_quiz_minutes"])),
                    2,
                ),
                "average_quiz_score_pct": round(
                    float(data.get("average_quiz_score_pct", defaults["average_quiz_score_pct"])),
                    1,
                ),
                "best_quiz_score_pct": round(
                    float(data.get("best_quiz_score_pct", defaults["best_quiz_score_pct"])),
                    1,
                ),
                "timed_quiz_history": list(
                    data.get("timed_quiz_history", defaults["timed_quiz_history"])
                ),
            }
        except Exception as exc:
            logger.warning(
                "Failed to read or parse study stats from %s (%s). Using defaults.",
                self.stats_file,
                exc,
            )
            self._stats = _default_stats()

        return deepcopy(self._stats)

    def save(self, stats: dict[str, Any] | None = None) -> None:
        """Atomically persist statistics to JSON on disk."""
        if stats is not None:
            self._stats = deepcopy(stats)

        try:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.stats_file.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._stats, f, indent=2)
            os.replace(tmp_path, self.stats_file)
        except Exception as exc:
            logger.error("Failed to atomically write study stats to %s: %s", self.stats_file, exc)

    def record_session(
        self,
        session_type: str,
        duration_seconds: int | float,
        *,
        cycle_number: int = 1,
        session_in_cycle: int = 1,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Record a completed session, update totals, streak, and history."""
        now = timestamp or datetime.now(timezone.utc)
        today_str = now.date().isoformat()
        duration_minutes = round(max(0.0, float(duration_seconds)) / 60.0, 2)

        if session_type == "focus":
            self._stats["completed_focus_sessions"] += 1
            self._stats["total_focus_minutes"] = round(
                self._stats["total_focus_minutes"] + duration_minutes, 2
            )

            # Update study streak
            last_date_str = self._stats.get("last_completed_date")
            if not last_date_str:
                self._stats["current_streak"] = 1
            elif last_date_str == today_str:
                # Same day: streak stays unchanged
                pass
            else:
                try:
                    last_date = date.fromisoformat(last_date_str)
                    if now.date() == last_date + timedelta(days=1):
                        self._stats["current_streak"] += 1
                    elif now.date() > last_date + timedelta(days=1):
                        self._stats["current_streak"] = 1
                except ValueError:
                    self._stats["current_streak"] = 1
            self._stats["last_completed_date"] = today_str
        else:
            self._stats["total_break_minutes"] = round(
                self._stats["total_break_minutes"] + duration_minutes, 2
            )

        record = {
            "session_id": str(uuid.uuid4())[:8],
            "session_type": session_type,
            "duration_seconds": int(duration_seconds),
            "duration_minutes": duration_minutes,
            "completed_at": now.isoformat(),
            "cycle_number": cycle_number,
            "session_in_cycle": session_in_cycle,
        }
        self._stats["session_history"].append(record)
        self.save()
        return deepcopy(self._stats)

    def record_challenge(
        self,
        title: str,
        planned_duration_seconds: int | float,
        actual_duration_seconds: int | float,
        *,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Record a completed study challenge, update totals and challenge history."""
        now = timestamp or datetime.now(timezone.utc)
        planned_sec = max(0, int(planned_duration_seconds))
        actual_sec = max(0, int(actual_duration_seconds))
        actual_mins = round(actual_sec / 60.0, 2)

        self._stats["completed_challenges"] += 1
        self._stats["total_challenge_minutes"] = round(
            self._stats["total_challenge_minutes"] + actual_mins, 2
        )

        record = {
            "challenge_id": str(uuid.uuid4())[:8],
            "title": str(title).strip() or "Study Challenge",
            "planned_duration_seconds": planned_sec,
            "actual_duration_seconds": actual_sec,
            "duration_minutes": actual_mins,
            "completed_at": now.isoformat(),
        }
        self._stats["challenge_history"].append(record)
        self.save()
        return deepcopy(self._stats)

    def record_timed_quiz(
        self,
        *,
        document_id: str,
        question_set_id: str,
        attempt_number: int,
        configured_duration_seconds: int | float,
        actual_elapsed_seconds: int | float,
        earned_points: float,
        total_points: float,
        percentage: float,
        answered_count: int,
        unanswered_count: int,
        auto_submitted: bool,
        attempt_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Record a successfully evaluated timed quiz attempt, updating totals, averages, and history."""
        stable_attempt_id = attempt_id or f"{document_id}_{question_set_id}_a{attempt_number}"

        # Deduplication: prevent duplicate recording of the exact same attempt
        existing_ids = {
            entry.get("attempt_id")
            for entry in self._stats.get("timed_quiz_history", [])
            if isinstance(entry, dict)
        }
        if stable_attempt_id in existing_ids:
            logger.info("Timed quiz attempt %s already recorded. Skipping duplicate.", stable_attempt_id)
            return deepcopy(self._stats)

        now = timestamp or datetime.now(timezone.utc)
        conf_sec = max(0, int(configured_duration_seconds))
        act_sec = max(0, int(actual_elapsed_seconds))
        act_mins = round(act_sec / 60.0, 2)
        pct = round(float(percentage), 1)

        self._stats["completed_timed_quizzes"] += 1
        self._stats["total_timed_quiz_attempts"] += 1
        self._stats["total_timed_quiz_minutes"] = round(
            self._stats["total_timed_quiz_minutes"] + act_mins, 2
        )

        current_best = float(self._stats.get("best_quiz_score_pct", 0.0))
        self._stats["best_quiz_score_pct"] = round(max(current_best, pct), 1)

        record = {
            "attempt_id": stable_attempt_id,
            "document_id": document_id,
            "question_set_id": question_set_id,
            "attempt_number": attempt_number,
            "configured_duration_seconds": conf_sec,
            "actual_elapsed_seconds": act_sec,
            "duration_minutes": act_mins,
            "earned_points": round(float(earned_points), 2),
            "total_points": round(float(total_points), 2),
            "percentage": pct,
            "answered_count": int(answered_count),
            "unanswered_count": int(unanswered_count),
            "auto_submitted": bool(auto_submitted),
            "completed_at": now.isoformat(),
        }
        self._stats["timed_quiz_history"].append(record)

        all_pcts = [
            float(r.get("percentage", 0.0))
            for r in self._stats["timed_quiz_history"]
            if isinstance(r, dict)
        ]
        self._stats["average_quiz_score_pct"] = (
            round(sum(all_pcts) / len(all_pcts), 1) if all_pcts else 0.0
        )

        self.save()
        return deepcopy(self._stats)

    def reset_statistics(self) -> dict[str, Any]:
        """Reset all tracked statistics to initial state."""
        self._stats = _default_stats()
        self.save()
        return deepcopy(self._stats)

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the current statistics dict."""
        return deepcopy(self._stats)
