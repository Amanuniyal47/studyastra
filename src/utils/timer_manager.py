import time
from typing import Any


class PomodoroTimer:
    """Accurate, monotonic deadline-based Pomodoro timer state machine."""

    FOCUS = "focus"
    SHORT_BREAK = "short_break"
    LONG_BREAK = "long_break"

    STATE_IDLE = "idle"
    STATE_RUNNING = "running"
    STATE_PAUSED = "paused"
    STATE_COMPLETED = "completed"

    def __init__(
        self,
        *,
        focus_minutes: int = 25,
        short_break_minutes: int = 5,
        long_break_minutes: int = 15,
        sessions_per_cycle: int = 4,
    ) -> None:
        self.focus_seconds = max(1, int(focus_minutes * 60))
        self.short_break_seconds = max(1, int(short_break_minutes * 60))
        self.long_break_seconds = max(1, int(long_break_minutes * 60))
        self.sessions_per_cycle = max(1, int(sessions_per_cycle))

        self.state: str = self.STATE_IDLE
        self.current_session_type: str = self.FOCUS
        self.cycle_number: int = 1
        self.session_in_cycle: int = 1
        self.completed_focus_sessions: int = 0

        self.remaining_seconds: float = float(self.focus_seconds)
        self.deadline_monotonic: float | None = None
        self.completion_handled: bool = False

    def get_configured_duration(self, session_type: str | None = None) -> int:
        """Return the total configured duration in seconds for a session type."""
        stype = session_type or self.current_session_type
        if stype == self.FOCUS:
            return self.focus_seconds
        elif stype == self.SHORT_BREAK:
            return self.short_break_seconds
        elif stype == self.LONG_BREAK:
            return self.long_break_seconds
        return self.focus_seconds

    def start(self, now_monotonic: float | None = None) -> None:
        """Start or resume countdown."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self.state in (self.STATE_IDLE, self.STATE_COMPLETED):
            self.remaining_seconds = float(self.get_configured_duration())
            self.deadline_monotonic = now + self.remaining_seconds
            self.state = self.STATE_RUNNING
            self.completion_handled = False
        elif self.state == self.STATE_PAUSED:
            self.deadline_monotonic = now + self.remaining_seconds
            self.state = self.STATE_RUNNING

    def pause(self, now_monotonic: float | None = None) -> None:
        """Pause the active countdown."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self.state == self.STATE_RUNNING:
            if self.deadline_monotonic is not None:
                self.remaining_seconds = max(0.0, self.deadline_monotonic - now)
            self.deadline_monotonic = None
            self.state = self.STATE_PAUSED

    def resume(self, now_monotonic: float | None = None) -> None:
        """Resume a paused countdown."""
        self.start(now_monotonic)

    def reset(self) -> None:
        """Reset the current session to its initial configured duration."""
        self.state = self.STATE_IDLE
        self.remaining_seconds = float(self.get_configured_duration())
        self.deadline_monotonic = None
        self.completion_handled = False

    def skip(self) -> str:
        """Skip the current session without recording focus completion, advancing to next session type."""
        skipped_type = self.current_session_type
        if skipped_type == self.FOCUS:
            if self.session_in_cycle >= self.sessions_per_cycle:
                self.current_session_type = self.LONG_BREAK
            else:
                self.current_session_type = self.SHORT_BREAK
        elif skipped_type == self.SHORT_BREAK:
            self.current_session_type = self.FOCUS
            self.session_in_cycle += 1
        elif skipped_type == self.LONG_BREAK:
            self.current_session_type = self.FOCUS
            self.session_in_cycle = 1
            self.cycle_number += 1

        self.state = self.STATE_IDLE
        self.remaining_seconds = float(self.get_configured_duration())
        self.deadline_monotonic = None
        self.completion_handled = False
        return skipped_type

    def update(self, now_monotonic: float | None = None) -> tuple[bool, str]:
        """Update timer state based on monotonic time.
        
        Returns:
            (is_just_completed, session_type)
            is_just_completed is True exactly once when the timer reaches 0.
        """
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self.state == self.STATE_RUNNING:
            if self.deadline_monotonic is not None:
                rem = self.deadline_monotonic - now
                if rem <= 0.0:
                    self.remaining_seconds = 0.0
                    self.state = self.STATE_COMPLETED
                    self.deadline_monotonic = None
                    if not self.completion_handled:
                        self.completion_handled = True
                        return True, self.current_session_type
                else:
                    self.remaining_seconds = rem

        return False, self.current_session_type

    def advance_to_next_session(self) -> tuple[str, int, int]:
        """Advance to the next session type in the cycle after completion."""
        finished_type = self.current_session_type
        if finished_type == self.FOCUS:
            self.completed_focus_sessions += 1
            if self.session_in_cycle >= self.sessions_per_cycle:
                self.current_session_type = self.LONG_BREAK
            else:
                self.current_session_type = self.SHORT_BREAK
        elif finished_type == self.SHORT_BREAK:
            self.current_session_type = self.FOCUS
            self.session_in_cycle += 1
        elif finished_type == self.LONG_BREAK:
            self.current_session_type = self.FOCUS
            self.session_in_cycle = 1
            self.cycle_number += 1

        self.state = self.STATE_IDLE
        self.remaining_seconds = float(self.get_configured_duration())
        self.deadline_monotonic = None
        self.completion_handled = False
        return self.current_session_type, self.cycle_number, self.session_in_cycle

    def format_time(self, seconds: float | None = None) -> str:
        """Format seconds into MM:SS string."""
        sec = int(round(self.remaining_seconds if seconds is None else seconds))
        sec = max(0, sec)
        mins = sec // 60
        secs = sec % 60
        return f"{mins:02d}:{secs:02d}"

    def get_progress(self) -> float:
        """Calculate progress ratio from 0.0 (just started) to 1.0 (completed)."""
        total = float(self.get_configured_duration())
        if total <= 0:
            return 1.0
        elapsed = total - self.remaining_seconds
        return min(1.0, max(0.0, elapsed / total))

    def to_dict(self) -> dict[str, Any]:
        """Serialize timer state to a dictionary for session_state."""
        return {
            "focus_seconds": self.focus_seconds,
            "short_break_seconds": self.short_break_seconds,
            "long_break_seconds": self.long_break_seconds,
            "sessions_per_cycle": self.sessions_per_cycle,
            "state": self.state,
            "current_session_type": self.current_session_type,
            "cycle_number": self.cycle_number,
            "session_in_cycle": self.session_in_cycle,
            "completed_focus_sessions": self.completed_focus_sessions,
            "remaining_seconds": self.remaining_seconds,
            "deadline_monotonic": self.deadline_monotonic,
            "completion_handled": self.completion_handled,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        """Restore timer state from a dictionary."""
        self.focus_seconds = int(data.get("focus_seconds", self.focus_seconds))
        self.short_break_seconds = int(data.get("short_break_seconds", self.short_break_seconds))
        self.long_break_seconds = int(data.get("long_break_seconds", self.long_break_seconds))
        self.sessions_per_cycle = int(data.get("sessions_per_cycle", self.sessions_per_cycle))
        self.state = str(data.get("state", self.STATE_IDLE))
        self.current_session_type = str(data.get("current_session_type", self.FOCUS))
        self.cycle_number = int(data.get("cycle_number", 1))
        self.session_in_cycle = int(data.get("session_in_cycle", 1))
        self.completed_focus_sessions = int(data.get("completed_focus_sessions", 0))
        self.remaining_seconds = float(data.get("remaining_seconds", self.focus_seconds))
        self.deadline_monotonic = data.get("deadline_monotonic")
        self.completion_handled = bool(data.get("completion_handled", False))


class ChallengeTimer:
    """Accurate, monotonic deadline-based countdown timer for custom study challenges."""

    STATE_IDLE = "idle"
    STATE_RUNNING = "running"
    STATE_PAUSED = "paused"
    STATE_COMPLETED = "completed"
    STATE_CANCELLED = "cancelled"

    def __init__(
        self,
        *,
        title: str = "Study Challenge",
        duration_minutes: int = 25,
    ) -> None:
        self.title: str = str(title).strip() or "Study Challenge"
        self.duration_seconds: int = max(1, int(duration_minutes * 60))
        self.state: str = self.STATE_IDLE
        self.remaining_seconds: float = float(self.duration_seconds)
        self.deadline_monotonic: float | None = None
        self.completion_handled: bool = False
        self.actual_elapsed_seconds: float = 0.0

    def set_challenge(self, title: str, duration_minutes: int) -> None:
        """Configure challenge title and duration when idle, completed, or cancelled."""
        self.title = str(title).strip() or "Study Challenge"
        self.duration_seconds = max(1, int(duration_minutes * 60))
        self.remaining_seconds = float(self.duration_seconds)
        self.state = self.STATE_IDLE
        self.deadline_monotonic = None
        self.completion_handled = False
        self.actual_elapsed_seconds = 0.0

    def start(self, now_monotonic: float | None = None) -> None:
        """Start or resume countdown."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self.state in (self.STATE_IDLE, self.STATE_COMPLETED, self.STATE_CANCELLED):
            self.remaining_seconds = float(self.duration_seconds)
            self.deadline_monotonic = now + self.remaining_seconds
            self.state = self.STATE_RUNNING
            self.completion_handled = False
            self.actual_elapsed_seconds = 0.0
        elif self.state == self.STATE_PAUSED:
            self.deadline_monotonic = now + self.remaining_seconds
            self.state = self.STATE_RUNNING

    def pause(self, now_monotonic: float | None = None) -> None:
        """Pause the active countdown."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self.state == self.STATE_RUNNING:
            if self.deadline_monotonic is not None:
                self.remaining_seconds = max(0.0, self.deadline_monotonic - now)
                self.actual_elapsed_seconds = max(0.0, self.duration_seconds - self.remaining_seconds)
            self.deadline_monotonic = None
            self.state = self.STATE_PAUSED

    def resume(self, now_monotonic: float | None = None) -> None:
        """Resume a paused countdown."""
        self.start(now_monotonic)

    def reset(self) -> None:
        """Reset the challenge to its initial configured state without recording completion."""
        self.state = self.STATE_IDLE
        self.remaining_seconds = float(self.duration_seconds)
        self.deadline_monotonic = None
        self.completion_handled = False
        self.actual_elapsed_seconds = 0.0

    def cancel(self, now_monotonic: float | None = None) -> None:
        """Cancel the active challenge without recording completion."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self.state == self.STATE_RUNNING and self.deadline_monotonic is not None:
            self.remaining_seconds = max(0.0, self.deadline_monotonic - now)
            self.actual_elapsed_seconds = max(0.0, self.duration_seconds - self.remaining_seconds)
        elif self.state == self.STATE_PAUSED:
            self.actual_elapsed_seconds = max(0.0, self.duration_seconds - self.remaining_seconds)
        self.state = self.STATE_CANCELLED
        self.deadline_monotonic = None
        self.completion_handled = False

    def update(self, now_monotonic: float | None = None) -> tuple[bool, str, float]:
        """Update timer state based on monotonic time.

        Returns:
            (is_just_completed, title, actual_elapsed_seconds)
            is_just_completed is True exactly once when the timer reaches 0.
        """
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self.state == self.STATE_RUNNING:
            if self.deadline_monotonic is not None:
                rem = self.deadline_monotonic - now
                if rem <= 0.0:
                    self.remaining_seconds = 0.0
                    self.actual_elapsed_seconds = float(self.duration_seconds)
                    self.state = self.STATE_COMPLETED
                    self.deadline_monotonic = None
                    if not self.completion_handled:
                        self.completion_handled = True
                        return True, self.title, self.actual_elapsed_seconds
                else:
                    self.remaining_seconds = rem
                    self.actual_elapsed_seconds = max(0.0, self.duration_seconds - rem)

        return False, self.title, self.actual_elapsed_seconds

    def format_time(self, seconds: float | None = None) -> str:
        """Format seconds into MM:SS string."""
        sec = int(round(self.remaining_seconds if seconds is None else seconds))
        sec = max(0, sec)
        mins = sec // 60
        secs = sec % 60
        return f"{mins:02d}:{secs:02d}"

    def get_progress(self) -> float:
        """Calculate progress ratio from 0.0 (just started) to 1.0 (completed)."""
        if self.duration_seconds <= 0:
            return 1.0
        elapsed = self.duration_seconds - self.remaining_seconds
        return min(1.0, max(0.0, elapsed / self.duration_seconds))

    def to_dict(self) -> dict[str, Any]:
        """Serialize challenge timer state."""
        return {
            "title": self.title,
            "duration_seconds": self.duration_seconds,
            "state": self.state,
            "remaining_seconds": self.remaining_seconds,
            "deadline_monotonic": self.deadline_monotonic,
            "completion_handled": self.completion_handled,
            "actual_elapsed_seconds": self.actual_elapsed_seconds,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        """Restore challenge timer state."""
        self.title = str(data.get("title", self.title))
        self.duration_seconds = int(data.get("duration_seconds", self.duration_seconds))
        self.state = str(data.get("state", self.STATE_IDLE))
        self.remaining_seconds = float(data.get("remaining_seconds", self.duration_seconds))
        self.deadline_monotonic = data.get("deadline_monotonic")
        self.completion_handled = bool(data.get("completion_handled", False))
        self.actual_elapsed_seconds = float(data.get("actual_elapsed_seconds", 0.0))


class QuizTimer:
    """Monotonic deadline-based countdown timer strictly scoped to a single quiz attempt."""

    STATE_IDLE = "idle"
    STATE_RUNNING = "running"
    STATE_SUBMITTED = "submitted"
    STATE_EXPIRED = "expired"

    def __init__(
        self,
        *,
        document_id: str,
        question_set_id: str,
        attempt_number: int = 1,
        duration_minutes: int = 10,
        warning_threshold_seconds: int = 60,
    ) -> None:
        self.document_id: str = str(document_id)
        self.question_set_id: str = str(question_set_id)
        self.attempt_number: int = int(attempt_number)
        self.duration_seconds: int = max(1, int(duration_minutes * 60))
        self.warning_threshold_seconds: int = max(0, int(warning_threshold_seconds))

        self.state: str = self.STATE_IDLE
        self.remaining_seconds: float = float(self.duration_seconds)
        self.deadline_monotonic: float | None = None
        self.completion_handled: bool = False
        self.actual_elapsed_seconds: float = 0.0

    @property
    def attempt_key(self) -> str:
        """Unique identifier scoping this timer to its exact document and attempt."""
        return f"{self.document_id}_{self.question_set_id}_a{self.attempt_number}"

    def start(self, now_monotonic: float | None = None) -> None:
        """Start countdown against monotonic deadline."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self.state in (self.STATE_IDLE, self.STATE_EXPIRED, self.STATE_SUBMITTED):
            self.remaining_seconds = float(self.duration_seconds)
            self.deadline_monotonic = now + self.remaining_seconds
            self.state = self.STATE_RUNNING
            self.completion_handled = False
            self.actual_elapsed_seconds = 0.0

    def stop(self, now_monotonic: float | None = None) -> float:
        """Stop countdown on manual submission, recording actual elapsed time."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self.state == self.STATE_RUNNING:
            if self.deadline_monotonic is not None:
                self.remaining_seconds = max(0.0, self.deadline_monotonic - now)
                self.actual_elapsed_seconds = max(0.0, self.duration_seconds - self.remaining_seconds)
            self.deadline_monotonic = None
            self.state = self.STATE_SUBMITTED
        return self.actual_elapsed_seconds

    def update(self, now_monotonic: float | None = None) -> tuple[bool, float]:
        """Update timer state based on monotonic time.

        Returns:
            (is_just_expired, actual_elapsed_seconds)
            is_just_expired is True exactly once when reaching zero.
        """
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self.state == self.STATE_RUNNING:
            if self.deadline_monotonic is not None:
                rem = self.deadline_monotonic - now
                if rem <= 0.0:
                    self.remaining_seconds = 0.0
                    self.actual_elapsed_seconds = float(self.duration_seconds)
                    self.state = self.STATE_EXPIRED
                    self.deadline_monotonic = None
                    if not self.completion_handled:
                        self.completion_handled = True
                        return True, self.actual_elapsed_seconds
                else:
                    self.remaining_seconds = rem
                    self.actual_elapsed_seconds = max(0.0, self.duration_seconds - rem)

        return False, self.actual_elapsed_seconds

    def is_warning(self) -> bool:
        """Check whether the active timer has entered the low-time warning threshold."""
        return (
            self.state == self.STATE_RUNNING
            and 0.0 < self.remaining_seconds <= float(self.warning_threshold_seconds)
        )

    def format_time(self, seconds: float | None = None) -> str:
        """Format seconds into MM:SS string."""
        sec = int(round(self.remaining_seconds if seconds is None else seconds))
        sec = max(0, sec)
        mins = sec // 60
        secs = sec % 60
        return f"{mins:02d}:{secs:02d}"

    def get_progress(self) -> float:
        """Calculate progress ratio from 0.0 (started) to 1.0 (completed/expired)."""
        if self.duration_seconds <= 0:
            return 1.0
        elapsed = self.duration_seconds - self.remaining_seconds
        return min(1.0, max(0.0, elapsed / self.duration_seconds))

    def to_dict(self) -> dict[str, Any]:
        """Serialize quiz timer state."""
        return {
            "document_id": self.document_id,
            "question_set_id": self.question_set_id,
            "attempt_number": self.attempt_number,
            "duration_seconds": self.duration_seconds,
            "warning_threshold_seconds": self.warning_threshold_seconds,
            "state": self.state,
            "remaining_seconds": self.remaining_seconds,
            "deadline_monotonic": self.deadline_monotonic,
            "completion_handled": self.completion_handled,
            "actual_elapsed_seconds": self.actual_elapsed_seconds,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        """Restore quiz timer state."""
        self.document_id = str(data.get("document_id", self.document_id))
        self.question_set_id = str(data.get("question_set_id", self.question_set_id))
        self.attempt_number = int(data.get("attempt_number", self.attempt_number))
        self.duration_seconds = int(data.get("duration_seconds", self.duration_seconds))
        self.warning_threshold_seconds = int(
            data.get("warning_threshold_seconds", self.warning_threshold_seconds)
        )
        self.state = str(data.get("state", self.STATE_IDLE))
        self.remaining_seconds = float(data.get("remaining_seconds", self.duration_seconds))
        self.deadline_monotonic = data.get("deadline_monotonic")
        self.completion_handled = bool(data.get("completion_handled", False))
        self.actual_elapsed_seconds = float(data.get("actual_elapsed_seconds", 0.0))


