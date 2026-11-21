"""Application instance locking and lifecycle management.

Guarantees single-instance execution for StudyAstra to prevent concurrent
processes from corrupting the persistent FAISS vector store or study statistics.

Uses OS kernel-level byte-range locking (msvcrt on Windows, fcntl on POSIX)
which ensures that locks are automatically freed by the operating system kernel
if a process terminates abnormally or crashes.
"""

from __future__ import annotations

import atexit
import datetime
import json
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any

from src import config

logger = logging.getLogger(__name__)

DEFAULT_LOCK_FILENAME = ".app.lock"
DEFAULT_INFO_FILENAME = ".app.lock.info"


class ApplicationLockError(RuntimeError):
    """Raised when an application lock operation encounters an internal error."""


class ApplicationLock:
    """Manages single-instance lifecycle locking using OS kernel byte locking.

    Attributes:
        lock_path: Absolute path to the OS lock file.
        info_path: Absolute path to the companion JSON metadata file.
    """

    def __init__(
        self,
        lock_dir: Path | str | None = None,
        lock_filename: str = DEFAULT_LOCK_FILENAME,
        info_filename: str = DEFAULT_INFO_FILENAME,
    ) -> None:
        base_dir = Path(lock_dir) if lock_dir is not None else config.PROJECT_ROOT
        self._lock_path = base_dir / lock_filename
        self._info_path = base_dir / info_filename

        self._thread_lock = threading.RLock()
        self._is_held: bool = False
        self._file_handle: Any = None
        self._holder_info: dict[str, Any] | None = None
        self._atexit_registered: bool = False

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    @property
    def info_path(self) -> Path:
        return self._info_path

    @property
    def is_held(self) -> bool:
        """Return True if this instance currently holds the OS lock."""
        with self._thread_lock:
            return self._is_held

    def acquire(self) -> tuple[bool, dict[str, Any] | None, str | None]:
        """Attempt to acquire the exclusive single-instance application lock.

        Returns:
            tuple[bool, dict | None, str | None]:
                - is_acquired (bool): True if the lock was acquired or already held.
                - holder_info (dict | None): Diagnostic metadata of the active holder.
                - error_message (str | None): Diagnostic message if acquisition failed.
        """
        with self._thread_lock:
            # Re-entrant in the current process
            if self._is_held:
                return True, self._holder_info, None

            # Ensure directory exists
            try:
                self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                msg = f"Failed to create lock directory '{self._lock_path.parent}': {exc}"
                logger.error(msg)
                return False, None, msg

            handle = None
            try:
                # Open lock file in read/write binary mode (create if not exists)
                handle = open(self._lock_path, "a+b")
                handle.seek(0)

                # Attempt non-blocking kernel byte-range lock
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            except (OSError, PermissionError) as exc:
                # Lock is held by another active process
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass

                existing_info = self._read_info()
                holder_desc = self._format_holder_diagnostic(existing_info)
                err_msg = (
                    f"Another StudyAstra process holds the application lock ({holder_desc})."
                )
                logger.warning(err_msg)
                return False, existing_info, err_msg
            except Exception as exc:
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass
                msg = f"Unexpected error acquiring application lock: {exc}"
                logger.exception(msg)
                return False, None, msg

            # Lock acquired successfully
            self._file_handle = handle
            self._is_held = True

            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
            info: dict[str, Any] = {
                "pid": os.getpid(),
                "started_at": now_iso,
                "lock_path": str(self._lock_path),
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "platform": sys.platform,
            }
            self._holder_info = info
            self._write_info(info)

            if not self._atexit_registered:
                atexit.register(self.release)
                self._atexit_registered = True

            logger.info(
                "Acquired single-instance application lock at %s (PID: %d)",
                self._lock_path,
                os.getpid(),
            )
            return True, info, None

    def release(self) -> None:
        """Release the application lock safely.

        Follows a strict release sequence to eliminate concurrency races:
        1. Acquire the in-process thread lock.
        2. Return immediately if this instance does not hold the application lock.
        3. While still holding the OS kernel lock, remove .app.lock.info if it
           belongs to the current process.
        4. Release the OS kernel byte lock.
        5. Close the file handle (guaranteed via finally block).
        6. Reset the in-process ownership state.
        7. Retain .app.lock permanently on disk as a safe, static lock anchor.
        """
        with self._thread_lock:
            # 2. Return immediately if this instance does not hold the lock
            if not self._is_held:
                return

            # 3. While still holding OS kernel lock, remove companion metadata if ours
            try:
                if self._info_path.exists():
                    existing = self._read_info()
                    if existing and existing.get("pid") == os.getpid():
                        self._info_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.debug("Exception removing lock info file: %s", exc)

            # 4. Release OS kernel byte lock & 5. Close file handle
            if self._file_handle is not None:
                try:
                    self._file_handle.seek(0)
                    if sys.platform == "win32":
                        import msvcrt

                        try:
                            msvcrt.locking(self._file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                        except OSError:
                            pass
                    else:
                        import fcntl

                        try:
                            fcntl.flock(self._file_handle.fileno(), fcntl.LOCK_UN)
                        except OSError:
                            pass
                except Exception as exc:
                    logger.debug("Exception while unlocking OS byte lock: %s", exc)
                finally:
                    # 5. Ensure file handle is closed even if unlocking raises
                    try:
                        self._file_handle.close()
                    except Exception:
                        pass
                    self._file_handle = None

            # 6. Reset in-process ownership state
            self._is_held = False
            self._holder_info = None

            # 7. Retain .app.lock permanently on disk.
            # Never unlink .app.lock during normal shutdown:
            # Eliminates file-deletion races, eliminates Windows [WinError 32]
            # sharing violations, and ensures subsequent processes lock against
            # the exact same filesystem object.
            logger.info("Released single-instance application lock at %s", self._lock_path)

    def _read_info(self) -> dict[str, Any] | None:
        """Safely read metadata from companion info file."""
        try:
            if self._info_path.exists():
                text = self._info_path.read_text(encoding="utf-8").strip()
                if text:
                    return json.loads(text)
        except Exception as exc:
            logger.debug("Could not read lock info file '%s': %s", self._info_path, exc)
        return None

    def _write_info(self, info: dict[str, Any]) -> None:
        """Write metadata to companion info file."""
        try:
            self._info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not write lock info file '%s': %s", self._info_path, exc)

    def _format_holder_diagnostic(self, info: dict[str, Any] | None) -> str:
        """Format holder diagnostic string."""
        if not info:
            return "holding process details unavailable"
        pid = info.get("pid", "unknown")
        started = info.get("started_at", "unknown")
        return f"PID: {pid}, Started: {started}"

    def __enter__(self) -> ApplicationLock:
        ok, _, err = self.acquire()
        if not ok:
            raise ApplicationLockError(err or "Failed to acquire application lock.")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


_GLOBAL_APP_LOCK: ApplicationLock | None = None
_GLOBAL_LOCK_MUTEX = threading.RLock()


def get_application_lock(lock_dir: Path | str | None = None) -> ApplicationLock:
    """Get or initialize the process-wide ApplicationLock singleton."""
    global _GLOBAL_APP_LOCK
    with _GLOBAL_LOCK_MUTEX:
        if _GLOBAL_APP_LOCK is None:
            _GLOBAL_APP_LOCK = ApplicationLock(lock_dir=lock_dir)
        return _GLOBAL_APP_LOCK


def reset_global_application_lock() -> None:
    """Release and clear the process-wide ApplicationLock singleton (for tests)."""
    global _GLOBAL_APP_LOCK
    with _GLOBAL_LOCK_MUTEX:
        if _GLOBAL_APP_LOCK is not None:
            try:
                _GLOBAL_APP_LOCK.release()
            except Exception:
                pass
            _GLOBAL_APP_LOCK = None
