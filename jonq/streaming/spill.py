"""Temporary chunk files with guaranteed cleanup.

Every temporary file the engine writes lives inside one directory created by
:class:`TempWorkspace`.  The directory is removed on normal exit, on exception,
and when the engine is interrupted (``SIGINT``/``SIGTERM``), so a failed query
never leaves spill files behind.  ``keep_temp`` disables cleanup for debugging.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import tempfile
import threading

from jonq.streaming.config import DEFAULT_CHUNK_DIR_PREFIX


class TempWorkspace:
    def __init__(self, *, base_dir: str | None = None, prefix: str = DEFAULT_CHUNK_DIR_PREFIX, keep: bool = False):
        self.keep = keep
        self.path = tempfile.mkdtemp(prefix=prefix, dir=base_dir)
        self._closed = False
        self._lock = threading.Lock()
        self._counter = 0
        self._previous_handlers = {}
        self._atexit_registered = True
        atexit.register(self.cleanup)
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        # Signal handlers can only be installed from the main thread; when the
        # engine runs in a worker thread, atexit + context management suffice.
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._on_signal)
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            self._previous_handlers.clear()

    def _restore_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig, handler in self._previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):  # pragma: no cover
                pass
        self._previous_handlers.clear()

    def _on_signal(self, signum, frame):
        self.cleanup()
        previous = self._previous_handlers.get(signum)
        if callable(previous):
            previous(signum, frame)
            return
        if previous == signal.SIG_IGN:
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def new_file(self, prefix: str = "chunk", suffix: str = ".jsonl") -> str:
        with self._lock:
            self._counter += 1
            name = f"{prefix}_{self._counter:06d}{suffix}"
            return os.path.join(self.path, name)

    def cleanup(self) -> None:
        with self._lock:
            if self._closed or self.keep:
                return
            self._closed = True
        self._restore_signal_handlers()
        shutil.rmtree(self.path, ignore_errors=True)
        if self._atexit_registered:
            self._atexit_registered = False
            try:
                atexit.unregister(self.cleanup)
            except (ValueError, RuntimeError):  # pragma: no cover
                pass

    def __enter__(self) -> "TempWorkspace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def __del__(self):  # pragma: no cover - best-effort safety net
        try:
            self.cleanup()
        except Exception:
            pass
