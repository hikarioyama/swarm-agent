"""GoalScheduler — shared/exclusive, K-capped admission for parallel goal executions.

Read-only goals fill the shared DecodeGate concurrently, up to K at once. Writing goals
run alone so they cannot collide on files or cwd. The scheduler is a readers-writers lock
with writer preference: once a writer is waiting, new readers stop entering so a steady
stream of read-only goals cannot starve it (PARALLEL_GOALS_PLAN §4.1).
"""
from __future__ import annotations

import contextlib
import threading


class GoalScheduler:
    def __init__(self, k: int) -> None:
        self._k = max(1, int(k))
        self._cond = threading.Condition()
        self._readers = 0            # active read-only executions
        self._writer = False         # a writer is currently executing
        self._waiting_writers = 0    # writers blocked waiting (writer-preference / no-starve)

    @property
    def k(self) -> int:
        return self._k

    # introspection for the manager's capacity check (no lock needed for a cheap read;
    # these are advisory snapshots used only to decide optimistic dispatch).
    @property
    def writer_active(self) -> bool:
        return self._writer

    @property
    def writer_pending(self) -> bool:
        return self._waiting_writers > 0

    @property
    def readers(self) -> int:
        return self._readers

    def acquire_readonly(self) -> None:
        with self._cond:
            # Wait while a writer holds/awaits the lane (writer-preference: a pending writer
            # blocks NEW readers so a steady reader stream can't starve it) or the reader cap
            # is full.
            while self._writer or self._waiting_writers > 0 or self._readers >= self._k:
                self._cond.wait()
            self._readers += 1

    def acquire_writer(self) -> None:
        with self._cond:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1

    def release(self, *, writer: bool) -> None:
        with self._cond:
            if writer:
                self._writer = False
            else:
                self._readers -= 1
            self._cond.notify_all()

    @contextlib.contextmanager
    def permit(self, *, readonly: bool):
        """Acquire the right permit for one fleet execution, release on exit.
        readonly=True -> shared reader slot (≤K); readonly=False -> exclusive writer."""
        if readonly:
            self.acquire_readonly()
        else:
            self.acquire_writer()
        try:
            yield
        finally:
            self.release(writer=not readonly)
