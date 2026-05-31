"""The blackboard — stigmergic coordination substrate.

Agents do NOT talk to each other or to a central conductor. They coordinate only
through this shared board: claim a ready task, write its result, which unlocks its
dependents. Coordination emerges from the queue + dependency state, so the system
scales to dozens of workers without a "main" agent that panics (no central context
accumulation, no serial supervision bottleneck).

v1 is an in-memory DAG queue. The same interface backs a SQLite/file board later
(for persistence across restarts and multi-process producers) — see TODO markers.
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class State(str, Enum):
    PENDING = "pending"     # waiting on dependencies
    READY = "ready"         # deps satisfied, can be claimed
    RUNNING = "running"     # claimed by a worker
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    prompt: str
    deps: List[str] = field(default_factory=list)   # task ids that must be DONE first
    lane: str = "worker"                            # role/context-class (router|worker|reducer|planner)
    state: State = State.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def spec(self) -> Dict[str, Any]:
        """Picklable payload handed to a worker process."""
        return {"id": self.id, "prompt": self.prompt, "lane": self.lane, "meta": self.meta}


class Board:
    """Thread-safe DAG work queue. The scheduler is the only writer of state
    transitions; producers may add tasks at any time (even while running)."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self.started_at = time.time()

    # ---- producer side --------------------------------------------------
    def add(self, task: Task) -> None:
        with self._lock:
            self._tasks[task.id] = task
            self._refresh(task)

    def add_many(self, tasks: List[Task]) -> None:
        for t in tasks:
            self.add(t)

    # ---- scheduler side -------------------------------------------------
    def _deps_done(self, t: Task) -> bool:
        return all(self._tasks.get(d) and self._tasks[d].state == State.DONE for d in t.deps)

    def _refresh(self, t: Task) -> None:
        if t.state == State.PENDING and self._deps_done(t):
            t.state = State.READY

    def claim_ready(self, n: int) -> List[Task]:
        """Atomically claim up to n READY tasks -> RUNNING."""
        with self._lock:
            for t in self._tasks.values():
                self._refresh(t)
            ready = [t for t in self._tasks.values() if t.state == State.READY][:n]
            for t in ready:
                t.state = State.RUNNING
                if t.deps:                       # inject upstream results so reducers can reduce
                    t.meta = dict(t.meta, dep_results={d: self._tasks[d].result
                                                       for d in t.deps if d in self._tasks})
            return ready

    def complete(self, tid: str, result: str) -> None:
        with self._lock:
            t = self._tasks[tid]
            t.state, t.result = State.DONE, result
            for o in self._tasks.values():       # unlock dependents
                self._refresh(o)

    def fail(self, tid: str, error: str, max_retries: int) -> bool:
        """Mark failed; requeue if retries remain. Returns True if requeued."""
        with self._lock:
            t = self._tasks[tid]
            if t.retries < max_retries:
                t.retries += 1
                t.state = State.PENDING
                self._refresh(t)
                return True
            t.state, t.error = State.FAILED, error
            return False

    # ---- introspection --------------------------------------------------
    def counts(self) -> Dict[str, int]:
        with self._lock:
            c = {s.value: 0 for s in State}
            for t in self._tasks.values():
                c[t.state.value] += 1
            return c

    def unfinished(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values()
                       if t.state not in (State.DONE, State.FAILED))

    def has_ready(self) -> bool:
        with self._lock:
            for t in self._tasks.values():
                self._refresh(t)
            return any(t.state == State.READY for t in self._tasks.values())

    def results(self) -> Dict[str, Task]:
        with self._lock:
            return dict(self._tasks)
