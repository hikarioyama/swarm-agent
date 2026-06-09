"""Persistent goal queue for completion-managed swarm tasks."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path


def _now() -> float:
    """Single clock seam for the store. Tests monkeypatch this to inject a fake clock
    so stuck/liveness assertions never sleep on the wall clock (SWARM_V2_TODO §0)."""
    return time.time()


class TaskStore:
    """JSON-backed, thread-safe queue of goals that survive TUI restarts."""

    def __init__(self, path: str | None = None, *, max_attempts: int | None = None) -> None:
        self.path = Path(path or os.environ.get(
            "SWARM_TASKS_PATH", str(Path.home() / ".cache" / "swarm-agent" / "tasks.json")))
        self.max_attempts = int(max_attempts if max_attempts is not None else
                                os.environ.get("SWARM_TASK_MAX_ATTEMPTS", "3"))
        self._lock = threading.RLock()
        self._records: list[dict] = []
        with self._lock:
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
            self._records = data if isinstance(data, list) else []
        except Exception:
            self._records = []
        recovered = False
        for rec in self._records:
            # Migrate legacy records that predate the inter-goal DAG (§2.1): no deps key → [].
            if "deps" not in rec:
                rec["deps"] = []
            if rec.get("state") == "running":
                rec["state"] = "pending"
                rec["started_at"] = None
                recovered = True
        if recovered:
            self._persist()

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._records, ensure_ascii=False, indent=2))
        os.replace(tmp, self.path)

    def _find(self, tid: str) -> dict | None:
        for rec in self._records:
            if rec.get("id") == tid:
                return rec
        return None

    def add(self, goal: str, *, deps=None, now: float | None = None) -> dict:
        """Append a pending goal. ``deps`` are the ids of OTHER queued goals whose result this
        one needs (inter-goal DAG edges, §A.2) — it stays unclaimable until they are ``done``."""
        with self._lock:
            now = _now() if now is None else now
            rec = {
                "id": f"task-{uuid.uuid4().hex[:8]}",
                "goal": goal,
                "state": "pending",
                "attempts": 0,
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
                "progress_at": now,
                "deps": [str(d) for d in (deps or [])],
            }
            self._records.append(rec)
            self._persist()
            return dict(rec)

    def next_pending(self) -> dict | None:
        with self._lock:
            for rec in self._records:
                if rec.get("state") == "pending":
                    return dict(rec)
        return None

    def claim_next(self) -> dict | None:
        """Atomically claim the oldest DISPATCHABLE pending goal: pending -> running, copy.

        Single locked transition (vs next_pending() + mark_running()) so two concurrent
        dispatchers can never claim the same goal (PARALLEL_GOALS_PLAN §4.3). Dependency-aware
        (§2.2): a goal is dispatchable only once EVERY one of its ``deps`` is ``done`` — a goal
        waiting on an unfinished (or failed) dep is skipped, so the oldest *ready* goal wins.
        Returns None when nothing is ready.
        """
        with self._lock:
            done = {r.get("id") for r in self._records if r.get("state") == "done"}
            for rec in self._records:
                if rec.get("state") != "pending":
                    continue
                if not all(dep in done for dep in (rec.get("deps") or [])):
                    continue                       # a dep is not done yet → not dispatchable
                now = _now()
                rec["state"] = "running"
                rec["started_at"] = now
                rec["progress_at"] = now
                self._persist()
                return dict(rec)
        return None

    def mark_running(self, tid: str) -> None:
        with self._lock:
            rec = self._find(tid)
            if rec is None:
                return
            now = _now()
            rec["state"] = "running"
            rec["started_at"] = now
            rec["progress_at"] = now
            self._persist()

    def touch(self, tid: str, *, now: float | None = None) -> None:
        """Advance a record's last-progress timestamp. Called on every fleet task event
        for a queued goal so a healthy long-running goal keeps a fresh ``progress_at`` —
        without this the manager cannot tell "running+progressing" from "running+stuck"
        (the whole point of §B.1: today ``progress_at`` only moves at start)."""
        with self._lock:
            rec = self._find(tid)
            if rec is None:
                return
            rec["progress_at"] = _now() if now is None else now
            self._persist()

    def seconds_since_progress(self, tid: str, now: float | None = None) -> float | None:
        """Pure read: seconds elapsed since the record last made progress, or None if the
        record is unknown. One definition of "stale" the manager's stuck-detection relies on
        (SWARM_V2_TODO §0.2). Falls back to ``created_at`` for a record never touched."""
        now = _now() if now is None else now
        with self._lock:
            rec = self._find(tid)
            if rec is None:
                return None
            base = rec.get("progress_at")
            if base is None:
                base = rec.get("created_at")
            if base is None:
                return None
            return max(0.0, now - float(base))

    def complete(self, tid: str, result: str) -> None:
        with self._lock:
            rec = self._find(tid)
            if rec is None:
                return
            now = _now()
            rec["state"] = "done"
            rec["finished_at"] = now
            rec["result"] = result
            rec["error"] = None
            rec["progress_at"] = now
            self._persist()

    def fail(self, tid: str, error: str) -> str:
        with self._lock:
            rec = self._find(tid)
            if rec is None:
                return ""
            now = _now()
            rec["attempts"] = int(rec.get("attempts") or 0) + 1
            rec["state"] = "pending" if rec["attempts"] < self.max_attempts else "failed"
            rec["started_at"] = None
            rec["finished_at"] = now if rec["state"] == "failed" else None
            rec["error"] = error
            rec["progress_at"] = now
            self._persist()
            return rec["state"]

    def mark_failed(self, tid: str, error: str) -> None:
        """Terminally fail a goal WITHOUT incrementing attempts or re-queuing (unlike ``fail``).

        Used by the manager for an unrecoverable situation — a DAG deadlock (a dep failed) or
        an escalated, budget-exhausted goal — where retrying is pointless and re-pending would
        loop the auditor. ``fail`` is for a genuine attempt that erred (and may retry)."""
        with self._lock:
            rec = self._find(tid)
            if rec is None:
                return
            now = _now()
            rec["state"] = "failed"
            rec["started_at"] = None
            rec["finished_at"] = now
            rec["error"] = error
            rec["progress_at"] = now
            self._persist()

    def requeue(self, tid: str) -> None:
        with self._lock:
            rec = self._find(tid)
            if rec is None:
                return
            rec["state"] = "pending"
            rec["started_at"] = None
            rec["finished_at"] = None
            rec["progress_at"] = _now()
            self._persist()

    def counts(self) -> dict:
        with self._lock:
            out = {"pending": 0, "running": 0, "done": 0, "failed": 0,
                   "total": len(self._records)}
            for rec in self._records:
                state = rec.get("state")
                if state in out:
                    out[state] += 1
            return out

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(rec) for rec in self._records]

    def has_unfinished(self) -> bool:
        with self._lock:
            return any(rec.get("state") in ("pending", "running") for rec in self._records)
