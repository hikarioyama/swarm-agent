"""Persistent goal queue for completion-managed swarm tasks."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path


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

    def add(self, goal: str) -> dict:
        with self._lock:
            now = time.time()
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
        """Atomically claim the oldest pending goal: pending -> running, returns a copy.

        Single locked transition (vs next_pending() + mark_running()) so two concurrent
        dispatchers can never claim the same goal (PARALLEL_GOALS_PLAN §4.3). Returns None
        when nothing is pending.
        """
        with self._lock:
            for rec in self._records:
                if rec.get("state") == "pending":
                    now = time.time()
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
            now = time.time()
            rec["state"] = "running"
            rec["started_at"] = now
            rec["progress_at"] = now
            self._persist()

    def touch(self, tid: str) -> None:
        with self._lock:
            rec = self._find(tid)
            if rec is None:
                return
            rec["progress_at"] = time.time()
            self._persist()

    def complete(self, tid: str, result: str) -> None:
        with self._lock:
            rec = self._find(tid)
            if rec is None:
                return
            now = time.time()
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
            now = time.time()
            rec["attempts"] = int(rec.get("attempts") or 0) + 1
            rec["state"] = "pending" if rec["attempts"] < self.max_attempts else "failed"
            rec["started_at"] = None
            rec["finished_at"] = now if rec["state"] == "failed" else None
            rec["error"] = error
            rec["progress_at"] = now
            self._persist()
            return rec["state"]

    def requeue(self, tid: str) -> None:
        with self._lock:
            rec = self._find(tid)
            if rec is None:
                return
            rec["state"] = "pending"
            rec["started_at"] = None
            rec["finished_at"] = None
            rec["progress_at"] = time.time()
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
