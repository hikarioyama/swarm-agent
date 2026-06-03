"""Background completion manager for persistent swarm goals."""
from __future__ import annotations

import json
import os
import threading
import time

from .runner import _parse_route


class CompletionManager:
    """Wake periodically or on queue events and keep persistent goals moving."""

    def __init__(self, runner, *, interval_s: float | None = None) -> None:
        self.runner = runner
        self.interval_s = float(interval_s if interval_s is not None else
                                os.environ.get("SWARM_MANAGER_INTERVAL", "180"))
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_eval = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="swarm-manager", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self.runner._wake.set()

    def notify(self) -> None:
        self.runner._wake.set()

    def _loop(self) -> None:
        while not self._stopped.is_set():
            self.runner._wake.wait(timeout=self.interval_s)
            self.runner._wake.clear()
            if self._stopped.is_set():
                break
            try:
                self._tick()
            except Exception as e:
                import traceback
                try:
                    self.runner.log.event("manager_error", error=repr(e),
                                          detail=traceback.format_exc())
                except Exception:
                    pass

    def _tick(self) -> None:
        snapshot = self.runner.tasks.snapshot()
        # Re-queue stalled work: a record marked 'running' whose turn is NOT actually in
        # flight (orphaned by a crash/dead thread). With parallel goals we can no longer use
        # a single 'not busy' flag — check the live in-flight set instead.
        active = self.runner.active_goal_ids()
        for rec in snapshot:
            if rec.get("state") == "running" and rec["id"] not in active:
                self.runner.tasks.requeue(rec["id"])
                self.runner.emit("manager", text=f"re-queued stalled task {rec['id']}")

        # Dispatch up to capacity. Probe the server ONCE: if it is down, hold all pending
        # work (back off to the heartbeat) rather than claim-then-requeue churn.
        if self.runner.tasks.has_unfinished() and self.runner.can_admit_goal():
            if not self._server_ok():
                # only announce if there is actually pending work to start
                if any(r.get("state") == "pending" for r in snapshot):
                    self.runner.emit(
                        "manager",
                        text="waiting for inference server before starting queued work")
            else:
                while self.runner.can_admit_goal():
                    rec = self.runner.tasks.claim_next()   # atomic pending -> running
                    if rec is None:
                        break
                    if not self.runner._setup_done:
                        self.runner.setup()
                    th = self.runner.submit_goal(rec)
                    if th is None:
                        # already in flight (shouldn't happen) — undo the claim
                        self.runner.tasks.requeue(rec["id"])
                        break
                    self.runner.emit("manager",
                                     text=f"starting queued task: {rec['goal'][:60]}")

        now = time.time()
        if self.runner.tasks.has_unfinished() and now - self._last_eval >= self.interval_s:
            self._last_eval = now
            self._evaluate(now)

    def _server_ok(self) -> bool:
        """Fast probe: is the inference server reachable? Defensive (False on error)."""
        try:
            from fleet import metrics, config
            return metrics.scrape(config.METRICS_URL, timeout=0.2) is not None
        except Exception:
            return False

    def _evaluate(self, now: float) -> None:
        rows = []
        snapshot = self.runner.tasks.snapshot()
        for rec in snapshot:
            age = max(0, int(now - float(rec.get("progress_at") or rec.get("created_at") or now)))
            rows.append({
                "id": rec.get("id"), "state": rec.get("state"),
                "attempts": rec.get("attempts"), "goal": str(rec.get("goal") or "")[:120],
                "seconds_since_progress": age,
            })
        prompt = """You manage a persistent swarm goal queue. Assess stalled work and
return ONLY JSON:
{"note":"<one sentence status>","requeue":["id",...],"escalate":["id",...]}
Tasks:
""" + json.dumps(rows, ensure_ascii=False)
        try:
            text = self.runner._run_agent("manager", prompt, "manage",
                                          max_iterations=1, max_tokens=1024)
        except self.runner._ServerDown:
            return
        data = _parse_route(text) or {}
        by_id = {rec.get("id"): rec for rec in snapshot}
        for tid in data.get("requeue") or []:
            rec = by_id.get(tid)
            if (rec and int(rec.get("attempts") or 0) < self.runner.tasks.max_attempts
                    and rec.get("state") in ("pending", "failed")):
                self.runner.tasks.requeue(tid)
        for tid in data.get("escalate") or []:
            rec = by_id.get(tid)
            if rec:
                self.runner.emit("error",
                                 text=f"task {tid} needs your attention: {rec.get('error') or rec.get('goal')}")
        self.runner.emit("manager", text=str(data.get("note") or "queue evaluated"))
