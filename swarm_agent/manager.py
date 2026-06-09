"""Background completion manager for persistent swarm goals."""
from __future__ import annotations

import json
import os
import threading
import time

from fleet import config

from . import audit
from . import worktree as _worktree
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
        # Goal ids already escalated to the human (TUI + Telegram), so a terminal failure is
        # announced exactly ONCE, not re-announced on every subsequent heartbeat (§B.3).
        self._escalated: set[str] = set()

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

    # ── Manager v2 auditor (SWARM_V2 §B) ─────────────────────────────────────────
    def _park_dir(self) -> str:
        return os.path.join(os.path.expanduser(config.WORKTREE_ROOT), "parked")

    def _drain_merges(self) -> None:
        """§B.4: merge completed writing-goal worktrees back to base ONE AT A TIME (sequential,
        even though the goals ran in parallel) so the base branch never sees an interleaved
        half-merge. Clean → remove the worktree; conflict → PARK it (never auto-resolve, never
        delete) and escalate with the conflicting paths."""
        while True:
            wt = self.runner.pop_pending_merge()
            if wt is None:
                return
            try:
                res = _worktree.merge_back(wt)
            except Exception as e:
                self.runner.emit("error", text=f"merge of {wt.branch} errored: {e!r}")
                continue
            if res.ok:
                try:
                    _worktree.remove(wt)
                except Exception:
                    pass
                self.runner.emit("manager",
                                 text=f"merged {wt.branch} into {wt.base_branch or 'base'}")
            else:
                parked = wt.path
                try:
                    parked = _worktree.park(wt, park_dir=self._park_dir())
                except Exception:
                    pass
                detail = ", ".join(res.conflicting_paths) or res.message
                self.runner.emit("error", text=(
                    f"merge conflict on {wt.branch} ({detail}); parked at {parked} — "
                    f"needs your attention (resolve and merge by hand)"))

    def _gc_worktrees(self) -> None:
        """§B.2 worktree-leak: reclaim worktrees belonging to no active goal — UNCHANGED ones
        pruned, CHANGED ones parked (never deleted). No-op unless parallel writes are on and we
        are inside a git repo."""
        if not config.PARALLEL_WRITES:
            return
        repo = _worktree.repo_root(os.getcwd())
        if not repo:
            return
        try:
            res = _worktree.gc_worktrees(
                config.WORKTREE_ROOT, self.runner.active_goal_ids(), repo=repo,
                park_dir=self._park_dir(), branch_prefix=config.GOAL_BRANCH_PREFIX)
        except Exception as e:
            self.runner.log.event("manager_error", error=f"worktree gc: {e!r}")
            return
        if res["pruned"] or res["parked"]:
            self.runner.emit("manager", text=(
                f"worktree GC: pruned {len(res['pruned'])} stale, "
                f"parked {len(res['parked'])} with changes"))

    def _audit(self, now: float, snapshot) -> list[str]:
        """Bounded auto-remediation over a snapshot (§B.2-B.3). Returns the goal ids escalated
        this pass. HANG (running + alive + no progress for > STUCK_SECONDS) → interrupt and
        bounded-requeue via ``fail`` (which terminally fails once the retry budget is spent);
        every NEWLY terminal goal (hang/thrash exhausted, deadlock) is escalated to the human
        exactly once. A starved decode gate is reported (held to the heartbeat)."""
        max_attempts = self.runner.tasks.max_attempts
        stuck_seconds = float(getattr(config, "STUCK_SECONDS", 600.0))
        active = self.runner.active_goal_ids()
        escalated: list[str] = []

        # 1) hang remediation (may push a wedged goal to terminal failure).
        for rec in snapshot:
            rid = rec.get("id")
            if (rec.get("state") == "running" and rid in active
                    and audit.is_stuck(rec, now, stuck_seconds)):
                # NB: interrupt() is currently process-global (no per-goal targeting yet), so it
                # also nudges other live agents — acceptable as a last resort for a goal silent
                # for > STUCK_SECONDS (>> the per-task TASK_TIMEOUT, i.e. genuinely wedged).
                try:
                    self.runner.interrupt()
                except Exception:
                    pass
                state = self.runner.tasks.fail(
                    rid, "hang: no progress for too long; interrupted by manager")
                if state != "failed":
                    self.runner.emit("manager", text=f"interrupted + requeued hung task {rid}")

        # 2) escalate every NEWLY terminal failure exactly once (hang/thrash exhausted, deadlock).
        for rec in self.runner.tasks.snapshot():
            rid = rec.get("id")
            if rec.get("state") == "failed" and rid not in self._escalated:
                self._escalated.add(rid)
                self.runner.emit("error", text=(
                    f"task {rid} failed and needs your attention: "
                    f"{rec.get('error') or rec.get('goal')}"))
                escalated.append(rid)

        # 3) gate starvation: workers waiting but nothing decoding → hold + note (server flap).
        stats = None
        try:
            stats = self.runner.gate.stats() if self.runner.gate is not None else None
        except Exception:
            stats = None
        if audit.gate_starved(stats):
            self.runner.emit("manager",
                             text="decode gate starved (server flap?) — holding dispatch")
        return escalated

    def _fail_deadlocked(self, snapshot) -> None:
        """§2.5 / §B.2: a PENDING goal whose deps include a ``failed`` (or missing) goal can
        never become dispatchable — terminally fail it with a reason naming the dep, rather
        than leaving it stuck in the queue forever. ``mark_failed`` is terminal (no requeue),
        so the auditor does not re-flag it each tick."""
        by_id = {r.get("id"): r for r in snapshot}
        for rec in snapshot:
            dep = audit.deadlocked_dep(rec, by_id)
            if dep is not None:
                self.runner.tasks.mark_failed(
                    rec["id"], f"dependency {dep} failed; goal can never run")
                self.runner.emit(
                    "error", text=f"task {rec['id']} deadlocked: dependency {dep} failed")

    def _tick(self) -> None:
        # §B.4: land any completed writing-goal worktrees first (sequential merge-back).
        self._drain_merges()
        snapshot = self.runner.tasks.snapshot()
        # Fail goals deadlocked behind a failed dependency BEFORE dispatch (so a deadlocked
        # goal never blocks the queue and is reported promptly).
        self._fail_deadlocked(snapshot)
        snapshot = self.runner.tasks.snapshot()
        # Re-queue stalled work: a record marked 'running' whose turn is NOT actually in
        # flight (orphaned by a crash/dead thread). With parallel goals we can no longer use
        # a single 'not busy' flag — check the live in-flight set instead.
        active = self.runner.active_goal_ids()
        for rec in snapshot:
            if rec.get("state") == "running" and rec["id"] not in active:
                self.runner.tasks.requeue(rec["id"])
                self.runner.emit("manager", text=f"re-queued stalled task {rec['id']}")
        # §B.2-B.3: bounded auto-remediation (hang interrupt+requeue, terminal escalation).
        self._audit(time.time(), self.runner.tasks.snapshot())

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

        # §B.2: reclaim leaked worktrees (prune unchanged / park changed). Cheap dir scan.
        self._gc_worktrees()

        now = time.time()
        if self.runner.tasks.has_unfinished() and now - self._last_eval >= self.interval_s:
            self._last_eval = now
            self._evaluate(now)

        # Periodic skill curator: only when IDLE (no queued work, no live turn) so it never
        # competes with the swarm, and only if the server is up (the consolidation pass runs
        # an LLM). should_run_now() self-gates to its own weekly interval, so this check is
        # cheap to make every heartbeat.
        if not self.runner.tasks.has_unfinished() and not self.runner.busy:
            try:
                from .skills import curator as _cur
                if _cur.should_run_now() and self._server_ok():
                    from .skills.llm import make_proposer
                    threading.Thread(
                        target=lambda: _cur.run_curator(make_proposer()),
                        name="swarm-curator", daemon=True).start()
                    self.runner.emit("manager", text="running skill curator (idle maintenance)")
            except Exception as e:
                self.runner.log.event("manager_error", error=f"curator: {e!r}")

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
