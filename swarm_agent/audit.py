"""Pure auditor signals + remediation policy for the Swarm v2 manager.

Every function here is a PURE predicate/decision over a task record (or a snapshot of
records) plus an injected ``now`` — NO threads, NO LLM, NO inference server, NO disk. That
is deliberate: the manager's "is it stuck / thrashing / deadlocked, and what do I do about
it" logic (SWARM_V2_TODO §B.2–B.3) is the part most worth testing in isolation, so it lives
here behind tiny inputs the tests can craft by hand. The manager (manager.py) wires these
detectors to real actions (interrupt/requeue/fail/escalate) and the worktree GC; the
filesystem/git half of the worktree-leak signal lives in ``worktree.py`` (it cannot be pure).
"""
from __future__ import annotations

from typing import Optional

# ── signal names (what an auditor tick can observe) ──────────────────────────
SIG_HANG = "hang"                     # running, but no task event for > STUCK_SECONDS
SIG_THRASH = "thrash"                 # fail→pending looping, attempts climbing
SIG_EMPTY = "empty_deliverable"       # done, but the reducer produced nothing
SIG_DEADLOCK = "dag_deadlock"         # a dep this goal needs has failed → can never run
SIG_MERGE_CONFLICT = "merge_conflict"  # a worktree branch failed to merge to base
SIG_SERVER_FLAP = "server_flap"       # gate pinned / inference server reachability flapping
SIG_WORKTREE_LEAK = "worktree_leak"   # a worktree for a no-longer-active goal still on disk

# ── remediation actions (the manager applies these) ──────────────────────────
ACT_REQUEUE = "requeue"               # interrupt + re-queue the goal
ACT_BACKOFF_REQUEUE = "backoff_requeue"  # re-queue with jitter after a backoff
ACT_REPLAN = "replan"                 # re-plan once (different planner seed)
ACT_FAIL = "fail"                     # mark failed (terminal) + notify
ACT_ESCALATE = "escalate"             # hand to the human (TUI + Telegram)
ACT_PARK = "park"                     # preserve aside, never delete
ACT_HOLD = "hold"                     # hold dispatch, back off to heartbeat


# ── detectors (pure predicates over a record + clock) ────────────────────────

def _progress_base(rec: dict) -> Optional[float]:
    base = rec.get("progress_at")
    if base is None:
        base = rec.get("created_at")
    return None if base is None else float(base)


def is_stuck(rec: dict, now: float, stuck_seconds: float) -> bool:
    """A RUNNING goal is stuck (hung) iff it has made no progress for > ``stuck_seconds``.

    This is the §B.1 fix in action: because ``progress_at`` now advances on every fleet
    task event (taskstore.touch via runner.push), a goal that is *running and progressing*
    has a fresh timestamp and is NOT flagged, while a *running and hung* one goes stale.
    A non-positive ``stuck_seconds`` disables hang detection (never flag)."""
    if rec.get("state") != "running" or stuck_seconds <= 0:
        return False
    base = _progress_base(rec)
    if base is None:
        return False
    return (now - base) > stuck_seconds


def is_thrashing(rec: dict, *, threshold: int = 2) -> bool:
    """A goal that keeps failing and re-queuing (``attempts`` climbing) without finishing.

    Counts only goals still in the retry loop (pending) — a ``failed`` record has already
    exhausted its budget (terminal), and a ``done`` one succeeded."""
    return rec.get("state") == "pending" and int(rec.get("attempts") or 0) >= threshold


def produced_nothing(rec: dict) -> bool:
    """``done`` but the reducer yielded no deliverable (defensive — the runner normally
    fails an empty-deliverable goal rather than marking it done, but a stale/imported record
    could still show this)."""
    return rec.get("state") == "done" and not str(rec.get("result") or "").strip()


def deadlocked_dep(rec: dict, by_id: dict) -> Optional[str]:
    """If this PENDING goal depends on a goal that can never become ``done`` (a ``failed`` or
    a missing dep), return that dep id — the goal is deadlocked and must be failed with a
    reason naming the dep (§A.2 / §B.2). Returns None when every dep can still complete."""
    if rec.get("state") != "pending":
        return None
    for dep in rec.get("deps") or []:
        d = by_id.get(dep)
        if d is None or d.get("state") == "failed":
            return dep
    return None


def gate_starved(stats: dict) -> bool:
    """Workers are waiting on the DecodeGate but nothing is actually decoding — a sign the
    inference server stopped draining (flap / outage), distinct from a healthy busy gate
    (in_flight > 0). Defensive on a missing/None stats dict."""
    if not stats:
        return False
    return int(stats.get("waiting") or 0) > 0 and int(stats.get("in_flight") or 0) == 0


def v3_reflex_triage(snapshot: list[dict], *, now: float, stuck_seconds: float,
                     max_attempts: int = 3) -> dict:
    """Cheap Swarm v3 reflex pass over task records.

    This is deliberately pure: it only inspects the injected snapshot/clock and returns
    whether the expensive manager cortex pass is worth running. Deterministic remediations
    are returned as ``(task_id, action)`` pairs for the manager to apply with its existing
    task-store methods.
    """
    records = list(snapshot or [])
    if not records:
        return {"needs_cortex": False, "auto": [], "signals": {}}

    by_id = {rec.get("id"): rec for rec in records}
    auto: list[tuple[str, str]] = []
    signals: dict[str, str] = {}
    needs_cortex = False

    for rec in records:
        tid = rec.get("id")
        if tid is None:
            continue
        tid = str(tid)
        rec_signals: list[str] = []

        if is_stuck(rec, now, stuck_seconds):
            rec_signals.append(SIG_HANG)
            needs_cortex = True
        if is_thrashing(rec):
            rec_signals.append(SIG_THRASH)
            needs_cortex = True
        if produced_nothing(rec):
            rec_signals.append(SIG_EMPTY)
            needs_cortex = True

        dep = deadlocked_dep(rec, by_id)
        if dep is not None:
            rec_signals.append(SIG_DEADLOCK)
            action = decide(
                SIG_DEADLOCK,
                attempts=int(rec.get("attempts") or 0),
                max_attempts=max_attempts,
            )
            auto.append((tid, action))

        if rec_signals:
            signals[tid] = rec_signals[0]

    return {"needs_cortex": needs_cortex, "auto": auto, "signals": signals}


# ── remediation policy (pure decision; the manager executes the action) ──────

def decide(signal: str, *, attempts: int = 0, max_attempts: int = 3) -> str:
    """Map an observed signal (+ the goal's retry count) to a bounded action (§B.3).

    Bounded auto-fix within the retry budget, then escalate/fail when exhausted. A merge
    conflict is NEVER auto-resolved (always escalate); a DAG deadlock always fails the
    dependent; a leaked worktree is always parked (preserve, never delete)."""
    if signal == SIG_MERGE_CONFLICT:
        return ACT_ESCALATE                 # human / dedicated reducer decides — never auto-merge
    if signal == SIG_DEADLOCK:
        return ACT_FAIL                     # a dep failed → this goal can never run
    if signal == SIG_WORKTREE_LEAK:
        return ACT_PARK                     # preserve experimental output, never rm -rf
    if signal == SIG_SERVER_FLAP:
        return ACT_HOLD                     # hold dispatch, back off to the heartbeat
    exhausted = int(attempts) >= int(max_attempts)
    if signal == SIG_HANG:
        return ACT_ESCALATE if exhausted else ACT_REQUEUE
    if signal == SIG_THRASH:
        return ACT_FAIL if exhausted else ACT_BACKOFF_REQUEUE
    if signal == SIG_EMPTY:
        return ACT_ESCALATE if exhausted else ACT_REPLAN
    return ACT_ESCALATE                     # unknown signal → hand to the human
