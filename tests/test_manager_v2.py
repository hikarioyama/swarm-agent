"""Offline tests for the Manager v2 auditor (SWARM_V2 Phase 3).

Pure-policy (decide / detectors) and manager-integration tests. No inference server: the LLM
``_evaluate`` path is never reached (we call the v2 methods directly), and the merge / GC tests
drive real-but-local git. TaskStore is always pointed at a tmp file (memory §0 isolation rule).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from swarm_agent import audit
from swarm_agent import worktree as wt_mod
from swarm_agent.manager import CompletionManager
from swarm_agent.runner import SwarmRunner
from swarm_agent.taskstore import TaskStore
from fleet import config


def _drain(runner) -> list[dict]:
    out = []
    while not runner.events.empty():
        out.append(runner.events.get_nowait())
    return out


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          check=True, capture_output=True, text=True)


def _init_repo(path) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True, text=True)
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "tester")
    (path / "base.txt").write_text("base\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "base")
    return str(path)


def _runner(tmp_path, monkeypatch) -> SwarmRunner:
    monkeypatch.setenv("SWARM_TASKS_PATH", str(tmp_path / "tasks.json"))
    runner = SwarmRunner(warm=False, admission="static")
    runner.tasks = TaskStore(str(tmp_path / "tasks.json"))
    return runner


# ── 3.1 decide() policy table ────────────────────────────────────────────────

def test_decide_policy_table() -> None:
    # merge conflict is NEVER auto-resolved, regardless of attempts
    assert audit.decide(audit.SIG_MERGE_CONFLICT, attempts=0, max_attempts=3) == audit.ACT_ESCALATE
    assert audit.decide(audit.SIG_MERGE_CONFLICT, attempts=9, max_attempts=3) == audit.ACT_ESCALATE
    # deadlock always fails the dependent; leak always parks; flap holds
    assert audit.decide(audit.SIG_DEADLOCK, attempts=0, max_attempts=3) == audit.ACT_FAIL
    assert audit.decide(audit.SIG_WORKTREE_LEAK, attempts=0, max_attempts=3) == audit.ACT_PARK
    assert audit.decide(audit.SIG_SERVER_FLAP, attempts=0, max_attempts=3) == audit.ACT_HOLD
    # bounded signals: act within budget, escalate/fail once spent
    assert audit.decide(audit.SIG_HANG, attempts=0, max_attempts=3) == audit.ACT_REQUEUE
    assert audit.decide(audit.SIG_HANG, attempts=3, max_attempts=3) == audit.ACT_ESCALATE
    assert audit.decide(audit.SIG_THRASH, attempts=1, max_attempts=3) == audit.ACT_BACKOFF_REQUEUE
    assert audit.decide(audit.SIG_THRASH, attempts=3, max_attempts=3) == audit.ACT_FAIL
    assert audit.decide(audit.SIG_EMPTY, attempts=0, max_attempts=3) == audit.ACT_REPLAN
    assert audit.decide(audit.SIG_EMPTY, attempts=5, max_attempts=3) == audit.ACT_ESCALATE


# ── 3.2 per-detector predicates ──────────────────────────────────────────────

def test_detectors_fire_only_when_they_should() -> None:
    # thrash: pending + attempts climbing past threshold
    assert audit.is_thrashing({"state": "pending", "attempts": 2})
    assert not audit.is_thrashing({"state": "pending", "attempts": 1})
    assert not audit.is_thrashing({"state": "running", "attempts": 5})   # not pending
    # empty deliverable: done with no result
    assert audit.produced_nothing({"state": "done", "result": ""})
    assert not audit.produced_nothing({"state": "done", "result": "x"})
    assert not audit.produced_nothing({"state": "running", "result": ""})
    # gate starvation: waiters but nothing decoding
    assert audit.gate_starved({"waiting": 4, "in_flight": 0})
    assert not audit.gate_starved({"waiting": 4, "in_flight": 2})
    assert not audit.gate_starved({"waiting": 0, "in_flight": 0})
    assert not audit.gate_starved(None)


# ── 3.4 bounded auto-remediation (hang) ──────────────────────────────────────

def _running_with_attempts(store: TaskStore, goal: str, attempts: int) -> str:
    rec = store.add(goal)
    rid = rec["id"]
    store.claim_next()                       # → running (attempts 0)
    for _ in range(attempts):                # fail/claim to climb attempts, end running
        store.fail(rid, "retry")
        store.claim_next()
    return rid


def test_audit_hang_requeues_below_budget_then_escalates_once(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path, monkeypatch)
    interrupts = {"n": 0}
    monkeypatch.setattr(runner, "interrupt", lambda *a, **k: interrupts.__setitem__("n", interrupts["n"] + 1))
    mgr = CompletionManager(runner)

    # Below budget: a hung goal (attempts 0, max 3) is interrupted + requeued, NOT escalated.
    rid = _running_with_attempts(runner.tasks, "do work", attempts=0)
    runner.tasks._records[0]["progress_at"] = 0.0           # make it stale
    with runner._busy_lock:
        runner._active[rid] = None                          # alive / in-flight
    _drain(runner)

    escalated = mgr._audit(now=10_000.0, snapshot=runner.tasks.snapshot())
    assert escalated == []
    assert interrupts["n"] == 1
    rec = {r["id"]: r for r in runner.tasks.snapshot()}[rid]
    assert rec["state"] == "pending" and rec["attempts"] == 1
    evs = _drain(runner)
    assert any(e.get("kind") == "manager" and "requeued hung" in e.get("text", "") for e in evs)
    assert not any(e.get("kind") == "error" for e in evs)

    # At budget: a goal that has already exhausted attempts hangs → escalates exactly once.
    runner.tasks = TaskStore(str(tmp_path / "tasks2.json"))
    rid2 = _running_with_attempts(runner.tasks, "wedged", attempts=2)   # one retry left
    runner.tasks._records[0]["progress_at"] = 0.0
    with runner._busy_lock:
        runner._active.clear()
        runner._active[rid2] = None
    interrupts["n"] = 0
    _drain(runner)

    escalated = mgr._audit(now=10_000.0, snapshot=runner.tasks.snapshot())
    assert escalated == [rid2]                              # exhausted → escalated
    assert interrupts["n"] == 1
    assert {r["id"]: r for r in runner.tasks.snapshot()}[rid2]["state"] == "failed"
    evs = _drain(runner)
    assert sum(1 for e in evs if e.get("kind") == "error") == 1   # exactly one escalate

    # Re-running the audit does NOT re-escalate the same terminal failure.
    again = mgr._audit(now=10_001.0, snapshot=runner.tasks.snapshot())
    assert again == []
    assert not any(e.get("kind") == "error" for e in _drain(runner))
    runner.shutdown()


# ── _tick integration: deadlock-fail + escalation flow through the real tick ──

def test_tick_fails_deadlocked_and_escalates_end_to_end(tmp_path, monkeypatch) -> None:
    # Proves the policy flows through the REAL _tick() (not just the _audit/_fail_deadlocked
    # helpers): a goal behind a failed dependency is terminally failed and escalated once,
    # with the LLM evaluator and dispatch disabled (no server).
    runner = _runner(tmp_path, monkeypatch)
    a = runner.tasks.add("impl")
    b = runner.tasks.add("tests", deps=[a["id"]])
    runner.tasks.mark_failed(a["id"], "boom")
    mgr = CompletionManager(runner)
    monkeypatch.setattr(mgr, "_evaluate", lambda now: None)     # no LLM evaluator
    monkeypatch.setattr(mgr, "_server_ok", lambda: False)       # no dispatch
    _drain(runner)

    mgr._tick()

    snap = {r["id"]: r for r in runner.tasks.snapshot()}
    assert snap[b["id"]]["state"] == "failed"                   # deadlocked dependent failed
    evs = _drain(runner)
    assert any(e.get("kind") == "error" and b["id"] in (e.get("text") or "") for e in evs)
    runner.shutdown()


# ── 3.5 sequential merge-back orchestration ──────────────────────────────────

def test_drain_merges_lands_clean_and_parks_conflict(tmp_path, monkeypatch) -> None:
    repo = _init_repo(tmp_path / "repo")
    (Path(repo) / "shared.txt").write_text("line1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add shared")
    root = str(tmp_path / "wts")
    monkeypatch.setattr(config, "WORKTREE_ROOT", root)

    wt1 = wt_mod.create("g1", repo=repo, worktree_root=root)
    (Path(wt1.path) / "shared.txt").write_text("g1 change\n")
    wt_mod.commit(wt1.path, "g1")
    wt2 = wt_mod.create("g2", repo=repo, worktree_root=root)     # forked from same base
    (Path(wt2.path) / "shared.txt").write_text("g2 change\n")
    wt_mod.commit(wt2.path, "g2")

    runner = _runner(tmp_path, monkeypatch)
    runner.enqueue_merge(wt1)
    runner.enqueue_merge(wt2)
    _drain(runner)
    mgr = CompletionManager(runner)
    mgr._drain_merges()

    # first merged clean → worktree removed; base has g1's content
    assert not Path(wt1.path).exists()
    assert (Path(repo) / "shared.txt").read_text() == "g1 change\n"
    # second conflicted → parked (preserved), escalated; base NOT corrupted by g2
    parked = Path(root) / "parked" / "wt-g2"
    assert parked.exists() and (parked / "shared.txt").read_text() == "g2 change\n"
    evs = _drain(runner)
    assert any(e.get("kind") == "error" and "conflict" in e.get("text", "") for e in evs)
    assert runner.pop_pending_merge() is None               # queue fully drained
    runner.shutdown()


# ── 3.3 worktree-leak GC through the manager ─────────────────────────────────

def test_manager_gc_parks_changed_and_skips_active(tmp_path, monkeypatch) -> None:
    repo = _init_repo(tmp_path / "repo")
    monkeypatch.chdir(repo)
    root = str(tmp_path / "wts")
    monkeypatch.setattr(config, "PARALLEL_WRITES", True)
    monkeypatch.setattr(config, "WORKTREE_ROOT", root)
    monkeypatch.setattr(config, "GOAL_BRANCH_PREFIX", "swarm/")

    clean = wt_mod.create("clean", repo=repo, worktree_root=root)
    dirty = wt_mod.create("dirty", repo=repo, worktree_root=root)
    (Path(dirty.path) / "wip.txt").write_text("scratch\n")
    live = wt_mod.create("live", repo=repo, worktree_root=root)

    runner = _runner(tmp_path, monkeypatch)
    with runner._busy_lock:
        runner._active["live"] = None                       # active goal → must be skipped
    mgr = CompletionManager(runner)
    _drain(runner)
    mgr._gc_worktrees()

    assert not Path(clean.path).exists()                    # unchanged → pruned
    assert (Path(root) / "parked" / "wt-dirty").exists()    # changed → parked, never deleted
    assert Path(live.path).exists()                         # active → untouched
    assert any(e.get("kind") == "manager" and "GC" in e.get("text", "") for e in _drain(runner))
    runner.shutdown()
