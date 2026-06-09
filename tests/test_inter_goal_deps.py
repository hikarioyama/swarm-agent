"""Offline tests for the inter-goal dependency DAG (SWARM_V2 Phase 2)."""
from __future__ import annotations

import json
import threading

from swarm_agent import audit
from swarm_agent import goal as goal_mod
from swarm_agent.goal import analyze_deps
from swarm_agent.manager import CompletionManager
from swarm_agent.runner import SwarmRunner
from swarm_agent.taskstore import TaskStore


# ── 2.1 deps field round-trips + legacy migration ────────────────────────────

def test_deps_round_trip_and_legacy_migration(tmp_path) -> None:
    path = tmp_path / "tasks.json"
    store = TaskStore(str(path))
    a = store.add("implement X")
    b = store.add("test X", deps=[a["id"]])
    assert b["deps"] == [a["id"]]

    reloaded = TaskStore(str(path))                  # fresh load from disk
    recs = {r["id"]: r for r in reloaded.snapshot()}
    assert recs[b["id"]]["deps"] == [a["id"]]        # deps survive persistence

    # A legacy record with NO deps key migrates to [].
    path.write_text(json.dumps([
        {"id": "task-legacy", "goal": "old", "state": "pending", "attempts": 0}]))
    legacy = TaskStore(str(path))
    assert legacy.snapshot()[0]["deps"] == []


# ── 2.2 dependency-aware claim_next ──────────────────────────────────────────

def test_claim_next_respects_deps(tmp_path) -> None:
    store = TaskStore(str(tmp_path / "tasks.json"))
    a = store.add("A")
    b = store.add("B", deps=[a["id"]])

    c1 = store.claim_next()
    assert c1 is not None and c1["id"] == a["id"]    # A is ready, B is blocked
    assert store.claim_next() is None                # B stays pending (A running, not done)
    store.complete(a["id"], "result A")
    c2 = store.claim_next()
    assert c2 is not None and c2["id"] == b["id"]    # A done → B now dispatchable


def test_claim_next_independent_goals_both_claimable(tmp_path) -> None:
    store = TaskStore(str(tmp_path / "tasks.json"))
    x = store.add("X")
    y = store.add("Y")
    assert store.claim_next()["id"] == x["id"]
    assert store.claim_next()["id"] == y["id"]


def test_claim_next_never_returns_dependent_before_dep_done(tmp_path) -> None:
    store = TaskStore(str(tmp_path / "tasks.json"))
    a = store.add("A")
    b = store.add("B", deps=[a["id"]])
    claimed: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        rec = store.claim_next()
        if rec is not None:
            with lock:
                claimed.append(rec["id"])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)
    assert claimed == [a["id"]]                       # only A; B blocked behind it
    store.complete(a["id"], "ok")
    assert store.claim_next()["id"] == b["id"]


# ── 2.3 analyze_deps (no server) ─────────────────────────────────────────────

def test_analyze_deps_parses_and_validates() -> None:
    existing = [{"id": "task-aa", "goal": "implement endpoint X"},
                {"id": "task-bb", "goal": "unrelated docs"}]

    def fake(lane, prompt, task_id, **kw):
        return '{"deps":["task-aa"]}'

    assert analyze_deps("write tests for endpoint X", existing, run_agent=fake) == ["task-aa"]
    # garbage reply → independent
    assert analyze_deps("g", existing, run_agent=lambda *a, **k: "totally not json") == []
    # unknown id is filtered out
    assert analyze_deps("g", existing, run_agent=lambda *a, **k: '{"deps":["task-zz"]}') == []
    # a raising run_agent fails open
    def boom(*a, **k):
        raise RuntimeError("server down")
    assert analyze_deps("g", existing, run_agent=boom) == []


def test_analyze_deps_skips_llm_when_no_candidates() -> None:
    called = {"n": 0}

    def counting(*a, **k):
        called["n"] += 1
        return '{"deps":[]}'

    assert analyze_deps("g", [], run_agent=counting) == []
    assert called["n"] == 0                            # no existing goals → no model call


# ── 2.4 enqueue_task wires analyze_deps onto the new record ──────────────────

def test_enqueue_task_records_analyzed_deps(tmp_path, monkeypatch) -> None:
    runner = SwarmRunner(warm=False, admission="static")
    runner.tasks = TaskStore(str(tmp_path / "tasks.json"))
    a = runner.enqueue_task("implement endpoint X", analyze=False)   # first → no deps
    assert a["deps"] == []

    monkeypatch.setattr(goal_mod, "analyze_deps",
                        lambda new, existing, *, run_agent: [existing[0]["id"]] if existing else [])
    b = runner.enqueue_task("write tests for endpoint X")            # depends on A
    assert b["deps"] == [a["id"]]
    runner.shutdown()


# ── 2.5 DAG-deadlock detection + manager failing the dependent ───────────────

def test_deadlocked_dep_detects_failed_dependency() -> None:
    by_id = {"A": {"id": "A", "state": "failed"},
             "B": {"id": "B", "state": "pending", "deps": ["A"]}}
    assert audit.deadlocked_dep(by_id["B"], by_id) == "A"
    assert audit.deadlocked_dep(by_id["A"], by_id) is None       # the failed dep itself isn't
    ok = {"A": {"id": "A", "state": "done"},
          "B": {"id": "B", "state": "pending", "deps": ["A"]}}
    assert audit.deadlocked_dep(ok["B"], ok) is None             # dep done → not deadlocked


def test_manager_fails_deadlocked_goal(tmp_path) -> None:
    runner = SwarmRunner(warm=False, admission="static")
    runner.tasks = TaskStore(str(tmp_path / "tasks.json"))
    a = runner.tasks.add("A")
    b = runner.tasks.add("B", deps=[a["id"]])
    runner.tasks.mark_failed(a["id"], "boom")

    mgr = CompletionManager(runner)
    mgr._fail_deadlocked(runner.tasks.snapshot())

    snap = {r["id"]: r for r in runner.tasks.snapshot()}
    assert snap[b["id"]]["state"] == "failed"
    assert a["id"] in (snap[b["id"]]["error"] or "")            # reason names the failed dep
    runner.shutdown()
