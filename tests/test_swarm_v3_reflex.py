from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from fleet import config, v3
from swarm_agent import audit
from swarm_agent.manager import CompletionManager
from swarm_agent.taskstore import TaskStore


@pytest.fixture(autouse=True)
def _clean_v3_flags(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SWARM_V3"):
            monkeypatch.delenv(key, raising=False)
    v3.reset_flags_cache()
    yield
    for key in list(os.environ):
        if key.startswith("SWARM_V3"):
            monkeypatch.delenv(key, raising=False)
    v3.reset_flags_cache()


def _rec(tid: str, state: str, **kw) -> dict:
    rec = {
        "id": tid,
        "goal": tid,
        "state": state,
        "attempts": 0,
        "created_at": 0.0,
        "progress_at": 0.0,
        "deps": [],
        "result": None,
        "error": None,
    }
    rec.update(kw)
    return rec


def test_reflex_triage_skips_all_running_progressing() -> None:
    triage = audit.v3_reflex_triage(
        [
            _rec("a", "running", progress_at=99.0),
            _rec("b", "running", progress_at=98.0),
        ],
        now=100.0,
        stuck_seconds=10.0,
    )

    assert triage == {"needs_cortex": False, "auto": [], "signals": {}}


def test_reflex_triage_escalates_stuck_running_goal() -> None:
    triage = audit.v3_reflex_triage(
        [_rec("a", "running", progress_at=1.0)],
        now=100.0,
        stuck_seconds=10.0,
    )

    assert triage["needs_cortex"] is True
    assert triage["auto"] == []
    assert triage["signals"] == {"a": audit.SIG_HANG}


def test_reflex_triage_escalates_thrashing_pending_goal() -> None:
    triage = audit.v3_reflex_triage(
        [_rec("a", "pending", attempts=2)],
        now=100.0,
        stuck_seconds=10.0,
    )

    assert triage["needs_cortex"] is True
    assert triage["auto"] == []
    assert triage["signals"] == {"a": audit.SIG_THRASH}


def test_reflex_triage_deadlock_auto_fails_without_cortex() -> None:
    triage = audit.v3_reflex_triage(
        [
            _rec("dep", "failed"),
            _rec("child", "pending", deps=["dep"]),
        ],
        now=100.0,
        stuck_seconds=10.0,
    )

    assert triage["needs_cortex"] is False
    assert ("child", audit.ACT_FAIL) in triage["auto"]
    assert triage["signals"] == {"child": audit.SIG_DEADLOCK}


def test_reflex_flag_is_real_v3_subflag(monkeypatch) -> None:
    monkeypatch.setenv("SWARM_V3", "1")
    monkeypatch.setenv("SWARM_V3_REFLEX", "1")
    v3.reset_flags_cache()

    assert v3.enabled("reflex") is True
    assert v3.any_on() is True
    assert v3.enabled("hebbian") is False
    assert v3.enabled("sleep") is False


class FakeRunner:
    class _ServerDown(Exception):
        pass

    def __init__(self, path):
        self.tasks = TaskStore(str(path), max_attempts=3)
        self.gate = None
        self.busy = False
        self._setup_done = True
        self.log = SimpleNamespace(event=lambda *a, **k: None)
        self.events: list[tuple[str, dict]] = []
        self.cortex_calls = 0

    def pop_pending_merge(self):
        return None

    def active_goal_ids(self) -> set[str]:
        return {r["id"] for r in self.tasks.snapshot() if r.get("state") == "running"}

    def can_admit_goal(self) -> bool:
        return False

    def emit(self, kind: str, **kw) -> None:
        self.events.append((kind, kw))

    def _run_agent(self, *args, **kwargs) -> str:
        self.cortex_calls += 1
        return '{"note":"ok","requeue":[],"escalate":[]}'


def _running_task(runner: FakeRunner, now: float) -> str:
    rec = runner.tasks.add("healthy", now=now)
    runner.tasks.claim_next()
    tid = rec["id"]
    runner.tasks.touch(tid, now=now)
    return tid


def _make_thrashing_pending(runner: FakeRunner) -> str:
    rec = runner.tasks.add("thrash", now=0.0)
    tid = rec["id"]
    for _ in range(2):
        claimed = runner.tasks.claim_next()
        assert claimed and claimed["id"] == tid
        runner.tasks.fail(tid, "retry")
    return tid


def _tick_sequence(tmp_path, monkeypatch, *, reflex: bool) -> int:
    monkeypatch.setattr(config, "PARALLEL_WRITES", False)
    if reflex:
        monkeypatch.setenv("SWARM_V3", "1")
        monkeypatch.setenv("SWARM_V3_REFLEX", "1")
    else:
        monkeypatch.setenv("SWARM_V3", "0")
        monkeypatch.delenv("SWARM_V3_REFLEX", raising=False)
    v3.reset_flags_cache()

    runner = FakeRunner(tmp_path / ("tasks-reflex.json" if reflex else "tasks-base.json"))
    _running_task(runner, now=1_000_000_000_000.0)
    mgr = CompletionManager(runner, interval_s=0.0)

    mgr._tick()
    thrashing = _make_thrashing_pending(runner)
    mgr._tick()
    runner.tasks.mark_failed(thrashing, "resolved")
    mgr._tick()

    return runner.cortex_calls


def test_reflex_on_calls_cortex_only_for_problem_ticks(tmp_path, monkeypatch) -> None:
    baseline_calls = _tick_sequence(tmp_path, monkeypatch, reflex=False)
    reflex_calls = _tick_sequence(tmp_path, monkeypatch, reflex=True)

    assert baseline_calls == 3
    assert reflex_calls == 1
    assert reflex_calls < baseline_calls


def test_reflex_unset_matches_baseline_cortex_pattern(tmp_path, monkeypatch) -> None:
    baseline_calls = _tick_sequence(tmp_path, monkeypatch, reflex=False)

    monkeypatch.setenv("SWARM_V3", "1")
    monkeypatch.delenv("SWARM_V3_REFLEX", raising=False)
    v3.reset_flags_cache()
    runner = FakeRunner(tmp_path / "tasks-unset.json")
    _running_task(runner, now=1_000_000_000_000.0)
    mgr = CompletionManager(runner, interval_s=0.0)

    mgr._tick()
    thrashing = _make_thrashing_pending(runner)
    mgr._tick()
    runner.tasks.mark_failed(thrashing, "resolved")
    mgr._tick()

    assert runner.cortex_calls == baseline_calls
