"""Offline tests for parallel queued-goal scheduling and isolation."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from fleet.board import Task
from swarm_agent.goal import classify_plan, namespace_tasks, validate_tasks
from swarm_agent.runner import SwarmRunner
from swarm_agent.scheduler import GoalScheduler
from swarm_agent.taskstore import TaskStore


def _start_acquire(acquire) -> tuple[threading.Thread, threading.Event]:
    acquired = threading.Event()

    def run() -> None:
        acquire()
        acquired.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, acquired


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def test_scheduler_caps_concurrent_readers_and_wakes_waiter() -> None:
    scheduler = GoalScheduler(3)
    for _ in range(3):
        scheduler.acquire_readonly()
    assert scheduler.readers == 3

    thread, acquired = _start_acquire(scheduler.acquire_readonly)
    assert not acquired.wait(0.05)
    scheduler.release(writer=False)
    assert acquired.wait(2.0)
    assert scheduler.readers == 3

    for _ in range(3):
        scheduler.release(writer=False)
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_scheduler_writer_preference_blocks_new_readers() -> None:
    scheduler = GoalScheduler(3)
    scheduler.acquire_readonly()

    writer, writer_acquired = _start_acquire(scheduler.acquire_writer)
    _wait_until(lambda: scheduler.writer_pending)
    reader, reader_acquired = _start_acquire(scheduler.acquire_readonly)
    assert not writer_acquired.wait(0.05)
    assert not reader_acquired.wait(0.05)

    scheduler.release(writer=False)
    assert writer_acquired.wait(2.0)
    assert scheduler.writer_active
    assert not reader_acquired.wait(0.05)

    scheduler.release(writer=True)
    assert reader_acquired.wait(2.0)
    scheduler.release(writer=False)
    writer.join(timeout=2.0)
    reader.join(timeout=2.0)
    assert not writer.is_alive() and not reader.is_alive()


def test_scheduler_permit_acquires_and_releases() -> None:
    scheduler = GoalScheduler(2)
    with scheduler.permit(readonly=True):
        assert scheduler.readers == 1
        assert not scheduler.writer_active
    assert scheduler.readers == 0

    with scheduler.permit(readonly=False):
        assert scheduler.writer_active
        assert scheduler.readers == 0
    assert not scheduler.writer_active


def test_taskstore_claim_next_moves_oldest_pending_to_running(tmp_path) -> None:
    store = TaskStore(str(tmp_path / "tasks.json"))
    first = store.add("first")
    second = store.add("second")

    claimed_first = store.claim_next()
    claimed_second = store.claim_next()
    assert claimed_first is not None and claimed_first["id"] == first["id"]
    assert claimed_first["state"] == "running"
    assert claimed_second is not None and claimed_second["id"] == second["id"]
    assert claimed_second["state"] == "running"
    assert store.claim_next() is None


def test_taskstore_claim_next_never_returns_duplicate_ids(tmp_path) -> None:
    store = TaskStore(str(tmp_path / "tasks.json"))
    n = 12
    for i in range(n):
        store.add(f"goal {i}")
    claimed: list[str] = []
    lock = threading.Lock()

    def claim() -> None:
        rec = store.claim_next()
        if rec is not None:
            with lock:
                claimed.append(rec["id"])

    threads = [threading.Thread(target=claim) for _ in range(2 * n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
    assert all(not thread.is_alive() for thread in threads)
    assert len(claimed) == n
    assert len(set(claimed)) == n


def test_namespace_tasks_rewrites_ids_and_deps_without_mutating_input() -> None:
    tasks = validate_tasks([
        Task(id="inspect", prompt="Inspect", lane="analyst"),
        Task(id="reduce", prompt="Reduce", deps=["inspect"], lane="reducer"),
    ])
    namespaced = namespace_tasks(tasks, "task-123")
    namespaced_twice = namespace_tasks(namespaced, "task-123")

    assert [task.id for task in tasks] == ["inspect", "reduce"]
    assert tasks[1].deps == ["inspect"]
    assert [task.id for task in namespaced] == ["task-123.inspect", "task-123.reduce"]
    assert namespaced[1].deps == ["task-123.inspect"]
    assert [task.id for task in namespaced_twice] == [
        "task-123.inspect", "task-123.reduce"]
    assert namespaced_twice[1].deps == ["task-123.inspect"]


def test_classify_plan_is_capability_based_and_fail_closed() -> None:
    # Read-only is decided by REAL tool capability, not lane name. Only genuinely
    # non-mutating lanes (writer/researcher/reducer) classify read-only.
    readonly = [
        Task(id="write", prompt="Draft", lane="writer"),       # [] tools
        Task(id="research", prompt="Research", lane="researcher"),  # web + skills
        Task(id="reduce", prompt="Reduce", lane="reducer"),    # [] tools
    ]
    assert classify_plan(readonly) == "read-only"
    # coder/code/worker mutate -> writing
    for lane in ("coder", "worker", "code"):
        assert classify_plan([Task(id="e", prompt="Edit", lane=lane)]) == "writing"
    assert classify_plan([]) == "writing"                      # empty -> fail closed


def test_classify_plan_treats_write_capable_inspection_lanes_as_writing() -> None:
    # analyst/reviewer carry the "file" toolset, which exposes write_file + patch — so they
    # CAN mutate the tree and MUST run exclusively (Codex review P1), not as concurrent
    # readers. This is the capability-based classification's whole point.
    from swarm_agent.goal import lane_writes
    assert lane_writes("analyst") and lane_writes("reviewer")
    assert not lane_writes("writer") and not lane_writes("researcher") and not lane_writes("reducer")
    assert classify_plan([Task(id="a", prompt="inspect", lane="analyst"),
                          Task(id="r", prompt="reduce", lane="reducer", deps=["a"])]) == "writing"
    assert classify_plan([Task(id="a", prompt="audit", lane="reviewer"),
                          Task(id="r", prompt="reduce", lane="reducer", deps=["a"])]) == "writing"


def test_classify_plan_fails_closed_on_unknown_lane() -> None:
    # An UNRECOGNISED lane (planner typo / future lane) falls back to the write-capable
    # `worker` tool profile in config.toolsets_for(), so it MUST be treated as writing
    # (exclusive) — never admitted as a concurrent reader.
    from fleet.config import toolsets_for
    assert "terminal" in toolsets_for("developer")        # unknown -> write-capable fallback
    unknown = [
        Task(id="x", prompt="do", lane="developer"),      # not a known read-only lane
        Task(id="r", prompt="reduce", lane="reducer", deps=["x"]),
    ]
    assert classify_plan(unknown) == "writing"


def test_runner_k1_capacity_busy_property_and_active_ids() -> None:
    runner = SwarmRunner()
    assert runner._max_goals == 1
    assert runner.can_admit_goal()
    with runner._busy_lock:
        runner._active["goal-1"] = None
    assert not runner.can_admit_goal()
    runner.busy = True
    assert runner.submit("second message") is None
    assert runner.active_goal_ids() == {"goal-1"}
    runner.busy = False
    with runner._busy_lock:
        runner._active.pop("goal-1")
    runner.shutdown()


def test_submit_and_submit_goal_enforce_k_cap_atomically(tmp_path, monkeypatch) -> None:
    # The K cap is enforced ATOMICALLY in submit()/submit_goal() — not only via the manager's
    # pre-check — so an interactive turn or a goal claimed during a slow setup window cannot
    # push the active set past K (Codex review P2). At K=1, one taken slot refuses both.
    runner = SwarmRunner(warm=False, admission="static")
    runner._max_goals = 1
    runner.tasks = TaskStore(str(tmp_path / "tasks.json"))

    def _fake_turn(rec):                                   # don't touch the network
        with runner._busy_lock:
            runner._active.pop(rec["id"], None)
    monkeypatch.setattr(runner, "_run_goal_turn", _fake_turn)

    with runner._busy_lock:
        runner._active["goal-x"] = None                   # K=1 slot taken
    assert runner.submit("hi") is None                    # interactive refused at cap
    rec = runner.tasks.add("another")
    assert runner.submit_goal(rec) is None                # queued refused at cap (atomic)
    with runner._busy_lock:
        runner._active.pop("goal-x")                      # free the slot
    th = runner.submit_goal(rec)                           # now admitted
    assert th is not None
    th.join(timeout=5.0)
    runner.shutdown()


def test_runner_k1_interactive_turn_blocks_queued_dispatch(monkeypatch) -> None:
    # DoD §9: at K=1 an in-flight INTERACTIVE turn occupies the single slot, so the manager
    # must NOT dispatch a queued goal alongside it — byte-for-byte the old "dispatch only
    # when not busy" behaviour. (The interactive sentinel is counted toward the K cap.)
    monkeypatch.setenv("FLEET_MAX_CONCURRENT_GOALS", "1")
    runner = SwarmRunner()
    runner._max_goals = 1
    assert runner.can_admit_goal()
    runner.busy = True                       # a typed turn is in flight
    assert not runner.can_admit_goal()       # queued dispatch is held
    assert runner.active_goal_ids() == set() # but the typed turn is not a queued goal
    runner.busy = False
    assert runner.can_admit_goal()
    runner.shutdown()


def test_runner_k3_admits_interactive_plus_two_goals_then_caps() -> None:
    # At K=3 the interactive turn is simply one of the three concurrent slots: it plus two
    # queued goals fill the cap, and a third queued goal is held until one drains.
    runner = SwarmRunner()
    runner._max_goals = 3
    runner.busy = True                       # slot 1: interactive
    with runner._busy_lock:
        runner._active["goal-a"] = None      # slot 2
        runner._active["goal-b"] = None      # slot 3
    assert not runner.can_admit_goal()       # full
    with runner._busy_lock:
        runner._active.pop("goal-b")
    assert runner.can_admit_goal()           # a slot freed
    assert runner.active_goal_ids() == {"goal-a"}
    runner.busy = False
    with runner._busy_lock:
        runner._active.pop("goal-a")
    runner.shutdown()


def test_runner_dispatches_k_readonly_goals_concurrently(tmp_path) -> None:
    # End-to-end (offline, fleet stubbed): three read-only queued goals dispatched via
    # submit_goal run their fleet executions CONCURRENTLY through the real GoalScheduler
    # (a threading.Barrier(3) proves all three are in flight at once), and all reach 'done'
    # in the store. Isolated task path so the real persistent queue is never touched.
    runner = SwarmRunner(warm=False, admission="static")
    runner._setup_done = True
    runner._max_goals = 3
    runner._goals = GoalScheduler(3)
    runner.tasks = TaskStore(str(tmp_path / "tasks.json"))

    barrier = threading.Barrier(3, timeout=3.0)

    def fake_run_swarm(goal_text, *, goal_id=None):
        with runner._goals.permit(readonly=True):   # exercise the real reader permit
            barrier.wait()                          # all 3 must be concurrent or this raises
        return (True, f"deliverable for {goal_text}")

    runner._run_swarm = fake_run_swarm
    for i in range(3):
        runner.tasks.add(f"summarise topic {i}")

    threads = []
    for _ in range(3):
        assert runner.can_admit_goal()
        rec = runner.tasks.claim_next()
        assert rec is not None
        th = runner.submit_goal(rec)
        assert th is not None
        threads.append(th)

    for th in threads:
        th.join(timeout=5.0)
    assert all(not th.is_alive() for th in threads)
    assert runner.tasks.counts()["done"] == 3
    assert runner.active_goal_ids() == set()
    assert runner.can_admit_goal()
    runner.shutdown()


def test_runner_writing_goal_runs_exclusively(tmp_path) -> None:
    # A writing goal takes the EXCLUSIVE writer permit; while it holds it, can_admit_goal is
    # False (writers drain everything first) and no second fleet runs concurrently.
    runner = SwarmRunner(warm=False, admission="static")
    runner._setup_done = True
    runner._max_goals = 3
    runner._goals = GoalScheduler(3)
    runner.tasks = TaskStore(str(tmp_path / "tasks.json"))

    in_writer = threading.Event()
    release = threading.Event()
    peak = {"n": 0}
    cur = {"n": 0}
    lock = threading.Lock()

    def writing_run(goal_text, *, goal_id=None):
        with runner._goals.permit(readonly=False):
            with lock:
                cur["n"] += 1
                peak["n"] = max(peak["n"], cur["n"])
            in_writer.set()
            release.wait(timeout=3.0)
            with lock:
                cur["n"] -= 1
        return (True, "wrote files")

    runner._run_swarm = writing_run
    for i in range(2):
        runner.tasks.add(f"edit file {i}")
    rec = runner.tasks.claim_next()
    th = runner.submit_goal(rec)
    assert in_writer.wait(3.0)
    assert runner._goals.writer_active
    assert not runner.can_admit_goal()       # writer active -> dispatch held
    release.set()
    th.join(timeout=5.0)
    assert peak["n"] == 1
    assert runner.can_admit_goal()
    runner.shutdown()


def test_runner_writer_pending_holds_dispatch() -> None:
    # While a writing goal is waiting for readers to drain (or executing), the manager must
    # stop dispatching so writers drain everything first (§4.4) — even with a free slot.
    runner = SwarmRunner()
    runner._max_goals = 3
    assert runner.can_admit_goal()
    runner._goals.acquire_writer()           # writer now active (no readers, so it grabs it)
    assert runner._goals.writer_active
    assert not runner.can_admit_goal()
    runner._goals.release(writer=True)
    assert runner.can_admit_goal()
    runner.shutdown()


def test_multiswarmview_routes_by_goal_id_and_retires_on_idle() -> None:
    from swarm_agent.dashboard import MultiSwarmView
    mv = MultiSwarmView()
    mv.ingest({"kind": "user", "text": "goal A", "goal_id": "g1"})
    mv.ingest({"kind": "planning", "goal_id": "g1"})
    mv.ingest({"kind": "user", "text": "goal B", "goal_id": "g2"})
    mv.ingest({"kind": "planning", "goal_id": "g2"})
    keys = [k for k, _ in mv.active_views()]
    assert set(keys) == {"g1", "g2"}
    assert mv.goal_label("g1") == "goal A"
    assert mv.active_views()[0][0] == "g2"           # most-recent first
    mv.ingest({"kind": "idle", "goal_id": "g1"})
    assert [k for k, _ in mv.active_views()] == ["g2"]   # g1 retired


def test_engine_abandoned_writer_registry_prunes_finished_futures() -> None:
    from concurrent.futures import Future
    from fleet import engine

    fut = Future()
    with engine._ABANDONED_LOCK:
        engine._ABANDONED_WRITERS.clear()
        engine._ABANDONED_WRITERS.add(fut)
    try:
        assert engine.abandoned_writers_alive() is True
        fut.set_result(None)
        assert engine.abandoned_writers_alive() is False
        with engine._ABANDONED_LOCK:
            assert not engine._ABANDONED_WRITERS
    finally:
        with engine._ABANDONED_LOCK:
            engine._ABANDONED_WRITERS.clear()


def test_runner_abandoned_writer_gate_waits_then_proceeds(monkeypatch) -> None:
    from concurrent.futures import Future
    from fleet import config
    from fleet import engine

    runner = SwarmRunner(warm=False, admission="static")
    fut = Future()
    monkeypatch.setattr(config, "ABANDONED_WRITER_WAIT_S", 0.5)
    with engine._ABANDONED_LOCK:
        engine._ABANDONED_WRITERS.clear()
        engine._ABANDONED_WRITERS.add(fut)
    try:
        t0 = time.monotonic()
        runner._await_abandoned_writers("g1")
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0
        events = []
        while not runner.events.empty():
            events.append(runner.events.get_nowait())
        assert any(ev.get("kind") == "status" for ev in events)
        assert any(ev.get("kind") == "error" for ev in events)

        with engine._ABANDONED_LOCK:
            engine._ABANDONED_WRITERS.clear()
        t0 = time.monotonic()
        runner._await_abandoned_writers("g2")
        assert time.monotonic() - t0 < 0.2
    finally:
        with engine._ABANDONED_LOCK:
            engine._ABANDONED_WRITERS.clear()
        runner.shutdown()


def test_lane_writes_is_shared_between_config_and_goal() -> None:
    from fleet.config import lane_writes as config_lane_writes
    from swarm_agent.goal import lane_writes as goal_lane_writes

    assert config_lane_writes("analyst") is True
    assert config_lane_writes("writer") is False
    assert goal_lane_writes("analyst") == config_lane_writes("analyst")
    assert goal_lane_writes("writer") == config_lane_writes("writer")


if __name__ == "__main__":
    test_scheduler_caps_concurrent_readers_and_wakes_waiter()
    test_scheduler_writer_preference_blocks_new_readers()
    test_scheduler_permit_acquires_and_releases()
    with TemporaryDirectory() as tmp:
        test_taskstore_claim_next_moves_oldest_pending_to_running(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_taskstore_claim_next_never_returns_duplicate_ids(Path(tmp))
    test_namespace_tasks_rewrites_ids_and_deps_without_mutating_input()
    test_classify_plan_is_capability_based_and_fail_closed()
    test_classify_plan_treats_write_capable_inspection_lanes_as_writing()
    test_classify_plan_fails_closed_on_unknown_lane()
    test_runner_k1_capacity_busy_property_and_active_ids()
    test_runner_k3_admits_interactive_plus_two_goals_then_caps()
    with TemporaryDirectory() as tmp:
        test_runner_dispatches_k_readonly_goals_concurrently(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_runner_writing_goal_runs_exclusively(Path(tmp))
    test_runner_writer_pending_holds_dispatch()
    test_multiswarmview_routes_by_goal_id_and_retires_on_idle()
    test_engine_abandoned_writer_registry_prunes_finished_futures()
    class _MonkeyPatch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)
    test_runner_abandoned_writer_gate_waits_then_proceeds(_MonkeyPatch())
    test_lane_writes_is_shared_between_config_and_goal()
    print("parallel goals offline smoke passed")
