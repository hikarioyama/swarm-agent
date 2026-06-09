"""End-to-end LIVE tests for Swarm v2 (the `Test (live)` items in SWARM_V2_TODO.md).

These drive a REAL SwarmRunner (real planner + worker agents) against the running Step-3.7
endpoint, inside a throwaway git repo in tmp — never the real project. Safety bounds: dangerous
shell is auto-DENIED (FLEET_AUTO_APPROVE=deny), a small decode gate, a short per-task timeout,
and tiny goals. Skipped automatically (via the ``live_endpoint`` fixture) when no endpoint is up.

Run with:  ~/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_live_swarm_v2.py -m live -q
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from fleet import config
from swarm_agent.goal import analyze_deps
from swarm_agent.manager import CompletionManager
from swarm_agent.runner import SwarmRunner
from swarm_agent.scheduler import GoalScheduler
from swarm_agent.taskstore import TaskStore
from swarm_agent.telegram import TelegramBridge

pytestmark = pytest.mark.live


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          check=True, capture_output=True, text=True)


def _init_repo(path) -> str:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True, text=True)
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "tester")
    (path / "README.md").write_text("base\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "base")
    return str(path)


def _live_runner(repo, tmp_path, monkeypatch, *, k=4) -> SwarmRunner:
    monkeypatch.chdir(repo)
    monkeypatch.setenv("FLEET_AUTO_APPROVE", "deny")        # safety: block rm -rf / sudo
    monkeypatch.setattr(config, "PARALLEL_WRITES", True)
    monkeypatch.setattr(config, "WORKTREE_ROOT", str(tmp_path / "wts"))
    monkeypatch.setattr(config, "GOAL_BRANCH_PREFIX", "swarm/")
    monkeypatch.setattr(config, "TASK_TIMEOUT_S", 120.0)
    runner = SwarmRunner(warm=False, admission="static", gate_start=8)
    runner._max_goals = k
    runner._goals = GoalScheduler(k)
    runner.tasks = TaskStore(str(tmp_path / "tasks.json"))
    runner.setup()
    return runner


def _dispatch_all(runner, timeout=240.0) -> None:
    """Claim + run every dispatchable queued goal (dependency-aware), then join."""
    threads = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = runner.tasks.claim_next()
        if rec is None:
            if not any(t.is_alive() for t in threads):
                break                                       # nothing ready and nothing running
            time.sleep(0.5)
            continue
        t = runner.submit_goal(rec)
        if t is not None:
            threads.append(t)
    for t in threads:
        t.join(timeout=max(1.0, deadline - time.monotonic()))


def _no_worktrees_left(tmp_path) -> bool:
    root = tmp_path / "wts"
    return not root.is_dir() or not list(root.glob("wt-*"))


# ── 1.6 live: parallel disjoint writers land + merge clean ───────────────────

def test_live_parallel_disjoint_writers(tmp_path, monkeypatch, live_endpoint) -> None:
    repo = _init_repo(tmp_path / "repo")
    runner = _live_runner(repo, tmp_path, monkeypatch, k=4)
    try:
        names = ["alpha.txt", "beta.txt", "gamma.txt"]
        for n in names:
            word = n.split(".")[0].upper()
            runner.enqueue_task(
                f"Run exactly one shell command to create a file: printf '{word}\\n' > {n} "
                f"in the current working directory. Then stop. Do not create any other file.",
                analyze=False)                              # independent → run in parallel
        _dispatch_all(runner)
        CompletionManager(runner)._drain_merges()           # sequential merge-back

        states = {r["goal"][:12]: r["state"] for r in runner.tasks.snapshot()}
        landed = [n for n in names if (Path(repo) / n).exists()]
        tracked = set(_git(repo, "ls-files").stdout.split())
        merged = [n for n in names if n in tracked]
        # core DoD: the disjoint files land in the base repo and the worktrees are reclaimed.
        assert len(landed) >= 2, f"expected ≥2 of {names} to land, got {landed}; states={states}"
        assert _no_worktrees_left(tmp_path), "worktrees should be gone after merge-back"
        # ≥2 of them arrived via a CLEAN worktree merge-back (committed/tracked). NB: an
        # occasional file can be written by a coder's write_file against the process cwd rather
        # than its worktree (a hermes file-tool cwd-resolution artifact — "only RELATIVE cwd is
        # isolated", per compat.sandbox_root); that surfaces as a stray untracked file, never as
        # merge corruption. We therefore assert merge CLEANLINESS, not a pristine working tree.
        assert len(merged) >= 2, f"expected ≥2 merged/tracked, got {merged}; states={states}"
        mh = subprocess.run(["git", "-C", repo, "rev-parse", "-q", "--verify", "MERGE_HEAD"],
                            capture_output=True, text=True)
        assert mh.returncode != 0, "no dangling half-merge after merge-back"
        for n in merged:
            assert "<<<<<<<" not in (Path(repo) / n).read_text()   # no conflict markers
    finally:
        runner.shutdown()


# ── 2.4 live: the analyzer marks the dependent goal (real model) ─────────────

def test_live_analyze_deps_marks_dependent(tmp_path, monkeypatch, live_endpoint) -> None:
    repo = _init_repo(tmp_path / "repo")
    runner = _live_runner(repo, tmp_path, monkeypatch, k=2)
    try:
        existing = [{"id": "task-impl", "goal": "Implement the HTTP endpoint GET /status in app.py"}]
        deps = analyze_deps(
            "Write unit tests for the GET /status endpoint implemented in app.py",
            existing, run_agent=runner._run_agent)
        assert deps == ["task-impl"], f"analyzer should mark the tests goal dependent; got {deps}"

        # and the dependency-aware claim holds it back until the dep is done
        a = runner.tasks.add("impl")
        b = runner.tasks.add("tests", deps=[a["id"]])
        assert runner.tasks.claim_next()["id"] == a["id"]
        assert runner.tasks.claim_next() is None            # b blocked behind a
        runner.tasks.complete(a["id"], "done")
        assert runner.tasks.claim_next()["id"] == b["id"]
    finally:
        runner.shutdown()


# ── 3.5 live: a same-file conflict escalates, nothing silently lost ──────────

def test_live_conflict_escalates_without_data_loss(tmp_path, monkeypatch, live_endpoint) -> None:
    repo = _init_repo(tmp_path / "repo")
    (Path(repo) / "notes.md").write_text("# notes\n\nfirst line\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add notes")
    runner = _live_runner(repo, tmp_path, monkeypatch, k=4)
    try:
        for who in ("One", "Two"):
            runner.enqueue_task(
                f"In the existing file notes.md, replace the line that reads 'first line' with "
                f"the line '{who} was here'. Change only that single line.", analyze=False)
        _dispatch_all(runner)
        events = []
        # capture events emitted during merge-back
        mgr = CompletionManager(runner)
        mgr._drain_merges()
        while not runner.events.empty():
            events.append(runner.events.get_nowait())

        # base repo must be intact: no merge in progress, no conflict markers, clean tree.
        assert _git(repo, "status", "--porcelain").stdout.strip() == ""
        mh = subprocess.run(["git", "-C", repo, "rev-parse", "-q", "--verify", "MERGE_HEAD"],
                            capture_output=True, text=True)
        assert mh.returncode != 0                            # no dangling half-merge
        assert "<<<<<<<" not in (Path(repo) / "notes.md").read_text()
        # if a conflict happened it was surfaced (parked + escalated), never silently dropped.
        conflict_events = [e for e in events if e.get("kind") == "error"
                           and "conflict" in (e.get("text") or "")]
        parked = list((tmp_path / "wts" / "parked").glob("wt-*")) if (tmp_path / "wts" / "parked").is_dir() else []
        if conflict_events:
            assert parked, "a reported conflict must leave a parked worktree (preserved work)"
        assert _no_worktrees_left(tmp_path) or parked        # active worktrees all resolved
    finally:
        runner.shutdown()


# ── 4.6 live: in-process mirror against the live system (no phone) ───────────

def test_live_telegram_mirror_roundtrip_in_process(tmp_path, monkeypatch, live_endpoint) -> None:
    # The literal phone round-trip needs a real bot token + handset (manual). This proves the
    # SAME-SESSION mechanics against the LIVE system with a fake transport: inbound enqueues a
    # real goal, and real session events mirror outbound through the renderer.
    class FakeTransport:
        def __init__(self):
            self.sent, self.edited, self._mid = [], [], 0

        def send_message(self, chat_id, text):
            self._mid += 1
            self.sent.append((chat_id, text))
            return self._mid

        def edit_message(self, chat_id, mid, text):
            self.edited.append((chat_id, mid, text))
            return True

        def get_updates(self, offset, timeout=0):
            return []

    repo = _init_repo(tmp_path / "repo")
    runner = _live_runner(repo, tmp_path, monkeypatch, k=2)
    bridge = TelegramBridge(runner, transport=FakeTransport(), allowed_chat_ids={42},
                            log_path=runner.log.path, poll_interval=0.05)
    try:
        assert bridge.configured and bridge.start()
        # inbound (allowed chat) enqueues a REAL goal — identical to TUI /task
        before = runner.tasks.counts()["total"]
        bridge.handle_inbound(42, "/task Summarize what the README.md file contains")
        assert runner.tasks.counts()["total"] == before + 1
        # foreign chat is ignored
        bridge.handle_inbound(7, "/task should be ignored")
        assert runner.tasks.counts()["total"] == before + 1

        # drive the queued goal to completion so real events flow to the logbook…
        _dispatch_all(runner, timeout=180.0)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not bridge._transport.sent:
            time.sleep(0.1)                                  # let the outbound tail catch up
        # …and the mirror produced outbound traffic (the live session was reflected).
        assert bridge._transport.sent, "outbound mirror produced no messages"
    finally:
        bridge.stop()
        runner.shutdown()
