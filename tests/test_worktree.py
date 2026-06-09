"""Offline tests for per-goal git worktree write-isolation (SWARM_V2 Phase 1).

All git here is real but fully local and deterministic: each test builds a throwaway repo in
``tmp_path``. No inference server, no LLM. The runner-integration tests stub ``ThreadFleet`` so
no agent ever runs — they only assert the worker cwd routing / permit / fallback decisions.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fleet import compat, config
from fleet.board import Task
from swarm_agent import worktree as wt_mod
from swarm_agent.goal import validate_tasks
from swarm_agent.runner import SwarmRunner
from swarm_agent.taskstore import TaskStore


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


# ── 1.1 create / commit / merge (clean) ──────────────────────────────────────

def test_worktree_create_commit_merge_clean(tmp_path) -> None:
    repo = _init_repo(tmp_path / "repo")
    root = tmp_path / "wts"
    wt = wt_mod.create("g1", repo=repo, worktree_root=str(root), branch_prefix="swarm/")
    assert Path(wt.path).is_dir()
    assert wt.branch == "swarm/g1"
    assert wt.base_branch is not None            # repo is on a named branch
    assert Path(wt.path).name == "wt-g1"

    (Path(wt.path) / "feature.txt").write_text("hello from g1\n")
    sha = wt_mod.commit(wt.path, "swarm goal g1: add feature")
    assert sha                                   # a real commit happened

    res = wt_mod.merge_back(wt)
    assert res.ok and not res.conflict
    landed = Path(repo) / "feature.txt"
    assert landed.exists() and landed.read_text() == "hello from g1\n"   # merged into base

    assert wt_mod.remove(wt)
    assert not Path(wt.path).exists()


def test_worktree_commit_returns_none_when_clean(tmp_path) -> None:
    repo = _init_repo(tmp_path / "repo")
    wt = wt_mod.create("empty", repo=repo, worktree_root=str(tmp_path / "wts"))
    assert wt_mod.commit(wt.path, "nothing to do") is None   # no changes → no commit


# ── 1.2 merge conflict is structured, base left non-corrupt ──────────────────

def test_worktree_merge_conflict_is_structured(tmp_path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (Path(repo) / "shared.txt").write_text("line1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add shared")
    root = tmp_path / "wts"

    wt1 = wt_mod.create("g1", repo=repo, worktree_root=str(root))
    (Path(wt1.path) / "shared.txt").write_text("g1 change\n")
    wt_mod.commit(wt1.path, "g1 edits shared")

    # wt2 forks from the SAME base (created before wt1 is merged back).
    wt2 = wt_mod.create("g2", repo=repo, worktree_root=str(root))
    (Path(wt2.path) / "shared.txt").write_text("g2 change\n")
    wt_mod.commit(wt2.path, "g2 edits shared")

    r1 = wt_mod.merge_back(wt1)
    assert r1.ok and not r1.conflict
    assert (Path(repo) / "shared.txt").read_text() == "g1 change\n"

    r2 = wt_mod.merge_back(wt2)
    assert not r2.ok and r2.conflict
    assert "shared.txt" in r2.conflicting_paths

    # base tree left NON-CORRUPT: no merge in progress, g1's content intact (g2 not applied).
    assert (Path(repo) / "shared.txt").read_text() == "g1 change\n"
    merge_head = subprocess.run(["git", "-C", repo, "rev-parse", "-q", "--verify", "MERGE_HEAD"],
                                capture_output=True, text=True)
    assert merge_head.returncode != 0           # MERGE_HEAD absent → no half-merge committed
    status = _git(repo, "status", "--porcelain").stdout
    assert "UU" not in status                    # no unmerged entries left behind


# ── 1.4 permit routing (pure decision) ───────────────────────────────────────

def test_permit_is_shared_truth_table() -> None:
    # read-only always shares
    assert wt_mod.permit_is_shared(True, "g1", parallel_writes=False, is_git=False)
    # writing + parallel-writes + git → shared (worktree-isolated)
    assert wt_mod.permit_is_shared(False, "g1", parallel_writes=True, is_git=True)
    # writing but flag off → exclusive
    assert not wt_mod.permit_is_shared(False, "g1", parallel_writes=False, is_git=True)
    # writing but not a git repo → exclusive (fallback)
    assert not wt_mod.permit_is_shared(False, "g1", parallel_writes=True, is_git=False)
    # writing but interactive (no goal_id) → exclusive
    assert not wt_mod.permit_is_shared(False, None, parallel_writes=True, is_git=True)


# ── 1.3 runner routes a writing goal into a worktree, read-only does not ──────

def _stub_fleet(monkeypatch, captured, gid):
    from swarm_agent import runner as runner_mod

    class FakeFleet:
        def __init__(self, board, gate, *, cfg=None, on_event=None):
            self._board = board

        def run(self):
            # what cwd a worker of THIS goal would resolve (namespaced "<gid>.<id>")
            captured["wt"] = compat.goal_worktree_for(f"{gid}.c")
            results = {
                "c": Task(id="c", prompt="edit", lane="coder"),
                "r": Task(id="r", prompt="reduce", deps=["c"], lane="reducer"),
            }
            results["r"].result = "done"
            return {"board_results": results, "counts": {}}

    monkeypatch.setattr(runner_mod, "ThreadFleet", FakeFleet)
    monkeypatch.setattr(runner_mod._skills_synth, "synthesize_async", lambda *a, **k: None)


def test_run_swarm_routes_writer_into_worktree(tmp_path, monkeypatch) -> None:
    repo = _init_repo(tmp_path / "repo")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(config, "PARALLEL_WRITES", True)
    monkeypatch.setattr(config, "WORKTREE_ROOT", str(tmp_path / "wts"))
    monkeypatch.setattr(config, "GOAL_BRANCH_PREFIX", "swarm/")

    runner = SwarmRunner(warm=False, admission="static")
    runner._setup_done = True
    runner.tasks = TaskStore(str(tmp_path / "tasks.json"))
    gid = "task-write01"

    writing = validate_tasks([
        Task(id="c", prompt="edit", lane="coder"),
        Task(id="r", prompt="reduce", deps=["c"], lane="reducer"),
    ])
    captured = {}
    _stub_fleet(monkeypatch, captured, gid)
    monkeypatch.setattr(runner, "_plan", lambda goal: writing)

    ok, _ = runner._run_swarm("edit the project", goal_id=gid)
    assert ok
    assert captured["wt"] is not None
    assert Path(captured["wt"]).name == "wt-task-write01"
    # FakeFleet wrote nothing in the worktree → nothing to commit → worktree reclaimed,
    # no pending merge queued.
    assert runner.pop_pending_merge() is None
    assert not Path(captured["wt"]).exists()
    assert compat.goal_worktree_for(f"{gid}.c") is None   # unregistered after the goal

    # A read-only goal must NOT get a worktree.
    readonly = validate_tasks([
        Task(id="w", prompt="explain", lane="writer"),
        Task(id="r", prompt="reduce", deps=["w"], lane="reducer"),
    ])
    captured.clear()
    monkeypatch.setattr(runner, "_plan", lambda goal: readonly)
    ok, _ = runner._run_swarm("explain something", goal_id="task-read01")
    assert ok
    assert captured["wt"] is None
    runner.shutdown()


# ── 1.5 non-git working dir → exclusive fallback with a loud status ──────────

def test_isolation_non_git_falls_back_to_exclusive(tmp_path, monkeypatch) -> None:
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    monkeypatch.chdir(plain)
    monkeypatch.setattr(config, "PARALLEL_WRITES", True)

    runner = SwarmRunner(warm=False, admission="static")
    mode, wt = runner._isolation_for(False, "task-x")
    assert mode == "exclusive" and wt is None
    drained = []
    while not runner.events.empty():
        drained.append(runner.events.get_nowait())
    assert any(ev.get("kind") == "status" and "not a git repo" in ev.get("text", "")
               for ev in drained)
    runner.shutdown()


def test_isolation_writing_without_parallel_writes_is_exclusive(tmp_path, monkeypatch) -> None:
    repo = _init_repo(tmp_path / "repo")
    monkeypatch.chdir(repo)
    monkeypatch.setattr(config, "PARALLEL_WRITES", False)   # flag off → today's behaviour
    runner = SwarmRunner(warm=False, admission="static")
    assert runner._isolation_for(False, "task-x") == ("exclusive", None)
    assert runner._isolation_for(True, "task-x") == ("readonly", None)
    runner.shutdown()


# ── 1.6 config validation ────────────────────────────────────────────────────

def test_config_validation_rejects_bad_swarm_v2_values(monkeypatch) -> None:
    monkeypatch.setattr(config, "STUCK_SECONDS", -5.0)
    with pytest.raises(ValueError, match="STUCK_SECONDS"):
        config.validate()
    monkeypatch.setattr(config, "STUCK_SECONDS", 600.0)
    monkeypatch.setattr(config, "GOAL_BRANCH_PREFIX", "")
    with pytest.raises(ValueError, match="GOAL_BRANCH_PREFIX"):
        config.validate()
    monkeypatch.setattr(config, "GOAL_BRANCH_PREFIX", "swarm/")
    config.validate()                            # restored values pass


# ── worktree-leak GC (supports §3.3): prune unchanged, park changed ──────────

def test_gc_prunes_unchanged_and_parks_changed(tmp_path) -> None:
    repo = _init_repo(tmp_path / "repo")
    root = str(tmp_path / "wts")
    park = str(tmp_path / "parked")

    clean = wt_mod.create("clean", repo=repo, worktree_root=root)         # no work → prune
    dirty = wt_mod.create("dirty", repo=repo, worktree_root=root)         # uncommitted → park
    (Path(dirty.path) / "scratch.txt").write_text("wip\n")
    committed = wt_mod.create("committed", repo=repo, worktree_root=root)  # unmerged commit → park
    (Path(committed.path) / "out.txt").write_text("result\n")
    wt_mod.commit(committed.path, "committed work")
    live = wt_mod.create("live", repo=repo, worktree_root=root)           # active goal → untouched

    res = wt_mod.gc_worktrees(root, {"live"}, repo=repo, park_dir=park)

    assert res["pruned"] == ["clean"]
    assert set(res["parked"]) == {"committed", "dirty"}
    assert not Path(clean.path).exists()                  # pruned (git worktree remove)
    assert Path(live.path).exists()                       # active → never touched
    # changed worktrees PRESERVED under park/, never rm -rf'd
    assert (Path(park) / "wt-dirty").exists()
    assert (Path(park) / "wt-committed" / "out.txt").read_text() == "result\n"
