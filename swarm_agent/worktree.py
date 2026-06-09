"""Per-goal git worktree isolation for parallel writing goals (SWARM_V2 §A.1, §B.4).

A writing goal runs inside its OWN ``git worktree`` (separate working tree, shared object
store) on a branch ``<prefix><goal_id>``. N disjoint writers therefore proceed truly in
parallel instead of being serialised by the old exclusive-writer lock, and — unlike the
empty-tempdir sandbox — a worktree is a REAL checkout, so relative paths and repo tooling
work (this fixes the known "isolate=1 → relative paths can't see the repo → hallucination"
failure rather than fighting it).

Everything here shells out to ``git`` (thin wrappers); nothing in this module imports
HermesAgent. The merge half is deliberately conflict-SURFACING, never conflict-resolving:
a clean merge lands the deliverable, a conflict is reported with its paths and the base tree
is left non-corrupt (the caller parks the worktree and escalates — §B.3).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_GIT_TIMEOUT = 60.0
# Identity stamped on swarm commits so a worktree commit never fails on a repo/global with no
# configured user.name/email (we pass it per-commit; it does not mutate any git config).
_COMMIT_ENV = ("-c", "user.name=swarm-agent", "-c", "user.email=swarm-agent@localhost")


def _git(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    """Run a git command, capturing output. Never raises on non-zero unless ``check``."""
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          timeout=_GIT_TIMEOUT, check=check)


def is_git_repo(path: str) -> bool:
    """True iff ``path`` is inside a git working tree."""
    try:
        cp = _git(["-C", str(path), "rev-parse", "--is-inside-work-tree"])
        return cp.returncode == 0 and cp.stdout.strip() == "true"
    except Exception:
        return False


def repo_root(path: str) -> Optional[str]:
    """The top-level dir of the git repo containing ``path``, or None if not a repo."""
    try:
        cp = _git(["-C", str(path), "rev-parse", "--show-toplevel"])
        return cp.stdout.strip() if cp.returncode == 0 and cp.stdout.strip() else None
    except Exception:
        return None


def current_head(repo: str) -> Optional[str]:
    cp = _git(["-C", str(repo), "rev-parse", "HEAD"])
    return cp.stdout.strip() if cp.returncode == 0 else None


def current_branch(repo: str) -> Optional[str]:
    """The checked-out branch name, or None if detached HEAD (no branch to merge back into)."""
    cp = _git(["-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"])
    name = cp.stdout.strip() if cp.returncode == 0 else ""
    return None if (not name or name == "HEAD") else name


@dataclass
class Worktree:
    goal_id: str
    path: str
    branch: str
    repo: str
    base_branch: Optional[str]
    base_sha: Optional[str]


@dataclass
class MergeResult:
    ok: bool
    conflict: bool = False
    conflicting_paths: list[str] = field(default_factory=list)
    merged_sha: Optional[str] = None
    message: str = ""


def create(goal_id: str, *, repo: str, worktree_root: str,
           branch_prefix: str = "swarm/", base: Optional[str] = None) -> Worktree:
    """Add a worktree on a fresh branch ``<branch_prefix><goal_id>`` forked from ``base``
    (default: the repo's current HEAD). Returns a :class:`Worktree`. Raises on git failure."""
    base_sha = base or current_head(repo)
    if not base_sha:
        raise RuntimeError(f"worktree.create: cannot resolve base HEAD of {repo!r}")
    base_branch = current_branch(repo)
    branch = f"{branch_prefix}{goal_id}"
    path = str(Path(worktree_root).expanduser() / f"wt-{goal_id}")
    Path(worktree_root).expanduser().mkdir(parents=True, exist_ok=True)
    # Stale leftover dir from a prior aborted run for this exact id → clear admin + dir first.
    if Path(path).exists():
        _git(["-C", repo, "worktree", "remove", "--force", path])
        shutil.rmtree(path, ignore_errors=True)
    _git(["-C", repo, "worktree", "prune"])
    cp = _git(["-C", repo, "worktree", "add", "-b", branch, path, base_sha])
    if cp.returncode != 0:
        raise RuntimeError(f"worktree.create failed for {goal_id}: {cp.stderr.strip()}")
    return Worktree(goal_id=goal_id, path=path, branch=branch, repo=repo,
                    base_branch=base_branch, base_sha=base_sha)


def commit(path: str, message: str) -> Optional[str]:
    """Stage everything in the worktree and commit. Returns the new commit sha, or None when
    there was nothing to commit (clean tree). Never raises on the empty-commit case."""
    _git(["-C", str(path), "add", "-A"])
    cp = _git(["-C", str(path), *_COMMIT_ENV, "commit", "-m", message])
    if cp.returncode != 0:
        # Most common non-zero is "nothing to commit" — treat as "no work produced".
        return None
    head = _git(["-C", str(path), "rev-parse", "HEAD"])
    return head.stdout.strip() if head.returncode == 0 else None


def merge_back(wt: Worktree, *, message: Optional[str] = None) -> MergeResult:
    """Merge the goal branch into its base branch in the MAIN repo, sequentially-safe.

    Clean → MergeResult(ok=True, merged_sha). Conflict → the merge is ABORTED (base tree left
    non-corrupt, no half-merge committed) and MergeResult(ok=False, conflict=True,
    conflicting_paths=[…]) is returned — the caller parks the worktree and escalates (§B.3:
    a merge conflict is never auto-resolved). Non-conflict failures return ok=False too."""
    repo = wt.repo
    base = wt.base_branch
    if not base:
        return MergeResult(ok=False, conflict=False,
                           message="detached HEAD / no base branch to merge into")
    # Land on the base branch (no-op if already there). A dirty base tree blocks checkout —
    # report rather than force.
    if current_branch(repo) != base:
        co = _git(["-C", repo, "checkout", base])
        if co.returncode != 0:
            return MergeResult(ok=False, conflict=False,
                               message=f"could not checkout base {base!r}: {co.stderr.strip()}")
    msg = message or f"swarm: merge {wt.branch}"
    mg = _git(["-C", repo, "merge", "--no-ff", "-m", msg, wt.branch])
    if mg.returncode == 0:
        return MergeResult(ok=True, merged_sha=current_head(repo), message="merged")
    # Conflict (or other failure) — collect unmerged paths, then ABORT to keep base clean.
    unmerged = _git(["-C", repo, "diff", "--name-only", "--diff-filter=U"])
    paths = [p for p in unmerged.stdout.splitlines() if p.strip()]
    _git(["-C", repo, "merge", "--abort"])
    if paths:
        return MergeResult(ok=False, conflict=True, conflicting_paths=paths,
                           message=f"merge conflict in {len(paths)} path(s)")
    return MergeResult(ok=False, conflict=False,
                       message=f"merge failed: {mg.stderr.strip() or 'unknown'}")


def has_changes(path: str, *, repo: str, branch: str) -> bool:
    """True iff the worktree holds work worth preserving: uncommitted changes, OR commits on
    its branch not already reachable from the repo's current HEAD (i.e. not fully merged).
    A worktree that is clean AND fully merged carries nothing unique → safe to prune."""
    st = _git(["-C", str(path), "status", "--porcelain"])
    if st.returncode != 0 or st.stdout.strip():
        return True  # dirty (or unreadable → be conservative, treat as changed)
    anc = _git(["-C", repo, "merge-base", "--is-ancestor", branch, "HEAD"])
    # rc 0 → branch is an ancestor of HEAD (fully merged) → no unique work.
    return anc.returncode != 0


def remove(wt: Worktree) -> bool:
    """Remove the worktree and delete its (now-merged) branch. Best-effort; True on success."""
    cp = _git(["-C", wt.repo, "worktree", "remove", "--force", wt.path])
    _git(["-C", wt.repo, "branch", "-D", wt.branch])
    _git(["-C", wt.repo, "worktree", "prune"])
    return cp.returncode == 0


def park(wt: Worktree, *, park_dir: str) -> str:
    """Move the worktree aside into ``park_dir`` (PRESERVE, never delete — house rule
    ``feedback_preserve_experiment_data``). Returns the parked path (or the original path if
    the move could not be performed)."""
    Path(park_dir).expanduser().mkdir(parents=True, exist_ok=True)
    dest = str(Path(park_dir).expanduser() / f"wt-{wt.goal_id}")
    if Path(dest).exists():
        dest = str(Path(park_dir).expanduser() / f"wt-{wt.goal_id}-{os.getpid()}")
    mv = _git(["-C", wt.repo, "worktree", "move", wt.path, dest])
    if mv.returncode == 0:
        return dest
    # git refused (e.g. locked); leave it in place rather than risk losing it.
    return wt.path


def gc_worktrees(worktree_root: str, active_goal_ids, *, repo: str, park_dir: str,
                 branch_prefix: str = "swarm/") -> dict:
    """Scan ``worktree_root`` for ``wt-<goal_id>`` dirs belonging to no ACTIVE goal and reclaim
    them (§B.2 worktree-leak): UNCHANGED ones are pruned; CHANGED ones are PARKED (moved aside,
    never rm -rf'd). Returns ``{"pruned": [...], "parked": [...]}`` of goal ids handled."""
    root = Path(worktree_root).expanduser()
    out = {"pruned": [], "parked": []}
    if not root.is_dir():
        return out
    active = set(active_goal_ids or ())
    for child in sorted(root.glob("wt-*")):
        if not child.is_dir():
            continue
        goal_id = child.name[len("wt-"):]
        if not goal_id or goal_id in active:
            continue
        branch = f"{branch_prefix}{goal_id}"
        wt = Worktree(goal_id=goal_id, path=str(child), branch=branch, repo=repo,
                      base_branch=None, base_sha=None)
        try:
            changed = has_changes(str(child), repo=repo, branch=branch)
        except Exception:
            changed = True  # unsure → preserve
        if changed:
            park(wt, park_dir=park_dir)
            out["parked"].append(goal_id)
        else:
            remove(wt)
            out["pruned"].append(goal_id)
    return out


def permit_is_shared(readonly: bool, goal_id, *, parallel_writes: bool, is_git: bool) -> bool:
    """Whether a goal takes the SHARED (reader-style, ≤K concurrent) scheduler permit instead
    of the EXCLUSIVE writer permit. Read-only goals always share; a writing goal shares ONLY
    when it is worktree-isolated (FLEET_PARALLEL_WRITES on, a queued goal, inside a git repo) —
    otherwise it stays exclusive (today's behaviour / non-git fallback). §A.3, §1.4/1.5."""
    if readonly:
        return True
    return bool(parallel_writes and goal_id is not None and is_git)
