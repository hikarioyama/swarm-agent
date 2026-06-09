# Swarm v2 — implementation TODO + test plan

**Companion to:** `SWARM_V2_CONCEPT.md` (the "why/what"; §-anchors below point into it) and
`PARALLEL_GOALS_PLAN.md` (the implemented Option A base). This doc is the **actionable
checklist**: each leaf item is a unit of work with a concrete **Test:** beside it. Check items
off as they land.

Legend: `- [ ]` todo · `- [x]` done · **Test (offline)** = no inference server needed ·
**Test (live)** = needs the Step-3.7 endpoint up.

> **STATUS (implemented 2026-06-09):** All phases (P0 liveness, P1 worktree write-isolation,
> P2 inter-goal DAG, P3 manager v2 auditor, P4 Telegram mirror) and the §6 regression guard are
> implemented with their **offline** tests green — full suite **148 passed** via the hermes
> venv pytest. New modules: `swarm_agent/worktree.py`, `swarm_agent/audit.py`,
> `swarm_agent/telegram/`. New tests: `tests/test_worktree.py`, `tests/test_inter_goal_deps.py`,
> `tests/test_manager_v2.py`, `tests/test_telegram_bridge.py` (+ additions to
> `tests/test_parallel_goals.py`). The **`Test (live)`** items below still need a running
> Step-3.7 endpoint (real 3-disjoint-writers / impl→tests ordering / phone round-trip) and are
> the only remaining verification.

---

## 0. How to run the tests (read first — repo-specific gotchas)

- **Interpreter:** use the **hermes venv** python, not system python
  (`~/.hermes/hermes-agent/venv/bin/python -m pytest`), matching the existing
  `tests/test_parallel_goals.py` / `tests/test_tui.py` setup. New deps go in via
  `uv pip` into that venv.
- **TaskStore isolation (critical):** `SwarmRunner`/`TaskStore` read the **real**
  `~/.cache/swarm-agent/tasks.json` by default. Every test MUST point them at a tmp file via
  `TaskStore(path=tmp)` or `SWARM_TASKS_PATH=<tmp>` — otherwise tests mutate the live queue
  (memory: `project_swarm_agent_parallel_goals`). Use a `pytest` fixture that sets a tmp path.
- **No-server seam for LLM calls:** offline tests must not hit the model. The new LLM touch
  points (`analyze_deps`, manager remediation) take an **injectable `run_agent` callable**
  (default = `runner._run_agent`) so tests pass a fake returning canned JSON. Same trick the
  router/planner tests use.
- **Git fixtures for worktree work:** worktree create/merge/conflict are pure git — test them
  against a throwaway repo built in `tmp_path` (`git init`, a couple of commits). Fully
  offline and deterministic; no LLM, no server.
- **Clock injection for liveness/manager:** stuck-detection compares timestamps; inject a fake
  `now()` so tests don't sleep. Don't use wall-clock waits.
- **Live tests:** bring up the Step-3.7 endpoint first (`~/bench/` infra: `launch_model`;
  `metrics.scrape(config.METRICS_URL)` must return non-None). Drive the swarm via `bin/swarm`
  with `FLEET_SANDBOX_ISOLATE=0` for repo-touching goals where applicable. Live tests are
  smoke/integration, kept out of the default `pytest` run (mark `@pytest.mark.live`).
- **Regression anchor:** the whole feature set must be inert when
  `FLEET_MAX_CONCURRENT_GOALS=1` **and** `FLEET_PARALLEL_WRITES=0` — that config must behave
  byte-for-byte like today. This is its own test (§6).

New test files to add: `tests/test_worktree.py`, `tests/test_inter_goal_deps.py`,
`tests/test_manager_v2.py`, `tests/test_telegram_bridge.py`, plus additions to
`tests/test_parallel_goals.py`.

---

## Phase 0 — liveness fix (tiny; unblocks "止まってないか" detection) — §B.1

- [x] **0.1** In `_run_swarm`'s `push` callback (`runner.py:616`), call
      `self.tasks.touch(goal_id)` on each task event so a per-goal **last-progress timestamp**
      actually advances during a fleet run (today `progress_at` only moves at start →
      a healthy long goal looks identical to a hung one).
  - **Test (offline):** unit-test that feeding N `push("task", ...)` events advances the
    record's `progress_at` (inject a fake clock; assert it equals the last event's time). Add
    to `tests/test_parallel_goals.py`.
- [x] **0.2** Add `TaskStore` helper `seconds_since_progress(tid, now)` (pure) used by the
      manager, so "stuck" has one definition.
  - **Test (offline):** record with `progress_at=t0`, assert `seconds_since_progress(.., t0+90)
    == 90`. Trivial, but pins the contract the manager relies on.
- [x] **0.3** Manager: distinguish **running+progressing** from **running+stuck** using 0.1/0.2
      (a running goal is "stuck" only if `seconds_since_progress > FLEET_STUCK_SECONDS`).
  - **Test (offline):** synthetic snapshot with two running goals (one freshly touched, one
    stale) → assert only the stale one is flagged stuck. Fake clock, no server.

## Phase 1 — worktree write-isolation — §A.1, §B.4

- [x] **1.1** New `swarm_agent/worktree.py`: `create(goal_id, base_sha) -> path`,
      `commit(path, msg)`, `merge_back(path, branch, base) -> MergeResult`, `remove(path)`,
      `park(path) -> parked_path` (move aside, don't delete). Thin wrappers over `git -C`.
  - **Test (offline):** git fixture repo → `create` makes a worktree on `swarm/<id>`; write a
    file, `commit`, `merge_back` → base HEAD contains the file, worktree removable. New
    `tests/test_worktree.py`.
- [x] **1.2** Conflict path: `merge_back` returns a structured conflict (conflicting paths)
      instead of raising; caller decides.
  - **Test (offline):** two worktrees edit the **same line** of the same file off a common
    base; merge the first (clean), merge the second → assert `MergeResult.conflict` with the
    path listed; base tree is left in a non-corrupt state (no half-merge committed).
- [x] **1.3** `runner.py::_run_swarm` writing branch: when `FLEET_PARALLEL_WRITES=1`, replace
      `compat.sandbox_root(mkdtemp(...))` (`runner.py:627-635`) with a **worktree root** for
      the goal; read-only goals unchanged.
  - **Test (offline):** monkeypatch `ThreadFleet.run` to a no-op that records `cwd`/sandbox
    root; assert a writing goal under `PARALLEL_WRITES=1` runs rooted in a worktree, a
    read-only goal does not. No server (fleet stubbed).
- [x] **1.4** Scheduler routing: with `PARALLEL_WRITES=1`, worktree-isolated writers acquire a
      **shared (reader-style) permit** (they no longer share a tree) instead of the exclusive
      writer permit; `PARALLEL_WRITES=0` keeps today's exclusive behaviour.
  - **Test (offline):** extend `GoalScheduler` tests — with the flag on, K worktree-writers
    admitted concurrently; with it off, a writer still excludes everyone. Pure threading test,
    barriers, no server (already the pattern in `tests/test_parallel_goals.py`).
- [x] **1.5** Non-git working dir → fallback to exclusive (today's path); log the downgrade.
  - **Test (offline):** point at a tmp dir that is **not** a git repo → assert the writer takes
    the exclusive permit and emits a "worktree unavailable, running exclusive" status.
- [x] **1.6** `config.py`: make `FLEET_PARALLEL_WRITES` load-bearing; add `FLEET_WORKTREE_ROOT`,
      `FLEET_GOAL_BRANCH_PREFIX` (default `swarm/`), `FLEET_STUCK_SECONDS`.
  - **Test (offline):** config-validation test (mirrors the existing `DECODE_GATE_*` /
    `MAX_CONCURRENT_GOALS` validators at `config.py:277-292`): bad values rejected with a clear
    message.
  - **Test (live):** `FLEET_PARALLEL_WRITES=1 FLEET_MAX_CONCURRENT_GOALS=4`, enqueue 3 writing
    `/task`s that touch **disjoint** files → all run concurrently (watch `/metrics` running >>
    one goal), each file lands, all branches merge clean, worktrees gone afterward.

## Phase 2 — inter-goal dependency DAG — §A.2

- [x] **2.1** `TaskStore`: add `deps: list[str]` to records (`add(goal, deps=…)`); migrate old
      records (missing `deps` → `[]`).
  - **Test (offline):** add with deps, reload from disk (`_load`) → deps round-trip; a legacy
    record with no `deps` key loads as `[]`. tmp path.
- [x] **2.2** `claim_next()` becomes **dependency-aware**: only return a `pending` goal whose
      every dep is `done` (replaces the plain oldest scan at `taskstore.py:78`).
  - **Test (offline):** queue A, then B(deps=[A]) → first `claim_next` returns A only; B stays
    pending until A is `complete`d, then `claim_next` returns B. Also: two independent goals →
    both claimable. Concurrency: two threads, B never claimed before A done.
- [x] **2.3** `goal.py::analyze_deps(new_goal, existing, *, run_agent)`: cheap LLM pass that
      returns the dep ids for a newly enqueued goal; tolerant JSON parse (reuse `_extract_json`);
      empty/parse-fail → `[]` (independent — fail open to parallel, conflicts caught at merge).
  - **Test (offline):** inject a fake `run_agent` returning `'{"deps":["task-aa"]}'` → deps
    parsed; returning garbage → `[]`. No server.
- [x] **2.4** Wire `analyze_deps` into `enqueue_task` (`runner.py:278`) against current
      `pending`/`running` records; store the result on the new record.
  - **Test (offline):** stub `analyze_deps`; enqueue B after A → B's record carries `deps=[A]`.
  - **Test (live):** enqueue "implement endpoint X", then "write tests for endpoint X" →
    analyzer marks the 2nd dependent on the 1st; manager does not start tests until impl is
    `done`; final test goal sees the implemented code.
- [x] **2.5** DAG-deadlock detection: a goal whose deps include a `failed` goal can never run.
  - **Test (offline):** A `failed`, B(deps=[A]) pending → manager flags B as deadlocked and
    fails it with a reason mentioning A (synthetic snapshot, no server).

## Phase 3 — Manager v2 auditor (bounded auto-fix → escalate) — §B.2–B.4

- [x] **3.1** Pull remediation into a **pure decision function**
      `decide(signal, record) -> Action` (Action ∈ {requeue, backoff_requeue, replan, fail,
      escalate, park}) so policy is testable without threads/LLM/server.
  - **Test (offline):** table test — each (signal, attempts) maps to the §B.3 action; merge
    conflict ⇒ always `escalate`; `attempts >= max_attempts` ⇒ `fail`+escalate. New
    `tests/test_manager_v2.py`.
- [x] **3.2** Signal detectors (each a pure predicate over a snapshot + clock): hang (§B.1
      liveness), thrash (`attempts` climbing), empty-deliverable, DAG-deadlock, gate-starvation,
      worktree-leak. (Merge-conflict comes from §1.2.)
  - **Test (offline):** one focused test per detector with a crafted snapshot → fires only when
    it should; quiet otherwise. Fake clock.
- [x] **3.3** Worktree-leak GC: scan worktree root vs `active_goal_ids()`; **unchanged** leaked
      worktrees → prune; **changed** ones → `park` (never delete — memory
      `feedback_preserve_experiment_data`).
  - **Test (offline):** git fixture with a leaked worktree (no active goal) → an unchanged one
    is pruned, one with uncommitted changes is moved to `parked/` and reported, neither is
    `rm -rf`'d.
- [x] **3.4** Bounded auto-remediation loop in `_tick`/`_evaluate`: apply `decide()`'s action,
      respecting `max_attempts`; on exhaustion escalate to TUI **and** Telegram (emit an
      `escalate`/`error` event the bridge will mirror).
  - **Test (offline):** drive `_tick` with a fake runner + stubbed store: a hung goal gets
    interrupted+requeued up to budget, then a single `escalate` event is emitted (assert via a
    recording `emit`). No server (LLM `_evaluate` stubbed or skipped).
- [x] **3.5** Sequential merge-back orchestration in the manager (§B.4): completed writing goals
      merge one at a time; clean → remove worktree; conflict → park + escalate.
  - **Test (offline):** two completed writing goals with conflicting branches (git fixture) →
    first merges clean and its worktree is removed; second escalates and is parked; base tree
    intact. (Builds on 1.2/3.3.)
  - **Test (live):** force a conflict — two writing `/task`s edit the same file's same region →
    one merges, the other escalates to TUI+Telegram with the path; nothing silently lost.

## Phase 4 — Telegram mirror (in-process, bidirectional) — §C

- [x] **4.1** `swarm_agent/telegram/` bridge: in-process daemon thread holding the `runner`
      ref, running its **own asyncio loop**; transport vendored from HermesAgent
      `gateway/platforms/telegram*.py`. Absent (no-op) if token/chat-id unset.
  - **Test (offline):** construct the bridge with a **fake transport** + fake runner; assert it
    starts/stops cleanly and is a no-op when config is missing. New
    `tests/test_telegram_bridge.py`.
- [x] **4.2** Inbound = **identical to TUI typing**: route a Telegram message through the same
      path as a keypress — `/task …` → `enqueue_task`; bare text → `submit` (turn) or `steer`
      mid-flight; `/stop` → `interrupt`; `/tasks` → snapshot.
  - **Test (offline):** feed fake inbound messages → assert the exact `runner` method is called
    with the right args (spy on a fake runner). `/task foo` → `enqueue_task("foo")`;
    mid-flight text → `steer(...)`; `/stop` → `interrupt()`.
- [x] **4.3** Busy contention (§C.3): a message arriving while a turn is in flight **steers**
      rather than drops; a `submit`-rejected (busy) input gets a clear "still working" reply.
  - **Test (offline):** fake runner reports busy → inbound text triggers `steer()` (not a
    dropped message) and a user-facing "busy" reply is sent via the fake transport.
- [x] **4.4** Outbound = **mirror all events** by tailing the logbook JSONL (reuse the
      `webui/tailer.LogSource` pattern); render conversational events as messages and the
      dashboard-style `task`/counts/gate events as a **single live-edited status message**
      (§C.4) — full info, no spam, nothing filtered.
  - **Test (offline):** feed a synthetic JSONL stream into the renderer → `reply`/`final`/
    `error`/`escalate` become discrete sends; a burst of `task` events collapses into **one**
    edited status message (assert send vs edit counts on the fake transport). Offset-resume:
    re-feeding from a saved offset doesn't double-send.
- [x] **4.5** Security: **chat_id allowlist** — ignore every chat id but Hikari's; token+id from
      env/config.
  - **Test (offline):** inbound from a non-allowed chat id → **no** `runner` method called, no
    reply. Allowed id → routed. (Security-critical; keep this test loud.)
- [x] **4.6** Lifecycle: bridge started alongside the runner (e.g. from `start_manager`/TUI
      boot), stopped on `shutdown()`; "TUI が生きてる限り同じセッション" holds.
  - **Test (offline):** start → `runner.shutdown()` → bridge thread joins/stops; no lingering
    asyncio loop.
  - **Test (live):** from a phone, `/task <goal>` → goal runs and its `final` arrives on
    Telegram; send a mid-flight message → it steers the running swarm (visible in TUI); a
    manager escalation appears on **both** TUI and Telegram; TUI keypress output also mirrors
    to Telegram. Kill the TUI → Telegram stops responding (same-session proof).

---

## 5. Consolidated test matrix

| Area | Offline unit | Live integration |
|---|---|---|
| Liveness / stuck detection (P0) | progress_at advances on task events; stuck only when stale | — |
| Worktree create/commit/merge/conflict (P1) | git fixture: clean merge, conflict surfaced, park-not-delete | 3 disjoint writers run + merge clean; conflict escalates |
| Scheduler write-parallel routing (P1) | flag on → K writers concurrent; off → exclusive | gate utilisation >> single goal under K=4 |
| Inter-goal deps (P2) | dep-aware `claim_next`; `analyze_deps` parse; deadlock flag | impl→tests ordering respected end-to-end |
| Manager v2 (P3) | `decide()` table; per-signal detectors; GC park/prune; bounded loop emits one escalate | forced merge conflict → escalation to TUI+Telegram |
| Telegram bridge (P4) | inbound routing; busy→steer; allowlist; outbound render (send vs edit); offset-resume | phone round-trip; same-session-dies-with-TUI proof |
| **Regression (§6)** | K=1 + PARALLEL_WRITES=0 == today | K=1 live run identical to current behaviour |

Run the offline suite with:
`~/.hermes/hermes-agent/venv/bin/python -m pytest tests/ -q`
(live tests gated behind `-m live` and a reachable Step-3.7 endpoint).

## 6. Regression guard (must stay green throughout)

- [x] **6.1** With `FLEET_MAX_CONCURRENT_GOALS=1` **and** `FLEET_PARALLEL_WRITES=0`: writing
      goals run exclusively (no worktree), no inter-goal deps change dispatch order beyond
      today, no Telegram bridge unless explicitly configured.
  - **Test (offline):** the existing `tests/test_parallel_goals.py` K=1 expectations still pass
    unchanged; add an assertion that `PARALLEL_WRITES=0` takes the legacy `sandbox_root` path,
    not a worktree.
  - **Test (live):** a single writing `/task` at K=1 behaves byte-for-byte like the current
    build (same artifacts, same exclusivity).
- [x] **6.2** HermesAgent upstream untouched — the Telegram transport is **vendored/copied**
      into `swarm_agent/telegram/`, not imported from `~/.hermes/...` at runtime (swarm stays
      self-contained, per the skills/vendoring precedent in
      `project_swarm_agent_skills`).
  - **Test (offline):** import-graph check — nothing under `swarm_agent/` imports from
    `hermes_agent`/`~/.hermes` at runtime.

---

## 7. Suggested build order (dependency-aware, like our own DAG)

```
P0 (liveness)  ─┬─▶ P3 (manager v2 needs real stuck signal)
P1 (worktree)  ─┼─▶ P3 (merge orchestration lives in manager)
P2 (deps)      ─┘
P4 (telegram)  ──── independent; can land any time after P0 (escalations to mirror)
```

P0 first (tiny, unblocks honest stuck-detection). P1 and P2 are independent of each other and
can go in parallel. P3 integrates P0+P1+P2. P4 is independent — ship it whenever; it only needs
P0's events to have something meaningful to escalate. Each phase has its own offline tests and
must keep §6 green.
