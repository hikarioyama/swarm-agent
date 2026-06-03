# Parallel goal consumption (Option A) — implementation plan

**Status:** implemented (Phases 1–3) · **Scope:** swarm-agent front door (`swarm_agent/`, `fleet/`) · **Author handoff:** Opus → impl

> **Implemented 2026-06-03.** Phase 1 (concurrency core), Phase 2 (multi-swarm UI), and
> Phase 3 (per-goal writer sandbox roots) are all in tree. Opt in with
> `FLEET_MAX_CONCURRENT_GOALS=2|3` (K defaults to 1 = unchanged). New module
> `swarm_agent/scheduler.py` (`GoalScheduler`); offline coverage in
> `tests/test_parallel_goals.py` + multi-swarm render tests in `tests/test_tui.py`.
> Two findings from the post-impl review were fixed: `classify_plan` is an ALLOWLIST so an
> unknown/typo lane fails closed to "writing" (it would otherwise inherit the write-capable
> `worker` toolset); and a finished goal no longer blanks the global TUI status while peers
> still run.

## 1. Goal

Let the completion manager run **multiple queued goals at the same time** instead of one
at a time, to fill the spare capacity of the shared `DecodeGate` (one small goal's swarm
only puts ~4–6 generations in flight, while the gate allows up to 96 — the GPU is
under-fed). This is the **safe subset (Option A)** discussed:

- **Read-only goals** (only `writer` / `analyst` / `researcher` / `reviewer` / `reducer`
  lanes — they read/think/synthesise but never mutate shared state) run **concurrently**,
  up to `K` at once.
- **Writing goals** (any `coder` / `code` / `worker` task — they edit files / run shell)
  run **exclusively** (alone), so they can never collide on files or cwd with another goal.

Out of scope here (later phases): recursive per-DAG-task sub-planners (Option B), and a
full multi-swarm dashboard. See §8.

## 2. Why it is safe (and why it helps)

- **Resource contention is already bounded.** The `DecodeGate` is process-global (created
  once in `SwarmRunner.setup` → `compat.apply(gate)`) and caps *concurrent generations* ==
  server `num_requests_running` == resident KV. Running 3 goals' swarms concurrently just
  fills the same gate; it cannot blow up VRAM/KV. The AIMD controller keeps retuning the
  gate from live `/metrics` regardless of how many goals feed it. **So the only real risk
  is shared *mutable state*, not resources.**
- **Each goal already gets its own execution graph.** `SwarmRunner._run_swarm` builds a
  **fresh `Board()` per call** and runs its own `ThreadFleet`. Two concurrent `_run_swarm`
  calls therefore have independent boards — no cross-goal task-graph coupling. This is the
  key enabler that makes Option A cheap.
- **The win:** with `K`≈2–3 read-only goals in flight, the gate goes from ~6/96 utilised to
  comfortably inside the measured 32–64 efficient region → much better dual-GPU saturation
  (aligns with the standing "both GPUs saturated" requirement).

## 3. What serialises today (the things to change)

| Mechanism | File | Current behaviour |
|---|---|---|
| `self.busy` + `self._busy_lock` | `swarm_agent/runner.py` `submit()` | single boolean; `submit` refuses a 2nd turn while busy |
| `self._run_lock` | `runner.py` `_run_turn()` | a plain `Lock` held for the WHOLE turn → strict one-turn-at-a-time |
| manager dispatch | `swarm_agent/manager.py` `_tick()` | dispatches ONE pending goal only when `not runner.busy` |
| `SwarmView` | `swarm_agent/dashboard.py` | models exactly ONE swarm (`planning`→`planned`→`task` events) |
| event stream | `runner.events` / `tui.App.pump` | events are untagged → assume a single active swarm |
| sandbox key | `fleet/compat.py` `worker_sandbox(task_id)` | keys the per-worker cwd on `spec["id"]` → **collides if two goals reuse the same task id** |
| queue claim | `TaskStore.next_pending` | returns the oldest pending but does NOT claim it atomically → two dispatchers could grab the same goal |

## 4. Design

### 4.1 Concurrency primitive — a `GoalScheduler` (shared/exclusive, K-capped)

Add a small scheduler that admits goal *executions* under the safe policy. Conceptually a
readers–writers lock with a cap of `K` simultaneous "readers" (read-only goals) and at most
one "writer" (writing goal), where a writer also excludes all readers.

```
acquire_readonly():  block until (no writer active AND active_count < K); active_count++
acquire_writer():    block until (active_count == 0); writer_active = True
release():           decrement / clear writer; wake waiters
```

Implementation: a `threading.Condition` guarding `active_count`, `writer_active`, and a
`pending_writer` flag (set `pending_writer` when a writer is waiting so new readers don't
starve it). Lives on `SwarmRunner` (e.g. `self._goals = GoalScheduler(K)`).

### 4.2 Classify a goal read-only vs writing

We only know the lanes after planning, but we must pick the right permit before executing.
**Plan first (cheap, concurrent — planning is a read-only LLM call), then classify from the
plan, then acquire the execution permit:**

1. Manager dispatches a goal → its turn **plans** (no execution permit needed; planning is
   safe to run concurrently — it is just `_run_agent("planner", …)`).
2. Inspect the plan's lanes. `WRITES = {"coder", "code", "worker"}`. If any task lane ∈
   `WRITES` → **writing goal**; else → **read-only goal**.
3. Acquire `acquire_writer()` or `acquire_readonly()` accordingly, THEN run the fleet
   (`ThreadFleet(board…).run()`), then `release()`.

This avoids lock *upgrades* (plan unlocked → execute locked). Edge cases:
- Plan parse fails → no execution; nothing acquired.
- Unknown/empty lanes → default to **writing** (exclusive) — fail safe.

### 4.3 Per-goal isolation (kill the remaining collisions)

- **Task-id namespacing.** Prefix every task id with the goal record id when building the
  Board, e.g. `f"{rec_id}.{task.id}"` (rewrite `deps` too), so `worker_sandbox(spec["id"])`
  (FIX#3) never shares a per-worker cwd across goals. Do this in `_run_swarm` when it
  receives a `record`, or in a small `namespace_tasks(tasks, prefix)` helper in `goal.py`.
- **Per-goal sandbox root.** For writing goals, set a unique `FLEET_SANDBOX_ROOT` (a temp
  build dir) for that goal so its coder workers write under a private root. Since writing
  goals are exclusive anyway, this is belt-and-suspenders; mainly it keeps a writing goal's
  artifacts from clobbering an interactive turn's. (Note: absolute-path writes the planner
  emits still land where told — exclusivity is what actually prevents cross-goal write
  races; the sandbox root only isolates *relative* cwd.)
- **TaskStore atomic claim.** Add `TaskStore.claim_next()` that atomically moves the oldest
  `pending` → `running` and returns it (under the existing `RLock`), replacing the
  `next_pending()` + later `mark_running()` two-step so two concurrent dispatch attempts
  can't grab the same goal.

### 4.4 Manager dispatch loop (`manager.py::_tick`)

Replace "dispatch one when `not busy`" with a slot-aware loop:

```
while runner has free capacity:
    if a writer is active or pending → stop (writers drain everything first)
    rec = store.claim_next()            # atomic; None when nothing pending
    if rec is None: break
    if not server_ok(): requeue(rec); break          # existing probe (unchanged)
    runner.submit_goal(rec)             # non-blocking: spawns the goal turn-thread
```

"Free capacity" = `active_readonly_count < K` (and no exclusive writer running). The
manager still wakes on `_wake` (enqueue / idle) + the ≤3-min heartbeat. The actual permit
(reader vs writer) is taken *inside* the turn after planning (§4.2), so the manager can
optimistically dispatch up to K and the scheduler enforces the real policy (a goal that
turns out to be writing will simply block on `acquire_writer()` until others drain — that's
correct, just slightly less eager; acceptable for v1).

### 4.5 Runner turn model (`runner.py`)

- Replace the single `self.busy` bool with an in-flight **count/set** of active goal turns
  (`self._active = {}` keyed by goal id) guarded by `_busy_lock`. `busy` becomes a derived
  property `len(self._active) > 0` (keep the name for the UI/tests).
- Drop `_run_lock` as a global serialiser for **queued-goal** turns; keep one-at-a-time only
  for **interactive** turns if desired (a user typically types one thing). Simplest v1:
  interactive turns also go through the scheduler as read-only (chat/btw) or writing (a
  user `/swarm` that plans coder tasks), so everything shares one policy.
- `submit_goal(rec)` (new): spawn a daemon thread running a goal turn that:
  `claim/mark running → emit user(goal) → plan → classify → acquire permit → run fleet →
  release → complete/fail in the store → emit final/error → _wake.set()`.
- Keep the `_ServerDown`/`Exception` handling and the budget rules from the current
  `_run_turn` (server-down → requeue without burning budget; real failure → `fail`).

### 4.6 Events + UI (`runner.events`, `dashboard.py`, `tui.py`)

Minimum viable multi-swarm display:
- **Tag every swarm event with `goal_id`** (`planning`/`planned`/`task`/`final`). `emit`
  already takes `**kw`; thread `goal_id=rec_id` through `_run_swarm`'s `push`.
- **`SwarmView` → per-goal sub-views.** Keep one `SwarmView` per active `goal_id` in a dict;
  the right panel renders the **active goals** compactly: one short block per goal
  (goal text · done/total · running count), and expands the most-recent/active one. On
  `final`/`error` for a goal, drop its sub-view. Fall back to the current single-swarm
  layout when only one goal is active (no visual regression).
- **Chat:** each goal's `final` posts to the chat tagged with the goal (e.g. a short
  `▸ <goal[:40]>` header before the deliverable) so concurrent results stay legible.
- `/btw` already independent; its `_situation_snapshot` should now summarise **all** active
  goals (loop over the sub-views), not just one.

### 4.7 Config (`fleet/config.py`)

```
FLEET_MAX_CONCURRENT_GOALS = _envi("FLEET_MAX_CONCURRENT_GOALS", 1)   # K; 1 == today's behaviour
FLEET_PARALLEL_WRITES      = env flag, default OFF (writing goals always exclusive)
```

Ship with **K defaulting to 1** (identical to current behaviour) so the change is inert
until opted in; set `K=2` or `3` to turn parallel consumption on. This is the safe rollout.

## 5. Collision matrix (and how each is handled)

| Collision | Handled by |
|---|---|
| KV / VRAM / generation contention | shared `DecodeGate` (already bounds it) — no change needed |
| Two goals editing the same file | **writing goals run exclusively** (§4.1/4.2) |
| Shared cwd / `FLEET_SANDBOX_ISOLATE=0` build dir | writing goals exclusive + **per-goal sandbox root** (§4.3) |
| Per-worker sandbox cwd shared via duplicate `task_id` | **task-id namespacing** per goal (§4.3) |
| Same pending goal dispatched twice | **`TaskStore.claim_next()`** atomic claim (§4.3) |
| Multiple reducers across goals | none — each goal has its own Board/reducer; no cross-goal merge in Option A |
| UI showing garbled interleaved swarms | **goal_id-tagged events + per-goal sub-views** (§4.6) |
| Writer starvation behind a stream of readers | `pending_writer` flag in the scheduler (§4.1) |

## 6. Implementation phases

**Phase 1 — concurrency core (no UI).** `GoalScheduler`; `TaskStore.claim_next`; runner
`_active` count + `submit_goal` + plan→classify→permit→run; task-id namespacing; manager
slot-aware dispatch; config `FLEET_MAX_CONCURRENT_GOALS` (default 1). Headless-testable.

**Phase 2 — UI.** goal_id-tagged events; per-goal `SwarmView`s + compact multi-goal panel;
chat tagging; `/btw` snapshot over all active goals.

**Phase 3 — polish.** per-goal sandbox roots for writers; manager LLM eval aware of multiple
goals; metrics line shows aggregate in-flight goals.

## 7. Testing / validation

Offline (no server):
- `GoalScheduler`: K readers admitted concurrently; a writer waits for readers to drain and
  blocks new readers; `pending_writer` prevents starvation; release wakes correctly.
- `TaskStore.claim_next`: atomic pending→running; two threads never claim the same id.
- `namespace_tasks`: ids + deps rewritten, still a valid DAG (`validate_tasks` passes).
- Classifier: a plan with a `coder` task ⇒ "writing"; all-`writer/analyst` ⇒ "read-only".

Live (Step-3.7 up):
- Enqueue 3 **read-only** goals with `K=3` → all three plan+run concurrently; `/metrics`
  shows running >> a single goal's footprint; all 3 reach `done`; deliverables coherent.
- Enqueue a **writing** goal among read-only ones → it runs alone (others pause/finish
  first), its file lands correctly, no cross-goal file race.
- `K=1` regression → behaviour byte-for-byte identical to today.

## 8. Risks / explicitly deferred

- **Multi-swarm UI** is the largest piece; Phase 1 is usable headless / via the queue even
  before the UI lands (results post to chat).
- **Interactive vs background fairness:** v1 routes everything through one scheduler; if a
  user-typed turn feels laggy behind background goals, add a priority lane later.
- **Recursive sub-planners (Option B)** — dynamic task insertion into a running Board +
  a reducer *tree* (relaxing the single-reducer-sink invariant + `final_result`) + depth
  caps + termination. Bigger; separate plan.
- **Writing-goal eagerness:** optimistic dispatch means a goal discovered to be writing
  blocks on the writer permit after planning; fine for v1 (planning is cheap), revisit if
  many writing goals queue up.

## 9. Definition of done (Phase 1)

`FLEET_MAX_CONCURRENT_GOALS=3` runs 3 read-only queued goals to `done` concurrently with the
gate visibly better utilised; a writing goal interleaved runs exclusively with no file
collision; `K=1` is identical to current behaviour; all offline tests green; HermesAgent
upstream untouched.
