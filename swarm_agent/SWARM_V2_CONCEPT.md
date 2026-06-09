# Swarm v2 — self-managing parallel queue + remote mirror (concept)

**Status:** concept / design (not yet implemented) · **Scope:** `swarm_agent/`, `fleet/`,
new `swarm_agent/telegram/` · **Successor to:** `PARALLEL_GOALS_PLAN.md` (Option A, Phases
1–3, implemented). This doc takes the items that plan **explicitly deferred** in its §8
(parallel *writing* goals; manager awareness of many goals) and adds two new pillars
(inter-goal dependency DAG; a Telegram mirror of the live session).

> This is a **concept memo** produced in a design session with Hikari. Four direction forks
> were decided up-front — see §0. Nothing here is built yet; this exists to be argued with
> and turned into an implementation plan.

---

## 0. Decisions locked (the four forks)

| # | Fork | Decision |
|---|---|---|
| 1 | How to run *writing* goals concurrently without workspace collision | **git worktree per writing goal** (true isolation; relative paths see a real checkout; sequential merge-back). |
| 2 | How to judge whether two separately-added `/task` items are parallelizable | **LLM dependency analysis at enqueue** → the queue becomes an inter-goal DAG (ordering deps), on top of worktree write-isolation. |
| 3 | How much authority the 3-min manager has when it finds a problem | **bounded auto-remediation, then escalate** when it exhausts retries / hits a merge conflict / can't diagnose. Escalations also go to Telegram. |
| 4 | What Telegram is | A **full bidirectional mirror of the live TUI session**: input is treated identically to TUI typing; *all* events the TUI sees are also pushed to Telegram. In-process, lives and dies with the runner. |

---

## 1. Motivation & one corrected mental model

The ask: *"`/task` で積んだタスクを、並列可能か判断して可能なら4つ以上同時に。128K で26並列持てるんだから。"*

There are **two layers of parallelism** and the request lives in the second:

- **Intra-goal (layer 1):** one goal → planner → task DAG → `ThreadFleet` fans the leaves
  across the process-global `DecodeGate` (start 40, AIMD ≤ 96 concurrent generations). **This
  is the "26 parallel in 128K" the user means.** One goal already fans out this wide.
- **Inter-goal (layer 2):** the `TaskStore` queue of goals (`/task` → `enqueue_task`), drained
  by `CompletionManager`, with cross-goal concurrency gated by `GoalScheduler`
  (`K = FLEET_MAX_CONCURRENT_GOALS`, **default 1**; writing goals exclusive).

**The correction that shapes everything:** for *writing* goals (implement / fix / build — the
common `/task`), the binding constraint is **NOT context budget**; it is the **workspace
collision** safety model that forces writers to run alone (`PARALLEL_GOALS_PLAN` §4.1/§4.2).
The `DecodeGate` is **shared across all goals**, so running 4 goals does not 4× the parallelism
— they draw from the same 40–96 slots. Therefore:

> The value of multi-goal is **pipelining**, not more total agents. One goal stuck in its
> serial tail (reducer wait / dependency chain) idles the gate; overlapping a second goal's
> leaves keeps the GPU fed, and independent `/task` items stop queuing behind a long one.

So "並列可能か判断" decomposes into **two independent judgments**:

1. **Ordering (semantic dependency):** does goal B consume goal A's result? → today: **no
   inter-goal dependency model at all** (goals are independent records claimed oldest-first).
2. **Mutual exclusion (write conflict):** do A and B touch the same working tree? → today:
   `classify_plan` is a coarse binary (any write-capable lane ⇒ "writing" ⇒ exclusive); it
   never asks whether two *different goals* touch disjoint files.

Pillar A solves both: worktree isolation removes the write-conflict reason for exclusivity;
the dependency DAG supplies the ordering.

---

## 2. Pillar A — parallel writing goals (worktree + inter-goal DAG)

Goal: flip `FLEET_PARALLEL_WRITES` from the reserved-OFF placeholder (`config.py:130`) into a
real, safe **ON** path, so ≥4 writing `/task` items can run at once.

### A.1 Worktree isolation (replaces exclusivity)

Each **writing** goal runs inside its **own `git worktree`** (separate working tree, shared
object store) on a goal branch `swarm/<goal_id>`:

```
git worktree add --detach <tmp>/wt-<goal_id> <base_sha>   # or a fresh branch swarm/<goal_id>
… run the goal's ThreadFleet with cwd rooted at the worktree …
git -C <wt> add -A && git -C <wt> commit -m "swarm goal <goal_id>: <goal[:60]>"
# merge-back handled by the manager (§B), worktree removed on success
```

Why worktree over the alternatives we considered:

- **vs. keeping writers exclusive:** that was the whole point — exclusivity is what we're
  removing. Worktree lets N writers proceed truly in parallel.
- **vs. predicted file-scope locks:** no reliance on the planner correctly predicting which
  paths a wandering agent will touch (it can't be trusted; a guard would be needed).
- **vs. the current empty-tempdir sandbox** (`FLEET_SANDBOX_ISOLATE`): a worktree is a **real
  checkout**, so relative paths and repo tooling work — this *fixes* the known
  "isolate=1 → relative paths can't see the repo → hallucination" failure
  (memory: `project_swarm_agent_sandbox_isolate`), rather than fighting it.

Replaces `compat.sandbox_root(mkdtemp(...))` in `_run_swarm` (`runner.py:630`) for writing
goals with a worktree root. Read-only goals need no worktree (they don't mutate) — they keep
running concurrently as today.

**Open edges:** non-git working dirs (fallback to today's exclusive behaviour, or a plain
copy); base SHA selection (branch from HEAD at dispatch); cost of N checkouts (cheap — shared
object store, only the working tree is duplicated).

### A.2 Inter-goal dependency analysis (queue → DAG)

At **enqueue** (`enqueue_task`), run a cheap LLM pass that compares the new goal against the
current `pending`/`running` goals and emits, for the new record:

```jsonc
{ "deps": ["task-ab12", ...],     // goals whose RESULT this one needs (ordering only)
  "rationale": "writes tests for the API that task-ab12 implements" }
```

This adds an **inter-goal `deps` field** to `TaskStore` records (distinct from the *intra*-goal
DAG the planner already builds inside `goal.py`). The manager then dispatches a goal only when
all its `deps` are `done`. This is the half of "並列可能か判断" that worktrees do **not** cover:
two goals can be write-isolated yet still be **ordered** (B needs A's output).

Conflict-vs-ordering split (so we don't conflate them):

| Relationship | Detected by | Effect |
|---|---|---|
| B needs A's *result* | A.2 LLM dep analysis (enqueue) | B waits for A `done` (DAG edge) |
| A and B mutate the *same files* | **not predicted** — handled by worktree isolation + merge | both run in parallel; conflict surfaces at *merge* (→ §B.4) |
| A and B are fully independent | default (empty deps) | run concurrently up to capacity |

Deliberately we do **not** try to statically predict file overlap (fork #1 rejected
scope-prediction). Overlap is allowed to happen and is resolved at merge time, loudly.

### A.3 What changes, by file

- `taskstore.py`: add `deps: list[str]` to records; `add(goal, deps=…)`; `claim_next()` must
  only return a goal whose deps are all `done` (dependency-aware claim) — replaces the plain
  oldest-pending scan (`taskstore.py:78`).
- `goal.py`: new small `analyze_deps(new_goal, existing: list[rec]) -> list[str]` (an LLM call
  on a cheap lane), reusing the tolerant JSON parsing already there (`_extract_json`).
- `runner.py` `_run_swarm`: writing branch acquires a **worktree root** instead of
  `sandbox_root(mkdtemp)` (`runner.py:627-635`); `GoalScheduler` writer-exclusivity is relaxed
  for worktree-isolated writers (they become "readers" w.r.t. the scheduler since they no
  longer share a tree) — i.e. `FLEET_PARALLEL_WRITES=1` reroutes writers through the K-capped
  shared lane, gated instead by `deps`.
- `config.py`: `FLEET_PARALLEL_WRITES` becomes load-bearing; add `FLEET_WORKTREE_ROOT`,
  `FLEET_GOAL_BRANCH_PREFIX` (`swarm/`).
- `scheduler.py`: still the K cap; the reader/writer distinction is now "needs the *shared*
  real tree" (rare — e.g. non-git fallback) vs "has its own worktree".

### A.4 Capacity note (set K honestly)

With writers parallelized, K (`FLEET_MAX_CONCURRENT_GOALS`) can sensibly be 4+. But remember
§1: all goals share the `DecodeGate`. K is about **how many independent DAG fronts** we keep
in flight to pipeline the gate, not a multiplier on agents. Suggested default once stable:
`K=4`, AIMD still owns the gate. Log when goals queue behind K so we can tune.

---

## 3. Pillar B — Manager v2 (3-min health auditor)

The "3分おきにタスク管理エージェントを起動して、止まってないか・バグってないか見る" ask is
**half-built already**: `CompletionManager` wakes every `SWARM_MANAGER_INTERVAL`
(**default 180s = 3 min**, `manager.py:17`) and runs an LLM evaluator (`_evaluate`,
`manager.py:116`) that returns `{note, requeue, escalate}`. v2 upgrades this watchdog into a
real auditor that understands worktrees + the inter-goal DAG, and can act (bounded).

### B.1 The liveness gap (must fix first)

`_evaluate` judges staleness by `seconds_since_progress = now - progress_at`, but **`progress_at`
is only set at goal start** — the intra-goal task events (`push` in `_run_swarm`,
`runner.py:616`) never call `tasks.touch(goal_id)`. So a healthy 8-minute goal and a hung goal
look identical to the manager (both show a monotonically growing age). **Today it literally
cannot tell "running and progressing" from "running and stuck."**

Fix: have the `push` task-event callback `touch(goal_id)` on each leaf done/fail, recording a
real **per-goal last-event timestamp**. Then "stuck" = no task event for N seconds *while
running*, which is a true hang signal. (Small change, unblocks the entire ask.)

### B.2 Detection signals (what each tick checks)

| Signal | Definition | Source |
|---|---|---|
| **Hang** | running, but no task event for `> N`s | B.1 liveness ts |
| **Thrash** | same goal `fail→pending` looping, `attempts` climbing | `taskstore` record |
| **Empty deliverable** | `done` but reducer produced nothing, repeatedly | `_run_swarm` final check (`runner.py:655`) |
| **DAG deadlock** | a goal's `deps` include a `failed` goal → it can never run | inter-goal DAG (§A.2) |
| **Merge conflict** | worktree branch fails to merge to base | §A.1 merge step |
| **Gate starvation / server flap** | `DecodeGate` pinned, or `/metrics` reachability flapping | `metrics.scrape` (already probed) |
| **Worktree leak** | worktree dir for a no-longer-active goal still on disk | filesystem scan vs `active_goal_ids()` |

### B.3 Remediation policy (bounded auto-fix → escalate)

Per fork #3: act automatically within a retry budget, then escalate (TUI **and** Telegram).

| Signal | Auto action (bounded) | Escalate when |
|---|---|---|
| Hang | interrupt + requeue the goal (≤ `max_attempts`) | budget exhausted |
| Thrash | back off + requeue with jitter | `attempts ≥ max_attempts` → `failed` + notify |
| Empty deliverable | re-plan once (different planner seed) | still empty |
| DAG deadlock | fail the dependents with a clear reason | always notify (a dep failed) |
| **Merge conflict** | **do NOT auto-resolve** | **always escalate** (human/dedicated reducer decides) |
| Server flap | hold dispatch, back off to heartbeat (existing) | prolonged outage |
| Worktree leak | **preserve, don't delete** — move aside to a `parked/` dir | report what was parked |

Two house rules folded in:
- **Never silently delete experimental output** (memory `feedback_preserve_experiment_data`,
  `feedback_abstract_failed_weights`): leaked/abandoned worktrees with changes are **parked**,
  not removed. Only unchanged worktrees are pruned.
- **auto-approve stance** (memory `project_swarm_agent_task_queue`): the manager is allowed to
  take its bounded remediation actions without prompting — consistent with the swarm's existing
  "dangerous commands auto-approve, don't re-litigate" choice.

### B.4 Merge orchestration (new manager duty)

When a writing goal's fleet completes, the manager (not the worker) merges
`swarm/<goal_id>` → base, **sequentially** (one merge at a time, even though goals ran in
parallel) so merges are linearizable. Clean merge → remove worktree. Conflict → leave the
branch, park the worktree, escalate with the conflicting paths. This keeps the parallel-write
risk (fork #1's known downside) contained to a loud, human-visible event instead of silent
corruption.

---

## 4. Pillar C — Telegram as a full session mirror

Per fork #4: Telegram is **not** a separate bot session (the HermesAgent model). It is a
**second front-end onto the one live `SwarmRunner`**, mirroring it both ways.

### C.1 The enabling discovery

The existing **WebUI is a read-only sidecar that tails the logbook JSONL** (`latest.jsonl`)
via `webui/tailer.py::LogSource` — it does **not** consume `runner.events`. `_publish`
(`runner.py:185`) writes every event to **both** `self.log` (JSONL) and `self.events` (queue),
so the JSONL is already a **durable, multi-subscriber event stream**. The single-queue
"two consumers steal from each other" problem is sidestepped by tailing the log.

### C.2 Architecture (in-process, bidirectional)

- **Outbound (session → Telegram): mirror *all* events.** A bridge **tails the same logbook
  JSONL** (reuse the `LogSource` pattern) and renders every event to Telegram — matching the
  "全イベント" decision. Tailing means no change to `self.events`, durability across bridge
  restarts (remember the byte offset), and exact parity with what the TUI shows.
- **Inbound (Telegram → session): identical to TUI typing.** The bridge runs **in the same
  process** as the runner (a daemon thread holding the `runner` reference), so a Telegram
  message goes through the **same path as a typed TUI turn** — `submit()` for a turn,
  `steer()` mid-flight, `enqueue_task()` for `/task`, `interrupt()` for stop. Because it shares
  the process, "TUI が起動している限り Telegram も同じセッション" is automatic: the bridge lives
  and dies with the runner.

```
  Telegram  ──in──▶  SwarmRunner.submit()/steer()/enqueue_task()   (same as TUI keypress)
            ◀─out──  tail(latest.jsonl) ──▶ render every event      (same as TUI pane)
```

### C.3 Contention (one interactive turn, two windows)

Inbound is "TUI と同じ扱い," and `submit()` allows **one interactive turn at a time**
(`_interactive` sentinel, `runner.py:294`). So a Telegram turn and a TUI turn genuinely
contend — correct for "one session, two windows." Behaviour when busy must mirror the TUI's:
a message arriving mid-turn should **steer** the in-flight turn (`steer()` fans it to live
agents) rather than be dropped, and `submit()`-rejected (busy) input gets a clear "still
working" reply on Telegram. (Exact busy semantics = whatever the TUI already does, kept
identical.)

### C.4 Rendering note (full mirror without spam)

"全イベント" to a chat app is firehose-y (dozens of `task done/fail` per goal). Honour the
decision but render smartly: a **single live-updating status message** (Telegram supports
editing a message in place) for the swarm dashboard-style events (`task`, counts, gate), and
the **chat stream** (`user`/`reply`/`final`/`error`/`manager`/`btw`/escalations) as normal
messages. Same information as the TUI, legible on a phone. This is a rendering choice, not a
filtering one — nothing is dropped.

### C.5 Reuse & security

- **Reuse:** HermesAgent `~/.hermes/hermes-agent/gateway/platforms/telegram.py` +
  `telegram_network.py` (python-telegram-bot v22) for transport/auth — "HermesAgent から
  モジュール取れば簡単" confirmed. It's asyncio, so the bridge runs its **own asyncio loop in a
  thread** beside the threaded runner.
- **Security (must-have):** **chat_id allowlist** — the bot is publicly addressable; ignore
  every chat id except Hikari's. Token + allowed id via env/config, bridge silently absent if
  unset (same soft-degrade as a down inference server).

---

## 5. Reuse map (build on, don't rebuild)

| Need | Existing thing to extend |
|---|---|
| Cross-goal concurrency cap + readers/writers | `swarm_agent/scheduler.py` `GoalScheduler` |
| Atomic queue claim, persistence, recovery | `taskstore.py` (add `deps`, dep-aware `claim_next`) |
| 3-min heartbeat + LLM queue eval | `manager.py` `CompletionManager` / `_evaluate` |
| Per-goal task-id namespacing (no cwd clash) | `goal.py::namespace_tasks` |
| Read-only vs writing classification | `goal.py::classify_plan`, `config.lane_writes` |
| Durable multi-subscriber event stream | logbook JSONL (`_publish`) + `webui/tailer.LogSource` |
| Telegram transport | HermesAgent `gateway/platforms/telegram*.py` (vendored, isolated) |
| Write-isolation primitive | replace `compat.sandbox_root` with `git worktree` for writers |

---

## 6. Open questions / risks

1. **Dep-analysis false positives/negatives.** A wrong "independent" verdict runs two truly
   ordered goals in parallel; a wrong "dependent" needlessly serializes. Mitigation: the
   analyzer is advisory + worktree+merge catches *write* conflicts anyway; consider a cheap
   "are these really independent?" second opinion only when goals look related.
2. **Merge-conflict frequency.** If `/task` items routinely touch the same files, parallel
   writes buy little (everything conflicts at merge). Worktree shines for **disjoint** edits
   (different modules/repos). Worth measuring before raising K high.
3. **Non-git working directories** — need a defined fallback (exclusive, as today).
4. **Telegram firehose** — even with live-edit rendering, very wide goals could be noisy;
   may want a per-goal collapse.
5. **Interactive fairness** — TUI vs Telegram vs background goals all share one interactive
   slot; if a typed turn lags behind background work, add a priority lane
   (already flagged in `PARALLEL_GOALS_PLAN` §8).
6. **Abandoned worktrees from killed workers** — subprocess that can't be killed may still
   hold a worktree; reuse the existing `_await_abandoned_writers` idea (`runner.py:560`).

---

## 7. Phased rollout

- **Phase 0 — liveness fix (tiny, unblocks the ask).** `touch(goal_id)` on task events (§B.1)
  so "stuck" is detectable at all. Independent of everything else.
- **Phase 1 — worktree write-isolation.** `FLEET_PARALLEL_WRITES=1` routes writers into
  worktrees + the K-capped lane; manager does sequential merge-back + conflict escalation
  (§A.1, §B.4). Read-only path unchanged.
- **Phase 2 — inter-goal DAG.** `deps` in `TaskStore`, `analyze_deps` at enqueue, dependency-
  aware `claim_next`, deadlock detection (§A.2, §B.2).
- **Phase 3 — Manager v2 auditor.** Full signal set + bounded remediation table + worktree GC
  (§B.2–B.4).
- **Phase 4 — Telegram mirror.** In-process bridge: outbound log-tail render, inbound
  TUI-identical, allowlist (§C). Can land independently after Phase 0.

Each phase is usable on its own; Phase 0 and Phase 4 don't depend on the parallel-write work.

## 8. Definition of done

≥4 disjoint writing `/task` items run concurrently in isolated worktrees and merge back
cleanly (conflicts escalate, never corrupt); the queue respects inter-goal `deps`; the 3-min
manager detects a real hang (not just elapsed time), auto-remediates within budget, and
escalates the rest to TUI **and** Telegram; Telegram mirrors the live session both ways
(input identical to TUI, all events out) and is restricted to Hikari's chat id; HermesAgent
upstream untouched; `K=1` + `FLEET_PARALLEL_WRITES=0` reproduces today's behaviour.
