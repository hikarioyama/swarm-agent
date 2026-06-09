# swarm-agent

> ⚠️ **Status: early, work-in-progress, and held together with tape.**
> This is a personal research harness under active and messy development —
> expect rough edges, half-finished features, breaking changes without notice,
> thin tests, and the occasional thing that is outright broken. It is shared
> as-is, with **no stability or support guarantees**. Poke around and learn from
> it, but don't build anything load-bearing on it yet.

A standalone **high-concurrency swarm harness** for Step-3.7-Flash. It reuses
HermesAgent's `run_agent.AIAgent` as a runtime module but owns its orchestration,
CLI, board, scheduling, and admission control outside the HermesAgent repo.

The public command is `swarm`. The original `fleet` package remains as the
measured engine compatibility layer while the goal-driven planner front door is
built.

Run `swarm` with no arguments to open the Hermes-inspired curses TUI. It can
launch the small demo, the invaders specialist DAG, or a free-form planner goal
while showing the live transcript and server metrics. The `❯` composer accepts
input immediately and sends normal messages through the planner front door;
it supports multi-line input (Shift+Enter inserts a newline).
Use `/goal TEXT` to save a planner goal without launching it, then `/run` to
launch it explicitly. Other commands include `/mode invaders`, `/gate 32`,
`/stop`, and `/help`.

## What's in the box

The `swarm_agent` package is the goal-driven front door; the `fleet` package is
the measured concurrency engine underneath it.

- **Conversational front door** (`swarm_agent/runner.py`) — an in-process router
  that turns a plain message into either a chat reply or a planned goal, then
  drives the fleet engine. No separate server process.
- **Planner + DAG** (`swarm_agent/goal.py`) — decomposes a goal into a task DAG
  (lanes: writer / coder / researcher / analyst / reviewer / reducer …) with an
  auto-repair pass that promotes a stray integration sink to `reducer`.
- **Parallel goal consumption** (`swarm_agent/scheduler.py`) — a reader-writer,
  K-capped scheduler runs read-only goals concurrently while serializing
  writing goals. Opt in with `FLEET_MAX_CONCURRENT_GOALS` (default 1 =
  legacy behaviour). See `swarm_agent/PARALLEL_GOALS_PLAN.md`.
- **Persistent task queue + completion manager** (`swarm_agent/taskstore.py`) —
  a durable `~/.cache/swarm-agent/tasks.json` queue with a background manager
  that drives queued goals to completion.
- **Conversation recall** (`swarm_agent/recall.py`) — a LanceDB hybrid
  (vector + BM25/FTS, CPU embeddings) store with JST time-window filters so the
  front door and planner can reference older turns instead of forgetting them.
  Fully fail-soft; disable with `SWARM_RECALL=0`.
- **Skill system** (`swarm_agent/skills/`) — self-contained skill synthesis on
  goal completion plus a weekly self-improving curator. The model proposes a
  structured plan and the harness applies it; no model-callable tools are added.
  Disable with `SWARM_SKILL_SYNTH=0` / `SWARM_CURATOR=0`.
- **Event log + web UI** (`swarm_agent/logbook.py`, `swarm_agent/webui/`) —
  every event is persisted as JSONL; a read-only FastAPI sidecar tails it and
  streams a live force-graph of the running DAG to the browser. The swarm core
  is never modified by the sidecar. See `swarm_agent/webui/PROTOCOL.md`.

### v2 — self-managing parallel swarm

These build on the front door above to run *multiple* goals at once, safely, and
to drive them without a human babysitting the TUI. Design notes:
[`swarm_agent/SWARM_V2_CONCEPT.md`](swarm_agent/SWARM_V2_CONCEPT.md).

- **Parallel writes via git worktrees** (`swarm_agent/worktree.py`) — each
  writing goal runs in its own isolated git worktree and is merged back
  sequentially, so several writers make progress concurrently without clobbering
  each other (structured conflict handling; park + escalate on conflict). This
  also fixes the relative-path/sandbox-isolation problem, since each worktree is
  a real checkout. Opt in with `FLEET_PARALLEL_WRITES`.
- **Inter-goal dependency DAG** (`swarm_agent/goal.py`, `swarm_agent/taskstore.py`)
  — at enqueue time an LLM pass infers semantic dependencies between queued
  goals, and a dep-aware `claim_next` runs independent goals in parallel while
  ordering dependent ones. Includes deadlock detection.
- **Self-managing manager** (`swarm_agent/audit.py`) — the background completion
  manager (180 s tick) audits in-flight goals, performs **bounded** auto-repair
  on hung work, and escalates exactly once when it cannot recover on its own.
- **Telegram mirror** (`swarm_agent/telegram/`) — an in-process, fully
  bidirectional mirror of the live TUI session (same process, not a separate bot
  session). Outbound events tail the event log; inbound phone messages route as
  TUI input (submit / steer / enqueue / interrupt). Enable by exporting
  `SWARM_TELEGRAM_TOKEN` plus a `SWARM_TELEGRAM_CHAT_IDS` allowlist; unset leaves
  it a complete no-op. No HermesAgent dependency.

### Subcommands
```bash
swarm                     # curses TUI (default)
swarm <tasks.jsonl> ...   # run a DAG / planner goal through the fleet engine
swarm logs [--errors|--tail N|--all|--path]   # inspect the persistent event log
swarm webui [--port 8765] [--replay <events-*.jsonl>]   # read-only graph UI
```

### Configuration

All optional; defaults preserve the original single-goal behaviour.

| Env var | Default | Effect |
|---|---|---|
| `FLEET_MAX_CONCURRENT_GOALS` | `1` | Parallel goal consumption cap (`1` = legacy serial). |
| `FLEET_PARALLEL_WRITES` | off | Run writing goals in isolated git worktrees and merge back. |
| `SWARM_TELEGRAM_TOKEN` | unset | Telegram bot token; set with chat-ids to enable the mirror. |
| `SWARM_TELEGRAM_CHAT_IDS` | unset | Comma-separated chat-id allowlist (required for Telegram). |
| `SWARM_RECALL` | `1` | Conversation recall (LanceDB hybrid); `0` disables. |
| `SWARM_SKILL_SYNTH` | `1` | Synthesize a skill on goal completion; `0` disables. |
| `SWARM_CURATOR` | `1` | Weekly skill curator; `0` disables. |

## Why

Single-stream Step-3.7 is ~125 tok/s; the GPU only earns its keep with **dozens of
agents in flight**. Measured efficient region for ~8K-token workers is **C32–C64**
(C32 ≈ 763 tok/s @ 23.8 tok/s/agent; C64 ≈ 1225 tok/s, ~9.8× single-stream). Full
data: `~/bench/step37-mtp/FLEET_OPTIMUM.md`.

## v0.2 — measured results (real agent path)

v0.2 drives the **real HermesAgent** through a resizable **DecodeGate** in a
**single-process thread pool** (HermesAgent is sync+threaded, not asyncio) with **AIMD**
dynamic admission. Live-measured through the gate (windowed 30 s, `results/operating_point.json`):

| in-flight N | tok/s | occupancy | tok/s/agent | KV% |
|---:|---:|---:|---:|---:|
| 16 | 947  | 1.00  | 59.2 | 11.3 |
| 32 | **1247** | 0.945 | 41.2 | 19.3 |
| 48 | 1246 | 0.919 | 28.2 | 27.9 |

- **Reaches/exceeds the synthetic C32–64 region through the real agent path** (N=32 → 1247 tok/s, ≈ the synthetic C64).
- **Throughput knee is ~C32** for realistic long agent outputs (32→48 is flat) — earlier than the short-output synthetic curve.
- **DecodeGate pins `num_requests_running` to N** (occupancy 0.92–1.0) — concurrency control validated.
- **KV bound validated**: measured KV% (11/19/28) tracks the prediction (11/20/30). Because HermesAgent is stateless
  (full-history resend), tool-executing workers hold **zero** server KV, so "parking" is automatic.
- **AIMD converges** to the knee (~C32) and oscillates without thrash/deadlock; **prefix-warm** lifts the role-prefix
  cache hit-rate 48% → 97%; **SQLite board** gives atomic claim + liveness-gated restart recovery.

A 6-agent adversarial review found and we fixed CRITICAL/MAJOR concurrency bugs (gate phantom-ticket leak, enrolled-clamp
inversion, kv summed across labels, AIMD saturation/EWMA, board write-guards + reset liveness, per-worker sandbox isolation).
The hermes-agent repo is **never modified** — all adaptation is runtime monkeypatch in `fleet/compat.py`. See `BUILD_SPEC.md`.

## Design (why it doesn't "panic")

- **Stigmergic coordination** — workers never talk to each other or to a central
  conductor. They coordinate only through a shared **Board** (claim a ready task,
  write its result, which unlocks dependents). No central context accumulation, no
  serial supervision bottleneck → scales to dozens.
- **Dumb fast scheduler** — an admission-control loop (pure code, no LLM) keeps
  `TARGET_INFLIGHT` ephemeral workers busy. The thing keeping the GPU fed is a
  `while`-loop, not an agent.
- **Ephemeral narrow-context workers** — each task runs in a fresh `AIAgent` and
  dies. State lives on the Board, not in any worker's context (short context =
  fits the KV budget + favourable throughput).

## Layout
```
fleet/config.py     measured operating point (server, target in-flight, ctx policy)
fleet/board.py      stigmergic blackboard: DAG work queue, task states + deps
fleet/worker.py     ephemeral worker (wraps AIAgent.run_conversation), picklable
fleet/scheduler.py  admission-control concurrency loop (ProcessPoolExecutor)
fleet/cli.py        `python -m fleet.cli tasks.jsonl --inflight 40`
plugin/             thin HermesAgent plugin exposing `/fleet` (deploy to ~/.hermes/plugins/)
```

## Run

Use the HermesAgent venv python (so `run_agent` + deps import inside workers):
```bash
cd ~/projects/step37-harness
./bin/swarm examples/tasks.jsonl --inflight 40
```
`tasks.jsonl` — one task per line: `{"id","prompt","deps":[],"lane":"worker"}`.

Or hand one goal to the planner front door:
```bash
swarm --goal "Audit this repository and synthesize a report" \
  --plan-out /tmp/audit-plan.jsonl --final-out /tmp/audit-report.md
```

Install the local launcher once to run it from any directory:
```bash
ln -s ~/projects/step37-harness/bin/swarm ~/.local/bin/swarm
swarm examples/tasks.jsonl --gate 32 --admission aimd
```

## Use the `/fleet` command inside HermesAgent (optional)
```bash
bash deploy.sh                       # symlink plugin into ~/.hermes/plugins/
hermes plugins enable fleet-orchestrator
# then in a chat:  /fleet status   |   /fleet run examples/tasks.jsonl 40
```

## Upstream-safe
Your code is in `~/projects/step37-harness/` (its own git). The plugin is
symlinked into `~/.hermes/plugins/` — **outside** `~/.hermes/hermes-agent/`, so
`git pull` in the HermesAgent repo never conflicts.

## v0.2 roadmap status (all landed + verified)
- **Single-process engine** — `engine.ThreadFleet` (thread pool; HermesAgent is sync+threaded). ✅
- **Decode-batch admission** — `compat.DecodeGate` pins concurrent generations == server KV. ✅
- **Dynamic admission** — `admission.AIMDController` targets the knee via `/metrics`. ✅
- **KV-portfolio lanes** — lane-priority gate acquire (`config.LANE_PRIORITY`). ✅
- **Prefix-warm** — `warm.warm_profiles` warms each role prefix (hit-rate 48%→97%). ✅
- **Persistent board** — `board.SqliteBoard` + `open_board` (atomic claim, restart-safe). ✅

Run the v0.2 thread engine: `python -m fleet.cli tasks.jsonl --engine thread --admission aimd --warm`.
Reproduce the operating point: `python scripts/throughput_probe.py --gate 32`.

### Still open
- **C96+** — needs an N-proportional warm-up to measure as steady state (gate MAX is 96).
- **Tree reduction** — log-depth fan-in reducers (the board supports it; not yet a helper).
- **Ungated summary path** — iteration-limit summary generations bypass the gate (rare; review #9, deferred).

### v2 status (landed, offline + live tested)
The self-managing parallel swarm — git-worktree writes, inter-goal dependency
DAG, manager audit/escalation, and the Telegram mirror — is implemented and
tested: the offline suite is green plus live runs against Step-3.7 on `:8001`.
Checklist and per-feature test notes:
[`swarm_agent/SWARM_V2_TODO.md`](swarm_agent/SWARM_V2_TODO.md).

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE). The bundled `swarm_agent/webui/static/vendor/d3.v7.min.js`
is third-party software under the ISC License (see `NOTICE`).
