# step37-harness

A **high-concurrency fleet orchestrator** for Step-3.7-Flash, built **on top of**
HermesAgent (imports `run_agent.AIAgent`) but living **outside** the HermesAgent
repo, so `git pull` upstream never touches it.

## Why

Single-stream Step-3.7 is ~125 tok/s; the GPU only earns its keep with **dozens of
agents in flight**. Measured efficient region for ~8K-token workers is **C32–C64**
(C32 ≈ 763 tok/s @ 23.8 tok/s/agent; C64 ≈ 1225 tok/s, ~9.8× single-stream). Full
data: `~/bench/step37-mtp/FLEET_OPTIMUM.md`.

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
/home/hikari/.hermes/hermes-agent/venv/bin/python -m fleet.cli examples/tasks.jsonl --inflight 40
```
`tasks.jsonl` — one task per line: `{"id","prompt","deps":[],"lane":"worker"}`.

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

## Roadmap (v2)
- **KV-portfolio lanes** — heterogeneous context by role (router ~1K / worker ~8-16K /
  reducer 8K→big / planner 40-60K), a priority waterfall over the 1.63M-token KV
  budget, worker lane as elastic basin (FLEET_OPTIMUM.md §5).
- **Dynamic admission** — target the throughput knee via `/metrics` instead of a
  fixed number (`decoding_now()` is already wired).
- **Tree reduction** — log-depth fan-in reducers instead of a star.
- **Persistent board** — SQLite/file backing for restart-safety + multi-producer.
- **Prefix-cache-aware dispatch** — share a common worker prefix to cut prefill
  (measurements were worst-case, cache off; real fleet should run hotter).
