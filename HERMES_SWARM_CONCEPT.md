# hermes-swarm — CONCEPT

**A fork of HermesAgent that boots swarm-first.** From the moment it starts, it's a swarm: throw it a single goal and the planner decomposes it, dozens of agents run concurrently as a matter of course, and tasks get cleared at the maximum throughput that saturates the GPU.

This is a **concept document** (pre-implementation). Every number and fact has been measured and grounded on step37-harness (§3, §4, §10). The predecessor, step37-harness, was designed to "wrap HermesAgent from the outside with no modifications," but this concept **forks it and makes swarm native** (user decision).

---

## 0. In one line

> `hermes-swarm "audit this repo and surface every vulnerability"` → planner decomposes into 12 subtasks →
> 40 agents execute via concurrent decode → reducer returns a consolidated report. **No conductor; swarm by default.**

---

## 1. The target experience (what becomes "the norm")

- **Startup = swarm.** Not a single-agent conversation: at boot time a board + dozens of disposable
  worker lanes + decode-gate are already standing. There is no "parallelize / don't parallelize" choice. Always a swarm.
- **Goal-driven.** Hand it a single high-level goal and the planner agent automatically decomposes it into a task DAG,
  the worker swarm executes in parallel, and the reducer aggregates over a tree to return the final answer. You don't hand-write the DAG.
- **Maximum throughput by default.** AIMD automatically tracks the concurrent decode count to the measured knee (~C32, §4). Users need no
  tuning; "dozens of agents running, and fast" is the default state.
- **Disposable, narrow context.** Each worker handles one subtask and dies. State persists on the board.
  Context is bundled to ~8K, and when it grows it's kept in check via compaction (staying on the throughput-favorable side).

Difference from the predecessor harness: the harness was "an execution substrate where you hand a DAG to `python -m fleet.cli tasks.jsonl` by hand."
hermes-swarm upgrades that into the core itself: "**swarm on startup + just throw it a goal**."

---

## 2. Why a fork (decision and rationale)

| Option | Assessment |
|---|---|
| External harness (unmodified monkeypatch) | ← the predecessor. Git-safe, but it only "wraps from the outside" and never makes swarm native. It can't reach part of the hot path (iteration-summary generation), so the gate gets bypassed (§7). |
| Modify the hermes core directly | Fully native, but the daily driver (Step3.7 hermes, production) carries upstream-pull conflict and breakage risk. |
| **Fork (chosen)** | Completely separate from upstream. The daily hermes stays untouched. Free to modify it for swarm purposes; upstream is merged manually. |

**What only a fork makes possible (§5.4)**: we can directly address areas that were too risky to touch under monkeypatch —
turning the decode-gate into a native admission layer in the runtime, properly implementing session/sandbox isolation,
making planner/reducer first-class swarm roles, and making Step3.7's verbose-reasoning control a default.

---

## 3. Grounded facts (recon — the 5 pillars that constrain the design)

A 5-agent recon nailed down the HermesAgent internals (full line-cited version in `BUILD_SPEC.md`). These are
the premises of the swarm design and physics that won't change even in the fork:

1. **Sync + threaded (not asyncio).** LLM calls are synchronous `openai.OpenAI`. Each turn blocks an OS
   thread, but releases the GIL during socket I/O → dozens of generations overlap within one process.
   → swarm workers are **threads** (not coroutines). 100-200 of them = heap only (no subprocess needed).
2. **Stateless full-history resend.** Every turn resends the full history; there's no server-side session. The server
   **holds KV only for in-flight requests** → a worker executing a tool holds zero KV.
   → **"parking" is automatic**. enrolled ≫ KV-resident holds. KV-bound ≈ concurrent decode count.
3. **Tools are in-process; MCP doesn't spawn per-agent.** Per-instance cost ≈ heap only.
4. **A single chokepoint for LLM calls** = `AIAgent._interruptible_(streaming_)api_call`. Hold this point and you can
   gate + measure every generation (streaming is consumed eagerly to completion before returning).
5. **Necessary conditions for thread-per-worker** (TS1-7): a unique session_id per worker (collisions cause sandbox/cwd
   sharing accidents) / sandbox isolation (terminal/file tools collapse task_id to "default") / tool_delay=0 /
   non-interactive mode / cache prewarm / handling of `_last_resolved_tool_names`. **In the fork these are solved by real implementation, not monkeypatch.**

---

## 4. Measured operating point (the numeric basis for "maximum throughput")

The step37-harness engine, measured live over the **real HermesAgent path** (via DecodeGate, windowed 30s,
`results/operating_point.json`):

| Concurrent in-flight N | tok/s | occupancy | tok/s/agent | KV% | Synthetic baseline (FLEET_OPTIMUM §4) |
|---:|---:|---:|---:|---:|---|
| 16 | 947  | 1.00  | 59.2 | 11.3 | 652 / 11% |
| 32 | **1247** | 0.945 | 41.2 | 19.3 | 763 / 20% |
| 48 | 1246 | 0.919 | 28.2 | 27.9 | 922 / 30% |

- **The gate pins the server's `num_requests_running` to N** (occupancy 0.92-1.0) = proving we can precisely control the
  swarm's concurrent decode count.
- **On the real-agent path it reaches and exceeds the synthetic C32-64 region** (1247 tok/s at N=32).
- **For long real outputs the throughput knee is ~C32** (nearly flat from 32→48). Earlier than the C64 of short synthetic outputs.
  → it's reasonable to set the swarm's default decode target at **~32** (tracked by AIMD).
- **KV model validation**: measured KV% (11/19/28) matches the prediction (11/20/30) → demonstrating §3-2's "the gate bounds KV."
- **AIMD convergence**: from gate 12, a sawtooth converges to the knee (~C32-46), KV<30%, with no thrash/deadlock.
- A single Step3.7 ≈ 125 tok/s. Drawing out **~10× the throughput** "by default" via the swarm is the point of hermes-swarm.

---

## 5. Architecture concept (swarm-native)

### 5.1 Swarm on startup (boot)
Stood up permanently at process startup:
- **Board** (DAG queue, states pending/ready/running/done, dependencies, upstream-result injection). The substrate for stigmergy.
  Persistence is SQLite (restart-resilient, atomic claim, liveness-gated recovery — implemented & verified).
- **DecodeGate** (variable semaphore, lane-priority) = pins concurrent generation count = server KV.
- **ThreadFleet** (single-process bounded ThreadPool) = oversubscribes enrolled workers (default
  oversub×3, enroll≤256). What keeps the GPU full is an LLM-free admission loop.
- **AIMD controller** = tracks the gate to the knee from /metrics (running, kv, preemptions, gen_tokens, waiting).
- **prefix-warm** = pre-warms each role's system+tools prefix so even worker #1 gets a cache hit (48→97% measured).

### 5.2 Goal-driven front door (the adopted flow)
```
goal(1 line) ─▶ planner agent ─▶ task DAG (add to board) ─▶ worker swarm parallel execution
                                                                      │
                          reducer (fan-in over a tree, larger context closer to the root)◀┘ ─▶ final answer
```
- **planner (director lane)**: reads the goal and generates a dependency-annotated task list (id/prompt/deps/lane) to
  submit to the board. Bursty, high-value. This is the heart of the "automatic decomposition" the predecessor harness lacked.
- **worker (the crowd)**: lean toolset, disposable, C32 operating point.
- **reducer**: actually aggregates upstream results (it's been demonstrated on step37-harness that the reducer consolidates real results).
- **router**: classification/dispatch, nearly free.
- Coordination emerges from the board's "queue + dependency state" (no messages). We don't put a conductor in the hot loop
  (Amdahl: if a supervisor touches it serially, the ceiling gets bound).

### 5.3 decode-gate admission (made native)
We implement the consequence of recon §3-2 as a runtime admission layer: a worker acquires the gate before entering generation,
and holds no permit while executing tools. This precisely maintains "always ~32 decode" independent of duty (in the predecessor it was a
forwarder monkeypatch; in the fork it's part of the agent runtime). lane-priority keeps planner/reducer from getting buried in the
worker swarm (the KV waterfall).

### 5.4 What the fork makes possible (things impossible with external monkeypatch)
- **Eradicating gate-bypass**: route every LLM call — including iteration-limit summary generation (§7) — through a single admission
  layer. Under monkeypatch we gave up on this for fear of breaking the client/socket lifecycle (TS6).
- **Real session/sandbox isolation**: fix the behavior that collapses task_id to "default," giving each worker a truly independent
  cwd/bash/env (externally this was bolted on via register_task_env_overrides).
- **Make planner/reducer first-class roles**: define per-role system prompt, context budget, and toolset
  natively (externally these were faked via AIAgent arguments).
- **Default control of Step3.7 verbose reasoning**: set `reasoning_effort` and output boundaries per role (workers
  concise, planner deliberative). Externally this was an env-gated bolt-on.
- **Dissolving SessionDB serialization**: rebuild the shared sqlite conn+lock (TS4) into per-thread WAL/pool, removing the
  persistent-write bottleneck of 120 agents.
- **Maximizing single-process efficiency**: redesign on the premise that the model client, tool registry, and prewarm are shared across the whole swarm.

---

## 6. Reused assets (step37-harness, verified)

Implementations portable to the fork (all confirmed live):
- `compat.DecodeGate` … variable, lane-priority, interrupt-safe semaphore (phantom-ticket
  leak fixed under adversarial review).
- `engine.ThreadFleet` … oversubscribe + enrolled clamp (ENROLL_MAX outer-cap fixed) + daemon metrics sampler.
- `admission.AIMDController` … judges saturation from gate in_flight, EWMA + dwell + KV/waiting backoff.
- `metrics.py` … /metrics scrape (kv is MAX across labels, with the "don't sum" fix) + duty/throughput derivation.
- `board.SqliteBoard` … atomic claim, liveness-gated restart recovery, busy-retry, WAL checkpoint.
- `warm.warm_profiles` … role-prefix pre-warming.
- Measurement infra … `scripts/throughput_probe.py` (operating-point reproduction), `scripts/aimd_probe.py` (convergence),
  `scripts/gen_tasks.py` (~8K task generation, tokenizer-calibrated).

→ The shortest path for hermes-swarm is **not to reimplement the engine, but to embed these into the fork's runtime and add the planner front
door and swarm-native boot**.

---

## 7. Known holes the fork closes (the predecessor's leftovers)

- **gate-bypass (review #9)**: the iteration-summary generation at `chat_completion_helpers.py:1433/1476` is
  ungated, bypassing the forwarder. Deferred in the predecessor as too risky for monkeypatch. **In the fork, resolved by integrating into the admission layer.**
- **iteration-limit / thinking-only loops**: Step3.7 is a verbose reasoner (>10K tok generation). An output cap alone gets stuck on
  continuation / thinking-only re-prefill. **In the fork, make per-role reasoning_effort and boundaries the default.**
- **Sandbox non-isolation**: cwd/bash sharing accidents among tool-using workers. **Per-task isolation by real implementation.**
- **SessionDB serialization**: 120 agents' persistence stalls on a shared conn. **Made per-thread.**

---

## 8. Risks and open questions

- **Upstream divergence**: the fork is merged manually. Heavy changes in the hermes upstream add follow-up cost (the more you touch the hot path, the more it hurts).
  → design to localize the touched surface to the admission layer + role definitions + boot, using the upstream core as-is as much as possible.
- **Planner quality**: decomposing goal → a good DAG depends on the planner's skill. If the decomposition is shallow / cyclic, the swarm spins idle
  (deadlock detection lives in the engine, but a planner self-verification loop is needed).
- **C96+**: still unmeasured due to fixed-window ramp contamination. Gate MAX is 96 for now. An N-proportional warm-up is needed.
- **Quality vs. speed**: whether skip_memory/skip_context/lean toolset degrade worker output quality is not yet quantified.
- **GPU contention with the daily driver**: while the swarm is running it saturates the Step3.7 server. A policy for co-running with production hermes
  (time slots / separate port / reserved gate) is needed.

---

## 9. Phases (concept-level roadmap)

1. **Fork & port**: fork hermes-agent, integrate step37-harness's 6 engine pieces (§6) into the runtime,
   `hermes-swarm` startup = boot (board+gate+fleet+aimd standing permanently).
2. **Goal-driven front door**: planner role (goal→DAG) + reducer aggregation + final answer. `hermes-swarm "<goal>"`.
3. **Native admission**: decode-gate into the runtime layer, eradicate gate-bypass (§7), real sandbox/session isolation.
4. **Role-native**: define system prompt, context budget, and reasoning_effort by default for planner/worker/reducer/router.
5. **Operations**: contention policy with daily hermes, observability (live operating-point dashboard), restart resilience.
6. **Validation**: re-measure §4's operating point on the fork, planner decomposition quality, end-to-end goal→report.

---

## 10. Provenance (where the real data lives)

- Grounding (HermesAgent internals): `~/projects/step37-harness/BUILD_SPEC.md` (5-agent recon, line-cited).
- Measured operating point: `~/projects/step37-harness/results/operating_point.json` + `scripts/throughput_probe.py` /
  `aimd_probe.py`. Synthetic baseline: `~/bench/step37-mtp/FLEET_OPTIMUM.md`.
- Verified engine: `~/projects/step37-harness/fleet/` (compat/engine/admission/metrics/board/warm).
- Design and intent: `~/projects/step37-harness/DESIGN.md` (§5 implementation status / §6 measurements).
- Adversarial review and fixes: end of DESIGN.md §5 (6 CRITICAL/MAJOR perspectives).
