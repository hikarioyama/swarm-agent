# step37-harness v0.2 — BUILD SPEC (recon-grounded, single source of truth)

This spec turns the DESIGN.md roadmap into an implementable contract, **corrected by
a 5-agent recon of the real HermesAgent internals** (cited inline as `file:line`).
Every implementer module MUST honour the interfaces and constraints here.

> Hard rule: **never modify the hermes-agent repo on disk.** All adaptation is runtime
> monkeypatching applied from `fleet/compat.py`. The harness stays git-pull-safe.

---

## 0. Recon verdict — the four facts that reshape the design

1. **HermesAgent is SYNC + THREADED, not asyncio.** The LLM call uses the *synchronous*
   `openai.OpenAI` client; every turn blocks an OS thread (`agent/chat_completion_helpers.py`
   sync `chat.completions.create`). So roadmap **#5 "single-process asyncio worker" is really
   a single-process THREAD POOL**: each worker is a thread, GIL is released during socket I/O,
   so dozens of generations overlap. There is no coroutine path to await.

2. **Stateless full-history resend.** Each turn rebuilds and resends the ENTIRE transcript
   (`agent/conversation_loop.py:1194` `_build_api_kwargs`, :919-961). There is **no server-side
   session**, so the server holds KV **only while a request is generating**. A worker executing
   a tool (between turns) holds **zero** server KV. Consequences:
   - **"parking" (roadmap) is automatic** — KV-resident ≈ count of *currently generating*
     requests, not enrolled agents. No eviction mechanism needed.
   - **KV pressure ≈ concurrent decode count × per-turn prompt** → bound it by bounding
     concurrent generations (the decode gate, §3).
   - `enrolled ≫ KV-resident` is natural → duty oversubscription is free.

3. **Tools run IN-PROCESS; no per-agent MCP / event-loop / subprocess at idle.** Tool dispatch
   is a plain Python call (`tools/registry.py:390-404`); MCP is process-global and only if
   configured (`tools/mcp_tool.py:3342`). Per-AIAgent cost ≈ Python heap only; transient child
   processes appear only while a terminal/search/code tool actually runs. → 100-200 instances as
   threads in ONE process is viable (vs. the old ProcessPool's ~0.8 GB/proc).

4. **The LLM call funnels through two instance methods** — the clean instrumentation + gating
   chokepoint for BOTH streaming and non-streaming and all api_modes:
   - `run_agent.AIAgent._interruptible_api_call(self, api_kwargs)` — `run_agent.py:3277`
   - `run_agent.AIAgent._interruptible_streaming_api_call(self, api_kwargs, *, on_first_delta=None)` — `run_agent.py:3448`
   The streaming one **consumes the SSE stream eagerly** (`agent/chat_completion_helpers.py:1758`
   `for chunk in stream:`) and returns the assembled response, so **wrapping it brackets the entire
   generation window** — exactly what a decode gate + decode_s timer need.

---

## 1. Thread-safety contract (MUST satisfy before fanning out threads)

From recon agents `thread-safety-and-state` + `tool-execution-and-mcp` + `model-call-path`:

| # | Hazard (cite) | Required mitigation (in harness) |
|---|---|---|
| TS1 | Auto session_id = `ts-sec + 6-hex-uuid4` → ~0.07%/200·s⁻¹ collision; collision shares sandbox/cwd file/process_registry (`agent_init.py:980-982`, `tools/environments/base.py:319`) | **Pass a globally-unique `session_id` to every `AIAgent(...)`** (init HAS a `session_id` param — verified). Also `set_current_session_id(uid)` in the worker thread so `get_session_env` reads the right ContextVar (`gateway/session_context.py:97,178`). |
| TS2 | `model_tools._last_resolved_tool_names` is a process-global list reassigned by every `get_tool_definitions()` and read by `execute_code` (`model_tools.py:213,835`) → cross-agent toolset bleed | **compat:** replace with a thread-local-backed proxy so each worker reads its own. Only `code` lane calls execute_code, but patch unconditionally. |
| TS3 | `_tool_defs_cache` / model_metadata caches unlocked → 120-thread cold-start stampede (`model_tools.py:254`, `agent/model_metadata.py:105-112`) | **Pre-warm** tool defs for every role profile + the OpenAI class import **once before fan-out** (compat.prewarm + warm.py). |
| TS4 | Single shared SessionDB sqlite conn+lock serializes all session/KV persistence (`hermes_state.py:373-376,430,473`) → process-wide ceiling | Workers run `skip_memory=True`, `skip_context_files=True`, `save_trajectories=False`. If persistence still bottlenecks under load, switch to per-thread WAL connections (measure first; document if accepted). |
| TS5 | Interactive sudo path mutates global env + bangs the TTY (`tools/terminal_tool.py:397,443`) | Run headless (no TTY in worker threads). **Unset `HERMES_KANBAN_TASK`** (it injects kanban tools into even empty profiles → breaks the zero-tool guarantee for router/reducer). `quiet_mode=True`. |
| TS6 | Per-request OpenAI+httpx client built & closed per call; FD-reuse-after-close race worsens at high concurrency (`run_agent.py:2868`, `chat_completion_helpers.py:166-215,217-267`) | Don't touch client/socket lifecycle in our patches (respect owner_tid discipline — wrap only AROUND the forwarder, never inside the client). Raise FD/NPROC ulimits in the launcher. |
| TS7 | Thread-local async-tool event loops accumulate; transient subprocess/FD storm if many tools fire at once (`model_tools.py:62-81`; `tools/environments/local.py:521-533`) | **Reuse** worker threads (bounded pool), don't churn. Cap concurrent *enrolled* workers; the decode gate caps generations separately. |

**Bug fix (independent of threading):** `config.WORKER_MAX_TURNS=12` exists but `worker.py` never
passes it → workers run at `max_iterations` default **90** (`agent_init.py`; contract recon).
**Pass `max_iterations` into `AIAgent(...)`.**

---

## 2. Architecture (v0.2) — corrected & unified

```
                 Board (DAG, in-mem default | SQLite persistence backend)
                                 │ claim_ready(n)  / lane-aware
            ┌────────────────────┴─────────────────────────┐
            │  Engine (single process, ThreadPool)          │
            │  • enrolled = oversub_factor × gate_limit     │  ← many threads, cheap (heap only)
            │  • each thread: Worker.run(task) → AIAgent     │
            └───────────────┬───────────────────────────────┘
                            │ every LLM generation passes through…
                  ┌─────────┴──────────┐
                  │  DecodeGate (compat)│  resizable semaphore, lane-priority acquire
                  │  permits == #concurrent generations == server num_requests_running
                  └─────────┬──────────┘
        AIMD Admission ◀────┤ resize(limit) from /metrics:
        controller          │   throughput knee ↑ while KV<85% & preempt stable
        (thread)            │   ×backoff on KV>85% / preempt jump / throughput drop
                  Metrics sampler ──▶ num_requests_running, kv_cache_usage_perc,
                                      num_preemptions_total, generation_tokens_total (tok/s)
```

- **decode-batch admission (DESIGN §3.6) is realized by the DecodeGate**, not by guessing duty.
  The gate pins `num_requests_running ≈ limit`; tool-executing workers wait on the gate, holding
  no server KV (fact #2). This *is* the unification of #2/#3/#4/§3.6.
- **#4 KV-waterfall lanes** → the gate serves waiters **highest-lane-priority first**
  (director > planner > reducer > worker > router), so reserved roles never starve behind the
  worker swarm. (Hard per-lane min-reservations are a documented refinement; priority acquire is v1.)

---

## 3. Module contracts (implement exactly these signatures)

### `fleet/config.py` (extended — AUTHORED IN FOUNDATION)
New/!changed keys (env-overridable). See the file; do not redefine.
- `MAX_ITERATIONS` (worker turn budget; default 12 — fixes the bug).
- `DECODE_GATE_*`: `ENABLED`, `START`, `MIN`, `MAX`, target band `KNEE_LO/KNEE_HI`.
- `OVERSUB_FACTOR` (enrolled = factor × gate_limit), `ENROLL_MAX`.
- `AIMD_*`: `STRIDE` (additive), `BACKOFF` (multiplicative), `KV_HI`, `INTERVAL_S`.
- `LANE_PRIORITY` (dict lane→int, higher = served first).
- `BOARD_PATH` (None = in-memory; else SQLite file).

### `fleet/compat.py` (AUTHORED IN FOUNDATION) — exports:
- `class DecodeGate`: `__init__(limit)`, `set_limit(n)`, `get_limit()`, `stats()->dict`,
  `acquire(lane="worker")` context manager (lane-priority, 0.5s re-check, never deadlocks when
  limit>0), per-instance counters (`acquired_total`, `wait_s_total`).
- `apply(gate: DecodeGate|None) -> None`: idempotent; (a) wrap the two forwarders to
  `with gate.acquire(self._fleet_lane): t0=…; try: return orig(); finally: self._fleet_decode_s+=…`
  (decode_s = generation only, excludes gate wait → tracked as `_fleet_gatewait_s`); (b) make
  `model_tools._last_resolved_tool_names` thread-local; (c) env hygiene (TS5). Safe if gate=None
  (timing only, no gating).
- `prewarm(profiles: list[list[str]]) -> None`: resolve tool defs per profile + warm OpenAI import,
  once, to kill the cold-start stampede (TS3).
- `make_agent(lane, *, base_url, api_key, model, ...) -> AIAgent`: constructs an AIAgent with the
  fleet-safe kwargs (unique `session_id`, `enabled_toolsets=toolsets_for(lane)`,
  `skip_context_files=True`, `skip_memory=True`, `save_trajectories=False`, `quiet_mode=True`,
  `max_iterations=config.MAX_ITERATIONS`), stamps `agent._fleet_lane=lane`, and sets the session
  ContextVar. Returns the agent. (Workers call this; keeps all TS mitigations in one place.)

### `fleet/metrics.py` (AUTHORED IN FOUNDATION) — exports:
- `scrape(url) -> dict|None`: parse vLLM prometheus text → keys: `running`, `waiting`,
  `kv` (0..1), `preemptions` (cumulative), `gen_tokens` (cumulative). Robust to missing lines.
- `class ThroughputMeter`: `update(scrape) -> tok_s|None` (differences gen_tokens over wall time).
- `class DutyIntegrator`: samples `running` over time → `mean_running`, and given enrolled count →
  fleet duty. Thread to be driven by the admission controller (no own thread needed).

### `fleet/admission.py` (CORE agent) — exports:
- `class AIMDController(Thread)`: `__init__(gate, metrics_url, cfg, on_sample=None)`. Loop every
  `AIMD_INTERVAL_S`: scrape; compute tok/s; **increase** `gate.set_limit(limit+STRIDE)` while
  `running >= limit*0.9` (gate saturated) AND `kv < KV_HI` AND preemptions stable AND tok/s not
  regressing; **back off** `limit*BACKOFF` on `kv>=KV_HI` OR preemption jump OR tok/s drop >X%.
  Clamp `[DECODE_GATE_MIN, DECODE_GATE_MAX]`. `start()/stop()`. Emit samples via `on_sample` for
  logging. **Must never set limit below MIN (workers would block forever).**

### `fleet/engine.py` (CORE agent) — exports:
- `class ThreadFleet`: `__init__(board, gate, cfg, on_event)`. Runs a single-process bounded
  ThreadPool. Maintains `enrolled = clamp(OVERSUB_FACTOR*gate.get_limit(), …, ENROLL_MAX)` workers
  in flight; claims lane-aware ready tasks; submits `Worker.run`; writes results to board (unlocks
  deps); requeues on failure (`MAX_RETRIES`). No central LLM agent on the hot path. `run()->summary`.
  Replaces the ProcessPool path; keep ProcessPool engine available via a flag for A/B.

### `fleet/worker.py` (CORE agent, rewrite) — exports:
- `run_task(spec) -> dict` (picklable result preserved for ProcessPool compat) AND an in-process
  `run_task_local(spec)` that uses `compat.make_agent`. Returns `{id, completed, text, api_calls,
  wall_s, decode_s, tool_s, gatewait_s, turns}`. decode_s/tool_s from the instance counters set by
  compat. Inject upstream `dep_results` into the prompt as today.

### `fleet/scheduler.py` (CORE agent, refactor)
- Keep `Scheduler` (ProcessPool, fixed inflight) for fallback/A-B. Add selection so `cli` can pick
  `--engine thread|process` and `--admission static|aimd`.

### `fleet/cli.py` (CORE agent, update)
- Flags: `--engine {thread,process}` (default thread), `--admission {static,aimd}` (default aimd),
  `--gate N` (start limit), `--no-gate`, `--warm/--no-warm`, `--board PATH`. Pretty progress incl.
  live `running/limit/kv%/tok_s`.

### `fleet/board.py` (LEAF-board agent, extend)
- Keep the in-memory `Board` API **unchanged**. Add a `SqliteBoard` (or pluggable backend) with the
  SAME public methods (`add/add_many/claim_ready/complete/fail/counts/unfinished/has_ready/results`)
  backed by a SQLite file (WAL) for restart-safety + multi-producer. `claim_ready` must be atomic
  across processes (UPDATE … WHERE state='ready' RETURNING, or a claimed-flag txn). Factory:
  `open_board(path|None)`.

### `fleet/warm.py` (LEAF-warm agent)
- `warm_profiles(roles, *, base_url, model, api_key) -> dict[role,float]`: for each distinct role
  profile, send ONE tiny request through `compat.make_agent(role)` (e.g. "ok") so vLLM prefix-caches
  that role's system+tools prefix → worker #1 of each role hits cache. Return per-role warm latency.
  Must call `compat.apply(None)`+`compat.prewarm(...)` first. Idempotent, fast.

### `fleet/roster.py` (LEAF-roster agent, correct the KV model)
- Rewrite the KV view to the **decode-resident** model (fact #2): resident KV ≈
  `Σ_lane (concurrent_decoding_lane × per_turn_tokens_lane)` where concurrent_decoding is bounded by
  the gate, NOT by enrolled. Show: gate budget vs KV; per-lane priority; and the OLD enrolled-KV view
  labelled "pessimistic (pre-stateless-insight)". Keep `python -m fleet.roster` runnable.

---

## 4. Acceptance criteria (what "optimal, verified" means — Phase 3 live)

Against the live `:8001` Step-3.7 server (load OK):
- **A. Correctness:** `examples/tasks.jsonl` completes via the thread engine; reducer receives real
  upstream results; 0 failures; results match the process engine.
- **B. Concurrency real:** with gate=N, server `num_requests_running` tracks ≈ N (±) under a worker
  swarm; thread engine holds the same N decoding with **far lower RAM** than N processes.
- **C. Operating point:** sustained fleet throughput in the measured **C32–C64** region
  (≥ ~760 tok/s at N=32, rising toward N=64) — i.e., we reproduce FLEET_OPTIMUM live through the
  REAL agent path, not just the synthetic probe.
- **D. Dynamic admission:** AIMD converges the gate into the knee band and **backs off** when KV>85%
  / preemptions rise (induce it by pushing the limit high) without thrashing or deadlock.
- **E. Duty measured:** report real per-worker `decode_s/tool_s` and fleet duty = mean(running)/enrolled
  — closes DESIGN §6 "実 duty 未測".
- **F. Prefix-warm:** worker #1 of each role shows a prefix-cache hit (server `prefix_cache_hits_total`
  rises on warm; first real worker's prefill is cheap).
- **G. Restart-safety:** SQLite board survives a mid-run kill and resumes remaining tasks.

Record all numbers; update DESIGN.md §5/§6 and README from hypotheses → measured.

---

## 5. Non-goals / guardrails
- No edits under `~/.hermes/hermes-agent/`. Patches live in `fleet/compat.py` only.
- Decode gate must be **toggleable** (`--no-gate`) so we can A/B it and fall back safely.
- Keep the ProcessPool engine working (fallback + cross-check for criterion A).
- Everything env-overridable; defaults encode the measured operating point.
