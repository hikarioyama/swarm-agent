# step37-harness — Design Decisions and Current State (DESIGN)

A summary of the **current state** and the **intent behind each decision** for the
harness that runs Step-3.7-Flash efficiently on 2× RTX PRO 6000 with "tens to hundreds
of agents in parallel." All numbers are measurement-derived
(`~/bench/step37-mtp/FLEET_OPTIMUM.md` + live measurement).

---

## 0. The Problem to Solve

- A single Step-3.7 stream is **~125 tok/s**. The only way to exploit the GPU is to keep **tens of agents in-flight simultaneously**
  (running one at a time throws away ~90% of the capability).
- The essential difficulty is **not throughput but "governing many parallel agents efficiently."**
  - A plain main-sub setup overloads the main (context explosion, conc=1 decode as the bottleneck, having to track N states at once).
  - Mimicking an agent team is also wrong (N² communication, shared state = the enemy of parallelism).

---

## 1. The Real Picture of Hardware and Model (measured)

| Item | Value | Notes |
|---|---|---|
| GPU | 2× RTX PRO 6000 Blackwell, 96GB×2 = 192GB | TP=2 |
| Model | StepFun Step-3.7-Flash, 198B MoE, **NVFP4 ~116GB** | Weights occupy most of VRAM |
| KV cache | **1,625,950 tokens** (fp8, ~41GB) | gpu-util 0.92 |
| Speculative decode | **MTP K=1**, acceptance ~0.79 | +14–44% across all parallelism |

**Why the KV is "light" (important)**: Step-3.7 uses **hybrid attention** — of its 45 layers,
**only 12 are full attention; the remaining 33 are sliding-window**. So full KV is only accumulated for those 12 layers,
at **~24KB/token**. A normal 45-layer full GQA would be ~90KB/token (3.7× heavier), fitting only ~460K tokens.
→ **The KV is not inefficient; the model is simply huge (116GB)**. 1.6M is reasonable. To increase it you can only free VRAM
(raise gpu-util / offload the display to the RX 9070 XT).

---

## 2. The Operating Point Established by Measurement (FLEET_OPTIMUM.md)

- **The efficient in-flight region for a worker (~8K context) = C32–C64**:
  C32=763 tok/s (23.8 tok/s/agent, ~6× single) / **C64=1225 tok/s (~9.8× single)**.
  Welch's test shows a significant increase from C16 up to C64. C96+ is **out of range** due to ramp contamination under the fixed window (B* is treated as plateauing at 64).
- **A worker context of ~8K is optimal**: 16K is ~18% slower, fills KV twice as fast, and plateaus at C32; 32K collapses at ~C8.
  → **The smaller the context is kept, the higher the efficiency** (when it grows, rein it in with compaction).
- **MTP is best at K=1** (K1>K2>K3; high K yields no gain because per-position acceptance drops).
- The server configuration is **already optimal; no change needed**.

> Note: the measurement is a conservative floor reached over 7 runs and 6 rounds of Codex review. Because it is a "worst case" with prefix-cache OFF (unique ctx per worker) and churn-trough included,
> a real fleet (shared prefix + cache) will perform better.
>
> **v0.2 measured the real-agent path → table in §6**. It exceeds the synthetic numbers above (httpx probe), reaching 1247 tok/s at N=32.
> For long real outputs the **knee is ~C32** (C48 is flat) — earlier than the synthetic curve above (which keeps rising to C64). The bottleneck is compute/DecodeGate, not KV.

---

## 3. Architecture Decisions and Intent

### 3.1 Stigmergic Coordination (via board) — No Conductor
**Intent**: take the smart main out of the hot loop. Amdahl — if a supervisor touches even 10% of the work serially, the ceiling is 10×.
- Workers talk neither to each other nor to a center. Coordination flows through a shared **Board** (DAG queue): claim task → write result → unlock dependents.
- Coordination emerges from "queue + dependency state" rather than from "messages" → the main's context explosion and serial supervision disappear.
- Implementation: `fleet/board.py` (states pending/ready/running/done/failed, dependencies, upstream-result injection).

### 3.2 A Zero-Intelligence Fast Dispatcher — Cycling N Agents Is a while Loop
**Intent**: what keeps the GPU full is admission-control code (no LLM needed), not an agent.
- Implementation: `fleet/scheduler.py`. Keeps TARGET_INFLIGHT agents always running, pulls ready tasks, and submits them to the process pool.

### 3.3 Lean, Prefix-Stable Workers — **The Biggest Lever (verified)**
**Intent**: giving every worker 39 tools + all MCP is a massive redundant prefill. Load only the minimal tools per role.
- Measured: the default 39 tools = **14,113 tok** (identical schema) re-prefilled **every time** by ~40 agents, with fleet prefix-cache **hit 0%**
  (because each worker's context-files/memory breaks the alignment of the system block).
- Fix (`AIAgent(enabled_toolsets=…, skip_context_files=True, skip_memory=True)`):
  coder[file,terminal,search]=3,328 tok (-77%) / researcher[web,search]=398 (-97%) / reducer[]=minimal.
  Same-role agents share a **byte-identical minimal prefix** → vLLM's auto prefix-cache hits from worker #1 onward → **effective -98%**.
- This is **the biggest immediate-effect lever**: a config change only (HermesAgent unmodified; AIAgent already accepts all the arguments).
- Implemented: `fleet/config.py` TOOL_PROFILES + `fleet/worker.py`.

### 3.4 A Heterogeneous Roster (KV Portfolio) — Not Everyone on the Same Context
**Intent**: different roles need different contexts. KV is a shared budget (1.6M tokens). Reserve a few large-context roles and make the worker lane an elastic basin.
**"The same ctx for everyone is optimal" is wrong** (as the user pointed out).
- The single-context curves (1K/8K/32K) are the **ingredients** for each lane; the fleet optimum is the **mixture** of them.
- Implementation: `fleet/config.py` ROSTER + `fleet/roster.py` (KV budget check). Details in §4.

### 3.5 Disposable Narrow-Context Workers — State Lives on the Board
**Intent**: short ctx = fits in the KV budget + sits on the favorable side of the throughput curve. Do not accumulate state in a worker's context.

### 3.6 Confine Admission to the decode-batch + duty oversubscription (designed, not implemented)
**Intent**: in-flight ≠ decoding. A worker waiting on a tool is not using the GPU.
- To keep ~40 agents **always decoding**, enrolled = B*/duty means **over-enrollment** (at duty 0.4, ~120 enrolled for ~48 decoding).
- Closed-loop control of B* via /metrics (num_requests_running, kv_cache_usage_perc, num_preemptions_total),
  with AIMD backoff when KV>85% / preemptions rise.
- **Verification caveat**: the mechanism is correct, but "+60-90%" is an **overestimate**. Real duty is unmeasured (~0.88 for light tools,
  dropping sharply with heavy browser/code use). Measuring real duty is the key to accuracy.

### 3.7 Bundle Worker Context at ~8K
**Intent**: as in §2, 16K/32K lose efficiency. When multi-stage work grows the context, hold it down to 8-16K with summarize/compaction.

---

## 4. Provisional Roster (current configuration)

The allocation against a KV budget of 1,625,950 tokens (reproduce with `python -m fleet.roster`):

| Role | context | count | tools | duty | Purpose |
|---|---|---|---|---|---|
| **director** | 128K | **1** | [todo] | 0.15 | Steers long-term direction. Holds goal+plan+state, **via the board**, not aboard the hot loop. The user's conversation partner |
| planner | 32K | 2 | [todo] | 0.5 | goal→DAG decomposition. Bursty |
| reducer | 16K | 6 | [] | 0.7 | Aggregates over a tree; grows the closer to the root |
| **worker** | 8K | **48** (in-flight) | [file,terminal,search] | 0.4 | **The swarm**. Disposable, lean, C32-64 operating point |
| router | 2K | 16 | [] | 0.2 | Classify/route, nearly free |

- KV: in-flight **44%** / enrolled (under duty over-enrollment) **96%** = fits (just barely).
- **One director at 128K = 8% of the KV budget**. At duty 0.15 it barely decodes → it merely holds KV without consuming GPU bandwidth.
- The worker "48" is the in-flight target. Depending on real duty, adjust enrolled to 60–130 to sustain "always ~48 decoding."

**The key intent**: what hits the GPU is the sum of `count×duty` across roles (= simultaneous decode). The 48 workers make up the bulk,
so **the whole aims at the measured knee (~48-64)**. The director **does not directly monitor the 48 workers** (that would overload it);
it reads the board's summary and updates direction = stigmergy.

---

## 5. Current Implementation Status

**Implemented (operation confirmed)**:
- `board.py` — DAG queue, dependencies, upstream-result injection (demonstrated that the reducer aggregates real results)
- `scheduler.py` — the admission loop over ProcessPoolExecutor (fixed inflight)
- `worker.py` — disposable AIAgent worker + **per-role lean toolset** (measured ~93% prefill reduction)
- `config.py` — measured operating point + TOOL_PROFILES + ROSTER
- `roster.py` — KV budget checker
- `cli.py` — `python -m fleet.cli tasks.jsonl --inflight N`
- `plugin/` — HermesAgent's `/fleet` command (symlinked into `~/.hermes/plugins/`, upstream repo untouched)

**Implemented and live-verified in v0.2 (full roadmap consumed)**:
| # | Item | Implementation | Status |
|---|---|---|---|
| 1 | Lean, prefix-stable workers | `compat.make_agent` per-role toolset | ✅ Done (prefill -77 to -98%) |
| 2 | decode-batch admission + AIMD | `compat.DecodeGate` (variable semaphore) + `admission.AIMDController` | ✅ The gate pins the server's `num_requests_running` to N (occupancy 0.92-1.0); AIMD converges at the knee (~C32) |
| 3 | Workers return decode_s/tool_s for real duty measurement | forwarder monkeypatch + `worker.run_task_local` | ✅ Measured (59/41/28 tok/s/agent @ N16/32/48) |
| 4 | Per-lane admission (KV waterfall) | `DecodeGate` lane-priority acquire (`config.LANE_PRIORITY`) | ✅ High lanes get priority and are not buried by the worker swarm |
| 5 | ~~Single-process asyncio worker~~ → **single-process ThreadPool** | `engine.ThreadFleet` (ThreadPoolExecutor) | ✅ Recon correction: HermesAgent is **sync+threaded** (no asyncio path). Per-instance state is heap only |
| - | prefix warm (pre-warm role #1 for each role) | `warm.warm_profiles` | ✅ role prefix hit-rate 48%→97% |
| - | parking (KV eviction while waiting on a tool) | **Unnecessary (automatic)** | ✅ Due to stateless full-history resend, KV is zero while a tool runs. Recon dissolved the parking mechanism |
| - | SQLite persistent board | `board.SqliteBoard` + `open_board` | ✅ atomic claim + liveness-gated restart recovery |

> Added in v0.2: `compat.py` (DecodeGate + non-invasive monkeypatch), `metrics.py` (/metrics + duty), `admission.py` (AIMD),
> `engine.py` (ThreadFleet), `warm.py`, `board.py` (SqliteBoard), `BUILD_SPEC.md` (the implementation contract grounded in recon).
> A 6-agent adversarial review found CRITICAL/MAJOR issues → all fixed (gate phantom-ticket leak, inverted enrolled clamp,
> kv label cross-sum, AIMD saturation/EWMA, board write-guard + reset liveness, sandbox isolation).
> The hermes-agent repo is untouched (all adaptation is runtime monkeypatch in `compat.py`).

---

## 6. Resolved Questions + Open Items (v0.2 live measurement)

**Resolved by v0.2 measurement (the real HermesAgent path measured over a windowed 30s via DecodeGate, `results/operating_point.json`)**:

| N | tok/s (measured) | mean_running | occupancy | tok/s/agent | KV% | Synthetic baseline (FLEET_OPTIMUM §4) |
|---:|---:|---:|---:|---:|---:|---|
| 16 | **947** | 16.0 | 1.00 | 59.2 | 11.3 | 652 / 0.97 / 11% |
| 32 | **1247** | 30.24 | 0.945 | 41.2 | 19.3 | 763 / 0.96 / 20% |
| 48 | **1246** | 44.12 | 0.919 | 28.2 | 27.9 | 922 / 0.96 / 30% |

- **Real throughput / operating point**: the real-agent path **reaches the C32-64 region and exceeds the synthetic numbers** (1247 tok/s at N=32 = synthetic C64 class).
  **New finding: for long real outputs the throughput knee is ~C32** (1247→1246, so 32→48 is nearly flat) — earlier than the short-output synthetic curve (C64).
- **Real duty cycle**: per-agent decode measured (59/41/28 tok/s/agent). The "unmeasured" gap is resolved.
- **KV model validation**: the measured KV% (11/19/28) closely matches the prediction (11/20/30) → the claim that **the gate bounds KV** is demonstrated.
  Due to stateless full-history resend, **KV is zero while a tool runs = automatic parking** (enrolled ≫ KV-resident).
- **AIMD convergence (criterion D)**: the gate grows additively from 12 and sawtooths at the knee (~C32-46), safe at KV<30%, with no deadlock/thrash.

**Open items**:
- **C96+**: still out of range due to fixed-window ramp contamination (an N-proportional warm-up is needed). DecodeGate MAX is 96.
- **Quality impact of skip_context_files/skip_memory**: assumed safe because the board holds state, but unquantified.
- **Known gate-bypass gap**: iteration-limit summary generation (`chat_completion_helpers.py:1433/1476`) does not go through the forwarder
  and is ungated (review #9, deferred). It does not fire on no-tool single-turn; rare and only a short single generation.
- **The value of increasing KV**: in the measured region (C32-64) KV reaches only 28% and is not the bottleneck (the bottleneck is DecodeGate/compute). Raising gpu-util / offloading the display is a separate matter.

---

## 7. Provenance

- Operating point, throughput curves, MTP, context regime: `~/bench/step37-mtp/FLEET_OPTIMUM.md`
  + measurement infra (`steady_probe.py`, `analyze_steady.py`, `fleet_sweep.py`). 6 rounds of Codex review.
- The 5-dimensional efficiency analysis (lean = biggest lever verified, duty controller, KV lanes, etc.): the efficiency workflow (9 agents).
- This harness: `~/projects/step37-harness/` (independent git). The plugin is symlinked into `~/.hermes/plugins/`.
