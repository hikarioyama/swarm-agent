# Swarm v3 — stigmergic coordination ("0 brain", chemical board)

> Status: **design locked** (5-round Opus↔Codex sparring, 2026-06-10). Commit-1 in flight.
> This doc is the implementation spec. Everything here is **additive + env-gated +
> fail-soft**: with all `SWARM_V3_*` flags off the system is **byte-identical to v2**
> (today's centralized-DAG behavior). A v3 path that cannot prove that identity is a bug.

## 0. Why (the ceiling v3 removes)

v2 is a **single-conductor DAG**: the planner decomposes a goal once, the `Board`
(`fleet/board.py`) hands ready tasks to a worker pool, the reducer takes a majority.
Workers are dumb executors that do not perceive each other. Two consequences:

1. **Herding** — N homogeneous workers fed the same context converge on the same
   *surface* hypothesis. A misleading majority (3 shallow signals → a decoy) beats one
   decisive-but-quiet signal → the reducer ships the decoy.
2. **No leaderless judgment** — "is this consensus real, or just correlated?" is never
   asked. 50 agreeing workers that share a blind spot are still wrong.

The design principle the sparring converged on:

> **meaning is local, metabolism is global, judgment is adversarial, memory sleeps.**

The base layer never reads task *meaning* (that would resurrect the conductor). It only
runs "physics": signal decay, priority, quorum counting. Meaning enters only through what
workers post and what verifiers refute.

## 1. The five mechanisms (and what ships when)

| # | Mechanism | Ships in | Extends |
|---|-----------|----------|---------|
| 1 | **Chemical board** — every result carries `chem` metadata (hypothesis/stance/evidence/confidence/contradictions); ready tasks are claimed in `priority_score` order | **commit-1** | `fleet/board.py`, `fleet/v3.py` |
| 2 | **Diversity quorum** — a reducer does not accept a single-stance majority; if stance diversity is too low it spawns a `contrarian`/`referee` and waits | **commit-1** | `fleet/board.py`, `fleet/engine.py` |
| 3 | **Reflex↔cortex escalation** — cheap reflex handles low-uncertainty/high-strength work; stale / contradictory / low-diversity-high-confidence escalates to a deep `referee` | **commit-2** | `swarm_agent/audit.py`, `manager.py` |
| 4 | **Hebbian credit** — a verified-correct (profile × domain) route is reinforced in a JSON sidecar and biases later referee-profile routing | **commit-3** | `fleet/v3_credit.py`, `fleet/board.py` |
| 5 | **Sleep consolidation** — repeated trap fingerprints compact into avoid-rules that down-weight known decoy stances during quorum/reduce | **commit-3** | `fleet/v3_sleep.py`, `fleet/v3.py`, `fleet/board.py` |

**Cut (not in v3):** full token-economy / GPU-tax, contract-net bidding, Context-DNA
breeding, weather-field board, jury-of-the-dead, predictive shadow-board. Reasons: base
rewrite, unproven convergence, or prefix-cache conflict. Revisit only after commit-1 earns
its keep on the live benchmark.

This doc specifies commit-1 through commit-3. The herding trap is defeated by 1+2, reflex
cuts unnecessary cortex passes in commit-2, and commit-3 adds learned referee routing plus
sleep-consolidated trap memory without changing the all-off path.

## 2. `fleet/v3.py` — the pure core (new file)

Self-contained, no swarm/fleet imports beyond stdlib. Pure functions + flag resolver, so
it is trivially unit-testable and import-safe. **All defaults OFF.**

### 2.1 Flags

```
SWARM_V3=0|1                 master switch (off ⇒ every sub-flag forced off)
SWARM_V3_CHEMICAL=0|1        mechanism 1 (priority claim + signal recording)
SWARM_V3_DIVERSITY=0|1       mechanism 2 (diversity quorum / referee spawn)
SWARM_V3_REFLEX=0|1          mechanism 3 (reflex triage before cortex)
SWARM_V3_HEBBIAN=0|1         mechanism 4 (learned referee-profile routing)
SWARM_V3_SLEEP=0|1           mechanism 5 (trap-fingerprint suppression)
```

Resolver contract: `v3.enabled("chemical") -> bool`. Returns False unless **both** the
master `SWARM_V3` and the sub-flag are truthy. Read once per process via a cached resolver
(env read is cheap but cache so a hot loop never re-parses). A helper `v3.any_on() -> bool`
lets callers early-out to the exact v2 code path when nothing is enabled.

### 2.2 The `chem` signal (what a worker result carries)

A worker's textual result MAY embed a fenced JSON block the harness parses into a `chem`
dict. Parsing is **fail-soft**: missing/invalid ⇒ `chem=None` and the task behaves exactly
like v2. Shape:

```json
{
  "hypothesis": "cache key ignores tenant_id",   // short claim string
  "stance_hash": "h:cache-tenant",               // canonical bucket of the claim (see 2.3)
  "evidence_ids": ["e3"],                          // shard ids the worker leaned on
  "confidence": 0.0-1.0,
  "contradictions": ["parser-regression"],         // stance_hashes this result argues against
  "toxins": []                                      // signals the worker flags as misleading
}
```

`stance_hash` is the load-bearing field: it is the **coarse equivalence class** of an
answer. Two workers that "independently" reach the same decoy share a `stance_hash`, so the
quorum can see that 3 votes are really 1 vote of information (Codex R2: "50 votes ≈ 3 votes").

### 2.3 Pure functions (deterministic, no I/O)

```
canonical_stance(hypothesis: str) -> str
    Lowercase, strip, collapse whitespace, keep salient tokens → stable bucket key.
    Deterministic; same hypothesis text → same hash. (Used when a worker omits stance_hash.)

stance_diversity(signals: list[chem]) -> float
    Effective-sample-size style: 1 - sum(p_i^2) over the stance_hash distribution
    (Simpson/Gini). 0.0 = total herding (one stance), →1.0 = many distinct stances.

priority_score(task_meta, *, now, crowding) -> float
    priority = base_strength * (1 + uncertainty) * (1 + novelty) / (1 + crowding)
    with time decay on strength: strength *= exp(-(now - last_reinforced)/HALFLIFE).
    Pure; reads only numbers already on the task. Missing fields default so a task with
    no chem still gets a stable, FIFO-equivalent score (see §3.1 identity requirement).

quorum_decision(signals, *, min_diversity, accept_conf, max_rounds, rounds_done)
    -> "accept" | "need_diversity" | "insufficient"
    BOUNDED. Returns "accept" when a stance clears accept_conf AND diversity>=min_diversity;
    "need_diversity" only while rounds_done < max_rounds; else terminal "insufficient".
    The bound is the anti-livelock guarantee (Codex R4 kill-criterion #2).
```

`HALFLIFE`, `min_diversity` (default 0.34), `accept_conf` (0.6), `max_rounds` (2) are env-
tunable constants on `v3` with sane defaults.

## 3. `fleet/board.py` integration (commit-1)

### 3.1 Priority claim (mechanism 1) — **identity-preserving**

`Board.claim_ready(n)` and `SqliteBoard.claim_ready(n)` today pick ready tasks in insertion
order (`seq` / dict order). Add a gated reorder:

```python
def claim_ready(self, n):
    ... promote ready ...
    ready = [tasks in current deterministic order]
    if v3.enabled("chemical"):
        ready = v3.order_by_priority(ready, now=...)   # STABLE sort by -priority_score
    ready = ready[:n]
    ... flip to running ...
```

**Hard requirement:** `v3.order_by_priority` is a **stable** sort, and when every task has
default/equal priority (the v2 world: no chem written yet) it must return the list **in the
original order**. With `SWARM_V3_CHEMICAL=0` the `if` is not entered at all → byte-identical
v2. The fallback test (§5.2) locks both: off-path untouched, and on-path-with-no-chem
equals off-path.

### 3.2 Signal recording (mechanism 1)

```python
def record_signal(self, tid, chem: dict|None) -> None      # both backends
    # no-op if chem is None or not v3.enabled("chemical")
    # else: t.meta["chem"] = chem  (in-mem); SqliteBoard writes meta JSON in one txn
```

`record_signal` is called by the engine after a worker returns (§4). It only ever *adds* a
`meta["chem"]` key, never alters state transitions, so it cannot change v2 outcomes.

### 3.3 Quorum read helpers (mechanism 2)

```python
def stance_signals(self, dep_ids: list[str]) -> list[chem]   # collect chem from given tasks
def diversity_of(self, dep_ids) -> float                     # = v3.stance_diversity(...)
```

Pure reads over existing rows; safe regardless of flags.

### 3.4 Quorum spawn (mechanism 2)

```python
def spawn_referee(self, reducer_tid, *, kind="contrarian") -> str|None
    # gated on v3.enabled("diversity"); idempotent (one referee per reducer per round,
    # tracked in reducer meta["v3_rounds"]); bounded by v3 max_rounds.
    # Adds a new Task (lane="referee") as a dep of the reducer, and re-pends the reducer
    # so it re-runs once the referee is DONE. Returns the new task id or None.
```

Idempotency + the `max_rounds` bound are mandatory (kill-criterion #2: quorum must never
loop). When diversity is already adequate, this is a no-op and the reducer proceeds exactly
as v2.

### 3.5 Hebbian referee routing (mechanism 4) — **shipped in commit-3**

`fleet/v3_credit.py` is a fail-soft JSON sidecar at
`SWARM_V3_CREDIT_PATH` (default `~/.cache/swarm-agent/v3_credit.json`) with atomic writes
and an optional `fcntl` lock:

```python
bump_credit(profile, domain, *, amount=1.0) -> None
credit_score(profile, domain) -> float
order_profiles(candidates, domain) -> list[str]   # stable: high credit first
reset() -> None                                   # test helper
```

Every API is a no-op/zero/identity unless `v3.enabled("hebbian")` is true. The real
consumption seam is `Board.spawn_referee()` / `SqliteBoard.spawn_referee()`: with Hebbian
off, the referee kind remains exactly the caller/default `contrarian`; with Hebbian on,
the board orders `["contrarian", "domain_expert"]` by `(profile, domain)` credit and writes
the winner into referee task meta as `v3_kind`.

Verified outcomes can reinforce the winning route through:

```python
board.credit_outcome(reducer_tid, winning_profile, domain)
```

That helper is also gated and fail-soft; it calls `v3_credit.bump_credit()` only when the
reducer exists and Hebbian is enabled.

### 3.6 Sleep suppression (mechanism 5) — **shipped in commit-3**

`fleet/v3_sleep.py` is a fail-soft JSON sidecar at
`SWARM_V3_SLEEP_PATH` (default `~/.cache/swarm-agent/v3_sleep.json`) with the same atomic
write / optional lock style:

```python
record_trap(domain, decoy_stance) -> None
consolidate() -> int                              # observations >= threshold become rules
is_suppressed(domain, stance) -> bool
reset() -> None                                   # test helper
```

The default consolidation threshold is `SWARM_V3_SLEEP_THRESHOLD=2`. Every API is a
no-op/empty/false unless `v3.enabled("sleep")` is true.

The real consumption seam is in quorum/reduce weighting. `fleet/v3.py` exposes pure helpers:

```python
suppressed_filter(signals, domain, *, is_suppressed=...) -> list[chem]
weighted_stance_counts(signals, domain, *, is_suppressed=...) -> dict[str, float]
```

With sleep off, these are identity/counting helpers. With sleep on, callers pass
`v3_sleep.is_suppressed`; matching stances are retained for traceability but receive
`weight=0.0` and `confidence=0.0`, so a known decoy cannot win purely by majority. Board
quorum calls also pass the suppression callback when sleep is enabled, which prevents a
suppressed high-confidence stance from clearing quorum acceptance unchanged.

## 4. `fleet/engine.py` integration (commit-1)

Single gated hook in the completion path of `ThreadFleet.run()` (mirror in
`scheduler.Scheduler` is optional; ThreadFleet is the hot path). After
`self.board.complete(tid, res["text"])`:

```python
if v3.any_on():
    chem = v3.parse_chem(res.get("text", ""))   # fail-soft fenced-JSON parse
    self.board.record_signal(tid, chem)
    if v3.enabled("diversity") and self._is_reducer_ready_with_low_diversity(...):
        self.board.spawn_referee(reducer_tid)    # holds reducer, adds contrarian
```

The diversity check runs only when a reducer's deps just completed. Everything is wrapped so
any exception degrades to "v2 behavior, signal dropped" (never crashes a worker harvest).
`worker_fn` stays injectable — the offline test passes a deterministic fake worker that
emits `chem` blocks; production uses the real `run_task_local` unchanged.

## 5. Tests (the practical acceptance — runs in hermes venv, no GPU)

Run: `PYTHONPATH=. ~/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_swarm_v3_*.py`

### 5.1 `tests/test_swarm_v3_offline.py` — **v3 beats v2** (deterministic)

A self-contained `SwarmSim` drives the **real** `Board` + **real** `fleet/v3.py` chemistry
with a seeded fake worker (no ThreadFleet threads → deterministic). Two modes: v2 (all flags
off) and v3 (chemical+diversity on).

Synthetic workload `deceptive_parallel_bughunt` — 40 scenarios, `seed=1337`:
- each: `truth_bug`, `decoy_bug`, `evidence_shards`, `misleading_majority` (3 surface
  signals → decoy), `decisive_evidence` (1 quiet signal → truth), `requires_escalation`.
- **24 herding traps**, **8 clean**, **8 escalation-required** (contradictory evidence).

Deterministic fake workers (RNG seeded by `hash(scenario_id, profile)`):
- `same_prior`: in a trap, returns the decoy ~80% (it follows the misleading majority) →
  this is what makes v2 herd. Shares a `stance_hash` with its clones.
- `contrarian`: searches against the majority; if a decisive shard exists, returns truth.
- `referee`: reads board contradictions/low-diversity and returns truth-or-escalate.

PASS thresholds (commit-1 scope = mechanisms 1+2):
- `accuracy_v3 >= 0.80`
- `accuracy_v2 <= 0.55`
- `accuracy_v3 - accuracy_v2 >= 0.30`
- herding subset: `v3_truth_rate >= 0.70` AND `v2_truth_rate <= 0.35`
- escalation subset: `escalation_recall >= 0.875` (≥7/8) with `false_escalation <= 1/8` on clean
- diversity quorum fired on `>= 20/24` herding scenarios
- overhead: v3 total fake-worker calls `<= 1.8 ×` v2 (referee spawns are bounded)

Hebbian and sleep have separate commit-3 epoch tests that prove the real consumption seams
improve measurable outcomes.

### 5.2 `tests/test_swarm_v3_fallback.py` — **all-off == v2** (the safety lock)

With every `SWARM_V3_*` unset/0:
1. `claim_ready` returns the **same task ids in the same order** as a pinned v2 baseline
   (drive both `Board` and `SqliteBoard`).
2. After a full sim run, **no `chem` / `v3_*` key exists** in any task `meta`.
3. `Board.results()` (id→state→result) and the final reducer text are identical across two
   runs and equal the v2 baseline.
4. Bonus: with `SWARM_V3_CHEMICAL=1` but **zero chem written**, `order_by_priority` returns
   the identity order (the §3.1 stable-sort requirement).

This file is authored independently (Codex) as a cross-check on the chemistry author.

### 5.3 `tests/test_swarm_v3_live_smoke.py` — reality check (commit-2, `@pytest.mark.live`)

3 scenarios against the real Step-3.7 server `:8001`: clean / herding-ish / ambiguous.
PASS = server reachable, all reach final-or-error, ≥1 v3 event recorded, `unfinished==0`, no
deadlock. Quality not strictly scored. Heavy 40-scenario live A/B is future work.

### 5.4 `tests/test_swarm_v3_hebbian.py` — learned routing improves epoch-2 cost

The epoch test first trains credit through `board.credit_outcome()` for the profiles that
actually resolve each domain trap, then replays epoch-2 traps through the real
`Board.spawn_referee()` seam. With Hebbian on, the effective profile routes first for each
domain and requires fewer simulated worker/referee attempts than a Hebbian-disabled control
that still spawns the default `contrarian`.

### 5.5 `tests/test_swarm_v3_sleep.py` — consolidated traps reduce decoy adoption

The epoch test records repeated decoy fingerprints, runs `v3_sleep.consolidate()`, then
replays the same majority-decoy reducer inputs. The reduce step uses
`v3.weighted_stance_counts(..., is_suppressed=v3_sleep.is_suppressed)`, so the known decoy
majority is down-weighted and the truth stance wins where the sleep-disabled control adopts
the decoy.

## 6. Kill criteria (stop — the design is wrong)

1. **Chemical board**: a positive-EV task class stays unclaimed for >N cycles while crowded
   low-EV tasks keep being claimed → priority feedback is starving. Stop.
2. **Quorum**: any quorum can exceed `max_rounds`/token bound or emit a non-terminal
   "need more evidence" → liveness broken. Stop.
3. **Reflex↔cortex** (commit-2): cortex queue age grows monotonically while reflex stays
   busy → priority inversion. Stop.
4. **Hebbian** (commit-3): a verifier-rejected path's descendants gain higher future
   selection than verified alternatives → reinforcing noise. Stop.
5. **Sleep** (commit-3): consolidation lowers replayable-trace success or raises known-
   regression recurrence → compacting away signal. Stop.

## 7. Fail-soft contract (non-negotiable)

- Any v3 exception (parse, sort, spawn, record) is caught and degrades to **v2 behavior**;
  it never crashes a worker, harvest, or the front door.
- No model-callable tool is added. Workers *emit* a JSON block in their text; the harness
  *reads* it. The model's tool surface is unchanged.
- All injection is on the **dynamic suffix** side; the static system prefix (prefix-cache)
  is untouched.
- `SWARM_V3=0` (default) ⇒ not one v3 branch is entered.
