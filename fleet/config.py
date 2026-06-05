"""Measured operating point + v0.2 fleet parameters for Step-3.7-Flash.

Provenance: ~/bench/step37-mtp/FLEET_OPTIMUM.md (live-measured, 2x RTX PRO 6000).
For ~8K-token workers the efficient in-flight region is C32 (latency-aware:
763 tok/s, 23.8 tok/s/agent) .. C64 (throughput-max: 1225 tok/s, ~9.8x single-stream).
Single-stream is ~125 tok/s, so the fleet's whole point is keeping dozens in flight.

v0.2 (recon-grounded — see BUILD_SPEC.md): HermesAgent is SYNC+THREADED and stateless
(full-history resend), so the server holds KV only for *currently generating* requests.
The harness therefore bounds concurrent *generations* with a resizable DecodeGate and
oversubscribes enrolled workers (tool-executing workers hold no server KV).
"""
import os


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _envi(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


# ── HermesAgent install (its venv must be the runtime python so run_agent imports) ──
HERMES_DIR = os.environ.get("HERMES_DIR", "/home/hikari/.hermes/hermes-agent")

# ── Inference server (the local Step-3.7-Flash NVFP4 + MTP K=1 endpoint) ──
BASE_URL = os.environ.get("FLEET_BASE_URL", "http://127.0.0.1:8001/v1")
MODEL = os.environ.get("FLEET_MODEL", "step3p7")
API_KEY = os.environ.get("FLEET_API_KEY", "EMPTY")
METRICS_URL = os.environ.get("FLEET_METRICS_URL", "http://127.0.0.1:8001/metrics")

# ── Worker turn budget (BUG FIX) ─────────────────────────────────────────────
# v0.1 defined WORKER_MAX_TURNS=12 but never passed it, so workers ran at the
# AIAgent default max_iterations=90. compat.make_agent now passes MAX_ITERATIONS.
MAX_ITERATIONS = _envi("FLEET_MAX_ITERATIONS", 12)
WORKER_MAX_TURNS = MAX_ITERATIONS  # backwards-compat alias

# ── Per-generation output cap (bounded-generation sweep) ─────────────────────
# FLEET_MAX_TOKENS caps each worker's output tokens so the sweep is one clean
# ~200-token generation per task (predictable decode batch, comparable points).
# None (unset) = the model/server default (no harness-imposed cap). AIAgent takes
# a native `max_tokens` kwarg (verified run_agent.py:390 -> init_agent), where
# None means "don't send max_tokens". compat.make_agent + worker.run_task read
# getattr(config, "MAX_TOKENS", None), so leaving this None preserves old behaviour.
def _envi_or_none(name):
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


MAX_TOKENS = _envi_or_none("FLEET_MAX_TOKENS")  # None = model default (no cap)

# ── Per-task wall-clock deadline (hard anti-wedge safety net) ─────────────────
# A worker that runs a never-returning FOREGROUND command (a server, a GUI app) blocks its
# thread forever; the engine cannot kill it (the terminal tool's own timeout + cleanup_vm do
# NOT interrupt an in-flight command — measured). So the engine ABANDONS a task that exceeds
# this many seconds (marks it failed, stops waiting) and runs workers on DAEMON threads so the
# abandoned one can never block run()/process exit. 0 or negative = disabled (no deadline).
TASK_TIMEOUT_S = _envf("FLEET_TASK_TIMEOUT", 300.0)

# How long a new WRITING goal waits for a still-alive ABANDONED write worker (from an earlier
# timed-out goal — its subprocess cannot be killed) to finish before starting, so it can't race
# the stale writer and corrupt the workspace. Then it proceeds with a logged warning (never a
# permanent stall). 0 = disabled (don't wait).
ABANDONED_WRITER_WAIT_S = _envf("FLEET_ABANDONED_WRITER_WAIT", 300.0)

# ── Planner output cap (separate from worker MAX_TOKENS) ─────────────────────
# The planner emits the WHOLE task-DAG JSON in one shot; a big/multi-topic goal can run to
# several thousand tokens. The old 6144 cap was too small — a large plan TRUNCATED mid-JSON
# and the agent loop then churned truncation→continuation→max-iterations (×3 plan retries),
# i.e. "planning seems to loop for minutes" (observed live). Give the planner generous
# headroom so a complete plan lands in ONE generation. Env-tunable.
PLANNER_MAX_TOKENS = _envi("FLEET_PLANNER_MAX_TOKENS", 16384)
# Iterations for the planner agent loop. It is a single JSON generation (no tool loop), so a
# small budget is right — and it BOUNDS the continuation churn if a plan ever still overflows
# (one continuation, not four) before _plan falls back to a retry.
PLANNER_MAX_ITERATIONS = _envi("FLEET_PLANNER_MAX_ITERATIONS", 2)

# ── Static admission (process-engine / --admission static) ───────────────────
# Target number of agents DECODING at once. Measured efficient region 32–64; 40 default.
TARGET_INFLIGHT = _envi("FLEET_INFLIGHT", 40)
MAX_RETRIES = _envi("FLEET_MAX_RETRIES", 1)

# ── DecodeGate (v0.2) — bounds concurrent *generations* == server num_requests_running ──
DECODE_GATE_ENABLED = os.environ.get("FLEET_DECODE_GATE", "1") not in ("0", "false", "False")
DECODE_GATE_START = _envi("FLEET_GATE_START", 40)   # initial concurrent-generation limit
DECODE_GATE_MIN = _envi("FLEET_GATE_MIN", 8)        # never below this (workers would block forever)
DECODE_GATE_MAX = _envi("FLEET_GATE_MAX", 96)       # ceiling for the AIMD search
KNEE_LO = _envi("FLEET_KNEE_LO", 32)                # measured efficient region (FLEET_OPTIMUM §4)
KNEE_HI = _envi("FLEET_KNEE_HI", 64)

# ── Oversubscription — enrolled workers >> gate limit (tool-executing hold no KV) ──
# Enrolled must stay AHEAD of the gate's real duty or the gate starves: at agent
# duty d (fraction of wall actually decoding), keeping the gate full needs roughly
# enrolled ≈ gate_limit / d_min surplus over the gate (tool-executing/blocked workers
# hold ZERO server KV — recon fact #3, threads are heap-only and cheap). At the
# default gate START=40 and a low real duty, OVERSUB=2.0/ENROLL_MAX=160 left the
# gate under-fed in the lead's sweep, so RAISE the defaults. Still env-overridable
# (the lead sets FLEET_OVERSUB / FLEET_ENROLL_MAX per sweep point).
OVERSUB_FACTOR = _envf("FLEET_OVERSUB", 3.0)        # enrolled = factor × gate_limit
ENROLL_MAX = _envi("FLEET_ENROLL_MAX", 256)         # hard cap on concurrent worker threads

# ── AIMD dynamic-admission controller (--admission aimd) ─────────────────────
AIMD_INTERVAL_S = _envf("FLEET_AIMD_INTERVAL", 4.0)  # control period
AIMD_STRIDE = _envi("FLEET_AIMD_STRIDE", 8)          # additive-increase step on the gate limit
AIMD_BACKOFF = _envf("FLEET_AIMD_BACKOFF", 0.7)      # multiplicative-decrease factor
AIMD_KV_HI = _envf("FLEET_AIMD_KV_HI", 0.85)         # KV usage backoff threshold (0..1)
AIMD_TPUT_DROP = _envf("FLEET_AIMD_TPUT_DROP", 0.07) # rel. tok/s regression that triggers backoff
AIMD_SATURATION = _envf("FLEET_AIMD_SAT", 0.9)       # only grow if running >= limit*this (gate full)

# ── Parallel goal consumption (Option A) — PARALLEL_GOALS_PLAN ───────────────
# K = how many QUEUED goals the completion manager may run at once. Read-only goals
# (writer/analyst/researcher/reviewer/reducer lanes) run concurrently up to K and fill the
# shared DecodeGate; writing goals (coder/code/worker lanes) always run EXCLUSIVELY. K
# defaults to 1 == today's strict one-at-a-time behaviour, so the feature is inert until
# opted in (set FLEET_MAX_CONCURRENT_GOALS=2 or 3 to enable). FLEET_PARALLEL_WRITES is
# reserved (default OFF): writing goals stay exclusive regardless in Phase 1.
MAX_CONCURRENT_GOALS = _envi("FLEET_MAX_CONCURRENT_GOALS", 1)
PARALLEL_WRITES = os.environ.get("FLEET_PARALLEL_WRITES", "0") not in ("0", "false", "False")

# ── Persistent board (None = in-memory) ──────────────────────────────────────
BOARD_PATH = os.environ.get("FLEET_BOARD_PATH") or None

# ── Per-role MINIMAL toolsets (the big lean-worker lever) ─────────────────────
# Default 39 tools ≈ 14,113 tok prefill re-paid by every worker every turn (stateless
# resend). Each lane loads ONLY what its role needs (verified valid leaf toolsets, recon
# `toolset-name-validation`): []=0 tools, todo=1, web=2, file+terminal+search=7, +code_exec=8.
# Same-role byte-identical prefix => vLLM auto prefix-cache serves it after worker #1.
TOOL_PROFILES = {
    # ── front door (persona + memory injected; see compat.make_agent) ──
    "director": ["todo"],                               # holds plan; minimal
    "planner":  ["todo"],                               # decompose; minimal
    "reducer":  [],                                     # pure synthesis of upstream results (0 tools)
    "manager":  [],                                     # queue oversight only (0 tools)
    "router":   [],                                     # classify / route only (0 tools)
    # ── worker swarm (lean, no persona/memory) ──
    # writer: pure reasoning/explanation/synthesis (0 tools). DEFAULT worker role.
    "writer":   [],
    # coder: file edit + shell + code-exec + skills. `skills` was deliberately omitted
    # for prefill leanness, but execution lanes need the run-app skill so servers/GUI
    # detach instead of foreground-blocking. The byte-identical same-role prefix means
    # vLLM prefix-cache serves the skill index after worker #1 (cost amortised).
    "coder":    ["file", "terminal", "search", "code_execution", "skills"],
    # researcher: web + skills. The `skills` toolset (skills_list/skill_view/
    # skill_manage) is the ONLY switch needed to expose ~/.hermes/skills/ —
    # system_prompt.py auto-injects the skill index when those tools are present.
    "researcher": ["web", "skills"],
    # analyst: read-MOSTLY inspection (read_file/search). No terminal, no code_exec.
    "analyst":  ["file", "search"],
    # reviewer: read repo + skills (code-review / security-review skill docs).
    "reviewer": ["file", "search", "skills"],
    # ── legacy aliases (KEEP — existing callers/plans still emit these) ──
    "worker":   ["file", "terminal", "search", "skills"],  # default lean coder (back-compat)
    "code":     ["file", "terminal", "search", "code_execution", "skills"],  # alias of coder
    "research": ["web"],                                # legacy research (no skills)
}


def toolsets_for(lane: str):
    return TOOL_PROFILES.get(lane, TOOL_PROFILES["worker"])


# Tool-PROFILE toolsets that grant working-tree / shell mutation (write_file/patch/terminal/
# code-exec). A lane is write-capable iff its profile contains ANY of these — classified by
# real capability, not name (so analyst/reviewer, whose "file" toolset includes write_file,
# count as writers; unknown lanes fall back to the write-capable worker profile = fail closed).
WRITE_TOOLSETS = {"file", "terminal", "code_execution"}


def lane_writes(lane: str) -> bool:
    """True iff ``lane`` can mutate the working tree / run shell, by its real tool capability."""
    return bool(set(toolsets_for(lane)) & WRITE_TOOLSETS)


# ── Front-door PERSONA lanes ─────────────────────────────────────────────────
# DEFAULT = none. swarm-agent now has its OWN identity, injected per-lane via
# fleet/prompts.py (SWARM_IDENTITY: "Hikari's hardware-optimised swarm, one mind across
# many parts"). Loading ~/.hermes/SOUL.md + ~/.hermes/memories/{MEMORY,USER}.md on the
# front-door lanes ACTIVELY HARMS that: the user's general HermesAgent persona/memory
# bleeds in and the model confabulates a wrong self-description — observed live, the chat
# claimed to be "a 27B main model + qwen36-35b-a3b workers via delegate_task" (those are
# the user's OTHER local-model notes, not swarm-agent's architecture). With persona OFF,
# the same probe answers cleanly as swarm-agent ("the parallel workers are all me, one
# mind"). So every lane stays lean and the only identity is the injected SWARM_IDENTITY.
# Still env-overridable: FLEET_PERSONA_LANES="router,reducer" re-enables SOUL+memory on
# those lanes (e.g. if you WANT the user's memory woven into final deliverables).
PERSONA_LANES = set(
    (os.environ.get("FLEET_PERSONA_LANES", "") or "")
    .replace(" ", "").split(",")
) - {""}


def is_persona_lane(lane: str) -> bool:
    """True iff `lane` is a front-door conversational lane that gets SOUL+memory."""
    return lane in PERSONA_LANES


# ── Lane priority for the DecodeGate waterfall (#4): higher = served first ────
# director/planner/reducer get decode slots ahead of the worker swarm so reserved
# roles never starve behind the bulk. (Per-lane hard reservations are a refinement.)
LANE_PRIORITY = {
    "director":   100,
    "planner":     80,
    "reducer":     60,
    "manager":     50,
    "coder":       45,
    "code":        45,   # legacy alias of coder
    "reviewer":    44,
    "researcher":  42,
    "research":    42,   # legacy alias
    "analyst":     41,
    "worker":      40,
    "writer":      40,
    "router":      20,
}


def lane_priority(lane: str) -> int:
    return LANE_PRIORITY.get(lane, LANE_PRIORITY["worker"])


# ── Agent ROSTER: heterogeneous context / role / count over the KV budget ─────
# KV (3,228,856 nvfp4-KV tokens; was 1,625,950 fp8) is a shared budget. Under the
# stateless-completions model (recon fact #2) resident KV ≈ concurrent-DECODING count ×
# per-turn tokens, NOT enrolled count — so the gate (not the roster) is what bounds KV. The
# roster sizes role mix + the decode-priority waterfall; see fleet/roster.py for both the
# resident and the (pessimistic) enrolled views.
# 2026-06-05: backend is now Step3.7 NVFP4 + NVFP4-KV (B2 in-kernel V de-swizzle) at
# util0.92 / max-len262144 / max-seqs128 — live-measured pool 3,228,856 tok (1.99x the old
# fp8 budget). The bigger pool buys longer-context tenants + more prefix-cache retention
# (re-prefill avoidance for the stateless full-resend model); it does NOT raise the decode
# knee (compute-bound at C32-64, gate ceiling 96), so the gate/AIMD params are unchanged.
KV_BUDGET = 3_228_856

ROSTER = {
    # 2026-06-05: per-turn `context` bumped to realistic sizes (worker 8K->32K, reducer
    # 16K->32K, router 2K->4K). These are KV-ACCOUNTING estimates of a typical turn's
    # transcript, NOT enforced caps (no context_length is passed to make_agent; a worker may
    # grow to the server max_model_len 262144). The old 8K underestimated real code work — a
    # single large file read blows past it. Honest against the 3.23M nvfp4-KV pool: enrolled
    # worst-case ~2.03M = 63%, still leaving ~1.2M for prefix cache + 256K interactive/Studio.
    # role:      context  count  tools                                  duty  flags / note
    "director": dict(context=131072, count=1,  tools=["todo"],                   duty=0.15,
                     persistent=True,
                     note="long-horizon steerer; 1 persistent agent holding goal+plan+state, board-driven, OFF the hot loop"),
    "planner":  dict(context=32768,  count=2,  tools=["todo"],                   duty=0.50,
                     note="goal -> task DAG; bursty, high value"),
    "reducer":  dict(context=32768,  count=6,  tools=[],                         duty=0.70,
                     note="tree fan-in; near-root reducers grow toward larger context"),
    "worker":   dict(context=32768,  count=48, tools=["file", "terminal", "search"], duty=0.40,
                     elastic=True,
                     note="the BULK; ephemeral lean coders, measured C32-64 operating point (count = decode target)"),
    "router":   dict(context=4096,   count=16, tools=[],                         duty=0.20,
                     note="classify/route; near-free, high churn"),
}


# ── Startup validation ───────────────────────────────────────────────────────
# Fail FAST and LOUD on an inconsistent env rather than deadlocking or thrashing at
# runtime. The DecodeGate never deadlocks while limit >= 1 (compat.DecodeGate), and
# the AIMD controller clamps into [MIN, MAX] — both rely on these invariants holding.
def validate() -> None:
    """Assert the operating-point invariants. Raises ValueError on bad env so a
    misconfigured sweep aborts at import, not 40 threads deep into a live run."""
    problems = []
    if DECODE_GATE_MIN < 1:
        problems.append(
            f"DECODE_GATE_MIN must be >= 1 (gate deadlocks at 0); got {DECODE_GATE_MIN} "
            f"(env FLEET_GATE_MIN)")
    if DECODE_GATE_MIN > DECODE_GATE_MAX:
        problems.append(
            f"DECODE_GATE_MIN ({DECODE_GATE_MIN}) must be <= DECODE_GATE_MAX "
            f"({DECODE_GATE_MAX}) (env FLEET_GATE_MIN / FLEET_GATE_MAX)")
    if ENROLL_MAX < 1:
        problems.append(
            f"ENROLL_MAX must be >= 1 (no worker threads otherwise); got {ENROLL_MAX} "
            f"(env FLEET_ENROLL_MAX)")
    if MAX_CONCURRENT_GOALS < 1:
        problems.append(
            f"MAX_CONCURRENT_GOALS must be >= 1; got {MAX_CONCURRENT_GOALS} "
            f"(env FLEET_MAX_CONCURRENT_GOALS)")
    if problems:
        raise ValueError(
            "fleet.config: invalid configuration:\n  - " + "\n  - ".join(problems))


# Validate at import so any importer (cli, engine, roster, tests) fails fast.
validate()
