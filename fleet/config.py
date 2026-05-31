"""Measured operating point for the Step-3.7-Flash fleet.

Provenance: ~/bench/step37-mtp/FLEET_OPTIMUM.md (live-measured, 2x RTX PRO 6000).
For ~8K-token workers the efficient in-flight region is C32 (latency-aware:
763 tok/s, 23.8 tok/s/agent) .. C64 (throughput-max: 1225 tok/s, ~9.8x single-stream).
Single-stream is ~125 tok/s, so the fleet's whole point is keeping dozens in flight.
"""
import os

# HermesAgent install (its venv must be the runtime python so `run_agent` imports).
HERMES_DIR = os.environ.get("HERMES_DIR", "/home/hikari/.hermes/hermes-agent")

# Inference server (the local Step-3.7-Flash NVFP4 + MTP K=1 endpoint).
BASE_URL = os.environ.get("FLEET_BASE_URL", "http://127.0.0.1:8001/v1")
MODEL = os.environ.get("FLEET_MODEL", "step3p7")
API_KEY = os.environ.get("FLEET_API_KEY", "EMPTY")
METRICS_URL = os.environ.get("FLEET_METRICS_URL", "http://127.0.0.1:8001/metrics")

# Target number of agents DECODING at once. The measured efficient region is 32–64;
# 40 is a balanced default. The scheduler holds this many in flight (admission control).
TARGET_INFLIGHT = int(os.environ.get("FLEET_INFLIGHT", "40"))

# Worker turn budget. Keep worker context BOUNDED (~8–16K): the data shows usable
# concurrency collapses as context grows (32K saturates ~C8 vs 8K's ~C48-64), so a
# real harness should compact/summarize rather than let a worker grow to 32K+.
WORKER_MAX_TURNS = int(os.environ.get("FLEET_MAX_TURNS", "12"))

# Per-task retry budget (failed tasks are requeued up to this many times).
MAX_RETRIES = int(os.environ.get("FLEET_MAX_RETRIES", "1"))

# --- Per-role MINIMAL toolsets (the big lean-worker lever) -------------------
# Giving every worker all ~39 tools + every MCP server is huge redundant prefill
# (a smoke run measured ~5K tokens/request, mostly tool schemas) AND makes the
# model reason over tools it will never use. Each lane loads ONLY what its role
# needs, via HermesAgent's AIAgent(enabled_toolsets=...). Pure-reasoning roles get
# NO tools. Keeping the per-role toolset STABLE + identical also lets vLLM
# prefix-cache the shared prefix across same-role workers (tool prefill paid once).
# Token cost of the system+tools prefix per profile is MEASURED on the live server
# (workflow analysis): default(39 tools)=14,113 tok; coder[file,terminal,search]=3,328
# (-77%); researcher[web,search]=398 (-97%); reducer[file]=1,459 (-90%). Same-role
# byte-identical prefix => vLLM auto prefix-cache serves it after worker #1 (~-98%).
TOOL_PROFILES = {
    "director": ["todo"],                               # holds plan; minimal
    "router":   [],                                     # classify / route only
    "reducer":  [],                                     # pure synthesis of upstream results
    "planner":  ["todo"],                               # decompose; minimal
    "worker":   ["file", "terminal", "search"],         # default coder (~3,328 tok, -77%)
    "code":     ["file", "terminal", "search", "code_execution"],
    "research": ["web", "search"],                      # (~398 tok, -97%)
}


def toolsets_for(lane: str):
    return TOOL_PROFILES.get(lane, TOOL_PROFILES["worker"])


# --- Agent ROSTER: heterogeneous context / role / count over the KV budget ---
# The fleet is NOT uniform-context. ONE long-horizon DIRECTOR (big context, steers
# the whole task via the board, rarely decodes) + a few planners/reducers + the bulk
# of lean ~8K workers + tiny routers. KV (1,625,950 fp8 tokens) is a shared budget:
# reserve the few-but-big roles first; the worker lane is the elastic remainder.
# Provenance of the worker operating point: ~/bench/step37-mtp/FLEET_OPTIMUM.md.
KV_BUDGET = 1_625_950

ROSTER = {
    # role:      context  count  tools                                  duty  flags / note
    "director": dict(context=131072, count=1,  tools=["todo"],                   duty=0.15,
                     persistent=True,
                     note="long-horizon steerer; 1 persistent agent holding goal+plan+state, board-driven, OFF the hot loop"),
    "planner":  dict(context=32768,  count=2,  tools=["todo"],                   duty=0.50,
                     note="goal -> task DAG; bursty, high value"),
    "reducer":  dict(context=16384,  count=6,  tools=[],                         duty=0.70,
                     note="tree fan-in; near-root reducers grow toward larger context"),
    "worker":   dict(context=8192,   count=48, tools=["file", "terminal", "search"], duty=0.40,
                     elastic=True,
                     note="the BULK; ephemeral lean coders, measured C32-64 operating point (count = in-flight target)"),
    "router":   dict(context=2048,   count=16, tools=[],                         duty=0.20,
                     note="classify/route; near-free, high churn"),
}
