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
