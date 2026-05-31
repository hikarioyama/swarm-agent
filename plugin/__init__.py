"""HermesAgent plugin: a thin `/fleet` command over the standalone step37-harness.

Deployed to ~/.hermes/plugins/fleet-orchestrator/ (OUTSIDE the hermes-agent repo,
so `git pull` upstream never touches it). The heavy logic lives in the external
package at ~/projects/step37-harness/fleet/ — this wrapper just launches it.
"""
from __future__ import annotations
import os
import subprocess
import urllib.request

HARNESS = os.path.expanduser("~/projects/step37-harness")
VENV_PY = os.environ.get("HERMES_VENV_PY", "/home/hikari/.hermes/hermes-agent/venv/bin/python")
METRICS = "http://127.0.0.1:8001/metrics"


def _decoding_now() -> str:
    try:
        raw = urllib.request.urlopen(METRICS, timeout=3).read().decode()
        for l in raw.splitlines():
            if l.startswith("vllm:num_requests_running") and not l.startswith("#"):
                return l.split()[-1]
    except Exception:
        return "?"
    return "?"


def _fleet_cmd(raw_args: str):
    parts = (raw_args or "").split()
    if not parts or parts[0] == "status":
        return (f"fleet status: ≈{_decoding_now()} requests decoding on :8001 | "
                f"operating region C32–C64 (target in-flight 40) | harness: {HARNESS}")
    if parts[0] == "run" and len(parts) >= 2:
        path, inflight = parts[1], (parts[2] if len(parts) > 2 else "40")
        log = "/tmp/fleet.log"
        p = subprocess.Popen([VENV_PY, "-m", "fleet.cli", path, "--inflight", inflight],
                             cwd=HARNESS, stdout=open(log, "a"), stderr=subprocess.STDOUT)
        return f"fleet launched (pid {p.pid}, in-flight {inflight}) — tail {log}"
    return "usage: /fleet [status | run <tasks.jsonl> [inflight]]"


def register(ctx) -> None:
    ctx.register_command(
        "fleet",
        handler=_fleet_cmd,
        description="Step-3.7 high-concurrency fleet: status | run <tasks.jsonl> [inflight]",
    )
