"""step37-harness — a high-concurrency fleet orchestrator for Step-3.7-Flash.

Sits ABOVE HermesAgent (imports `run_agent.AIAgent`), keeping dozens of ephemeral,
narrow-context workers in flight against the local Step-3.7 endpoint. Coordination
is stigmergic (a shared Board), dispatch is a dumb fast admission-control loop —
no central "main" agent on the critical path. Operating point is the measured
C32–C64 region for ~8K workers (see fleet/config.py & ~/bench/step37-mtp/FLEET_OPTIMUM.md).

v0.2 hot path: a single-process `ThreadFleet` (a thread per worker, the GIL
released during socket I/O) bounded by a resizable `DecodeGate`, with an
`AIMDController` widening/narrowing the gate toward the throughput knee from live
/metrics. The ProcessPool `Scheduler` stays as the fallback / A-B comparator.
"""
from .board import Board, Task, State
from .scheduler import Scheduler, decoding_now, build_engine
from .engine import ThreadFleet
from .admission import AIMDController
from . import config

__all__ = ["Board", "Task", "State", "Scheduler", "decoding_now", "build_engine",
           "ThreadFleet", "AIMDController", "config"]

# `open_board` is added by the board backend (SQLite) teammate; re-export it if present.
try:  # pragma: no cover - depends on teammate's board.py
    from .board import open_board  # noqa: F401
    __all__.append("open_board")
except Exception:
    pass

__version__ = "0.2.0"
