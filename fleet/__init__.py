"""step37-harness — a high-concurrency fleet orchestrator for Step-3.7-Flash.

Sits ABOVE HermesAgent (imports `run_agent.AIAgent`), keeping dozens of ephemeral,
narrow-context workers in flight against the local Step-3.7 endpoint. Coordination
is stigmergic (a shared Board), dispatch is a dumb fast admission-control loop —
no central "main" agent on the critical path. Operating point is the measured
C32–C64 region for ~8K workers (see fleet/config.py & ~/bench/step37-mtp/FLEET_OPTIMUM.md).
"""
from .board import Board, Task, State
from .scheduler import Scheduler, decoding_now
from . import config

__all__ = ["Board", "Task", "State", "Scheduler", "decoding_now", "config"]
__version__ = "0.1.0"
