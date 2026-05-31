"""The ProcessPool concurrency controller — the v0.1 fallback engine.

This is deliberately DUMB and FAST: no LLM, no reasoning. It just keeps
TARGET_INFLIGHT workers busy by pulling ready tasks off the Board and feeding a
process pool, writing each result back (which unlocks dependents). Because the
intelligence (decomposition, reduction) lives elsewhere and this loop is pure
code, there is no central "main" agent on the critical path to panic — the thing
that keeps dozens of requests in flight is an admission-control loop, not an agent.

Admission control holds exactly `inflight` requests in flight (the measured
efficient region for ~8K workers is C32–C64; default 40). A `/metrics` peek is
provided so a later version can target the throughput knee dynamically instead of
a fixed number.

v0.2 keeps this engine intact as the fallback / A-B comparator. The default hot
path is the single-process `engine.ThreadFleet` (one process, a thread per worker,
a resizable DecodeGate + AIMD admission); the CLI selects between them via
`--engine {thread,process}`. Use `build_engine(...)` to construct whichever the
caller picked.
"""
from __future__ import annotations
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import Callable, Dict, Optional

from . import config
from .board import Board
from .worker import run_task


def decoding_now() -> Optional[float]:
    """Requests currently running on the server (for dynamic admission / observability)."""
    try:
        raw = urllib.request.urlopen(config.METRICS_URL, timeout=3).read().decode()
        for l in raw.splitlines():
            if l.startswith("vllm:num_requests_running") and not l.startswith("#"):
                return float(l.split()[-1])
    except Exception:
        return None
    return None


class Scheduler:
    def __init__(self, board: Board, inflight: int = config.TARGET_INFLIGHT,
                 max_retries: int = config.MAX_RETRIES,
                 on_event: Optional[Callable[..., None]] = None) -> None:
        self.board = board
        self.inflight = max(1, inflight)
        self.max_retries = max_retries
        self.on_event = on_event or (lambda *a, **k: None)
        self.t0 = time.time()

    def _emit(self, kind: str, tid: Optional[str], **extra) -> None:
        self.on_event(kind, tid, counts=self.board.counts(), **extra)

    def run(self) -> Dict[str, object]:
        with ProcessPoolExecutor(max_workers=self.inflight) as ex:
            futs: Dict[object, str] = {}                       # future -> task id
            while self.board.unfinished() > 0 or futs:
                slots = self.inflight - len(futs)
                if slots > 0:
                    for t in self.board.claim_ready(slots):    # admission control
                        futs[ex.submit(run_task, t.spec())] = t.id
                        self._emit("dispatch", t.id)
                if not futs:
                    if self.board.unfinished() > 0:
                        self._emit("deadlock", None)            # unmet deps or all-failed
                    break
                done, _ = wait(list(futs), return_when=FIRST_COMPLETED)
                for f in done:
                    tid = futs.pop(f)
                    try:
                        res = f.result()
                        self.board.complete(tid, res.get("text", ""))
                        self._emit("done", tid, wall_s=res.get("wall_s"))
                    except Exception as e:
                        requeued = self.board.fail(tid, repr(e), self.max_retries)
                        self._emit("requeue" if requeued else "fail", tid, error=repr(e)[:160])
        return {"counts": self.board.counts(), "wall_s": round(time.time() - self.t0, 1),
                "results": self.board.results()}


def build_engine(engine: str, board, *, gate=None, inflight: int = config.TARGET_INFLIGHT,
                 on_event: Optional[Callable[..., None]] = None):
    """Factory the CLI uses to pick the hot path.

    ``engine="thread"``  -> ``engine.ThreadFleet`` (single-process ThreadPool +
                            DecodeGate; the v0.2 default). Requires a ``gate``.
    ``engine="process"`` -> this ``Scheduler`` (ProcessPool, fixed inflight;
                            the v0.1 fallback / A-B comparator). Ignores ``gate``.

    Returned object always exposes ``.run() -> summary``. ThreadFleet is imported
    lazily so importing this module never pulls the thread engine if unused.
    """
    if engine == "process":
        return Scheduler(board, inflight=inflight, on_event=on_event)
    if engine == "thread":
        from .engine import ThreadFleet
        return ThreadFleet(board, gate, cfg=config, on_event=on_event)
    raise ValueError(f"unknown engine {engine!r} (expected 'thread' or 'process')")
