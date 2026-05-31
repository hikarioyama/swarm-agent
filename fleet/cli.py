"""CLI: drive a fleet from a tasks file.

    <hermes-venv>/bin/python -m fleet.cli tasks.jsonl              # thread + aimd (default)
    <hermes-venv>/bin/python -m fleet.cli tasks.jsonl --engine process --inflight 40
    <hermes-venv>/bin/python -m fleet.cli tasks.jsonl --gate 32 --admission static --no-warm

tasks.jsonl: one JSON object per line, e.g.
    {"id": "a", "prompt": "Summarize X."}
    {"id": "b", "prompt": "Using A's result, do Y.", "deps": ["a"]}

Run with the HermesAgent venv python so `run_agent` imports inside workers.

v0.2 default is the single-process THREAD engine with AIMD admission:
  * one process, a thread per enrolled worker (cheap — Python heap only)
  * a resizable DecodeGate bounds concurrent generations == server KV
  * AIMD widens/narrows the gate toward the throughput knee from live /metrics
The ProcessPool engine (`--engine process`) stays as the fallback / A-B comparator.
"""
from __future__ import annotations
import argparse
import json
import sys

from . import board as board_mod
from .board import Task
from . import config, metrics


def load_tasks(path: str):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Task(id=str(d.get("id", i)), prompt=d["prompt"],
                            deps=[str(x) for x in d.get("deps", [])],
                            lane=d.get("lane", "worker"), meta=d.get("meta", {})))
    return out


def _open_board(path):
    """Defensive board factory. A teammate adds `board.open_board` (SQLite
    backend); until then fall back to the in-memory Board. Either way the public
    API is identical, so the engines don't care which they got."""
    open_board = getattr(board_mod, "open_board", None)
    if callable(open_board):
        return open_board(path)
    return board_mod.Board()


def _build_progress(total, gate, get_tput):
    """A live one-liner: completed/total, plus running/limit/kv%/tok_s scraped
    from /metrics. `get_tput` is a ThroughputMeter-backed closure so the rate is
    differenced across calls, not a single instantaneous reading."""
    seen = {"n": 0}

    def line():
        sc = metrics.scrape(config.METRICS_URL)
        tps = get_tput(sc)
        limit = gate.get_limit() if gate is not None else "-"
        running = int(sc.get("running", 0)) if sc else "?"
        kv = f"{sc.get('kv', 0) * 100:.0f}%" if sc else "?"
        tps_s = f"{tps:.0f}" if tps else "-"
        return f"running={running}/{limit} kv={kv} tok_s={tps_s}"

    def prog(kind, tid, counts=None, **extra):
        counts = counts or {}
        if kind == "done":
            seen["n"] += 1
            ds, ts = extra.get("decode_s"), extra.get("tool_s")
            timing = f" decode={ds}s tool={ts}s" if ds is not None else ""
            print(f"  [{seen['n']}/{total}] ✓ {tid} {extra.get('wall_s')}s{timing}  "
                  f"{line()}  ready={counts.get('ready')}", flush=True)
        elif kind in ("fail", "deadlock"):
            print(f"  ✗ {kind} {tid} {extra.get('error', '')}", flush=True)
        elif kind == "requeue":
            print(f"  ↻ requeue {tid}", flush=True)

    return prog


def _report(out, args, gate, controller):
    """Operating-point summary: mean running (≈ decode batch), fleet duty, tok/s,
    gate history, wall — the numbers DESIGN §6 wanted measured."""
    print(f"\ndone in {out['wall_s']}s | {out['counts']}", flush=True)
    if args.engine == "thread":
        print(f"operating point: mean_running={out.get('mean_running')} "
              f"peak_running={out.get('peak_running')} duty={out.get('duty')}", flush=True)
        gs = out.get("gate_stats")
        if gs:
            print(f"gate: limit={gs.get('limit')} peak_in={gs.get('peak_in')} "
                  f"acquired={gs.get('acquired_total')} wait_s={gs.get('wait_s_total')}",
                  flush=True)
        if controller is not None and controller.history:
            limits = [h["limit_after"] for h in controller.history]
            backoffs = sum(1 for h in controller.history if h["action"] == "backoff")
            print(f"aimd: gate {limits[0]}→{limits[-1]} (min {min(limits)} max {max(limits)}), "
                  f"{backoffs} backoff(s) over {len(controller.history)} ticks", flush=True)


def _exit_code(out, engine: str) -> int:
    """Map an engine summary to a process exit code.

    A task is FAILED (terminal: retries exhausted) or STRANDED (never finished:
    still pending/ready/running when the engine's run loop returned). The v0.1 code
    returned 0 whenever failed==0, so a DEADLOCK — the engine broke out with unmet
    deps / an all-failed frontier leaving tasks unfinished — was silently reported as
    SUCCESS. The engine now surfaces those as non-terminal counts in `out["counts"]`,
    so treat unfinished>0 as a failure too (and announce it). Print is here (not the
    caller) so both fall-throughs go through one place.
    """
    counts = out.get("counts", {}) or {}
    failed = counts.get("failed", 0)
    # unfinished == anything that never reached a terminal (done/failed) state.
    stranded = counts.get("pending", 0) + counts.get("ready", 0) + counts.get("running", 0)
    if stranded > 0:
        print(f"DEADLOCK: {stranded} tasks stranded "
              f"(pending={counts.get('pending', 0)} ready={counts.get('ready', 0)} "
              f"running={counts.get('running', 0)})", flush=True)
    return 0 if (failed == 0 and stranded == 0) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fleet", description="Step-3.7 high-concurrency fleet")
    ap.add_argument("tasks", help="JSONL file of tasks ({id, prompt, deps?, lane?})")
    ap.add_argument("--engine", choices=("thread", "process"), default="thread",
                    help="thread = single-process ThreadPool + DecodeGate (default); "
                         "process = ProcessPool fallback / A-B")
    ap.add_argument("--admission", choices=("static", "aimd"), default="aimd",
                    help="aimd = dynamically resize the gate toward the knee (default); "
                         "static = fixed gate")
    ap.add_argument("--gate", type=int, default=None,
                    help=f"initial decode-gate limit (default {config.DECODE_GATE_START})")
    ap.add_argument("--no-gate", action="store_true",
                    help="disable the decode gate (timing only) — A/B the gate's effect")
    ap.add_argument("--warm", dest="warm", action="store_true", default=True,
                    help="prefix-warm each role (default on)")
    ap.add_argument("--no-warm", dest="warm", action="store_false",
                    help="skip prefix-warm")
    ap.add_argument("--board", default=config.BOARD_PATH,
                    help="SQLite board path (default in-memory)")
    ap.add_argument("--inflight", type=int, default=config.TARGET_INFLIGHT,
                    help=f"process-engine fixed in-flight (default {config.TARGET_INFLIGHT})")
    ap.add_argument("--out", default="fleet_results.jsonl")
    args = ap.parse_args(argv)

    board = _open_board(args.board)
    board.add_many(load_tasks(args.tasks))
    total = board.unfinished()

    # ── process engine: the v0.1 fallback path (no gate / no AIMD) ───────────
    if args.engine == "process":
        from .scheduler import Scheduler
        print(f"fleet[process]: {total} tasks | inflight {args.inflight} | "
              f"{config.BASE_URL} ({config.MODEL})", flush=True)
        tput = metrics.ThroughputMeter()
        prog = _build_progress(total, None, tput.update)
        out = Scheduler(board, inflight=args.inflight, on_event=prog).run()
        _write_results(out, args.out)
        _report(out, args, None, None)
        return _exit_code(out, "process")

    # ── thread engine: the v0.2 default path (gate + AIMD) ───────────────────
    from . import compat
    from .engine import ThreadFleet
    from .admission import AIMDController

    gate = None
    if not args.no_gate and config.DECODE_GATE_ENABLED:
        gate = compat.DecodeGate(args.gate or config.DECODE_GATE_START)

    # apply the runtime monkeypatch (gate-acquire + decode_s timing) and kill the
    # cold-start cache stampede BEFORE fanning out threads.
    compat.apply(gate)
    compat.prewarm(list(config.TOOL_PROFILES.values()))

    if args.warm:
        try:
            from . import warm as warm_mod
            roles = sorted({t.lane for t in board.results().values()}) or ["worker"]
            lat = warm_mod.warm_profiles(roles, base_url=config.BASE_URL,
                                         model=config.MODEL, api_key=config.API_KEY)
            print(f"warmed roles: {lat}", flush=True)
        except Exception as e:  # warm.py may not exist yet / server may be down
            print(f"  (warm skipped: {e!r})", flush=True)

    controller = None
    if gate is not None and args.admission == "aimd":
        controller = AIMDController(gate, config.METRICS_URL, config)
        controller.start()

    gate_lbl = gate.get_limit() if gate is not None else "off"
    print(f"fleet[thread/{args.admission}]: {total} tasks | gate={gate_lbl} | "
          f"oversub×{config.OVERSUB_FACTOR} enroll≤{config.ENROLL_MAX} | "
          f"{config.BASE_URL} ({config.MODEL})", flush=True)

    tput = metrics.ThroughputMeter()
    prog = _build_progress(total, gate, tput.update)
    try:
        out = ThreadFleet(board, gate, cfg=config, on_event=prog).run()
    finally:
        if controller is not None:
            controller.stop()

    _write_results(out, args.out)
    _report(out, args, gate, controller)
    return _exit_code(out, "thread")


def _write_results(out, path) -> None:
    with open(path, "w") as f:
        for tid, t in out["board_results"].items() if "board_results" in out else out["results"].items():
            # ThreadFleet -> board_results is {id: Task}; Scheduler -> results is {id: Task}
            state = t.state.value if hasattr(t, "state") else None
            result = t.result if hasattr(t, "result") else None
            error = t.error if hasattr(t, "error") else None
            f.write(json.dumps({"id": tid, "state": state, "result": result,
                                "error": error}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
