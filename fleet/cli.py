"""CLI: drive a fleet from a tasks file.

    <hermes-venv>/bin/python -m fleet.cli tasks.jsonl --inflight 40

tasks.jsonl: one JSON object per line, e.g.
    {"id": "a", "prompt": "Summarize X."}
    {"id": "b", "prompt": "Using A's result, do Y.", "deps": ["a"]}

Run with the HermesAgent venv python so `run_agent` imports inside workers.
"""
from __future__ import annotations
import argparse
import json
import sys

from .board import Board, Task
from .scheduler import Scheduler, decoding_now
from . import config


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fleet", description="Step-3.7 high-concurrency fleet")
    ap.add_argument("tasks", help="JSONL file of tasks ({id, prompt, deps?, lane?})")
    ap.add_argument("--inflight", type=int, default=config.TARGET_INFLIGHT,
                    help=f"target agents in flight (measured region 32-64; default {config.TARGET_INFLIGHT})")
    ap.add_argument("--out", default="fleet_results.jsonl")
    args = ap.parse_args(argv)

    board = Board()
    board.add_many(load_tasks(args.tasks))
    total = board.unfinished()
    print(f"fleet: {total} tasks | target in-flight {args.inflight} | {config.BASE_URL} ({config.MODEL})",
          flush=True)

    seen = {"n": 0}

    def prog(kind, tid, counts=None, **extra):
        if kind == "done":
            seen["n"] += 1
            dn = decoding_now()
            print(f"  [{seen['n']}/{total}] ✓ {tid} {extra.get('wall_s')}s  "
                  f"decoding~{dn}  ready={counts.get('ready')} running={counts.get('running')}", flush=True)
        elif kind in ("fail", "deadlock"):
            print(f"  ✗ {kind} {tid} {extra.get('error', '')}", flush=True)
        elif kind == "requeue":
            print(f"  ↻ requeue {tid}", flush=True)

    out = Scheduler(board, inflight=args.inflight, on_event=prog).run()

    with open(args.out, "w") as f:
        for tid, t in out["results"].items():
            f.write(json.dumps({"id": tid, "state": t.state.value,
                                "result": t.result, "error": t.error}, ensure_ascii=False) + "\n")
    print(f"\ndone in {out['wall_s']}s | {out['counts']} | -> {args.out}", flush=True)
    return 0 if out["counts"].get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
