"""Live AIMD convergence probe (criterion D): does the controller move the gate toward
the throughput knee and hold without thrashing/deadlock? Starts the gate LOW, enrolls a
verbose load, runs AIMDController, logs (t, limit, running, kv, tok_s) for a window, exits.

    PYTHONPATH=. <venv>/python scripts/aimd_probe.py --start 12 --secs 70
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, ".")
os.environ.setdefault("FLEET_MAX_TOKENS", "8192")
os.environ.setdefault("FLEET_NO_CONTINUE", "1")
os.environ.setdefault("FLEET_AIMD_INTERVAL", "3.0")
from fleet import compat, config, metrics  # noqa: E402
from fleet.admission import AIMDController  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=12)
    ap.add_argument("--secs", type=float, default=70.0)
    ap.add_argument("--enroll", type=int, default=140)
    ap.add_argument("--tasks", default="examples/sweep_8k.jsonl")
    a = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(a.tasks)]
    gate = compat.DecodeGate(a.start)
    compat.apply(gate)
    compat.prewarm([[]])

    stop = threading.Event()
    ctr = {"i": 0}
    lk = threading.Lock()

    def worker():
        while not stop.is_set():
            with lk:
                i = ctr["i"]; ctr["i"] += 1
            ag = compat.make_agent("reducer", task_id=f"a{i}")
            try:
                ag.run_conversation(prompts[i % len(prompts)], task_id=f"a{i}")
            except Exception:
                pass

    for _ in range(a.enroll):
        threading.Thread(target=worker, daemon=True).start()

    ctrl = AIMDController(gate, config.METRICS_URL, config)
    ctrl.start()

    t0 = time.time()
    print(f"aimd: start gate={a.start} (KNEE {config.KNEE_LO}-{config.KNEE_HI}, "
          f"MIN {config.DECODE_GATE_MIN} MAX {config.DECODE_GATE_MAX})", flush=True)
    while time.time() - t0 < a.secs:
        sc = metrics.scrape(timeout=2.0)
        lim = gate.get_limit()
        run = sc["running"] if sc else "?"
        kv = f"{sc['kv']*100:.0f}%" if sc else "?"
        print(f"  t={time.time()-t0:5.1f}s gate={lim:3d} running={run} kv={kv}", flush=True)
        time.sleep(3.0)
    ctrl.stop()
    hist = getattr(ctrl, "history", [])
    limits = [h.get("limit_after") for h in hist if h.get("limit_after") is not None]
    backoffs = sum(1 for h in hist if h.get("action") == "backoff")
    print(f"aimd DONE: limits {limits[:1]}..{limits[-1:]} range[{min(limits) if limits else '-'},"
          f"{max(limits) if limits else '-'}] backoffs={backoffs} ticks={len(hist)}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
