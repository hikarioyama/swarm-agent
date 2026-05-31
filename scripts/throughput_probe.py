"""Windowed aggregate-throughput probe for ONE fixed gate N (self-terminating).

Measures the operating point the FLEET_OPTIMUM way — a time-average of the server's
generation throughput over a steady window — instead of waiting for verbose Step-3.7
tasks to complete. Enrolls 2N worker threads through the DecodeGate; the gate pins
concurrent generations to N; a sampler integrates running + differences gen_tokens
over [warm, warm+measure]; then os._exit abandons the in-flight threads (no clean join
needed — we only want the window's metrics).

    PYTHONPATH=. <venv>/python scripts/throughput_probe.py --gate 32 --warm 10 --measure 30
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, ".")
# long single generations keep the server saturated through the window with minimal
# re-prefill churn; cap high + neutralise truncation so a task never multi-continues.
os.environ.setdefault("FLEET_MAX_TOKENS", "8192")
os.environ.setdefault("FLEET_NO_CONTINUE", "1")
from fleet import compat, config, metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=int, required=True)
    ap.add_argument("--tasks", default="examples/sweep_8k.jsonl")
    ap.add_argument("--warm", type=float, default=10.0)
    ap.add_argument("--measure", type=float, default=30.0)
    ap.add_argument("--enroll-mult", type=float, default=2.0)
    a = ap.parse_args()

    N = a.gate
    prompts = [json.loads(l)["prompt"] for l in open(a.tasks)]
    gate = compat.DecodeGate(N)
    compat.apply(gate)
    compat.prewarm([[]])

    stop = threading.Event()
    counter = {"i": 0}
    clock = threading.Lock()

    def worker():
        while not stop.is_set():
            with clock:
                i = counter["i"]; counter["i"] += 1
            ag = compat.make_agent("reducer", task_id=f"tp{i}")
            try:
                ag.run_conversation(prompts[i % len(prompts)], task_id=f"tp{i}")
            except Exception:
                pass

    enroll = int(a.enroll_mult * N)
    for _ in range(enroll):
        threading.Thread(target=worker, daemon=True).start()

    # sample the server through warm + measure
    samples = []  # (t, running, kv, gen_tokens)
    t0 = time.time()
    end = t0 + a.warm + a.measure
    while time.time() < end:
        sc = metrics.scrape(timeout=2.0)
        if sc:
            samples.append((sc["t"], sc["running"], sc["kv"], sc.get("gen_tokens", 0.0)))
        time.sleep(1.0)
    stop.set()

    lo = t0 + a.warm
    win = [s for s in samples if s[0] >= lo]
    if len(win) < 2:
        print(json.dumps({"gate": N, "error": "too few samples", "n": len(win)}))
        os._exit(0)
    tok_s = (win[-1][3] - win[0][3]) / (win[-1][0] - win[0][0])
    area = span = 0.0
    for (ta, ra, _, _), (tb, rb, _, _) in zip(win, win[1:]):
        dt = tb - ta; area += 0.5 * (ra + rb) * dt; span += dt
    mean_run = area / span if span else win[-1][1]
    kv_peak = max(s[2] for s in win)
    out = {"gate": N, "enroll": enroll, "tok_s": round(tok_s),
           "mean_running": round(mean_run, 2), "occupancy": round(mean_run / N, 3),
           "tok_s_per_agent": round(tok_s / mean_run, 1) if mean_run else None,
           "kv_peak_pct": round(kv_peak * 100, 1), "samples": len(win)}
    print("RESULT " + json.dumps(out), flush=True)
    os._exit(0)  # abandon in-flight worker threads; we have the window's metrics


if __name__ == "__main__":
    main()
