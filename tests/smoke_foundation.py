"""Foundation smoke test — run with the HermesAgent venv python:

    cd ~/projects/step37-harness
    PYTHONPATH=. /home/hikari/.hermes/hermes-agent/venv/bin/python tests/smoke_foundation.py

Validates, before any higher layer is built on it:
  1. DecodeGate limit + lane-priority + resize (pure, no server).
  2. metrics.scrape() against the live server.
  3. compat.apply()+prewarm()+make_agent(): ONE real agent call end-to-end, asserting
     decode_s was captured and the server's num_requests_running rose during the call.
"""
import sys
import threading
import time

sys.path.insert(0, ".")
from fleet import compat, metrics, config  # noqa: E402


def test_decode_gate():
    g = compat.DecodeGate(limit=2)
    order = []
    olock = threading.Lock()
    started = threading.Event()

    def hold(lane, dur):
        with g.acquire(lane):
            with olock:
                order.append(lane)
            started.set()
            time.sleep(dur)

    # saturate the 2 permits with two workers
    a = threading.Thread(target=hold, args=("worker", 0.4))
    b = threading.Thread(target=hold, args=("worker", 0.4))
    a.start(); b.start()
    started.wait(1.0)
    time.sleep(0.05)
    assert g.stats()["in_flight"] == 2, g.stats()
    # now queue a low-priority router and a high-priority director; director must go first
    lo = threading.Thread(target=hold, args=("router", 0.05))
    time.sleep(0.01)
    hi = threading.Thread(target=hold, args=("director", 0.05))
    lo.start(); time.sleep(0.02); hi.start()
    for t in (a, b, lo, hi):
        t.join(3.0)
    # director enqueued AFTER router but higher priority => served before router
    assert order[2] == "director", f"priority broken: {order}"
    # resize down then up doesn't deadlock
    g.set_limit(1); assert g.get_limit() == 1
    g.set_limit(8); assert g.get_limit() == 8
    print("  [1] DecodeGate priority + resize OK:", order, g.stats())


def test_metrics():
    sc = metrics.scrape()
    assert sc is not None, "server unreachable at " + config.METRICS_URL
    assert "running" in sc and "kv" in sc, sc
    tm = metrics.ThroughputMeter(); tm.update(sc)
    print(f"  [2] metrics OK: running={sc['running']} kv={sc['kv']:.3f} "
          f"gen_tokens={sc.get('gen_tokens')} preempt={sc.get('preemptions')}")


def test_real_agent():
    gate = compat.DecodeGate(limit=4)
    compat.apply(gate)
    compat.prewarm(list(config.TOOL_PROFILES.values()))

    peak = {"running": 0.0}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            sc = metrics.scrape(timeout=2.0)
            if sc:
                peak["running"] = max(peak["running"], sc["running"])
            time.sleep(0.2)

    s = threading.Thread(target=sampler, daemon=True); s.start()
    agent = compat.make_agent("worker", task_id="smoke")
    t0 = time.time()
    result = agent.run_conversation(
        "Reply with exactly the single word: ok", task_id="smoke-1")
    wall = time.time() - t0
    stop.set(); s.join(1.0)

    decode_s = getattr(agent, "_fleet_decode_s", 0.0)
    text = ""
    for m in reversed(result.get("messages") or []):
        if m.get("role") == "assistant":
            c = m.get("content")
            text = c if isinstance(c, str) else "".join(
                b.get("text", "") for b in c if isinstance(b, dict))
            break
    print(f"  [3] real agent OK: completed={result.get('completed')} wall={wall:.2f}s "
          f"decode_s={decode_s:.2f}s gatewait_s={getattr(agent,'_fleet_gatewait_s',0):.2f}s "
          f"peak_running={peak['running']} api_calls={result.get('api_calls')} "
          f"reply={text!r}")
    assert decode_s > 0, "forwarder patch did not capture decode_s"
    assert peak["running"] >= 1, "server never showed a running request (gate/agent path?)"


if __name__ == "__main__":
    print("foundation smoke:")
    test_decode_gate()
    test_metrics()
    test_real_agent()
    print("ALL FOUNDATION SMOKE PASSED")
