"""vLLM /metrics scraping + throughput / duty derivation.

The AIMD controller (fleet/admission.py) and the CLI progress line read these. All
ground-truth for the operating point comes from the server, not from guessing duty:
  running     == vllm:num_requests_running     (concurrent generations == what the gate pins)
  waiting     == vllm:num_requests_waiting      (queued, not yet decoding)
  kv          == vllm:kv_cache_usage_perc       (0..1; AIMD backs off above KV_HI)
  preemptions == vllm:num_preemptions_total      (cumulative; a jump => KV thrash)
  gen_tokens  == vllm:generation_tokens_total    (cumulative; differenced => tok/s)
  prefix hit  == prefix_cache_hits/queries_total (cumulative; verifies prefix-warm)
"""
from __future__ import annotations

import time
import urllib.request
from typing import Dict, Optional

from . import config

# Counts (num_requests_*) are summed across label sets — with >1 engine/replica the
# total concurrency is the sum. Percent/ratio gauges (kv usage, 0..1) are NOT summable:
# summing two label sets (or the kv_cache_usage_perc + gpu_cache_usage_perc *alias* for
# the SAME quantity) would push kv past 1.0 and pin the AIMD gate at MIN forever
# (review fix #1). We therefore MAX percent gauges across label sets and resolve the two
# kv aliases to ONE source.
# Engine-agnostic: vLLM and SGLang expose the same quantities under different metric names.
# We map BOTH name spaces to one internal key set so scrape() works against either backend
# (only one engine's names are present at a time, so the aliases never collide). SGLang names
# verified against the live sglang.launch_server /metrics on SM120 (sglang:* prefix).
_COUNT_GAUGES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "sglang:num_running_reqs": "running",   # SGLang == concurrent generations
    "sglang:num_queue_reqs": "waiting",     # SGLang queued (not yet decoding)
}
# percent/ratio gauges: take the MAX across label sets, never the sum.
_PERCENT_GAUGES = {
    "vllm:kv_cache_usage_perc": "kv",
    "vllm:gpu_cache_usage_perc": "kv",   # older vLLM name == same quantity (alias)
    "sglang:token_usage": "kv",          # SGLang KV-pool usage fraction (0..1)
}
# Preferred source per percent key, in priority order: the first present alias wins, so a
# second alias for the SAME quantity never double-counts. (kv_cache_usage_perc preferred;
# gpu_cache_usage_perc is the older name and only a fallback.)
_PERCENT_PREFERRED = {
    "kv": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "sglang:token_usage"),
}
_COUNTERS = {
    "vllm:num_preemptions_total": "preemptions",
    "vllm:generation_tokens_total": "gen_tokens",
    "vllm:prefix_cache_hits_total": "prefix_hits",
    "vllm:prefix_cache_queries_total": "prefix_queries",
    "sglang:generation_tokens_total": "gen_tokens",   # SGLang cumulative gen tokens
    "sglang:cached_tokens_total": "prefix_hits",       # best-effort prefix/radix-cache hits
    "sglang:prompt_tokens_total": "prefix_queries",    # denominator proxy (verify on live)
}


def scrape(url: str = None, timeout: float = 3.0) -> Optional[Dict[str, float]]:
    """Return a dict of the parsed metrics, or None if the server is unreachable.

    Counts (running/waiting) and cumulative counters sum across label sets; percent
    gauges (kv, 0..1) take the MAX across label sets and resolve aliases to one source
    (review fix #1 — summing them pinned the AIMD gate at MIN).
    """
    url = url or config.METRICS_URL
    try:
        raw = urllib.request.urlopen(url, timeout=timeout).read().decode()
    except Exception:
        return None
    out: Dict[str, float] = {}
    # per-percent-key per-metric-name running MAX across label sets, so we can later pick
    # the preferred alias and never sum two readings of the same 0..1 quantity.
    pct_by_name: Dict[str, float] = {}
    for line in raw.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            val = float(line.rsplit(None, 1)[-1])
        except ValueError:
            continue
        if name in _PERCENT_GAUGES:
            # MAX across label sets for this exact metric name (not sum).
            pct_by_name[name] = max(pct_by_name.get(name, val), val)
            continue
        key = _COUNT_GAUGES.get(name) or _COUNTERS.get(name)
        if key is None:
            continue
        # counts + cumulative counters: sum across label sets (multi-engine total).
        out[key] = out.get(key, 0.0) + val

    # resolve each percent key from its preferred alias (first present wins), so the
    # kv_cache_usage_perc + gpu_cache_usage_perc alias pair collapses to ONE value.
    for key, names in _PERCENT_PREFERRED.items():
        for name in names:
            if name in pct_by_name:
                out[key] = pct_by_name[name]
                break

    out.setdefault("running", 0.0)
    out.setdefault("waiting", 0.0)
    out.setdefault("kv", 0.0)
    out["t"] = time.time()
    return out


class ThroughputMeter:
    """Differences a cumulative completion-token counter over wall time → live tok/s.

    Two attribution modes (review fix #4):
      * SERVER-WIDE (default, ``update``): differences the server's
        ``vllm:generation_tokens_total``. Simple but it mixes the fleet with any
        co-tenant on the same vLLM (e.g. vLLM Studio), so the rate over-reports when
        someone else is generating. This stays as the fallback.
      * FLEET-ATTRIBUTED (``update_fleet``): differences a fleet-only cumulative token
        count fed by the caller (the controller/engine sums worker ``completion_tokens``).
        This is the number the AIMD loop actually wants — its own throughput, not the
        box's.
    Both share the same differencer, but the differencer remembers which token-space it
    is baselined on and RE-BASELINES (one None sample, no rate) when the source switches,
    so a fleet diff is only ever taken against a prior fleet sample and a server diff
    against a server sample — the two counter spaces (and their clocks) never mix.
    ``attribution`` records which one produced the last sample so the report can label it.

    Partial / unreadable samples (review fix #2): if a tick has no usable counter we
    ADVANCE the timestamp/counter baseline to that tick instead of leaving it stale, so
    the NEXT good diff spans only the real new interval — not the combined gap across the
    dropout (which would mis-state tok/s).
    """

    def __init__(self):
        self._last_tok: Optional[float] = None
        self._last_t: Optional[float] = None
        self._last_source: Optional[str] = None  # token-space the baseline belongs to
        self.attribution: str = "none"   # "server" | "fleet" | "none" (last sample's source)

    def _diff(self, tok: Optional[float], t: float, *, source: str) -> Optional[float]:
        """Core differencer. On a partial sample (tok is None) advance only the time
        baseline so we don't stretch the next interval across the gap (fix #2). On a
        SOURCE switch re-baseline (no rate) so we never difference fleet tokens against a
        server baseline (or vice-versa) — different counter spaces and clocks (fix #4)."""
        if tok is None:
            # Partial/None tick: keep the counter baseline (we have no new reading) but
            # move the clock forward so a later good diff measures a real interval.
            if self._last_t is not None:
                self._last_t = t
            return None
        tps = None
        same_source = (self._last_source == source)
        if (same_source and self._last_tok is not None
                and self._last_t is not None and t > self._last_t):
            dt = t - self._last_t
            dtok = tok - self._last_tok
            if dt > 0 and dtok >= 0:
                tps = dtok / dt
            elif dtok < 0:
                # counter went backwards (server restart) → re-baseline, no rate.
                tps = None
        # else: first sample, or a source switch → just adopt this as the new baseline.
        self._last_tok, self._last_t, self._last_source = tok, t, source
        if tps is not None:
            self.attribution = source
        return tps

    def update_source(self, tok: Optional[float], t: Optional[float] = None, *,
                      source: str = "server") -> Optional[float]:
        """Difference a cumulative counter from `source` over wall time. The caller is
        responsible for keeping `source` consistent with the counter space it passes; the
        meter re-baselines whenever `source` changes (fix #4)."""
        return self._diff(tok, t if t is not None else time.time(), source=source)

    def update(self, sc: Optional[Dict[str, float]]) -> Optional[float]:
        """Server-wide tok/s from ``gen_tokens`` (fallback attribution). Used by the CLI
        progress line, which only has the scrape dict."""
        t = (sc or {}).get("t", time.time())
        tok = sc.get("gen_tokens") if (sc and "gen_tokens" in sc) else None
        return self._diff(tok, t, source="server")

    def update_fleet(self, fleet_tokens: Optional[float],
                     t: Optional[float] = None) -> Optional[float]:
        """Fleet-attributed tok/s. ``fleet_tokens`` is the fleet's CUMULATIVE completion
        tokens (monotone); the caller sums worker ``completion_tokens`` as they finish.
        Differenced against the prior fleet sample (fix #4)."""
        return self.update_source(fleet_tokens, t, source="fleet")


class DutyIntegrator:
    """Time-weighted mean of `running` (concurrent decode) → fleet duty vs enrolled.

    duty = mean(running) / enrolled  — the fraction of enrolled workers actually decoding;
    closes DESIGN §6 '実 duty 未測'."""

    def __init__(self):
        self._area = 0.0     # ∫ running dt
        self._span = 0.0     # ∫ dt
        self._last_t: Optional[float] = None
        self._last_running: Optional[float] = None
        self.peak_running = 0.0

    def update(self, sc: Optional[Dict[str, float]]) -> None:
        if not sc:
            return
        t, r = sc.get("t", time.time()), sc.get("running", 0.0)
        self.peak_running = max(self.peak_running, r)
        if self._last_t is not None and t > self._last_t:
            dt = t - self._last_t
            # trapezoidal: average of the two endpoint running counts
            self._area += 0.5 * (r + (self._last_running or r)) * dt
            self._span += dt
        self._last_t, self._last_running = t, r

    def mean_running(self) -> float:
        return self._area / self._span if self._span > 0 else (self._last_running or 0.0)

    def duty(self, enrolled: float) -> Optional[float]:
        if enrolled <= 0:
            return None
        return self.mean_running() / enrolled


def prefix_hit_rate(before: Dict[str, float], after: Dict[str, float]) -> Optional[float]:
    """Hit rate over the window [before, after] (verifies prefix-warm carry-over)."""
    if not before or not after:
        return None
    dq = after.get("prefix_queries", 0.0) - before.get("prefix_queries", 0.0)
    dh = after.get("prefix_hits", 0.0) - before.get("prefix_hits", 0.0)
    return (dh / dq) if dq > 0 else None
