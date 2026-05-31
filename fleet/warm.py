"""Prefix-warm each distinct role profile so worker #1 of every role hits the cache.

WHY this exists (BUILD_SPEC §3, acceptance criterion F):
HermesAgent is stateless — every turn re-sends the WHOLE transcript, so each worker
re-pays its system+tools prefix on every generation (config.TOOL_PROFILES: the default
39-tool prefill is ~14,113 tok). vLLM auto prefix-caches a byte-identical prefix, but the
FIRST request of a given prefix pays full prefill and only POPULATES the cache. By firing
one tiny request per distinct role profile up-front, we pay that cold prefill ONCE here,
single-threaded, so the first REAL worker of each role lands on a warm prefix-cache hit
(observable as `vllm:prefix_cache_hits_total` rising — see `metrics.prefix_hit_rate`).

The cacheable prefix is determined by the role's *toolset* (config.TOOL_PROFILES) plus the
fixed fleet-safe agent construction (skip_context_files / skip_memory => no per-agent
SOUL.md/AGENTS.md bytes). Two roles with the same toolset signature share the same prefix,
so we DEDUPE by toolset signature and warm each distinct prefix exactly once.

Idempotent and fast: ~one tiny "Reply ok" generation per DISTINCT profile (config currently
has 3 distinct toolset signatures across 7 roles). Robust: a role that errors records None
and we keep going — warming is best-effort and must never block fleet start-up.

Run standalone to warm everything and see the server-side prefix-cache delta::

    PYTHONPATH=. /home/hikari/.hermes/hermes-agent/venv/bin/python -m fleet.warm
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

from . import compat, config, metrics


def _toolset_signature(role: str) -> tuple:
    """The cache-key of a role's prefix == its (order-insensitive) toolset.

    config.toolsets_for falls back to the 'worker' profile for unknown lanes, matching
    exactly what compat.make_agent will build, so roles that resolve to the same toolset
    share a byte-identical server prefix and only need warming once.
    """
    return tuple(sorted(config.toolsets_for(role)))


def _warm_one(role: str, *, base_url: Optional[str], model: Optional[str],
              api_key: Optional[str]) -> Optional[float]:
    """Send ONE tiny request through the real agent path for `role`.

    Returns the wall-clock latency in seconds, or None if anything went wrong (the
    server is down, the toolset failed to resolve, the model erred…). We swallow the
    error on purpose: a missing warm-up only costs the first real worker a cold prefill,
    it must never abort fleet start-up.
    """
    t0 = time.perf_counter()
    try:
        agent = compat.make_agent(
            role, base_url=base_url, model=model, api_key=api_key,
            task_id=f"warm-{role}",
        )
        # Tiny prompt + the bounded max_iterations make this a single short generation
        # whose only job is to PASS the role's system+tools prefix through the server so
        # vLLM caches it. We don't care about the answer, only that the prefix is resident.
        agent.run_conversation("Reply ok", task_id=f"warm-{role}")
        return time.perf_counter() - t0
    except Exception:
        return None


def warm_profiles(roles: Sequence[str], *, base_url: str = None, model: str = None,
                  api_key: str = None) -> Dict[str, float]:
    """Prefix-warm each DISTINCT role profile in `roles`; return {role: warm_latency_s}.

    For every role we resolve its toolset signature and warm each distinct signature once
    (the first role seen for a signature is the one actually sent; the rest inherit its
    latency, since they share the identical server prefix). The order of `roles` therefore
    decides which role "owns" each shared prefix — pass them most-important-first if it
    matters (it usually doesn't, the prefix is identical).

    Always runs `compat.apply()` (ensure forwarders patched + env hygiene, WITHOUT touching
    the installed gate) and `compat.prewarm(...)` first, so tool-definition / OpenAI-import
    caches are warm BEFORE any request — both kills the cold-start stampede (TS3) and keeps
    the measured warm latency about prefill, not Python import cost. Using `apply()` (no arg)
    rather than `apply(None)` means calling warm AFTER the engine installed the real gate does
    NOT silently disable gating.

    Idempotent: re-running just re-confirms the cache is warm (a fast hit each time).
    Robust: a role that errors gets a None entry and warming continues.
    """
    roles = list(roles)
    profiles: List[List[str]] = [list(config.toolsets_for(r)) for r in roles]

    # One-time, single-threaded foundation warm-up (safe to call repeatedly — both are
    # idempotent). apply() = ensure forwarders patched WITHOUT touching the installed gate.
    compat.apply()
    compat.prewarm(profiles)

    out: Dict[str, float] = {}
    seen: Dict[tuple, str] = {}  # toolset signature -> the role that warmed it
    for role in roles:
        sig = _toolset_signature(role)
        if sig in seen:
            # Same prefix already warmed by an earlier role → inherit its latency
            # (a real request now would be a cache hit; no need to spend another).
            out[role] = out.get(seen[sig])
            continue
        seen[sig] = role
        out[role] = _warm_one(role, base_url=base_url, model=model, api_key=api_key)
    return out


def _fmt(v: Optional[float], width: int = 9) -> str:
    return ("—".rjust(width) if v is None else f"{v:>{width}.3f}")


def main() -> int:
    """Warm every config.TOOL_PROFILES role; print a table + the server prefix-cache delta."""
    roles = list(config.TOOL_PROFILES.keys())

    before = metrics.scrape() or {}
    latencies = warm_profiles(roles)
    after = metrics.scrape() or {}

    # Re-derive the signatures so the table can show which roles SHARED a warmed prefix.
    sig_owner: Dict[tuple, str] = {}
    for r in roles:
        sig_owner.setdefault(_toolset_signature(r), r)

    print("\nprefix-warm — one tiny request per DISTINCT role profile")
    print(f"  server : {config.BASE_URL}   model: {config.MODEL}")
    print()
    print(f"  {'role':<10} {'tools':<34} {'warm_s':>9}  note")
    print(f"  {'-'*10} {'-'*34} {'-'*9}  {'-'*24}")
    for role in roles:
        sig = _toolset_signature(role)
        owner = sig_owner.get(sig, role)
        toolset = ", ".join(config.toolsets_for(role)) or "(none)"
        note = "warmed" if owner == role else f"shares prefix w/ '{owner}'"
        print(f"  {role:<10} {toolset:<34} {_fmt(latencies.get(role))}  {note}")

    distinct = len(sig_owner)
    failed = [r for r, v in latencies.items() if v is None and sig_owner.get(_toolset_signature(r)) == r]
    print()
    print(f"  distinct profiles warmed : {distinct}  (roles total: {len(roles)})")
    if failed:
        print(f"  WARN: profiles that errored (recorded None): {', '.join(failed)}")

    # Server-side prefix-cache hit-rate over the warm window (verifies criterion F).
    rate = metrics.prefix_hit_rate(before, after)
    dq = (after.get("prefix_queries", 0.0) - before.get("prefix_queries", 0.0))
    dh = (after.get("prefix_hits", 0.0) - before.get("prefix_hits", 0.0))
    print()
    if rate is None:
        print("  prefix-cache : no prefix_cache_* metrics exposed (or no queries in window)")
    else:
        print(f"  prefix-cache : Δqueries={dq:.0f}  Δhits={dh:.0f}  hit-rate={rate*100:.1f}%")
        print("                 (low on a COLD warm = expected: warming POPULATES the cache;")
        print("                  the first real worker of each role is what reads the hit)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
