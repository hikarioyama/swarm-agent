"""Pure Swarm v3 commit-1 mechanics.

This module is intentionally import-safe: stdlib only, no fleet/swarm imports, and all
features default off behind cached environment flags.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional


_TRUTHY = {"1", "true", "yes", "on", "y", "t"}
_RESERVED = {"hebbian", "sleep"}


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def halflife() -> float:
    return max(0.001, _env_float("SWARM_V3_HALFLIFE", 3600.0))


def min_diversity() -> float:
    return _env_float("SWARM_V3_MIN_DIVERSITY", 0.34)


def accept_conf() -> float:
    return _env_float("SWARM_V3_ACCEPT_CONF", 0.6)


def max_rounds() -> int:
    return max(0, _env_int("SWARM_V3_MAX_ROUNDS", 2))


@functools.lru_cache(maxsize=1)
def _flag_snapshot() -> Dict[str, bool]:
    master = _truthy(os.environ.get("SWARM_V3"))
    flags: Dict[str, bool] = {}
    for name in ("chemical", "diversity", "reflex"):
        flags[name] = master and _truthy(os.environ.get(f"SWARM_V3_{name.upper()}"))
    for name in _RESERVED:
        flags[name] = False
    return flags


def reset_flags_cache() -> None:
    """Clear cached env flags. Intended for deterministic tests."""
    _flag_snapshot.cache_clear()


def enabled(name: str) -> bool:
    return bool(_flag_snapshot().get(str(name).strip().lower(), False))


def any_on() -> bool:
    flags = _flag_snapshot()
    return any(flags.get(name, False) for name in ("chemical", "diversity", "reflex"))


_FENCE_RE = re.compile(r"```(?:json|chem|chemical)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
}


def canonical_stance(hypothesis: str) -> str:
    text = _WS_RE.sub(" ", str(hypothesis or "").strip().lower())
    tokens = [t for t in _TOKEN_RE.findall(text) if t not in _STOP]
    if not tokens:
        tokens = _TOKEN_RE.findall(text) or ["empty"]
    salient = " ".join(tokens[:12])
    digest = hashlib.sha1(salient.encode("utf-8")).hexdigest()[:12]
    return f"h:{digest}"


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if isinstance(v, (str, int, float))]


def parse_chem(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first fenced JSON object in worker text; invalid/missing is None."""
    try:
        match = _FENCE_RE.search(text or "")
        if not match:
            return None
        raw = json.loads(match.group(1))
        if not isinstance(raw, dict):
            return None
        raw_hypothesis = raw.get("hypothesis")
        raw_stance = raw.get("stance_hash")
        hypothesis = raw_hypothesis.strip() if isinstance(raw_hypothesis, str) else ""
        stance = raw_stance.strip() if isinstance(raw_stance, str) else ""
        if not hypothesis and not stance:
            return None
        raw_confidence = raw.get("confidence")
        if (
            "confidence" not in raw
            or isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
            or not math.isfinite(float(raw_confidence))
        ):
            return None
        if not stance:
            stance = canonical_stance(hypothesis)
        confidence = max(0.0, min(1.0, float(raw_confidence)))
        return {
            "hypothesis": hypothesis,
            "stance_hash": stance,
            "evidence_ids": _string_list(raw.get("evidence_ids")),
            "confidence": confidence,
            "contradictions": _string_list(raw.get("contradictions")),
            "toxins": _string_list(raw.get("toxins")),
        }
    except Exception:
        return None


def _stance(chem: Dict[str, Any]) -> str:
    stance = str((chem or {}).get("stance_hash") or "").strip()
    if stance:
        return stance
    return canonical_stance(str((chem or {}).get("hypothesis") or ""))


def stance_diversity(signals: Iterable[Dict[str, Any]]) -> float:
    counts: Dict[str, int] = {}
    total = 0
    for sig in signals or []:
        if not isinstance(sig, dict):
            continue
        counts[_stance(sig)] = counts.get(_stance(sig), 0) + 1
        total += 1
    if total <= 0:
        return 0.0
    return 1.0 - sum((n / total) ** 2 for n in counts.values())


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def priority_score(task_meta: Optional[Dict[str, Any]], *, now: Optional[float] = None,
                   crowding: float = 0.0) -> float:
    meta = task_meta if isinstance(task_meta, dict) else {}
    chem = meta.get("chem") if isinstance(meta.get("chem"), dict) else {}
    source = dict(chem)
    source.update({k: v for k, v in meta.items() if k in {
        "strength", "uncertainty", "novelty", "last_reinforced",
    }})

    strength = _num(source.get("strength"), 1.0)
    uncertainty = _num(source.get("uncertainty"), 0.0)
    novelty = _num(source.get("novelty"), 0.0)
    last_reinforced = _num(source.get("last_reinforced"), now if now is not None else time.time())
    ts = time.time() if now is None else float(now)
    age = max(0.0, ts - last_reinforced)
    decayed_strength = strength * math.exp(-age / halflife())
    return decayed_strength * (1.0 + uncertainty) * (1.0 + novelty) / (1.0 + max(0.0, crowding))


def quorum_decision(signals: Iterable[Dict[str, Any]], *, min_diversity: Optional[float] = None,
                    accept_conf: Optional[float] = None, max_rounds: Optional[int] = None,
                    rounds_done: int = 0) -> str:
    sigs = [s for s in (signals or []) if isinstance(s, dict)]
    if not sigs:
        return "insufficient"
    min_div = min_diversity if min_diversity is not None else globals()["min_diversity"]()
    acc = accept_conf if accept_conf is not None else globals()["accept_conf"]()
    max_r = max_rounds if max_rounds is not None else globals()["max_rounds"]()

    conf_by_stance: Dict[str, float] = {}
    for sig in sigs:
        conf_by_stance[_stance(sig)] = max(conf_by_stance.get(_stance(sig), 0.0),
                                           _num(sig.get("confidence"), 0.0))
    clears = any(conf >= acc for conf in conf_by_stance.values())
    if clears and stance_diversity(sigs) >= min_div:
        return "accept"
    if rounds_done < max_r:
        return "need_diversity"
    return "insufficient"


def order_by_priority(tasks: Iterable[Any], *, now: Optional[float] = None) -> List[Any]:
    items = list(tasks or [])
    if len(items) < 2:
        return items

    ts = time.time() if now is None else float(now)
    stance_counts: Dict[str, int] = {}
    for t in items:
        meta = getattr(t, "meta", None)
        chem = meta.get("chem") if isinstance(meta, dict) and isinstance(meta.get("chem"), dict) else None
        if chem:
            stance_counts[_stance(chem)] = stance_counts.get(_stance(chem), 0) + 1

    scored = []
    for idx, t in enumerate(items):
        meta = getattr(t, "meta", None)
        chem = meta.get("chem") if isinstance(meta, dict) and isinstance(meta.get("chem"), dict) else None
        crowd = float(max(0, stance_counts.get(_stance(chem), 1) - 1)) if chem else 0.0
        scored.append((priority_score(meta if isinstance(meta, dict) else {}, now=ts, crowding=crowd), idx, t))

    first = scored[0][0]
    if all(score == first for score, _, _ in scored):
        return items
    return [t for _, _, t in sorted(scored, key=lambda row: (-row[0], row[1]))]
