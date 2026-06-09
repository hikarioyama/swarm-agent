"""Swarm v3 Hebbian credit sidecar.

The sidecar reinforces verified-correct (profile, domain) routes. All public
operations are fail-soft and gated by ``SWARM_V3_HEBBIAN``.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from . import v3

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def _path() -> Path:
    raw = os.environ.get("SWARM_V3_CREDIT_PATH")
    if raw:
        return Path(raw).expanduser()
    return Path("~/.cache/swarm-agent/v3_credit.json").expanduser()


@contextmanager
def _lock():
    path = _path()
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as fd:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
    except Exception:
        logger.debug("v3_credit lock failed", exc_info=True)
        yield


def _load() -> Dict[str, Any]:
    path = _path()
    if not path.exists():
        return {"routes": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("v3_credit read failed: %s", e)
        return {"routes": {}}
    if not isinstance(data, dict):
        return {"routes": {}}
    routes = data.get("routes")
    if not isinstance(routes, dict):
        data["routes"] = {}
    return data


def _save(data: Dict[str, Any]) -> None:
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".v3_credit_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("v3_credit write failed: %s", e, exc_info=True)


def _clean(value: str, default: str = "default") -> str:
    text = str(value or "").strip()
    return text or default


def bump_credit(profile: str, domain: str, *, amount: float = 1.0) -> None:
    if not v3.enabled("hebbian"):
        return
    profile = _clean(profile, "")
    domain = _clean(domain)
    if not profile:
        return
    try:
        delta = float(amount)
    except (TypeError, ValueError):
        return
    if delta <= 0:
        return
    try:
        with _lock():
            data = _load()
            routes = data.setdefault("routes", {})
            prof = routes.setdefault(profile, {})
            current = prof.get(domain, 0.0)
            try:
                current = float(current)
            except (TypeError, ValueError):
                current = 0.0
            prof[domain] = current + delta
            _save(data)
    except Exception as e:
        logger.debug("v3_credit bump failed: %s", e, exc_info=True)


def credit_score(profile: str, domain: str) -> float:
    if not v3.enabled("hebbian"):
        return 0.0
    try:
        data = _load()
        raw = data.get("routes", {}).get(_clean(profile, ""), {}).get(_clean(domain), 0.0)
        score = float(raw)
        return score if score > 0 else 0.0
    except Exception as e:
        logger.debug("v3_credit score failed: %s", e, exc_info=True)
        return 0.0


def order_profiles(candidates: List[str], domain: str) -> List[str]:
    items = list(candidates or [])
    if not v3.enabled("hebbian") or len(items) < 2:
        return items
    scored = [(credit_score(profile, domain), idx, profile) for idx, profile in enumerate(items)]
    if all(score == scored[0][0] for score, _, _ in scored):
        return items
    return [profile for _, _, profile in sorted(scored, key=lambda row: (-row[0], row[1]))]


def reset() -> None:
    if not v3.enabled("hebbian"):
        return
    try:
        with _lock():
            path = _path()
            if path.exists():
                path.unlink()
    except Exception as e:
        logger.debug("v3_credit reset failed: %s", e, exc_info=True)
