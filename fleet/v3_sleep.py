"""Swarm v3 sleep consolidation sidecar.

Recent trap observations are compacted into stable avoid-rules. All public
operations are fail-soft and gated by ``SWARM_V3_SLEEP``.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

from . import v3

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


def _path() -> Path:
    raw = os.environ.get("SWARM_V3_SLEEP_PATH")
    if raw:
        return Path(raw).expanduser()
    return Path("~/.cache/swarm-agent/v3_sleep.json").expanduser()


def _threshold() -> int:
    try:
        return max(1, int(os.environ.get("SWARM_V3_SLEEP_THRESHOLD", "2")))
    except (TypeError, ValueError):
        return 2


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
        logger.debug("v3_sleep lock failed", exc_info=True)
        yield


def _load() -> Dict[str, Any]:
    path = _path()
    if not path.exists():
        return {"observations": {}, "rules": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("v3_sleep read failed: %s", e)
        return {"observations": {}, "rules": {}}
    if not isinstance(data, dict):
        return {"observations": {}, "rules": {}}
    if not isinstance(data.get("observations"), dict):
        data["observations"] = {}
    if not isinstance(data.get("rules"), dict):
        data["rules"] = {}
    return data


def _save(data: Dict[str, Any]) -> None:
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".v3_sleep_", suffix=".tmp")
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
        logger.debug("v3_sleep write failed: %s", e, exc_info=True)


def _clean(value: str, default: str = "default") -> str:
    text = str(value or "").strip()
    return text or default


def record_trap(domain: str, decoy_stance: str) -> None:
    if not v3.enabled("sleep"):
        return
    domain = _clean(domain)
    stance = _clean(decoy_stance, "")
    if not stance:
        return
    try:
        with _lock():
            data = _load()
            observations = data.setdefault("observations", {})
            domain_obs = observations.setdefault(domain, {})
            try:
                current = int(domain_obs.get(stance, 0) or 0)
            except (TypeError, ValueError):
                current = 0
            domain_obs[stance] = current + 1
            _save(data)
    except Exception as e:
        logger.debug("v3_sleep record failed: %s", e, exc_info=True)


def consolidate() -> int:
    if not v3.enabled("sleep"):
        return 0
    try:
        threshold = _threshold()
        with _lock():
            data = _load()
            observations = data.setdefault("observations", {})
            rules = data.setdefault("rules", {})
            for domain, stances in list(observations.items()):
                if not isinstance(stances, dict):
                    continue
                domain_rules = set(str(s) for s in rules.get(domain, []) if str(s))
                for stance, count in stances.items():
                    try:
                        n = int(count)
                    except (TypeError, ValueError):
                        n = 0
                    if n >= threshold and str(stance):
                        domain_rules.add(str(stance))
                rules[domain] = sorted(domain_rules)
            _save(data)
            return sum(len(v) for v in rules.values() if isinstance(v, list))
    except Exception as e:
        logger.debug("v3_sleep consolidate failed: %s", e, exc_info=True)
        return 0


def is_suppressed(domain: str, stance: str) -> bool:
    if not v3.enabled("sleep"):
        return False
    try:
        data = _load()
        rules = data.get("rules", {})
        domain_rules = rules.get(_clean(domain), [])
        return str(_clean(stance, "")) in {str(s) for s in domain_rules}
    except Exception as e:
        logger.debug("v3_sleep suppressed check failed: %s", e, exc_info=True)
        return False


def reset() -> None:
    if not v3.enabled("sleep"):
        return
    try:
        with _lock():
            path = _path()
            if path.exists():
                path.unlink()
    except Exception as e:
        logger.debug("v3_sleep reset failed: %s", e, exc_info=True)
